import os
import glob
import json
import pickle
import random
from typing import List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow as pa
import pyarrow.csv
import pyarrow.parquet as pq
from tqdm import tqdm
import math

def split_csv_to_parquet(
        csv_path: str,
        out_dir: str,
        id_col: str = "ID",
        timestamp_col: str = "ExecutionTime",
        target_cols: List[str] = ["high", "low", "close", "volume"],
        chunksize: int = 200_000,
        force: bool = False,
):
    """
    Converts a large CSV file into a directory of smaller, per-asset Parquet files.
    """
    os.makedirs(out_dir, exist_ok=True)
    assets_seen = set()
    temp_dir = os.path.join(out_dir, "_append")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"[split] streaming CSV {csv_path} in chunksize={chunksize}")

    total_rows = sum(1 for _ in open(csv_path)) - 1
    total_chunks = math.ceil(total_rows / chunksize)
    print(f"[split] Found {total_rows} rows in {total_chunks} chunks.")

    # --- PASS 1: Stream the large CSV and split into temporary per-asset CSVs. ---
    chunk_iterator = pd.read_csv(
        csv_path,
        parse_dates=[timestamp_col],
        chunksize=chunksize
    )

    for chunk in tqdm(chunk_iterator, total=total_chunks, desc="[split] PASS 1: Reading CSV"):
        chunk[timestamp_col] = pd.to_datetime(chunk[timestamp_col], utc=True)
        store_cols = [timestamp_col] + target_cols
        for asset, g in chunk.groupby(id_col):
            assets_seen.add(str(asset))
            temp_path = os.path.join(temp_dir, f"{asset}.csv")
            header = not os.path.exists(temp_path)
            g[store_cols].to_csv(temp_path, mode="a", header=header, index=False)

    print("[split] PASS 1 complete.")
    
    # --- PASS 2: Convert each temporary CSV to a sorted, compressed Parquet file. ---
    print(f"[split] PASS 2: converting {len(assets_seen)} temp CSVs to sorted parquet...")
    assets = sorted(list(assets_seen))

    convert_options = pa.csv.ConvertOptions(
        column_types={timestamp_col: pa.timestamp('ns', tz='UTC')},
        include_columns=[timestamp_col] + target_cols
    )
    read_options = pa.csv.ReadOptions(autogenerate_column_names=False)

    for asset in tqdm(assets, desc="[split] Converting assets to parquet"):
        out_path = os.path.join(out_dir, f"{asset}.parquet")
        if os.path.exists(out_path) and not force:
            continue
        tmp_csv = os.path.join(temp_dir, f"{asset}.csv")
        if not os.path.exists(tmp_csv):
            continue
        
        # Use pyarrow for fast CSV reading, with a pandas fallback for robustness.
        try:
            table = pa.csv.read_csv(tmp_csv, read_options=read_options, convert_options=convert_options)
        except Exception as e:
            print(f"[split] pyarrow failed for {tmp_csv}: {e}. Falling back to pandas.")
            df = pd.read_csv(tmp_csv, parse_dates=[timestamp_col])
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
            df = df[[timestamp_col] + target_cols]
            table = pa.Table.from_pandas(df)
        
        # Sort by timestamp before saving to ensure data is chronological.
        sorted_table = table.sort_by([(timestamp_col, 'ascending')])
        pq.write_table(sorted_table, out_path, compression='snappy')
        os.remove(tmp_csv)
    
    # Clean up the temporary directory.
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
        
    # Save the list of processed assets for later use.
    final_assets = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(out_dir, "*.parquet"))]
    assets_json = os.path.join(out_dir, "assets.json")
    with open(assets_json, "w") as f:
        json.dump(final_assets, f)
    print(f"[split] created {len(final_assets)} per-asset parquet files in {out_dir}")
    return final_assets

class ElectricityPriceIterableDataset(IterableDataset):
    """
    A streaming dataset that reads and yields time series windows from pre-processed
    Parquet files.

    Yields:
        A tuple containing:
        - past (torch.Tensor): The input window. Shape: (input_chunk_length, num_features).
        - past_mask (torch.Tensor): The mask for the input window. Shape: (input_chunk_length, num_features).
        - future (torch.Tensor): The target window. Shape: (output_chunk_length, num_features).
        - future_mask (torch.Tensor): The mask for the target window. Shape: (output_chunk_length, num_features).
        - asset_id (str): The ID of the asset.
        - past_ts (np.ndarray): The timestamps for the input window.
        - past_covariates (torch.Tensor, optional): Past covariates. Shape: (input_chunk_length, num_past_covariates).
        - future_covariates (torch.Tensor, optional): Future covariates. Shape: (output_chunk_length, num_future_covariates).
    """
    def __init__(
            self,
            parquet_dir: str,
            scalers_dir: str,
            assets_list: Optional[List[str]] = None,
            input_chunk_length: int = 48,
            output_chunk_length: int = 10,
            target_cols: List[str] = ["high", "low", "close", "volume"],
            timestamp_col: str = "ExecutionTime",
            shuffle_buffer: int = 0,
            stride: int = 1,
            min_length: Optional[int] = None,
            return_covariates: bool = False,
            past_covariate_cols: Optional[List[str]] = None,
            future_covariate_cols: Optional[List[str]] = None,
            past_time_covariate_cols: Optional[List[str]] = None,
    ):
        super().__init__()
        self.parquet_dir = parquet_dir
        self.scalers_dir = scalers_dir
        assets_json = os.path.join(parquet_dir, "assets.json")
        if assets_list is None:
            if os.path.exists(assets_json):
                with open(assets_json, "r") as f:
                    self.assets = json.load(f)
            else:
                self.assets = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(parquet_dir, "*.parquet"))]
        else:
            self.assets = assets_list

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.target_cols = target_cols
        self.timestamp_col = timestamp_col
        self.shuffle_buffer = shuffle_buffer
        self.stride = stride
        self.min_length = min_length or (input_chunk_length + output_chunk_length)
        self.asset_paths = {a: os.path.join(parquet_dir, f"{a}.parquet") for a in self.assets}
        self._epoch = 0
        self.return_covariates = return_covariates
        self.past_covariate_cols = past_covariate_cols or []
        self.future_covariate_cols = future_covariate_cols or []
        self.past_time_covariate_cols = past_time_covariate_cols or []

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _get_worker_cache(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        cache_attr = f"_worker_cache_{worker_id}"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {"scalers": {}})
        return getattr(self, cache_attr)

    def _load_scaler_for_asset(self, asset: str):
        cache = self._get_worker_cache()
        scaler_map = cache["scalers"]
        if asset in scaler_map:
            return scaler_map[asset]
        path = os.path.join(self.scalers_dir, f"{asset}_scaler.joblib")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    scaler = pickle.load(f)
                scaler_map[asset] = scaler
                return scaler
            except Exception:
                scaler_map[asset] = None
                return None
        else:
            scaler_map[asset] = None
            return None

    def _iter_for_assets(self, assets_subset: List[str]):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = 1234 + self._epoch * 100 + worker_id
        rnd = random.Random(seed)
        buffer: List[Tuple[Any, ...]] = []

        for asset in assets_subset:
            path = self.asset_paths.get(asset)
            if not path or not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if self.timestamp_col not in df.columns:
                continue
                
            arr = df[self.target_cols].values.astype(np.float32)
            timestamps = df[self.timestamp_col].values
            n = arr.shape[0]
            if n < self.min_length:
                continue

            trading_mask = df["is_trading"].values.astype(np.float32)

            scaler = self._load_scaler_for_asset(asset)
            L_in = self.input_chunk_length
            L_out = self.output_chunk_length

            for start in range(0, n - L_in - L_out + 1, self.stride):
                x = arr[start : start + L_in]
                y = arr[start + L_in : start + L_in + L_out]
                past_ts = timestamps[start : start + L_in]

                past_trading = trading_mask[start : start + L_in]
                future_trading = trading_mask[start + L_in : start + L_in + L_out]

                if scaler is not None:
                    try:
                        x = scaler.transform(x)
                        y = scaler.transform(y)
                    except Exception:
                        pass

                past_mask = np.repeat(past_trading.reshape(-1, 1), repeats=len(self.target_cols), axis=1).astype(np.float32)
                future_mask = np.repeat(future_trading.reshape(-1, 1), repeats=len(self.target_cols), axis=1).astype(np.float32)

                # Extract covariates if requested
                if self.return_covariates:
                    # Extract past covariates (market features)
                    past_cov_list = []
                    for col in self.past_covariate_cols:
                        if col in df.columns:
                            past_cov_list.append(df[col].values[start : start + L_in])
                    past_covariates = np.stack(past_cov_list, axis=1).astype(np.float32) if past_cov_list else np.zeros((L_in, 0), dtype=np.float32)

                    # Extract past time covariates (time features for past window)
                    past_time_cov_list = []
                    for col in self.past_time_covariate_cols:
                        if col in df.columns:
                            past_time_cov_list.append(df[col].values[start : start + L_in])
                    past_time_covariates = np.stack(past_time_cov_list, axis=1).astype(np.float32) if past_time_cov_list else np.zeros((L_in, 0), dtype=np.float32)

                    # Extract future covariates (time features for future window)
                    future_cov_list = []
                    for col in self.future_covariate_cols:
                        if col in df.columns:
                            future_cov_list.append(df[col].values[start + L_in : start + L_in + L_out])
                    future_covariates = np.stack(future_cov_list, axis=1).astype(np.float32) if future_cov_list else np.zeros((L_out, 0), dtype=np.float32)

                    sample = (torch.from_numpy(x).float(),
                              torch.from_numpy(past_mask).float(),
                              torch.from_numpy(y).float(),
                              torch.from_numpy(future_mask).float(),
                              asset,
                              past_ts,
                              torch.from_numpy(past_covariates).float(),
                              torch.from_numpy(past_time_covariates).float(),
                              torch.from_numpy(future_covariates).float())
                else:
                    sample = (torch.from_numpy(x).float(),
                              torch.from_numpy(past_mask).float(),
                              torch.from_numpy(y).float(),
                              torch.from_numpy(future_mask).float(),
                              asset,
                              past_ts)

                if self.shuffle_buffer > 0:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer:
                        idx = rnd.randrange(len(buffer))
                        yield buffer.pop(idx)
                else:
                    yield sample
        
        while self.shuffle_buffer > 0 and len(buffer) > 0:
            idx = rnd.randrange(len(buffer))
            yield buffer.pop(idx)

    def __iter__(self):
        assets_to_process = list(self.assets)
        rnd = random.Random(1234 + self._epoch)
        rnd.shuffle(assets_to_process)
        worker_info = get_worker_info()
        if worker_info is None:
            assets_subset = assets_to_process
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            assets_subset = [a for i, a in enumerate(assets_to_process) if i % num_workers == worker_id]
        yield from self._iter_for_assets(assets_subset)

def collate_fn(batch):
    """
    A custom collate function for the DataLoader.

    Returns:
        A tuple of batched tensors and lists:
        - past (torch.Tensor)
        - past_mask (torch.Tensor)
        - future (torch.Tensor)
        - future_mask (torch.Tensor)
        - asset_ids (list)
        - past_ts_list (list)
        - past_covariates (torch.Tensor, optional) - market features for past window
        - past_time_covariates (torch.Tensor, optional) - time features for past window
        - future_covariates (torch.Tensor, optional) - time features for future window
    """
    # Check if batch includes all covariates (9 elements), old format (8 elements), or no covariates (6 elements)
    if len(batch[0]) == 9:
        # With all covariates (market + time for past, time for future)
        past_list, past_mask_list, future_list, future_mask_list, asset_list, ts_list, past_cov_list, past_time_cov_list, future_cov_list = zip(*batch)
        past = torch.stack(past_list, dim=0)
        past_mask = torch.stack(past_mask_list, dim=0)
        future = torch.stack(future_list, dim=0)
        future_mask = torch.stack(future_mask_list, dim=0)
        past_covariates = torch.stack(past_cov_list, dim=0)
        past_time_covariates = torch.stack(past_time_cov_list, dim=0)
        future_covariates = torch.stack(future_cov_list, dim=0)
        return past, past_mask, future, future_mask, list(asset_list), list(ts_list), past_covariates, past_time_covariates, future_covariates
    elif len(batch[0]) == 8:
        # Old format with covariates (backward compatible)
        past_list, past_mask_list, future_list, future_mask_list, asset_list, ts_list, past_cov_list, future_cov_list = zip(*batch)
        past = torch.stack(past_list, dim=0)
        past_mask = torch.stack(past_mask_list, dim=0)
        future = torch.stack(future_list, dim=0)
        future_mask = torch.stack(future_mask_list, dim=0)
        past_covariates = torch.stack(past_cov_list, dim=0)
        future_covariates = torch.stack(future_cov_list, dim=0)
        return past, past_mask, future, future_mask, list(asset_list), list(ts_list), past_covariates, future_covariates
    else:
        # Without covariates (backward compatible)
        past_list, past_mask_list, future_list, future_mask_list, asset_list, ts_list = zip(*batch)
        past = torch.stack(past_list, dim=0)
        past_mask = torch.stack(past_mask_list, dim=0)
        future = torch.stack(future_list, dim=0)
        future_mask = torch.stack(future_mask_list, dim=0)
        return past, past_mask, future, future_mask, list(asset_list), list(ts_list)

if __name__ == "__main__":
    # This block demonstrates the primary purpose of this script: converting the
    # raw CSV data into per-asset Parquet files.
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    raw_csv = os.path.join(repo_root, "data", "TRAIN_Reco_2021_2022_2023.csv")
    parquet_dir = os.path.join(repo_root, "data", "per_asset_parquet")
    print("STEP 1: splitting CSV -> per-asset parquet")
    split_csv_to_parquet(csv_path=raw_csv, out_dir=parquet_dir, chunksize=200_000, force=True)
    print("done.")
