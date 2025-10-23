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
from sklearn.preprocessing import StandardScaler
import pyarrow as pa
import pyarrow.csv
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------
# Part A: CSV -> per-asset parquet
# ---------------------------

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
    Memory-safe CSV -> per-asset parquet splitter using a two-pass approach.
    Pass 1: Stream CSV in chunks, append rows to per-asset *temporary CSVs*.
    Pass 2: Convert each temporary CSV to a *sorted* parquet file using pyarrow.

    - csv_path: path to large CSV file
    - out_dir: output directory (created if needed)
    - id_col: column name containing asset info
    - timestamp_col: column name containing datetime info
    - chunksize: number of rows per chunk to read (adjust to memory)
    - force: if True, re-write all parquet files even if they exist
    """
    os.makedirs(out_dir, exist_ok=True)
    assets_seen = set()
    temp_dir = os.path.join(out_dir, "_tmp_append")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"[split] streaming CSV {csv_path} in chunksize={chunksize}")

    # --------------------------------------------------------------------------
    # PASS 1: Stream main CSV and append to per-asset temporary CSVs
    # --------------------------------------------------------------------------

    # Iterate through CSV in chunks and append rows to per-asset temp CSVs (then convert to parquet)
    for chunk in pd.read_csv(csv_path, parse_dates=[timestamp_col], chunksize=chunksize):

        # Ensure timestamp is datetime and UTC-aware
        chunk[timestamp_col] = pd.to_datetime(chunk[timestamp_col], utc=True)

        # compute is_trading mask once
        chunk["is_trading"] = (chunk[target_cols].abs().sum(axis=1) > 0).astype(np.float32)

        store_cols = [timestamp_col] + target_cols + ["is_trading"]

        # group by asset and append to small per-asset CSVs in temp_dir
        for asset, g in chunk.groupby(id_col):
            assets_seen.add(str(asset))
            temp_path = os.path.join(temp_dir, f"{asset}.csv")
            # append without header if file exists
            header = not os.path.exists(temp_path)
            g[store_cols].to_csv(temp_path, mode="a", header=header, index=False)

    print("[split] PASS 1 complete.")

    # --------------------------------------------------------------------------
    # PASS 2: Convert each temp CSV to a SORTED parquet file using pyarrow
    # --------------------------------------------------------------------------

    print(f"[split] PASS 2: converting {len(assets_seen)} temp CSVs to sorted parquet...")
    assets = sorted(list(assets_seen))

    # Define pyarrow CSV read options
    # We must tell it which column to parse as a timestamp
    convert_options = pa.csv.ConvertOptions(
        column_types={timestamp_col: pa.timestamp('ns', tz='UTC')},
        include_columns=[timestamp_col] + target_cols + ["is_trading"]
    )
    read_options = pa.csv.ReadOptions(autogenerate_column_names=False)

    for asset in tqdm(assets, desc="[split] Converting assets to parquet"):
        out_path = os.path.join(out_dir, f"{asset}.parquet")
        if os.path.exists(out_path) and not force:
            continue

        tmp_csv = os.path.join(temp_dir, f"{asset}.csv")
        if not os.path.exists(tmp_csv):
            continue


        try:
            table = pa.csv.read_csv(tmp_csv, read_options=read_options, convert_options=convert_options)
        except Exception as e:
            print(f"[split] pyarrow failed for {tmp_csv}: {e}. Falling back to pandas.")
            df = pd.read_csv(tmp_csv, parse_dates=[timestamp_col])
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
            df = df[[timestamp_col] + target_cols + ["is_trading"]]
            table = pa.Table.from_pandas(df)


        # Sort the table in-memory by timestamp
        sorted_table = table.sort_by([(timestamp_col, 'ascending')])

        # Write the sorted table to a parquet file
        pq.write_table(sorted_table, out_path, compression='snappy')

        # remove temp csv to save space
        os.remove(tmp_csv)

    # cleanup temp dir
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass # Dir not empty if some files failed, which is fine

    # write assets.json (list what's actually on disk)
    final_assets = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(out_dir, "*.parquet"))]
    assets_json = os.path.join(out_dir, "assets.json")
    with open(assets_json, "w") as f:
        json.dump(final_assets, f)

    print(f"[split] created {len(final_assets)} per-asset parquet files in {out_dir}")
    return final_assets


# ---------------------------
# Part A.2: Fit & save per-asset scalers (train-only)
# ---------------------------

def fit_and_save_scalers(
        parquet_dir: str,
        scalers_dir: str,
        target_cols: List[str] = ["high", "low", "close", "volume"],
        timestamp_col: str = "ExecutionTime",
        min_nonzero: int = 10,
        train_time_cutoff: Optional[pd.Timestamp] = None,
):
    """
    Fit per-asset StandardScaler on non-zero training rows and save as pickle:
      scalers_dir/<asset>_scaler.pkl

    - parquet_dir: directory containing per-asset parquet files
    - train_time_cutoff: if given, only rows with timestamp <= cutoff are used for fitting
    - min_nonzero: minimum number of nonzero rows needed to fit a scaler (otherwise no scaler saved)
    """
    os.makedirs(scalers_dir, exist_ok=True)
    assets_json = os.path.join(parquet_dir, "assets.json")
    if os.path.exists(assets_json):
        with open(assets_json, "r") as f:
            assets = json.load(f)
    else:
        assets = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(parquet_dir, "*.parquet"))]

    for asset in tqdm(assets, desc="[scalers] Fitting scalers"):
        p = os.path.join(parquet_dir, f"{asset}.parquet")
        if not os.path.exists(p):
            continue

        df = pd.read_parquet(p)
        if train_time_cutoff is not None:
            # keep only rows <= cutoff
            if timestamp_col not in df.columns: # <-- ADD CHECK
                print(f"Warning: {timestamp_col} not in {p}, skipping cutoff.")
            else:
                df = df[df[timestamp_col] <= train_time_cutoff]

        # use is_trading (volume > 0) mask if present; fallback to non-all-zero rows
        if "is_trading" in df.columns:
            train_mask = (df["is_trading"] > 0).values
        else:
            train_mask = (df[target_cols].abs().sum(axis=1) > 0).values

        save_path = os.path.join(scalers_dir, f"{asset}_scaler.pkl")

        if train_mask.sum() < min_nonzero:
            # do not save small scalers - remove if exists
            if os.path.exists(save_path):
                os.remove(save_path)
            continue

        scaler = StandardScaler()
        scaler.fit(df.loc[train_mask, target_cols].values.astype(np.float32))
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)

    print(f"[scalers] saved scalers (where applicable) to {scalers_dir}")


# ---------------------------
# Part B: IterableDataset (streaming)
# ---------------------------

class ElectricityPriceIterableDataset(IterableDataset):
    """
    Streaming IterableDataset that reads per-asset parquet files one-by-one and yields windows.

    Yields: (past: Tensor L_in x C, future: Tensor L_out x C, mask: Tensor L_out x C, asset_id: str, past_ts: np.ndarray)
    - past/future are scaled (if scalers_dir provided)
    - mask: 1.0 for trading steps in the FUTURE, 0.0 for non-trading (used for masked sMAPE)
    """

    def __init__(
            self,
            parquet_dir: str,
            assets_list: Optional[List[str]] = None,
            input_chunk_length: int = 96,
            output_chunk_length: int = 10,
            target_cols: List[str] = ["high", "low", "close", "volume"],
            timestamp_col: str = "ExecutionTime",
            scalers_dir: Optional[str] = None,
            shuffle_buffer: int = 0,
            stride: int = 1,
            min_length: Optional[int] = None,
    ):
        super().__init__()
        self.parquet_dir = parquet_dir
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
        self.scalers_dir = scalers_dir
        self.shuffle_buffer = shuffle_buffer
        self.stride = stride
        self.min_length = min_length or (input_chunk_length + output_chunk_length)
        self.asset_paths = {a: os.path.join(parquet_dir, f"{a}.parquet") for a in self.assets}

        # per-worker caches will be created lazily in worker process space
        self._worker_scaler_cache = None
        self._epoch = 0  # set by set_epoch

    def set_epoch(self, epoch: int):
        """Set epoch index (useful to make shuffling deterministic across epochs)."""
        self._epoch = int(epoch)

    def _get_worker_cache(self):
        """Create per-worker cache dictionary in process memory to avoid repeated IO."""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # create a unique attribute per worker / process
        cache_attr = f"_worker_cache_{worker_id}"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {"scalers": {}})
        return getattr(self, cache_attr)

    def _load_scaler_for_asset(self, asset: str):
        """Load scaler for an asset using per-worker cache."""
        if self.scalers_dir is None:
            return None
        cache = self._get_worker_cache()
        scaler_map = cache["scalers"]
        if asset in scaler_map:
            return scaler_map[asset]
        path = os.path.join(self.scalers_dir, f"{asset}_scaler.pkl")
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
        """
        Internal generator: iterate through given assets and yield windows.
        """
        # set per-worker seeds deterministically using epoch to get reproducible shuffling
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = 1234 + self._epoch * 100 + worker_id
        rnd = random.Random(seed)

        buffer: List[Tuple[Any, ...]] = []

        for asset in assets_subset:
            path = self.asset_paths.get(asset)
            if not path or not os.path.exists(path):
                continue

            # read parquet into pandas (per-asset file; should be reasonably sized)
            try:
                df = pd.read_parquet(path)
            except Exception:
                # skip unreadable
                continue

            # quick sanity checks
            if self.timestamp_col not in df.columns:
                continue

            arr = df[self.target_cols].values.astype(np.float32)  # (N, C)
            timestamps = df[self.timestamp_col].values
            n = arr.shape[0]
            if n < self.min_length:
                continue

            # trading mask from precomputed column
            if "is_trading" in df.columns:
                trading_mask = df["is_trading"].values.astype(np.float32)
            else:
                trading_mask = (np.abs(arr).sum(axis=1) > 0).astype(np.float32)

            scaler = self._load_scaler_for_asset(asset)

            L_in = self.input_chunk_length
            L_out = self.output_chunk_length

            # sliding windows
            for start in range(0, n - L_in - L_out + 1, self.stride):
                x = arr[start : start + L_in]   # (L_in, C)
                y = arr[start + L_in : start + L_in + L_out]  # (L_out, C)
                past_ts = timestamps[start : start + L_in]
                future_trading = trading_mask[start + L_in : start + L_in + L_out]  # (L_out,)

                # apply scaler (fit on training non-zero rows beforehand)
                if scaler is not None:
                    try:
                        x = scaler.transform(x)
                        y = scaler.transform(y)
                    except Exception:
                        # fallback to identity if scaler fails
                        pass

                mask = np.repeat(future_trading.reshape(-1, 1), repeats=len(self.target_cols), axis=1).astype(np.float32)

                sample = (torch.from_numpy(x).float(),
                          torch.from_numpy(y).float(),
                          torch.from_numpy(mask).float(),
                          asset,
                          past_ts)

                if self.shuffle_buffer > 0:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer:
                        # pop a random element
                        idx = rnd.randrange(len(buffer))
                        yield buffer.pop(idx)
                else:
                    yield sample

        # flush buffer
        while self.shuffle_buffer > 0 and len(buffer) > 0:
            idx = rnd.randrange(len(buffer))
            yield buffer.pop(idx)

    def __iter__(self):
        """
        Partition assets among workers deterministically and iterate assigned subset.
        """
        # build the asset ordering per epoch (deterministic given epoch)
        assets_to_process = list(self.assets)
        rnd = random.Random(1234 + self._epoch)  # epoch-level seed
        rnd.shuffle(assets_to_process)

        worker_info = get_worker_info()
        if worker_info is None:
            assets_subset = assets_to_process
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            assets_subset = [a for i, a in enumerate(assets_to_process) if i % num_workers == worker_id]

        yield from self._iter_for_assets(assets_subset)


# ---------------------------
# Collate function for DataLoader
# ---------------------------

def collate_fn(batch):
    """
    Collate function stacks fixed-length windows into batched tensors.
    batch: list of tuples from dataset: (past, future, mask, asset_id, past_ts)
    returns: past (B, L_in, C), future (B, L_out, C), mask (B, L_out, C), asset_ids (list), past_ts_list (list)
    """
    past_list, future_list, mask_list, asset_list, ts_list = zip(*batch)
    past = torch.stack(past_list, dim=0)
    future = torch.stack(future_list, dim=0)
    mask = torch.stack(mask_list, dim=0)
    return past, future, mask, list(asset_list), list(ts_list)


# ---------------------------
# Quick test / usage example (run as script)
# ---------------------------

if __name__ == "__main__":
    # quick demo of preprocessing + scaler + dataloader
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    raw_csv = os.path.join(repo_root, "data", "TRAIN_Reco_2021_2022_2023.csv")
    parquet_dir = os.path.join(repo_root, "data", "per_asset_parquet")
    scalers_dir = os.path.join(repo_root, "data", "scalers")

    # Step 1: split CSV -> parquet (chunked)
    print("STEP 1: splitting CSV -> per-asset parquet")
    split_csv_to_parquet(csv_path=raw_csv, out_dir=parquet_dir, chunksize=200_000, force=False)

    # Step 2: fit scalers using train cutoff (optional: pass a pd.Timestamp)
    print("STEP 2: fitting scalers")
    fit_and_save_scalers(parquet_dir=parquet_dir, scalers_dir=scalers_dir)

    # Step 3: test streaming dataset
    print("STEP 3: testing streaming IterableDataset")
    ds = ElectricityPriceIterableDataset(
        parquet_dir=parquet_dir,
        input_chunk_length=96,
        output_chunk_length=10,
        target_cols=["high", "low", "close", "volume"],
        timestamp_col="ExecutionTime",
        scalers_dir=scalers_dir,
        shuffle_buffer=128,
        stride=1,
    )

    # DataLoader (multi-worker safe)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4, collate_fn=collate_fn)

    for i, (past, future, mask, assets, ts) in enumerate(loader):
        print(f"[batch {i}] past {past.shape} future {future.shape} mask {mask.shape} asset {assets[0]}")
        if i >= 2:
            break

    print("done.")
