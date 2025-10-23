import os
import pickle
from typing import Optional, List, Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
# Import dataset implementation from src.data_loader
from src.data_loader import ElectricityPriceIterableDataset, collate_fn


# --- Utility metric and inverse transform helpers --- #

def masked_smoothed_smape(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """
    y_true, y_pred, mask: tensors shape (B, T, C)
    mask elements are 1.0 for trading (include in metric) and 0.0 for ignore
    Returns scalar tensor (mean sMAPE over masked elements)
    """
    num = torch.abs(y_true - y_pred)
    den = torch.abs(y_true) + torch.abs(y_pred) + eps
    smape = 2.0 * num / den  # (B, T, C)
    masked = smape * mask
    # avoid division by zero
    denom = mask.sum()
    if denom.item() == 0:
        # if no trading steps, return zero
        return torch.tensor(0.0, device=y_true.device)
    return masked.sum() / denom


def inverse_transform_batch(preds: np.ndarray, asset_ids: List[str], scalers_dir: str) -> np.ndarray:
    """
    preds: np.array shape (B, T, C) in scaled space
    asset_ids: list length B corresponding to each row
    scalers_dir: directory where scalers are saved as <asset>_scaler.pkl
    Returns preds_inv: np.array (B, T, C) in original scale (if scaler missing, returns same values)
    """
    B, T, C = preds.shape
    out = np.empty_like(preds)
    for i, asset in enumerate(asset_ids):
        scaler_path = os.path.join(scalers_dir, f"{asset}_scaler.pkl") if scalers_dir else None
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                # scaler expects 2D array (N, C), so reshape
                reshaped = preds[i].reshape(-1, C)
                inv = scaler.inverse_transform(reshaped)
                out[i] = inv.reshape(T, C)
            except Exception as e:
                # on error, fallback to identity
                out[i] = preds[i]
        else:
            out[i] = preds[i]
    return out


# --- DataModule --- #

class ElectricityDataModule(pl.LightningDataModule):
    """
    DataModule wrapping the streaming IterableDataset.
    Expects per-split parquet directories:
      parquet_root/train/*.parquet
      parquet_root/val/*.parquet
      parquet_root/test/*.parquet

    Args:
        parquet_root: root directory containing 'train', 'val', 'test' subdirs (or pass specific paths)
        scalers_dir: directory with per-asset scaler pickles (<asset>_scaler.pkl)
        batch_size, num_workers: dataloader params
        dataset_kwargs: forwarded to ElectricityPriceIterableDataset constructor
    """

    def __init__(
            self,
            train_parquet: str,
            val_parquet: str,
            test_parquet: Optional[str],
            scalers_dir: Optional[str],
            batch_size: int = 16,
            num_workers: int = 4,
            dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.train_parquet = train_parquet
        self.val_parquet = val_parquet
        self.test_parquet = test_parquet
        self.scalers_dir = scalers_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs or {}
        # dataset placeholders (created in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # Cache for scalers, populated on first use
        self._scaler_cache: Dict[str, Optional[StandardScaler]] = {}

    def _load_scaler(self, asset_id: str) -> Optional[StandardScaler]:
        """Helper to load a scaler from disk with caching."""
        if asset_id in self._scaler_cache:
            return self._scaler_cache[asset_id]

        if not self.scalers_dir:
            self._scaler_cache[asset_id] = None
            return None

        scaler_path = os.path.join(self.scalers_dir, f"{asset_id}_scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                self._scaler_cache[asset_id] = scaler
                return scaler
            except Exception:
                pass

        self._scaler_cache[asset_id] = None
        return None

    def setup(self, stage: Optional[str] = None):
        """Create dataset instances. Called on every GPU worker by Lightning."""
        # Important: we pass scalers_dir to dataset so it can apply scaling on the fly
        ds_kwargs = dict(self.dataset_kwargs)
        ds_kwargs["scalers_dir"] = self.scalers_dir

        if stage in (None, "fit"):
            self.train_dataset = ElectricityPriceIterableDataset(
                parquet_dir=self.train_parquet,
                **ds_kwargs
            )
            self.val_dataset = ElectricityPriceIterableDataset(
                parquet_dir=self.val_parquet,
                **ds_kwargs
            )
        if stage in (None, "test") and self.test_parquet is not None:
            self.test_dataset = ElectricityPriceIterableDataset(
                parquet_dir=self.test_parquet,
                **ds_kwargs
            )

    def train_dataloader(self):
        # We pass collate_fn that stacks fixed-length windows into batches
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=max(1, self.num_workers // 2),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=max(1, self.num_workers // 2),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def on_train_epoch_start(self):
        """Called by Lightning at the start of each training epoch.
        We call set_epoch on the dataset so it can perform epoch-level shuffling deterministically.
        """
        epoch = int(self.trainer.current_epoch) if self.trainer else 0
        if hasattr(self.train_dataset, "set_epoch"):
            try:
                self.train_dataset.set_epoch(epoch)
            except Exception:
                # some IterableDataset implementations may not implement set_epoch
                pass

    def _inverse_transform_batch_cached(self, preds: np.ndarray, asset_ids: List[str]) -> np.ndarray:
        """Internal helper that uses the class's scaler cache."""
        B, T, C = preds.shape
        out = np.empty_like(preds)
        for i, asset in enumerate(asset_ids):
            # --- USE THE CACHED LOADER ---
            scaler = self._load_scaler(asset)
            if scaler is not None:
                try:
                    reshaped = preds[i].reshape(-1, C)
                    inv = scaler.inverse_transform(reshaped)
                    out[i] = inv.reshape(T, C)
                except Exception:
                    out[i] = preds[i] # Fallback
            else:
                out[i] = preds[i]
        return out

    # Optional helper to inverse-transform and compute sMAPE for a whole validation batch
    def compute_validation_metrics(self, preds_tensor: torch.Tensor, targets_tensor: torch.Tensor, mask_tensor: torch.Tensor, asset_ids: List[str]):
        """
        preds_tensor, targets_tensor, mask_tensor: torch tensors on CPU (B,T,C)
        asset_ids: list of asset ids length B
        Returns sMAPE scalar (float)
        """
        # Move to numpy
        preds_np = preds_tensor.detach().cpu().numpy()
        targets_np = targets_tensor.detach().cpu().numpy()
        mask_np = mask_tensor.detach().cpu().numpy()

        # Inverse-transform per-asset
        preds_inv = self._inverse_transform_batch_cached(preds_np, asset_ids)
        targets_inv = self._inverse_transform_batch_cached(targets_np, asset_ids)

        # Convert to torch for metric computation
        preds_inv_t = torch.from_numpy(preds_inv).to(mask_tensor.device)
        targets_inv_t = torch.from_numpy(targets_inv).to(mask_tensor.device)
        mask_t = torch.from_numpy(mask_np).to(mask_tensor.device)

        smape = masked_smoothed_smape(targets_inv_t, preds_inv_t, mask_t)
        return smape.item()


# --- Example usage in training script --- #
if __name__ == "__main__":
    # Paths (example)
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    train_parquet = os.path.join(base_dir, "data", "per_asset_parquet", "train")
    val_parquet = os.path.join(base_dir, "data", "per_asset_parquet", "val")
    test_parquet = os.path.join(base_dir, "data", "per_asset_parquet", "test")
    scalers_dir = os.path.join(base_dir, "data", "scalers")

    dataset_kwargs = dict(
        input_chunk_length=96,
        output_chunk_length=10,
        target_cols=["high", "low", "close", "volume"],
        timestamp_col="ExecutionTime",
        shuffle_buffer=128,
        stride=1,
    )

    print("--- Testing DataModule ---")
    dm = ElectricityDataModule(
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        test_parquet=test_parquet,
        scalers_dir=scalers_dir,
        batch_size=16,
        num_workers=4,
        dataset_kwargs=dataset_kwargs,
    )

    print("Setting up stage 'fit'...")
    dm.setup(stage="fit")

    # Manually call this since there is no Trainer
    dm.on_train_epoch_start()

    print("Fetching one train batch...")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    past, future, mask, assets, ts = train_batch
    print(f"  Train batch OK: past.shape={past.shape}, asset={assets[0]}")

    print("Fetching one val batch...")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    past_v, future_v, mask_v, assets_v, ts_v = val_batch
    print(f"  Val batch OK: past.shape={past_v.shape}, asset={assets_v[0]}")

    print("Testing validation metric helper...")
    # Use the cached version if you implemented the suggestion above
    smape = dm.compute_validation_metrics(past_v, future_v, mask_v, assets_v)
    print(f"  Example sMAPE: {smape:.4f}")

    print("\n--- DataModule test complete ---")
