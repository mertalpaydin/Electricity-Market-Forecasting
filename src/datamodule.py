import os
import pickle
from typing import Optional, List, Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
# Import dataset implementation from src.data_loader
from .data_loader import ElectricityPriceIterableDataset, collate_fn


# --- Utility metric and inverse transform helpers --- #

def masked_smoothed_smape(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (sMAPE), masked for
    non-trading periods.

    sMAPE is a percentage-based error metric that is less sensitive to outliers
    than MAPE. This version is "smoothed" by adding a small epsilon to the
    denominator to prevent division by zero.

    Args:
        y_true (torch.Tensor): The ground truth values. Shape: (B, T, C).
        y_pred (torch.Tensor): The predicted values. Shape: (B, T, C).
        mask (torch.Tensor): A binary tensor where 1.0 indicates a trading
            period to be included in the metric, and 0.0 indicates a non-trading
            period to be ignored. Shape: (B, T, C).
        eps (float): A small epsilon value to avoid division by zero.

    Returns:
        torch.Tensor: A scalar tensor representing the mean sMAPE over all
            masked elements.
    """
    num = torch.abs(y_true - y_pred)
    # Add epsilon for numerical stability, preventing division by zero.
    den = torch.abs(y_true) + torch.abs(y_pred) + eps
    smape = 2.0 * num / den
    masked = smape * mask
    denom = mask.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=y_true.device)
    return masked.sum() / denom


def inverse_transform_batch(preds: np.ndarray, asset_ids: List[str], scalers_dir: str) -> np.ndarray:
    """
    Applies the inverse scaling transformation to a batch of predictions.

    This function loads the appropriate pre-fitted scaler for each asset in the
    batch and applies the inverse transformation. If a scaler is not found for an
    asset, the data for that asset is returned unchanged.

    Args:
        preds (np.ndarray): A numpy array of predictions in scaled space.
            Shape: (B, T, C).
        asset_ids (List[str]): A list of asset IDs corresponding to each item
            in the batch.
        scalers_dir (str): The directory where the scalers are saved.

    Returns:
        np.ndarray: The predictions transformed back to their original scale.
            Shape: (B, T, C).
    """
    B, T, C = preds.shape
    out = np.empty_like(preds)
    for i, asset in enumerate(asset_ids):
        scaler_path = os.path.join(scalers_dir, f"{asset}_scaler.joblib") if scalers_dir else None
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                reshaped = preds[i].reshape(-1, C)
                inv = scaler.inverse_transform(reshaped)
                out[i] = inv.reshape(T, C)
            except Exception as e:
                # For robustness, if a scaler is missing or corrupt, use the scaled values.
                out[i] = preds[i]
        else:
            out[i] = preds[i]
    return out


# --- DataModule --- #

class ElectricityDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the electricity price dataset.

    This DataModule encapsulates all the data loading and preparation logic. It
    uses the streaming `ElectricityPriceIterableDataset` to handle large data
    efficiently. It expects the data to be pre-split into `train`, `val`, and
    `test` directories containing Parquet files.

    Args:
        train_parquet (str): Path to the directory of training Parquet files.
        val_parquet (str): Path to the directory of validation Parquet files.
        test_parquet (Optional[str]): Path to the directory of test Parquet files.
        scalers_dir (Optional[str]): Path to the directory of pre-fitted scalers.
        batch_size (int): The size of each batch.
        num_workers (int): The number of worker processes for data loading.
        dataset_kwargs (Optional[Dict[str, Any]]): Additional arguments to be
            forwarded to the `ElectricityPriceIterableDataset` constructor.
    """

    def __init__(
            self,
            train_parquet: str,
            val_parquet: str,
            test_parquet: Optional[str] = None,
            scalers_dir: Optional[str] = None,
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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._scaler_cache: Dict[str, Optional[StandardScaler]] = {}

    def _load_scaler(self, asset_id: str) -> Optional[StandardScaler]:
        """
        Loads a scaler for a given asset, using an in-memory cache to avoid
        repeated disk I/O.
        """
        if asset_id in self._scaler_cache:
            return self._scaler_cache[asset_id]

        if not self.scalers_dir:
            self._scaler_cache[asset_id] = None
            return None

        scaler_path = os.path.join(self.scalers_dir, f"{asset_id}_scaler.joblib")
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
        """
        Prepares the datasets for training, validation, or testing. This method
        is called automatically by PyTorch Lightning.
        """
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

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation set."""
        # Use fewer workers for validation as it's less I/O intensive than training.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=max(1, self.num_workers // 2),
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the test set."""
        if self.test_dataset is None:
            return None
        # Use fewer workers for testing as it's less I/O intensive than training.
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=max(1, self.num_workers // 2),
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def on_train_epoch_start(self):
        """
        A PyTorch Lightning hook that is called at the beginning of each
        training epoch. It notifies the iterable dataset of the new epoch
        to ensure reproducible shuffling.
        """
        epoch = int(self.trainer.current_epoch) if self.trainer else 0
        if hasattr(self.train_dataset, "set_epoch"):
            try:
                self.train_dataset.set_epoch(epoch)
            except Exception:
                pass

    def _inverse_transform_batch_cached(self, preds: np.ndarray, asset_ids: List[str]) -> np.ndarray:
        """
        An internal helper to inverse-transform a batch of predictions using the
        DataModule's scaler cache.
        """
        B, T, C = preds.shape
        out = np.empty_like(preds)
        for i, asset in enumerate(asset_ids):
            scaler = self._load_scaler(asset)
            if scaler is not None:
                try:
                    reshaped = preds[i].reshape(-1, C)
                    inv = scaler.inverse_transform(reshaped)
                    out[i] = inv.reshape(T, C)
                except Exception:
                    out[i] = preds[i]
            else:
                out[i] = preds[i]
        return out

    def compute_validation_metrics(self, preds_tensor: torch.Tensor, targets_tensor: torch.Tensor, mask_tensor: torch.Tensor, asset_ids: List[str]) -> float:
        """
        A helper function to compute the validation sMAPE for a batch of predictions.

        It first inverse-transforms both the predictions and targets to their
        original scale and then computes the masked sMAPE.

        Args:
            preds_tensor (torch.Tensor): The model's predictions.
            targets_tensor (torch.Tensor): The ground truth targets.
            mask_tensor (torch.Tensor): The mask for non-trading periods.
            asset_ids (List[str]): The list of asset IDs for the batch.

        Returns:
            float: The computed sMAPE score.
        """
        # Step 1: Move data to CPU and convert to NumPy for scaling.
        preds_np = preds_tensor.detach().cpu().numpy()
        targets_np = targets_tensor.detach().cpu().numpy()
        mask_np = mask_tensor.detach().cpu().numpy()

        # Step 2: Inverse-transform predictions and targets to their original scale.
        preds_inv = self._inverse_transform_batch_cached(preds_np, asset_ids)
        targets_inv = self._inverse_transform_batch_cached(targets_np, asset_ids)

        # Step 3: Convert back to tensors and compute sMAPE on original-scale data.
        preds_inv_t = torch.from_numpy(preds_inv).to(mask_tensor.device)
        targets_inv_t = torch.from_numpy(targets_inv).to(mask_tensor.device)
        mask_t = torch.from_numpy(mask_np).to(mask_tensor.device)

        smape = masked_smoothed_smape(targets_inv_t, preds_inv_t, mask_t)
        return smape.item()