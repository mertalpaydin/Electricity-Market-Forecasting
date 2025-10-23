## Completed Tasks

### Phase 0: Project Setup & Environment
- Created `.gitignore` (already existed and was configured).
- Created folders: `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `results/`.
- Generated `requirements.txt` with specified libraries.

### Phase 0.5.1: GPU Acceleration & VRAM Management Strategy
- Verified CUDA availability with `torch.cuda.is_available()`.

### Phase 0.5.2: Data Loading and Streaming Strategy
- Implemented a custom `IterableDataset` (`ElectricityPriceIterableDataset`) in `src/data_loader.py` that:
    - Reads preprocessed parquet chunks per asset directly from disk.
    - Yields `(past, future, mask, asset_id, past_ts)` tuples lazily.
    - Minimizes memory footprint by only keeping active batches in RAM.
- Integrated with `torch.utils.data.DataLoader` and `pytorch_lightning.LightningDataModule` for efficient batching and GPU feeding.

### Phase 1: Data Exploration, Preprocessing, and Feature Engineering (Partial)
- **Preprocessing Script (`src/data_loader.py` and `src/datamodule.py`):**
    - `split_csv_to_parquet` function created to transform raw CSV data into per-asset parquet files.
    - `fit_and_save_scalers` function created to fit `StandardScaler` for each asset on non-zero training data and save them.
    - Data loading now handles `ExecutionTime` as datetime and groups by `ID`.
    - Targets (`high`, `low`, `close`, `volume`) are extracted.
    - `is_trading` binary flag is computed based on `volume > 0`.
    - Scaling is applied on the fly within the `IterableDataset`.
- **Data Splitting (Conceptual):** The `ElectricityDataModule` is designed to work with `train`, `val`, and `test` parquet directories, implying a chronological split will be handled by preparing these directories.
