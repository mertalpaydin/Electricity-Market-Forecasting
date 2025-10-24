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
- Refactored `data_loader.py` to be solely responsible for converting the raw CSV to per-asset parquet files.

### Phase 1: Data Exploration, Preprocessing, and Feature Engineering
- **Preprocessing Script (`src/preprocess.py`):**
    - Created a dedicated `preprocess.py` script to handle all feature engineering, data splitting, and scaler fitting.
    - `feature_engineer` function created to add time-based features (`hour_of_day`, `day_of_week`, etc.) and an `is_trading` flag.
    - `split_data_by_year` function created to split the data into `train` (2021-2022) and `val` (2023) sets.
    - `fit_and_save_scalers` function created to fit `StandardScaler` for each asset on non-zero training data and save them to `data/scalers`.
- **Documentation:**
    - Added comprehensive docstrings and comments to all functions and classes in `src/data_loader.py`, `src/datamodule.py`, and `src/preprocess.py`.