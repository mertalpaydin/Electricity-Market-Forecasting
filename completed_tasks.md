## Completed Tasks

### Phase 0: Project Setup & Environment
- Created `.gitignore` (already existed and was configured).
- Created folders: `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `results/`.
- Generated `requirements.txt` with specified libraries.

### Phase 0.5.1: GPU Acceleration & VRAM Management Strategy
- Verified CUDA availability with `torch.cuda.is_available()`.

### Phase 0.5.2: Data Loading and Streaming Strategy
- Implemented a custom `IterableDataset` (`ElectricityPriceIterableDataset`) in `src/data_loader.py`.
- Integrated with `torch.utils.data.DataLoader` and `pytorch_lightning.LightningDataModule`.
- Updated the `IterableDataset` and `collate_fn` to yield a `past_observed_mask` for compatibility with transformer-based models.

### Phase 1: Data Exploration, Preprocessing, and Feature Engineering
- **Preprocessing Script (`src/preprocess.py`):**
    - Implemented robust feature engineering for time-based and cross-contract features.
    - Corrected the cross-contract feature implementation to be forward-looking safe and added new features (`close_lag_adj_k`, `close_delta_adj_k`, `volume_adj_k`, `nearest_liquid_contract_close`, `cross_contract_mean`).
    - Added the `time_to_delivery` feature.
    - Implemented data splitting and scaling.
- **Initial Data Loading and EDA (in `notebooks/01_EDA.ipynb`):**
    - Performed initial exploratory data analysis.
- **Documentation:**
    - Added comprehensive docstrings and comments to `src/data_loader.py`, `src/datamodule.py`, and `src/preprocess.py`.

### Phase 1.5: Subsampling Strategy Development and Testing
- Created dedicated notebook (`notebooks/01.5_Subsampling_Strategy.ipynb`) for testing subsampling strategies.
- Implemented 4 subsampling strategies:
  - **Trading-Only**: Keep only trading periods (is_trading == 1)
  - **Stratified**: All trading + 10% random non-trading
  - **Boundary-Aware**: Trading + periods around trading boundaries (±4 rows)
  - **Hybrid**: Boundary-aware + 5% random non-trading
- Implemented trading segment extraction and analysis functions.
- Created evaluation framework: trains simple LSTM with each strategy to measure training time, convergence, and validation performance.
- Strategy comparison and recommendation system.
- Results saved to `results/subsampling_recommendation.json`.
- **Status**: Notebook ready for execution.

### Phase 2: Baseline Modeling
- Created comprehensive baseline models notebook (`notebooks/03_Baseline_Models.ipynb`) with **strict time limits**:
  - **Time Limit Implementation**: Custom `TimeLimitCallback` for PyTorch Lightning enforcing 2-hour hard cap.
  - **Naive Baselines**: Last-value and mean forecasts (fast, no limit).
  - **LSTM Baseline**: Trains with best hyperparameters from Phase 4, stops after 2 hours or if beats naive baseline.
- All evaluations use `masked_smoothed_smape` from `src/datamodule.py`.
- Applies recommended subsampling strategy from Phase 1.5.
- Compares LSTM against naive baselines and saves results to `results/baseline_summary.json`.
- **Status**: Notebook ready for execution (requires Phase 1.5 results).

### Phase 3: Advanced Deep Learning Modeling (Chronos2)
- Created complete Chronos2 training notebook (`notebooks/04_Chronos2_Training.ipynb`):
  - Loads pretrained `amazon/chronos-2` model via `Chronos2Pipeline`.
  - Implements all helper functions for tensor conversion and shape alignment:
    - `_to_tensor()`, `_extract_context_target_mask()`, `_to_torch()`, `align_forecast_to_target()`
  - **Data Strategy**: Reads subsampling recommendation from Phase 1.5 and applies to **full dataset** (all assets, not just Optuna subset).
  - Implements 4 subsampling strategies (Trading-Only, Stratified, Boundary-Aware, Hybrid) and auto-applies best strategy.
  - Uses `ElectricityDataModule` with subsampled filtered data.
  - **Zero-shot evaluation**: Quick evaluation (100 batches) to assess pretrained performance.
  - **3-Stage Fine-Tuning** with architecture-aware gradual unfreezing:
    - Stage 1 (5 epochs): Train only `output_patch_embedding`
    - Stage 2 (8 epochs): Unfreeze last 3 encoder blocks with differential LR
    - Stage 3 (12 epochs): Full fine-tuning with layerwise LR decay
  - Uses best hyperparameters from Optuna (lr=0.0003, batch_size=64, max_context=96, quantiles=[0.05,0.25,0.5,0.75,0.95]).
  - **Full validation evaluation**: Complete validation set evaluation after fine-tuning.
  - Compares against baseline models from Phase 2.
  - Generates loss evolution plots and comparison visualizations.
  - Saves results to `results/chronos2_finetuned_results.json`, `results/chronos2_finetuning_loss.png`, `results/chronos2_final_comparison.png`.
- Fixed `notebooks/helper.py`: Added missing `torch` and `numpy` imports.
- **Status**: ✅ Notebook complete and ready for execution (requires Phase 1.5 subsampling results and Phase 2 baseline results).

### Phase 4: Systematic Hyperparameter Tuning
- Created dedicated notebook (`notebooks/02_Hyperparameter_Tuning.ipynb`) for hyperparameter optimization.
- Implemented trading segment extraction and trading-only data filtering.
- Used **Optuna** with TPE sampler and median pruner to tune:
  - **LSTM**: hidden_dim, n_rnn_layers, dropout, lr, batch_size
  - **Chronos2**: quantile_levels, batch_size, lr, max_context_length
- Saved best hyperparameters to:
  - `results/best_params_lstm.json`
  - `results/best_params_chronos2.json`
  - `results/best_hyperparameters_tst.json` (combined)
- Created filtered trading-only parquet files in `data/train_trading_only/` and `data/val_trading_only/`.
- **Status**: ✅ Completed with results saved.