# --- MARKDOWN CELL ---
# # Phase 2: Baseline Modeling
# 
# This notebook establishes baseline performance using a variety of models, from simple naive forecasts to standard deep learning architectures. The goal is to create a set of benchmarks against which more advanced models can be compared.
# 
# The models to be implemented are:
# 1.  **Naive Forecasts**: Last-value and seasonal mean baselines.
# 2.  **LightGBM**: A tree-based model using tabularized time series data.
# 3.  **RNN/LSTM**: A basic recurrent neural network baseline.
# 4.  **N-BEATS**: A block-based deep learning model.
# 
# All models will be trained on the preprocessed data from Phase 1 and evaluated on the validation set using the masked sMAPE metric.

# --- CODE CELL ---
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from darts import TimeSeries
from darts.models import NaiveSeasonal, NaiveMean, LightGBMModel, RNNModel, NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape
import matplotlib.pyplot as plt

# Add src directory to path to import custom modules
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.datamodule import ElectricityDataModule, masked_smoothed_smape

# --- MARKDOWN CELL ---
# ## 1. Setup and Data Loading
# 
# First, we define the paths to our data and instantiate the `ElectricityDataModule`.

# --- CODE CELL ---
# Define project paths
BASE_DIR = ".." 
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SCALERS_DIR = os.path.join(DATA_DIR, "scalers")
MODELS_DIR = os.path.join(BASE_DIR, "models", "baselines")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)

# Configuration
BATCH_SIZE = 32
INPUT_CHUNK_LENGTH = 96
OUTPUT_CHUNK_LENGTH = 10
TARGET_COLS = ["high", "low", "close", "volume"]

# --- CODE CELL ---
# Load the best hyperparameters found during tuning
with open(os.path.join(RESULTS_DIR, 'best_hyperparameters.json'), 'r') as f:
    best_hyperparams = json.load(f)

print("Loaded best hyperparameters:")
print(json.dumps(best_hyperparams, indent=4))

# --- CODE CELL ---
# Instantiate the DataModule
dm = ElectricityDataModule(
    train_parquet=TRAIN_DIR,
    val_parquet=VAL_DIR,
    test_parquet=None,
    scalers_dir=SCALERS_DIR,
    batch_size=BATCH_SIZE,
    dataset_kwargs={
        "input_chunk_length": INPUT_CHUNK_LENGTH,
        "output_chunk_length": OUTPUT_CHUNK_LENGTH,
        "target_cols": TARGET_COLS,
        "stride": 10,
    }
)
dm.setup(stage="fit")

# --- CODE CELL ---
# Initialize a dictionary to store the results
results = {}

# --- MARKDOWN CELL ---
# ## 2. Naive Forecasts

# --- CODE CELL ---
val_asset_files = [f for f in os.listdir(VAL_DIR) if f.endswith('.parquet')]
np.random.seed(42)
SAMPLE_ASSETS = np.random.choice(val_asset_files, 5, replace=False)

def evaluate_naive_models(sample_assets):
    all_smape_last_value = []
    all_smape_mean = []

    for asset_file in sample_assets:
        train_df = pd.read_parquet(os.path.join(TRAIN_DIR, asset_file))
        val_df = pd.read_parquet(os.path.join(VAL_DIR, asset_file))
        
        train_df['ExecutionTime'] = pd.to_datetime(train_df['ExecutionTime'])
        val_df['ExecutionTime'] = pd.to_datetime(val_df['ExecutionTime'])

        ts_train = TimeSeries.from_dataframe(train_df, 'ExecutionTime', TARGET_COLS, freq='15min')
        ts_val = TimeSeries.from_dataframe(val_df, 'ExecutionTime', TARGET_COLS, freq='15min')

        last_vals = [ts_train[c][ts_train[c].values() != 0].last_value() if not ts_train[c][ts_train[c].values() != 0].is_empty() else 0 for c in TARGET_COLS]
        pred_last_value = TimeSeries.from_values(np.tile(last_vals, (len(ts_val), 1)), columns=TARGET_COLS, start=ts_val.start_time(), freq=ts_val.freq)

        train_df_copy = train_df[train_df['is_trading'] == 1].copy()
        train_df_copy['month'] = train_df_copy['ExecutionTime'].dt.month
        train_df_copy['day_of_week'] = train_df_copy['ExecutionTime'].dt.dayofweek
        train_df_copy['hour_of_day'] = train_df_copy['ExecutionTime'].dt.hour
        mean_map = train_df_copy.groupby(['month', 'day_of_week', 'hour_of_day'])[TARGET_COLS].mean().to_dict('index')

        mean_preds = []
        for ts in val_df['ExecutionTime']:
            key = (ts.month, ts.dayofweek, ts.hour)
            default_mean = [train_df_copy[c].mean() for c in TARGET_COLS]
            mean_vals = list(mean_map.get(key, default_mean))
            mean_preds.append(mean_vals)
        pred_mean = TimeSeries.from_values(np.array(mean_preds), columns=TARGET_COLS, start=ts_val.start_time(), freq=ts_val.freq)

        mask = (ts_val.values() != 0).astype(float)
        all_smape_last_value.append(masked_smoothed_smape(torch.from_numpy(ts_val.values()), torch.from_numpy(pred_last_value.values()), torch.from_numpy(mask)).item())
        all_smape_mean.append(masked_smoothed_smape(torch.from_numpy(ts_val.values()), torch.from_numpy(pred_mean.values()), torch.from_numpy(mask)).item())

    return {"Naive (Last Value)": np.mean(all_smape_last_value), "Naive (Mean)": np.mean(all_smape_mean)}

results.update(evaluate_naive_models(SAMPLE_ASSETS))

# --- MARKDOWN CELL ---
# ## 3. LightGBM Baseline

# --- CODE CELL ---
def evaluate_lightgbm_models(sample_assets, lgbm_params):
    all_smape_scores = {target: [] for target in TARGET_COLS}
    for asset_file in sample_assets:
        train_df = pd.read_parquet(os.path.join(TRAIN_DIR, asset_file))
        val_df = pd.read_parquet(os.path.join(VAL_DIR, asset_file))
        train_df['ExecutionTime'] = pd.to_datetime(train_df['ExecutionTime'])
        val_df['ExecutionTime'] = pd.to_datetime(val_df['ExecutionTime'])

        covariate_cols = ['hour_of_day', 'day_of_week', 'week_of_year', 'month', 'is_weekend', 'time_to_delivery']
        ts_train_covs = TimeSeries.from_dataframe(train_df, 'ExecutionTime', covariate_cols, freq='15min')
        ts_val_covs = TimeSeries.from_dataframe(val_df, 'ExecutionTime', covariate_cols, freq='15min')

        cov_scaler = Scaler()
        ts_train_covs_scaled = cov_scaler.fit_transform(ts_train_covs)
        ts_val_covs_scaled = cov_scaler.transform(ts_val_covs)

        for target in TARGET_COLS:
            ts_train_target = TimeSeries.from_dataframe(train_df, 'ExecutionTime', [target], freq='15min')
            ts_val_target = TimeSeries.from_dataframe(val_df, 'ExecutionTime', [target], freq='15min')
            target_scaler = Scaler()
            ts_train_target_scaled = target_scaler.fit_transform(ts_train_target)

            # --- Use tuned hyperparameters ---
            model_lgbm = LightGBMModel(lags=96, lags_past_covariates=48, output_chunk_length=OUTPUT_CHUNK_LENGTH, lgbm_params=lgbm_params)
            model_lgbm.fit(series=ts_train_target_scaled, past_covariates=ts_train_covs_scaled, verbose=False)
            preds_scaled = model_lgbm.predict(n=len(ts_val_target), series=ts_train_target_scaled, past_covariates=ts_val_covs_scaled)
            preds_unscaled = target_scaler.inverse_transform(preds_scaled)
            mask = (ts_val_target.values() != 0).astype(float)
            all_smape_scores[target].append(masked_smoothed_smape(torch.from_numpy(ts_val_target.values()), torch.from_numpy(preds_unscaled.values()), torch.from_numpy(mask)).item())
            
    return {f"LightGBM ({t})": np.mean(s) for t, s in all_smape_scores.items()}

# Add fixed params and pass to the evaluation function
lgbm_params = best_hyperparams['lightgbm']
lgbm_params['objective'] = 'regression_l1'
lgbm_params['metric'] = 'mae'
lgbm_params['device_type'] = 'gpu'
lgbm_params['n_jobs'] = -1
results.update(evaluate_lightgbm_models(SAMPLE_ASSETS, lgbm_params))

# --- MARKDOWN CELL ---
# ## 4. RNN/LSTM Baseline

# --- CODE CELL ---
# --- Use tuned hyperparameters ---
pl_trainer_kwargs_rnn = {"accelerator": "gpu", "devices": 1, "precision": "16-mixed", "callbacks": [EarlyStopping(monitor="val_loss", patience=5, mode="min"), ModelCheckpoint(dirpath=os.path.join(MODELS_DIR, "rnn"), filename="best_model", monitor="val_loss", save_top_k=1, mode="min")]}

lstm_params = best_hyperparams['lstm']
optimizer_kwargs = {"lr": lstm_params.pop('lr')}

model_rnn = RNNModel(
    model="LSTM", 
    input_chunk_length=INPUT_CHUNK_LENGTH, 
    output_chunk_length=OUTPUT_CHUNK_LENGTH, 
    n_epochs=50, 
    loss_fn=torch.nn.L1Loss(), 
    optimizer_kwargs=optimizer_kwargs,
    pl_trainer_kwargs=pl_trainer_kwargs_rnn, 
    model_name="RNN_baseline", 
    save_checkpoints=True, 
    force_reset=True, 
    random_state=42,
    **lstm_params # Pass the rest of the tuned parameters
)
model_rnn.fit(dm)

best_rnn = RNNModel.load_from_checkpoint(model_name="RNN_baseline", work_dir=MODELS_DIR, best=True)
trainer_rnn = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed")
predictions_rnn = trainer_rnn.predict(best_rnn, dm.val_dataloader())

all_preds_rnn = np.concatenate([p[0] for p in predictions_rnn])
all_targets_rnn = np.concatenate([p[1] for p in predictions_rnn])
all_masks_rnn = np.concatenate([p[2] for p in predictions_rnn])
asset_ids_rnn = [a for p in predictions_rnn for a in p[3]]
preds_inv_rnn = dm._inverse_transform_batch_cached(all_preds_rnn, asset_ids_rnn)
targets_inv_rnn = dm._inverse_transform_batch_cached(all_targets_rnn, asset_ids_rnn)
final_smape_rnn = masked_smoothed_smape(torch.from_numpy(targets_inv_rnn), torch.from_numpy(preds_inv_rnn), torch.from_numpy(all_masks_rnn))
results["RNN/LSTM"] = final_smape_rnn.item()

# --- MARKDOWN CELL ---
# ## 5. N-BEATS Baseline

# --- CODE CELL ---
pl_trainer_kwargs_nbeats = {"accelerator": "gpu", "devices": 1, "precision": "16-mixed", "callbacks": [EarlyStopping(monitor="val_loss", patience=5, mode="min"), ModelCheckpoint(dirpath=os.path.join(MODELS_DIR, "nbeats"), filename="best_model", monitor="val_loss", save_top_k=1, mode="min")]}
model_nbeats = NBEATSModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=OUTPUT_CHUNK_LENGTH, num_stacks=4, num_blocks=2, layer_widths=256, n_epochs=50, loss_fn=torch.nn.L1Loss(), optimizer_kwargs={"lr": 1e-4}, pl_trainer_kwargs=pl_trainer_kwargs_nbeats, model_name="NBEATS_baseline", save_checkpoints=True, force_reset=True, random_state=42)
model_nbeats.fit(dm)

best_nbeats = NBEATSModel.load_from_checkpoint(model_name="NBEATS_baseline", work_dir=MODELS_DIR, best=True)
trainer_nbeats = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed")
predictions_nbeats = trainer_nbeats.predict(best_nbeats, dm.val_dataloader())

all_preds_nbeats = np.concatenate([p[0] for p in predictions_nbeats])
all_targets_nbeats = np.concatenate([p[1] for p in predictions_nbeats])
all_masks_nbeats = np.concatenate([p[2] for p in predictions_nbeats])
asset_ids_nbeats = [a for p in predictions_nbeats for a in p[3]]
preds_inv_nbeats = dm._inverse_transform_batch_cached(all_preds_nbeats, asset_ids_nbeats)
targets_inv_nbeats = dm._inverse_transform_batch_cached(all_targets_nbeats, asset_ids_nbeats)
final_smape_nbeats = masked_smoothed_smape(torch.from_numpy(targets_inv_nbeats), torch.from_numpy(preds_inv_nbeats), torch.from_numpy(all_masks_nbeats))
results["N-BEATS"] = final_smape_nbeats.item()

# --- MARKDOWN CELL ---
# ## 6. Model Evaluation Summary
# 
# Here we collect the validation sMAPE scores from all the baseline models and present them in a summary table for comparison.

# --- CODE CELL ---
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Validation sMAPE'])
results_df = results_df.sort_values('Validation sMAPE', ascending=True)

print("--- Baseline Model Evaluation Summary ---")
print(results_df)