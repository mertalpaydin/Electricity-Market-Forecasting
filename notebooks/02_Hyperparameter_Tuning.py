# --- MARKDOWN CELL ---
# # Phase 4: Systematic Hyperparameter Tuning
#
# In this notebook, we use Optuna to find the optimal hyperparameters for our baseline models before running the final training and evaluation. As per the project plan, we will tune the deep learning models on a representative subsample of the data to save time, while tree-based models can be tuned on a larger set if needed.
#
# The process is as follows:
# 1.  **Asset Subsampling**: We will analyze the liquidity of all assets and create a small, representative subset for tuning.
# 2.  **Optuna Objective Functions**: We will define an objective function for each model architecture (LSTM, LightGBM, PatchTSMixer) that Optuna will seek to minimize (the validation sMAPE or loss).
# 3.  **Run Studies**: We will execute the tuning studies for a predefined number of trials.
# 4.  **Save Best Parameters**: The best hyperparameters found will be saved for use in the next phase.

# --- CODE CELL ---
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import optuna
from darts import TimeSeries
from darts.models import RNNModel, LightGBMModel
from transformers import PatchTSMixerForPrediction, PatchTSMixerConfig

# Add src directory to path
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.datamodule import ElectricityDataModule, masked_smoothed_smape

# --- MARKDOWN CELL ---
# ## 1. Setup
# 
# Define paths and constants for the tuning process.

# --- CODE CELL ---
# Define project paths
BASE_DIR = ".."
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SCALERS_DIR = os.path.join(DATA_DIR, "scalers")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
INPUT_CHUNK_LENGTH = 96
OUTPUT_CHUNK_LENGTH = 10
TARGET_COLS = ["high", "low", "close", "volume"]
N_TRIALS_LSTM = 30
N_TRIALS_LGBM = 50
N_TRIALS_PATCH = 30

# --- MARKDOWN CELL ---
# ## 2. Asset Subsampling for Efficient Tuning

# --- CODE CELL ---
def get_asset_liquidity():
    """Calculates the trading liquidity for each asset in the training set."""
    asset_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.parquet')]
    liquidity_map = {}
    for asset_file in asset_files:
        df = pd.read_parquet(os.path.join(TRAIN_DIR, asset_file))
        liquidity = (df['volume'] > 0).mean()
        liquidity_map[asset_file.replace('.parquet', '')] = liquidity
    return pd.Series(liquidity_map)

liquidity = get_asset_liquidity()

high_liquidity_threshold = liquidity.quantile(0.80)
low_liquidity_threshold = liquidity.quantile(0.20)
high_liquidity_assets = liquidity[liquidity >= high_liquidity_threshold].index.tolist()
medium_liquidity_assets = liquidity[(liquidity > low_liquidity_threshold) & (liquidity < high_liquidity_threshold)].index.tolist()
low_liquidity_assets = liquidity[liquidity <= low_liquidity_threshold].index.tolist()

np.random.seed(42)
subsample_high = np.random.choice(high_liquidity_assets, 3, replace=False).tolist()
subsample_medium = np.random.choice(medium_liquidity_assets, 3, replace=False).tolist()
subsample_low = np.random.choice(low_liquidity_assets, 2, replace=False).tolist()
TUNING_ASSETS = subsample_high + subsample_medium + subsample_low
print(f"Selected {len(TUNING_ASSETS)} assets for tuning:", TUNING_ASSETS)

# --- MARKDOWN CELL ---
# ## 3. PatchTSMixer Hyperparameter Tuning

# --- CODE CELL ---
class PatchTSMixerLit(pl.LightningModule):
    def __init__(self, model_config, lr):
        super().__init__()
        self.model = PatchTSMixerForPrediction(model_config)
        self.lr = lr
        self.criterion = torch.nn.L1Loss()

    def training_step(self, batch, batch_idx):
        past_values, past_mask, future_values, _, _, _ = batch
        outputs = self.model(past_values=past_values, past_observed_mask=past_mask, future_values=future_values)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        past_values, past_mask, future_values, _, _, _ = batch
        outputs = self.model(past_values=past_values, past_observed_mask=past_mask, future_values=future_values)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def objective_patchtsmixer(trial: optuna.trial.Trial) -> float:
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    config = PatchTSMixerConfig(
        context_length=INPUT_CHUNK_LENGTH,
        prediction_length=OUTPUT_CHUNK_LENGTH,
        num_input_channels=len(TARGET_COLS),
        d_model=d_model,
        num_layers=n_layers,
        dropout=dropout,
        mode="common_channel"
    )

    model = PatchTSMixerLit(config, lr)
    dm_subsample = ElectricityDataModule(
        train_parquet=TRAIN_DIR, val_parquet=VAL_DIR, scalers_dir=SCALERS_DIR, batch_size=32,
        dataset_kwargs={"input_chunk_length": INPUT_CHUNK_LENGTH, "output_chunk_length": OUTPUT_CHUNK_LENGTH, "target_cols": TARGET_COLS, "assets_list": TUNING_ASSETS}
    )
    
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="16-mixed", max_epochs=20, logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")]
    )
    trainer.fit(model, dm_subsample)
    return trainer.callback_metrics["val_loss"].item()

study_patch = optuna.create_study(direction="minimize", study_name="PatchTSMixer_tuning")
study_patch.optimize(objective_patchtsmixer, n_trials=N_TRIALS_PATCH)
best_params_patch = study_patch.best_params
print("--- Best PatchTSMixer Hyperparameters ---", best_params_patch)

# --- MARKDOWN CELL ---
# ## 4. LSTM Hyperparameter Tuning

# --- CODE CELL ---
def objective_lstm(trial: optuna.trial.Trial) -> float:
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, log=True)
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    model = RNNModel(
        model="LSTM", input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=OUTPUT_CHUNK_LENGTH,
        hidden_dim=hidden_dim, n_rnn_layers=n_rnn_layers, dropout=dropout, n_epochs=30,
        loss_fn=torch.nn.L1Loss(), optimizer_kwargs={"lr": lr}, model_name=f"LSTM_trial_{trial.number}",
        save_checkpoints=False, force_reset=True, random_state=42
    )
    
    dm_subsample = ElectricityDataModule(
        train_parquet=TRAIN_DIR, val_parquet=VAL_DIR, scalers_dir=SCALERS_DIR, batch_size=32,
        dataset_kwargs={"input_chunk_length": INPUT_CHUNK_LENGTH, "output_chunk_length": OUTPUT_CHUNK_LENGTH, "target_cols": TARGET_COLS, "assets_list": TUNING_ASSETS}
    )
    
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="16-mixed", max_epochs=30, logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")]
    )
    trainer.fit(model, dm_subsample)
    return trainer.callback_metrics["val_loss"].item()

study_lstm = optuna.create_study(direction="minimize", study_name="LSTM_tuning")
study_lstm.optimize(objective_lstm, n_trials=N_TRIALS_LSTM)
best_params_lstm = study_lstm.best_params
print("--- Best LSTM Hyperparameters ---", best_params_lstm)

# --- MARKDOWN CELL ---
# ## 5. LightGBM Hyperparameter Tuning

# --- CODE CELL ---
def objective_lgbm(trial: optuna.trial.Trial) -> float:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "objective": "regression_l1", "metric": "mae", "device_type": "gpu", "n_jobs": -1,
    }
    all_smape_scores = []
    for asset_name in TUNING_ASSETS:
        train_df = pd.read_parquet(os.path.join(TRAIN_DIR, f"{asset_name}.parquet"))
        val_df = pd.read_parquet(os.path.join(VAL_DIR, f"{asset_name}.parquet"))
        train_df['ExecutionTime'] = pd.to_datetime(train_df['ExecutionTime'])
        val_df['ExecutionTime'] = pd.to_datetime(val_df['ExecutionTime'])
        for target in TARGET_COLS:
            ts_train_target = TimeSeries.from_dataframe(train_df, 'ExecutionTime', [target], freq='15min')
            ts_val_target = TimeSeries.from_dataframe(val_df, 'ExecutionTime', [target], freq='15min')
            model = LightGBMModel(lags=INPUT_CHUNK_LENGTH, output_chunk_length=OUTPUT_CHUNK_LENGTH, lgbm_params=params)
            model.fit(series=ts_train_target)
            preds = model.predict(n=len(ts_val_target))
            mask = (ts_val_target.values() != 0).astype(float)
            all_smape_scores.append(masked_smoothed_smape(torch.from_numpy(ts_val_target.values()), torch.from_numpy(preds.values()), torch.from_numpy(mask)).item())
    return np.mean(all_smape_scores)

study_lgbm = optuna.create_study(direction="minimize", study_name="LightGBM_tuning")
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS_LGBM)
best_params_lgbm = study_lgbm.best_params
print("--- Best LightGBM Hyperparameters ---", best_params_lgbm)

# --- MARKDOWN CELL ---
# ## 6. Save Best Hyperparameters

# --- CODE CELL ---
best_params = {
    "patchtsmixer": best_params_patch,
    "lstm": best_params_lstm,
    "lightgbm": best_params_lgbm
}
output_path = os.path.join(RESULTS_DIR, "best_hyperparameters.json")
with open(output_path, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Best hyperparameters saved to: {output_path}")
print("\n--- Contents ---")
print(json.dumps(best_params, indent=4))
