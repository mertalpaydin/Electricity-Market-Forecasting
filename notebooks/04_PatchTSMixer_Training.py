# --- MARKDOWN CELL ---
# # Phase 3: Advanced Deep Learning Modeling (PatchTSMixer)
#
# This notebook implements the advanced `PatchTSMixer` model from the `transformers` library. We will use the optimal hyperparameters found during the tuning phase to configure the model.
#
# The process is as follows:
# 1.  **Setup**: Load libraries and the best hyperparameters from the JSON file.
# 2.  **Data Loading**: Instantiate the `ElectricityDataModule` for the full dataset.
# 3.  **Model Definition**: Define the `PatchTSMixerLit` wrapper and instantiate the model with the tuned parameters.
# 4.  **Training**: Train the model on the full training set using the PyTorch Lightning `Trainer`.
# 5.  **Evaluation**: Evaluate the final model's performance on the full validation set.

# --- CODE CELL ---
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import PatchTSMixerForPrediction, PatchTSMixerConfig

# Add src directory to path
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.datamodule import ElectricityDataModule, masked_smoothed_smape

# --- MARKDOWN CELL ---
# ## 1. Setup and Configuration

# --- CODE CELL ---
# Define project paths
BASE_DIR = ".."
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SCALERS_DIR = os.path.join(DATA_DIR, "scalers")
MODELS_DIR = os.path.join(BASE_DIR, "models", "patchtsmixer")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)

# Load the best hyperparameters
with open(os.path.join(RESULTS_DIR, 'best_hyperparameters.json'), 'r') as f:
    best_hyperparams = json.load(f)

print("Loaded best hyperparameters for PatchTSMixer:")
print(json.dumps(best_hyperparams['patchtsmixer'], indent=4))

# Configuration
BATCH_SIZE = 32
INPUT_CHUNK_LENGTH = 96
OUTPUT_CHUNK_LENGTH = 10
TARGET_COLS = ["high", "low", "close", "volume"]

# --- MARKDOWN CELL ---
# ## 2. Data Loading
# 
# We instantiate the `ElectricityDataModule` to feed data to the model in a streaming fashion.

# --- CODE CELL ---
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
    }
)
dm.setup(stage="fit")

# --- MARKDOWN CELL ---
# ## 3. Model Definition
# 
# We define the `PatchTSMixerLit` wrapper class and instantiate the model using the tuned hyperparameters.

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

# Get tuned parameters
params = best_hyperparams['patchtsmixer']

# Create model config
config = PatchTSMixerConfig(
    context_length=INPUT_CHUNK_LENGTH,
    prediction_length=OUTPUT_CHUNK_LENGTH,
    num_input_channels=len(TARGET_COLS),
    d_model=params['d_model'],
    num_layers=params['n_layers'],
    dropout=params['dropout'],
    mode="common_channel"
)

# Instantiate the Lightning model
model = PatchTSMixerLit(config, lr=params['lr'])

# --- MARKDOWN CELL ---
# ## 4. Training
# 
# We'll use the PyTorch Lightning `Trainer` to handle the training process, including GPU acceleration, mixed precision, and checkpointing.

# --- CODE CELL ---
# Define trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=100,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(
            dirpath=MODELS_DIR, 
            filename="best_model", 
            monitor="val_loss", 
            save_top_k=1, 
            mode="min"
        )
    ]
)

# Train the model
print("--- Training PatchTSMixer Model ---")
trainer.fit(model, dm)

# --- MARKDOWN CELL ---
# ## 5. Evaluation
# 
# Finally, we load the best model from the checkpoint and evaluate its performance on the full validation set.

# --- CODE CELL ---
print("--- Evaluating PatchTSMixer Model ---")

# Load the best model
best_model = PatchTSMixerLit.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

# Get predictions on the validation set
predictions = trainer.predict(best_model, dm.val_dataloader())

# The predict loop returns a list of batches, so we need to re-assemble them
all_preds = []
all_targets = []
all_masks = []
all_asset_ids = []

for batch in predictions:
    # The model output is just the prediction tensor
    all_preds.append(batch)
    # We need to get the corresponding targets and masks from the dataloader
    # This is a bit tricky as predict doesn't return them. We'll re-iterate the val_dataloader.

# Re-iterate to get all data for sMAPE calculation
val_loader = dm.val_dataloader()
for batch in val_loader:
    _, _, future_values, future_mask, asset_ids, _ = batch
    all_targets.append(future_values.cpu().numpy())
    all_masks.append(future_mask.cpu().numpy())
    all_asset_ids.extend(asset_ids)

all_preds = np.concatenate([p.cpu().numpy() for p in all_preds])
all_targets = np.concatenate(all_targets)
all_masks = np.concatenate(all_masks)

# Inverse transform and calculate sMAPE
preds_inv = dm._inverse_transform_batch_cached(all_preds, all_asset_ids)
targets_inv = dm._inverse_transform_batch_cached(all_targets, all_asset_ids)

final_smape = masked_smoothed_smape(
    torch.from_numpy(targets_inv),
    torch.from_numpy(preds_inv),
    torch.from_numpy(all_masks)
)

print(f"\n--- PatchTSMixer Validation Performance ---")
print(f"Overall sMAPE on Validation Set: {final_smape.item():.4f}")
