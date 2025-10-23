# Project Plan: Electricity Market Forecasting

This document outlines a comprehensive, step-by-step plan for the electricity-market forecasting assignment.
It incorporates strategies for data handling, advanced modeling, GPU acceleration, and systematic improvement to ensure a rigorous and high-performing solution.

----------

### Phase 0: Project Setup & Environment

The goal is to create a structured project environment and install all necessary libraries.

-   create `.gitignore`,  add `.venv` and `.idea` as well as other common files and folder locations.
-   Create folders: `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `results/`.
    
-   Generate `requirements.txt`. 
    ```
		pandas
		numpy
		scikit-learn
		matplotlib
		lightgbm
		darts[pytorch]
		jupyterlab
		optuna
		pytorch
    ```
----------

### Phase 0.5.1: GPU Acceleration & VRAM Management Strategy

This phase ensures all intensive tasks run on your NVIDIA GPU and provides a plan to manage the 8GB VRAM constraint.

-   **Action 1: Configure Libraries for GPU Execution.**
    
    -   **Darts/PyTorch:** Verify CUDA is available with `torch.cuda.is_available()`. When training, pass `pl_trainer_kwargs={"accelerator":"gpu","devices":1,"precision":"16-mixed"}` to the `.fit()` method.
        
    -   **LightGBM:** Install the GPU-supported version and instantiate the model with the `device='gpu'` parameter.
        
-   **Action 2: VRAM tactics (to be tried in order in case of Memory Issues)**
    
	-   reduce `batch_size` (16 → 8 → 4),
       
	-   smaller models (decrease `d_model`, `n_layers`, `n_heads`),
     
	-   out-of-core training for LightGBM if needed.
        
----------

### Phase 0.5.2: Data Loading and Streaming Strategy

To efficiently handle large multivariate datasets without exceeding RAM limits, this project will implement **disk-based streaming** through a **PyTorch DataLoader** compatible with the Darts backend.

#### **Action 1: Streaming via IterableDataset**

-   Implement a custom `IterableDataset` in `src/data_loader.py` that:
    
    -   Reads preprocessed parquet or Zarr chunks per asset directly from disk.
        
    -   Yields `(input_window, target_window, exogenous_features)` tuples lazily.
        
    -   Minimizes memory footprint by only keeping active batches in RAM.
        
-   Use `torch.utils.data.DataLoader` with:
    
    `DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)` 
    
    to feed batches to GPU automatically through Darts / PyTorch Lightning.
    

#### **Action 2: Integration with Darts**

-   When using Darts models, ensure data is provided via a Lightning-compatible DataLoader rather than preloaded `TimeSeries` objects.
    
-   For baselines (LSTM, PatchTST), data will stream batch-by-batch from disk to GPU, allowing out-of-core training on 8GB VRAM.
    

#### **Action 3: Subsampling and Full-Training Protocol**

-   During Optuna tuning, use representative subsets of assets (based on liquidity tiers) loaded into memory.
    
-   For final training, switch to streaming DataLoader mode for full Train+Validation (`2021–2023`) runs.

----------

###  Phase 1: Data Exploration, Preprocessing, and Feature Engineering

The objective is to understand the data's unique structure and engineer features that will guide the models.

**Action 1: Initial Data Loading and EDA.** 		

	- Load the data and visualize a few sample contracts to understand the lifecycle of trading periods interspersed with zeros. Check for `NaNs` and analyze the basic statistics.
	- Visualize H/L/C/V and `Volume>0` coverage per asset. Compute liquidity (fraction of non-zero records).
	- Correlation heatmaps: within-asset correlations between H/L/C/V and cross-asset correlations between neighboring delivery contracts.
    
**Action 2: Create a Preprocessing Script (`src/preprocess.py`).**

This script will contain functions to transform the data.

- Create standard pipeline functions: `load()`, `clean()`, `feature_engineer()`, `split()`, `scale()`. 
- **Targets**: treat `High, Low, Close, Volume` as **multivariate targets** for every asset. Save data layout spec (asset × timestamp × 4).
    
   1.  **Feature Engineering:** Add columns for:
        
        -   Keep zero records (they encode non-trading). Add **`is_trading`:** A binary flag based on `Volume > 0`.
            
        -   Time-based Features: -   `hour_of_day`, `minute_of_day`, `day_of_week`, `week_of_year`, `month`, `is_weekend`,   `time_to_delivery`. For a contract like Tue11Q4 (delivering Tuesday from 11:45 to 12:00), at any given timestamp (e.g., Monday at 09:00), time_to_delivery would be the duration between these two points. `daylight_indicator` (simple proxy: hour between sunrise/sunset — approximate by month+hour.

	- Cross-contract & relational features: For each asset at time t, include last known `Close` / `Volume` of adjacent delivery contracts (e.g., ±3 contract window) as exogenous features. Rolling stats for each target (computed only across non-zero windows): 4-step MA, 8-step MA, 12-step std, etc.
            
    2.  **Data Splitting:** Implement a strict chronological split: **Train** (`2021-2022`), **Validation** (`2023`), **Test** (`2024`).
        
    3.  **Scaling:** Fit a `StandardScaler` for each asset **only on its non-zero training data**. 

**Note:** Save these scalers to be applied to the validation and test sets. Save processed artifacts: feature lists, and split indices.
        

----------

### Phase 2: Baseline Modeling

Establish strong baselines to prove the value of more complex models.

-   **Naive Forecasts**
    
    -   **Last-value** baseline: predict next 10 steps with last observed non-zero Close/High/Low/Volume separately.
        
    -   **Mean baseline**: monthly average for that (weekday,hour) pair.
        
-   **LightGBM (tabular baseline)**
    
	- Implement a Darts `LightGBM` model.

    -   Create sliding-window features (lookback = e.g., 4, 8, 24 steps) and include exogenous features (hour, month).
        
    -   Approach: a single LightGBM per target using asset-id as categorical feature.
        
    -   Use out-of-fold validation along the time axis (no leakage).
        
    -   Use GPU-enabled LightGBM and early stopping.
        
-   **RNN/LSTM Baseline (basic deep baseline)**

	-   Implement a Darts `RNNModel`.
	    
	-   Example config: `input_chunk_length=96` (24h), `output_chunk_length=10`, `hidden_size=128`, `n_layers=2`.
	    
	-   Train with early stopping on validation sMAPE. Train on L1 loss (MAE); validate on masked sMAPE.

-   **N-BEATS**

	-   Use Darts `NBEATSModel` with multivariate TimeSeries (4 channels H/L/C/V). If multivariate mode runs out of memory or is unsupported in your version, train four independent models (one per target).

	- Recommended config: `input_chunk_length = 96`, `output_chunk_length = 10`, `num_stacks = 4`, `num_blocks = 2`, `layer_widths = 256`.

	- GPU acceleration enabled through PyTorch backend.

	- Training loss = L1; monitor masked sMAPE on validation.

----------

###  Phase 3: Advanced Deep Learning Modeling — Pretrained PatchTSMixer (Fine-tuning)

Leverage a foundation-level, patch-based time-series model using the `transformers.PatchTSMixerForPrediction` class.

#### **Model overview**

-   Supports multivariate forecasting with configurable context length, patch length, and prediction horizon.
    
-   Efficient and VRAM-friendly (backbone `d_model ≈ 16–64`, 3–6 layers).
    
-   Handles `observed_mask` for missing or zero-trading periods.
    
-   Outputs shape `(batch, prediction_length, num_input_channels)` — ideal for H/L/C/V × 10-step forecast.

#### **Recommended configuration (starting point)**

`from transformers import PatchTSMixerConfig
config = PatchTSMixerConfig(
    context_length = 96, # 24 h lookback (15-min cadence) patch_length   = 8, # 2 h per patch patch_stride   = 8, # non-overlapping patches num_input_channels = 4, # H,L,C,V prediction_length = 10,
    d_model = 32,
    num_layers = 4,
    dropout = 0.15,
    mode = "common_channel", # or "mix_channel" if cross-correlation matters )` 

**If VRAM issues arise (8 GB GPU):**

-   Reduce `d_model` → 16 or 24.
    
-   Reduce `num_layers` → 2 or 3.
    
-   Lower `batch_size` → 8 → 4.
    
-   Enable mixed precision (`precision="16-mixed"`).
    
#### **Lightning integration (summary)**

-   Wrap model in `pl.LightningModule` (`src/models/patchtsmixer_module.py`).
    
-   Use custom `IterableDataset` (`src/data_loader.py`) yielding `(past, future, mask)`.
    
-   Implement training with:
    
    -   **Loss:** L1 (MAE) for training.
        
    -   **Metric:** masked sMAPE for validation/Optuna.
        
    -   **Gradient clipping:** `clip_grad_norm_(1.0)`.
        
    -   **Mixed precision:** `precision="16-mixed"`.
        
    -   **Early stopping:** `monitor="val_smape"`.
        
    -   **Checkpointing:** save best model per stage.
        
#### **Fine-tuning procedure (gradual unfreezing)**

1.  **Feature-extraction stage**  
    - Freeze all backbone layers; train head only (~ 3–5 epochs, LR ≈ 1e-4).
    
2.  **Partial unfreeze**  
    - Unfreeze last 1–2 mixer layers; use discriminative LR groups (head 5e-4, last layers 5e-5); train 5–10 epochs.
    
3.  **Full fine-tune**  
    - Unfreeze all layers; LR schedule (head 1e-4, backbone early 1e-5) with cosine decay and warm-up; 10–20 epochs.  
    - Apply weight decay (1e-4) + gradient clipping.
    
#### **How to use (pretrained workflow)**

`from transformers import PatchTSMixerForPrediction import torch

model = PatchTSMixerForPrediction.from_pretrained("ibm/patchtsmixer-etth1-pretrain", config=config)
model.cuda()

past_values   = torch.randn(4, 96, 4).cuda() # (B, context_length, channels) future_values = torch.randn(4, 10, 4).cuda() # targets (optional) observed_mask = (past_values != 0).float()

out = model(past_values=past_values,
            future_values=future_values,
            observed_mask=observed_mask,
            return_dict=True,
            return_loss=False)
preds = out.prediction_outputs # (B, 10, 4)` 

#### **Lightning training loop (essentials)**

`class  PatchTSMixerLit(pl.LightningModule): def  __init__(self, model, lr_head=1e-4, lr_backbone=5e-5): super().__init__()
        self.model = model
        self.lr_head, self.lr_backbone = lr_head, lr_backbone
        self.criterion = torch.nn.L1Loss() def  forward(self, past, future=None, mask=None):
        out = self.model(past_values=past, future_values=future, return_dict=True) return out.prediction_outputs def  training_step(self, batch, batch_idx):
        past, future, mask = batch
        preds = self(past)
        loss = self.criterion(preds, future)
        self.log("train_loss", loss) return loss def  validation_step(self, batch, batch_idx):
        past, future, mask = batch
        preds = self(past)
        # masked_smoothed_smape will be a custom utility function 
        # defined in src/utils.py that implements the masked sMAPE logic.
        smape = masked_smoothed_smape(future, preds, mask) 
        self.log("val_smape", smape, prog_bar=True) return smape def  configure_optimizers(self):
        head_params = [p for n,p in self.model.named_parameters() if  "head"  in n]
        backbone_params = [p for n,p in self.model.named_parameters() if  "head"  not  in n]
        opt = torch.optim.AdamW([
            {"params": head_params, "lr": self.lr_head},
            {"params": backbone_params, "lr": self.lr_backbone}
        ], weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10) return [opt], [sch]` 

#### **Validation metric**

Use **masked sMAPE** (computed only where `is_trading = 1`) for model selection and Optuna tuning.  
Training loss remains L1 for stability.

----------

###  Phase 4: Systematic Hyperparameter Tuning

Ensure your models are performing optimally by tuning their key parameters. Use a library like **Optuna** for an efficient search but **constrain search complexity**:

- Tune LightGBM on full training set.
- Tune neural models on **subsampled assets** or shorter time windows (select representative, liquid assets).
- Subsample selection strategy:
	- Compute asset liquidity (fraction non-zero). Partition assets into High, Medium, Low liquidity buckets (e.g., top 20%, mid 60%, bottom 20%).
	- Choose a representative subset for tuning: e.g., 3 from High, 3 from Medium, 2 from Low.
	- Use time windows representative of different months or seasons (e.g., winter block and summer block) if seasonality is strong.
- Key params:
	-   PatchTSMixer: `d_model`, `num_layers`, `dropout`, `mode`, `learning_rate`, `batch_size`, `patch_length`.    
	-   LSTM: `hidden_size`, `n_layers`, `dropout`.    
	-   LightGBM: `num_leaves`, `max_depth`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda`.
- Training Loss: For neural networks (LSTM, PatchTST), use a smooth, gradient-friendly loss function like Mean Absolute Error (MAE / L1Loss) for more stable training.
- Run Optuna on the representative subsample(s). Select best hyperparams per-architecture (LSTM / PatchTST / LightGBM). Validate transferability: 
	- Retrain the chosen hyperparams on a different subsample to check for consistent validation sMAPE. If performance varies widely, consider architecture-specific hyperparams per-liquidity bucket.
- Optimization target: minimize **validation masked sMAPE** (only for time steps where true value ≠ 0).
- Use a budgeted search (50–100 trials for LightGBM; 20–50 for each neural model on subsample).
- Save best trials and transfer best hyperparams to full training.

----------

###  Phase 5: Iteration and Improvement Scenarios

-   **Scenario 1: Enhanced Feature Engineering.**
    
	-   Add additional rolling windows, cross-contract lags, and interaction features (e.g., `close - lag_close_of_adjacent_contract`).
	    
	-   Test `is_trading` as input and/or mask out predictions where `is_trading` is false for sMAPE calculation.
        
-   **Scenario 2: Experiment with the Lookback Window.**
    
	-   Try `input_chunk_length`: 24, 96, 288 (short → medium → long history).
	    
	-   Log GPU/time tradeoffs; pick best validation sMAPE vs compute.
        
-   **Scenario 3: Create a Model Ensemble.** Combine the strengths of fundamentally different models. The best candidates are:
    
	-   Stack LightGBM + PatchTST + LSTM predictions.
	    
	-   Meta-models: simple Ridge / Linear Regression, or small NN trained on validation fold.
	    
	-   Ensemble at per-target level (one meta model per H/L/C/V may work best).
        
-   **Scenario 4: Stabilize with Multiple Seeds.** 
    
	-   Train best architectures with 3–5 different seeds; average predictions.
	    
	-   Record mean ± std of validation sMAPE.

----------

###  Phase 6: Final Evaluation & Reporting 

Execute the final run of all models on the test set and present your findings clearly.

-   **Action 1: Implement Masked sMAPE Metric.** Create your evaluation function that calculates sMAPE **only on time steps where the true value is not zero**.
    
-   **Action 2: Final Model Training and Prediction.**
          
    1.  **Re-train all baseline and other worthy models on the combined training and validation data (`2021-2023`).**
        
    2.  Generate final predictions on the unseen `2024` test set.
        
-   **Action 3: Generate Final Results Table & Conclude.** Present a clear table comparing all attempted models on the validation and test sets. Discuss your findings, justify why the winning model performed best, and reflect on the challenges of the project.

-   **Re-train final selected models** on combined Train+Validation (`2021-2023`) using the chosen hyperparams.
    
-   **Final predictions** on **test 2024** (only once).
    
-   **Evaluation metric**
    -   Implement **masked sMAPE** that ignores zero true values 
    -   Compute sMAPE per asset and per target (H,L,C,V), then aggregate (mean, median) across assets.
     
- **Results table (must include per-variable breakdown)** — example columns:
    
    `Model | Target | Val  sMAPE | Test  sMAPE | Train  time | Notes` 
    
    -   Provide per-model, per-target rows (or nested table) so graders can see how models perform across H/L/C/V.
        
- **Ablations & Diagnostics**
    
    -   Liquidity sensitivity: performance on high-liquidity vs low-liquidity assets.
        
    -   Cross-contract feature ablation: model with vs without cross-contract inputs.
        
    -   Error time-of-day and weekday analysis.

###  Phase 7 — Reproducibility & Submission

-   Store code/notebooks and `requirements.txt`, plus:
    
    -   `models/` with best checkpoints,
        
    -   `scalers/` per asset,
        
    -   `results/` with predictions (CSV) and evaluation scripts.
        
-   Short `report.pdf` with:
    
    -   problem summary, modeling choices, final results table, key diagnostics, caveats.
        
-   Include a short appendix with commands to reproduce the final run (one-liner scripts or `run.sh`).