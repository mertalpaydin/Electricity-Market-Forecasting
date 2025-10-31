# Project Plan: Electricity Market Forecasting

This document outlines a comprehensive, step-by-step plan for the electricity-market forecasting assignment.
It incorporates strategies for data handling, advanced modeling, GPU acceleration, and systematic improvement to ensure a rigorous and high-performing solution.

>**Workflow Note:** The core, reusable logic (like data loaders and preprocessing functions) resides in the `src/` directory. However, the main workflow execution, from data exploration to model training and evaluation, is orchestrated through Jupyter Notebooks in the `notebooks/` directory. This allows for interactive development and clear reporting.

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
        
-   **Action 2: VRAM tactics (to be tried in order in case of Memory Issues)**
    
	-   reduce `batch_size` (16 → 8 → 4),
       
	-   smaller models (decrease `d_model`, `n_layers`, `n_heads`),
        
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
    
-   For data heavy models (LSTM, Chronos etc.), data will stream batch-by-batch from disk to GPU, allowing out-of-core training on 8GB VRAM.
    

#### **Action 3: Subsampling and Full-Training Protocol**

-   During Optuna tuning, use representative subsets of assets (based on liquidity tiers) loaded into memory.
    
-   For final training, switch to streaming DataLoader mode for full Train+Validation (`2021–2023`) runs.

----------

###  Phase 1: Data Exploration, Preprocessing, and Feature Engineering

The objective is to understand the data's unique structure and engineer features that will guide the models.

**Action 1: Initial Data Loading and EDA (in `notebooks/01_EDA.ipynb`).** 		

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

        -   Cross contract features - add H, L, C, V from previous and next contracts
            
       2.  **Data Splitting:** Implement a strict chronological split: **Train** (`2021-2022`), **Validation** (`2023`), **Test** (`2024`).
            
       3.  **Scaling:** Fit a `StandardScaler` for each asset **only on its non-zero training data**. 

**Note:** Save these scalers to be applied to the validation and test sets. Save processed artifacts: feature lists, and split indices.
        
----------

### Phase 1.5: Come up with a Supsampling strategy

- The data is quite large and fill with rows `is_trading` = 0.
- Training all the full dataset is not possible and possible does not make sense since non trading data doesn't have much signal.
- Come up with a clever supsampling strategy (i.e seperate data into trading non_trading times, keep only small portion of non trading)
- Test different supsampling approaches with respect to training speed and convergence of Chronos as it is the main model before full scale implementation.


----------

### Phase 2: Baseline Modeling (in `notebooks/02_Baseline_Models.ipynb`)

Establish strong baselines to prove the value of more complex models.
Implementation of baseline models should allow hyperparameter tuning (Phase 4) before training.

-   **Naive Forecasts**
    
    -   **Last-value** baseline: predict next 10 steps with last observed non-zero Close/High/Low/Volume separately.
        
    -   **Mean baseline**: monthly average for that (weekday,hour) pair.
        
-   **RNN/LSTM Baseline (basic deep baseline)**

	-   Implement a Darts `RNNModel`.
	    
	-   Example config: `input_chunk_length=96` (24h), `output_chunk_length=10`, `hidden_size=128`, `n_layers=2`.
	    
	-   Train with early stopping on validation sMAPE. Train on L1 loss (MAE); validate on masked sMAPE.

    -   Train LSTM maximum for 2 hours. Train LSTM Until it beats sMAPE of the naive forecast.
    -   If it cannot beat it in 2 hours, stop training. If it can, train for another 5-10 epochs.

----------

###  Phase 3: Advanced Deep Learning Modeling (in `notebooks/04_Chronos_Training.ipynb`)

Leverage a foundation-level, Chronos2.
Implementation of this model should allow hyperparameter tuning (Phase 4) before training.

NOTE: 
 
#### **Lightning integration (summary)**

-   Wrap model in `pl.LightningModule` (`src/models/chronos.py`).
    
-   Use custom `IterableDataset` (`src/data_loader.py`) yielding `(past, future, mask)`.
    
-   Implement training with:
    
    -   **Loss:** L1 (MAE) for training.
        
    -   **Metric:** masked sMAPE for validation/Optuna.
        
    -   **Early stopping:** `monitor="val_smape"`.
        
    -   **Checkpointing:** save best model per stage.
    -   **Training Time** Hardcapped at 3 hours. Try to beat the sMAPE of the best baseline model. If beaten early test suggestions from Phase 5.
        
#### **Fine-tuning procedure (gradual unfreezing)**

1.  **Feature-extraction stage**  
    - Freeze all backbone layers; train head only (~ 3–5 epochs, LR ≈ 1e-4).
    
2.  **Partial unfreeze**  
    - Unfreeze last 1–2 mixer layers; use discriminative LR groups (head 5e-4, last layers 5e-5); train 5–10 epochs.
    
3.  **Full fine-tune**  
    - Unfreeze all layers; LR schedule (head 1e-4, backbone early 1e-5) with cosine decay and warm-up; 10–20 epochs.  
    - Apply weight decay (1e-4) + gradient clipping.

#### **Validation metric**

Use **masked sMAPE** (computed only where `is_trading = 1`) for model selection and Optuna tuning.  
Training loss remains L1 for stability.

----------

###  Phase 4: Systematic Hyperparameter Tuning

Ensure your models are performing optimally by tuning their key parameters. Use a library like **Optuna** for an efficient search but **constrain search complexity**:

- Tune neural models on **subsampled assets** or shorter time windows (select representative, liquid assets).
- Subsample selection strategy:
	- Compute asset liquidity (fraction non-zero). Partition assets into High, Medium, Low liquidity buckets (e.g., top 20%, mid 60%, bottom 20%).
	- Choose a representative subset for tuning: e.g., 3 from High, 3 from Medium, 2 from Low.
	- Use time windows representative of different months or seasons (e.g., winter block and summer block) if seasonality is strong.
- Key params:
	-   Chronos: `quantile_levels`, `batch_size`, `lr`, `max_context_length`.    
	-   LSTM: `hidden_size`, `n_layers`, `dropout`.
- Training Loss: For neural networks (LSTM, Chronos), use a smooth, gradient-friendly loss function like Mean Absolute Error (MAE / L1Loss) for more stable training.
- Run Optuna on the representative subsample(s). Select best hyperparams per-architecture (LSTM / Chronos ). Validate transferability: 
	- Retrain the chosen hyperparams on a different subsample to check for consistent validation sMAPE. If performance varies widely, consider architecture-specific hyperparams per-liquidity bucket.
- Optimization target: minimize **validation masked sMAPE** (only for time steps where true value ≠ 0).
- Save best trials and transfer best hyperparams to full training.

----------

###  Phase 5: Iteration and Improvement Scenarios (If time permits)

-   **Scenario 1: Enhanced Feature Engineering.**
    
	-   Add interaction features (e.g., `close - lag_close_of_adjacent_contract`).
	    
	-   Test `is_trading` as input and/or mask out predictions where `is_trading` is false for sMAPE calculation.
        
-   **Scenario 2: Experiment with the Lookback Window.**
    
	-   Try `input_chunk_length`: 24, 96, 288 (short → medium → long history).
	    
	-   Log GPU/time tradeoffs; pick best validation sMAPE vs compute. 
            
-   **Scenario 4: Stabilize with Multiple Seeds.** 
    
	-   Train best architectures with 3–5 different seeds; average predictions.
	    
	-   Record mean ± std of validation sMAPE.

----------

###  Phase 6: Final Evaluation & Reporting (in `notebooks/04_Evaluation.ipynb`)

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
        
    -   Error time-of-day and weekday analysis.

- - **Explanation of Executive Choices**

    -   Review back all the code, prepare a readme on how to run it and explain all the decisions made with its reasoning.

###  Phase 7 — Reproducibility & Submission

-   Store code/notebooks and `requirements.txt`, plus:
    
    -   `models/` with best checkpoints,
        
    -   `scalers/` per asset,
        
    -   `results/` with predictions (CSV) and evaluation scripts.
        
-   Short `report.pdf` with:
    
    -   problem summary, modeling choices, final results table, key diagnostics, caveats.
        
    -   Include a short appendix with commands to reproduce the final run (one-liner scripts or `run.sh`).