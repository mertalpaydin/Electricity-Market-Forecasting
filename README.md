
# ⚡ Electricity Price Forecasting

A comprehensive deep learning project to forecast electricity market prices and volumes from high-frequency (15-minute) trading data.

This repository was developed for the **Deep Learning Course (Autumn 2024)** assignment. It implements an end-to-end pipeline, including a sophisticated multi-stage data processing workflow, baseline modeling (Naive, LSTM), an AutoGluon hybrid ensemble model, and fine-tuning of the **Chronos2** foundation model from Amazon Science.


## 🧠 Project Overview

The goal is to **forecast the next 10 trading intervals** (15 minutes each) for every active contract, across **four targets per asset**:

-   **High**, **Low**, **Close**, and **Volume**

The project builds and compares multiple models, from classical baselines to advanced Transformers, on their ability to predict these values simultaneously. The final evaluation is based on the **masked Symmetric Mean Absolute Percentage Error (sMAPE)**.

## 📊 Dataset Summary

-   **Source:** European Power Exchange (EPEX). Curated and provided by the course instructor.
    
-   **Frequency:** 15-minute HLCV (High, Low, Close, Volume) candles
    
-   **Structure:**
    
    -   Each _contract_ represents a specific delivery time (e.g., “Tue11Q4” = every Tuesday 11:45–12:00).
        
    -   Non-trading periods are filled with **zeros**, representing an "inactive" state.
        
    -   Multiple contracts are traded in parallel, creating complex cross-contract correlations.
        
-   **Target horizon:** **10 steps ahead (2.5 hours)**
    

## 🏗️ Project Structure

This project separates the core, reusable Python library (`src/`) from the executable workflow, which is managed in Jupyter Notebooks (`notebooks/`). This allows for clear, step-by-step execution and easy conversion of the workflow into final reports.

```
electricity-price-forecasting/
│
├── data/                  # Raw and preprocessed data (Parquet format)
│   ├── train/             # Training data (2021-2022)
│   ├── val/               # Validation data (2023)
│   ├── test/              # Test data (2024)
│   └── train_continuous/  # Dense trading-only slices for model training
│   └── val_continuous/
│
├── notebooks/             # Executable workflow and reports
│   ├── 01_EDA.ipynb                         # Initial data exploration
│   ├── 02_Hyperparameter_Tuning.ipynb       # LSTM hyperparameter tuning
│   ├── 03_Baseline_Models.ipynb             # Naive baseline vs tuned LSTM
│   ├── 04_Ensemble_Model.ipynb              # AutoGluon hybrid two-stage model
│   ├── 05_Chronos2_Training.ipynb           # Fine-tuning Chronos2 foundation model
│   └── 06_Final_Evaluation.ipynb  # Final test set evaluation
│
├── src/                   # Core Python library (imported by notebooks)
│   ├── data_loader.py         # Custom PyTorch IterableDataset for streaming
│   ├── datamodule.py          # PyTorch Lightning DataModule
│   ├── preprocess.py          # Feature engineering, scaling, and data splitting
│   └── prepare_continuous_data.py  # Extract dense trading sequences
│
├── results/               # Model outputs and performance metrics
│   ├── best_params_lstm.json
│   ├── lstm_loss_history.png
│   └── chronos2_multivariate_finetuning_results.json
│
├── project_report.md      # Comprehensive project report
├── requirements.txt
├── README.md
└── LICENSE

```


## ⚙️ Environment Setup

1.  **Clone the repo**

    Bash

    ```
    git clone https://github.com/mertalpaydin/Electricity-Market-Forecasting.git
    cd electricity-market-forecasting
    ```
    
2.  **Create virtual environment**
    
    Bash
    
    ```
    python -m venv .venv
    source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)    
    ```
    
3.  **Install dependencies**
    
    Bash
    
    ```
    pip install -r requirements.txt    
    ```
    
4.  **Verify GPU support**
    
    Python
    
    ```
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")    
    ```
    

## 💿 Data Processing Pipeline

The project implements a sophisticated multi-stage data processing pipeline designed to handle large-scale time-series data efficiently:

### **Stage 1: CSV to Parquet Conversion**
-   Massive CSV files are split by individual assets and converted to Parquet format for efficient columnar access
-   Significantly improves read performance and enables per-asset processing

### **Stage 2: Feature Engineering**
-   **Time-based features:** `hour_of_day`, `day_of_week`, `month`, `time_to_delivery`
-   **`is_trading` flag:** Binary indicator for active trading periods
-   **Cross-contract features:** Relationships between neighboring contracts (`close_lag_adj_k`, `nearest_liquid_contract_close`)
-   All features engineered before splitting to ensure consistency

### **Stage 3: Data Splitting and Scaling**
-   Time-based split: Train (2021-2022), Validation (2023), Test (2024)
-   **Selective scaling:** `StandardScaler` fitted only on trading periods to avoid zero-value bias

### **Stage 4: Continuous Data Preparation**
-   Extract dense, information-rich sequences of continuous trading blocks
-   Creates `train_continuous` and `val_continuous` datasets for efficient model training
-   Dramatically reduces training time by focusing on active trading periods
        

## 🚀 Modeling Workflow

### **Phase 1 — Data Exploration (01_EDA.ipynb)**
-   Load HLCV data, visualize asset lifecycles, and identify trading patterns
-   Understand the high sparsity of trading periods vs non-trading periods

### **Phase 2 — LSTM Hyperparameter Tuning (02_Hyperparameter_Tuning.ipynb)**
-   Systematic hyperparameter search for LSTM on continuous trading data
-   Optimize: `hidden_dim`, `n_rnn_layers`, `dropout`, `learning_rate`
-   Best params: `hidden_dim=128`, `n_rnn_layers=2`, `dropout=0.1`, `lr=0.001`

### **Phase 3 — Baseline Modeling (03_Baseline_Models.ipynb)**
-   **Naive Last-Value:** Simple heuristic predicting last known value
    -   Validation sMAPE: **48.85%** (on continuous data)
-   **LSTM:** Trained with tuned hyperparameters using L1 Loss
    -   Validation sMAPE: **91.34%** (significantly underperformed)
    -   Excluded from final evaluation due to poor performance

### **Phase 4 — AutoGluon Hybrid Ensemble (04_Ensemble_Model.ipynb)**
A sophisticated two-stage approach:
-   **Stage 1 (Classifier):** DeepAR model predicts `is_trading` binary flag
-   **Stage 2 (Regressor):** TemporalFusionTransformer predicts HLCV values for trading periods
-   Validation sMAPE: **41.85%** (on continuous trading data)

### **Phase 5 — Chronos2 Fine-Tuning (05_Chronos2_Training.ipynb)**
-   **Model:** Amazon's `chronos-2` foundation model
-   **Progressive unfreezing strategy:**
    1.  Head-only fine-tuning (10 epochs)
    2.  Partial encoder unfreezing (last 2 blocks, 10 epochs)
    3.  Full model burst training with early stopping
-   Best validation sMAPE: **34.87%** (on full dataset)

### **Phase 6 — Final Evaluation (06_Final_Evaluation.ipynb)**
-   Test set (2024) evaluation on 25 random assets
-   **Winner: Naive Baseline (7.16% sMAPE)**
-   AutoGluon Hybrid: 21.36% sMAPE
-   Key insight: Simple models excel when data has dominant simple patterns (high sparsity)
    

## 🧩 Models Implemented

| Model | Description | Framework | Test sMAPE                  |
|-------|-------------|-----------|-----------------------------|
| **Naive Last-Value** | Simple heuristic: predict last known value | NumPy | **7.16%** ⭐                 |
| **LSTM** | Recurrent neural network (2 layers, hidden_dim=128) | PyTorch Lightning | 91.34% (trading heavy data) |
| **AutoGluon Hybrid** | Two-stage: DeepAR classifier + TFT regressor | AutoGluon | 21.36%                      |
| **Chronos2** | Fine-tuned Amazon foundation model | Hugging Face + PyTorch Lightning | 35.61% (5 sample only)      |


## 📈 Evaluation Metric

Masked sMAPE

The standard sMAPE formula is applied, but the loss is only calculated for time steps where the true value $y_i$ is not zero (i.e., when trading was active).

$$\text{sMAPE} = \frac{100\%}{N_{\text{active}}} \sum_{i \in \text{Active}} \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$


## ⚡ GPU and VRAM (8GB) Strategy

-   **Mixed Precision:** All deep models are trained with `precision="16-mixed"` to halve VRAM usage.
    
-   **VRAM Tactics:** If `CUDA out of memory` errors occur, the first step is to **reduce `batch_size`** (e.g., 16 $\rightarrow$ 8 $\rightarrow$ 4). The second is to reduce model complexity (e.g., `d_model`).
    
-   **Data Streaming:** The `IterableDataset` ensures that only the current batch resides in memory, not the full dataset.
    

## 📚 Results & Discussion

### Final Test Set Performance (2024)

The project evaluated multiple models on an unseen test set (2024) comprising 25 randomly selected assets. The results were surprising:

| Rank | Model | Test sMAPE | Notes |
|------|-------|------------|-------|
| 🥇 1st | Naive Baseline | **7.16%** | Captures zero patterns perfectly |
| 🥈 2nd | AutoGluon Hybrid | **21.36%** | Strong on trading data, struggles with sparsity |

### Key Findings

1. **Simplicity Wins on Sparse Data**: The naive last-value forecast dominated because the dataset contains predominantly non-trading periods (zeros). A simple heuristic perfectly captures this pattern.

2. **Training-Only Performance Misleading**: AutoGluon's regressor achieved 41.85% sMAPE on continuous trading data (better than naive's 48.85%), but this advantage disappeared on the full sparse test set.

3. **LSTM Struggled**: Despite systematic hyperparameter tuning, LSTM achieved 91.34% sMAPE, indicating potential capacity limitations or misalignment between L1 loss training and sMAPE evaluation.

4. **Chronos2 Promise**: The foundation model showed strong performance (34.87% on validation, 35.61% on test sample) but wasn't included in the final comparison due to limited test coverage.

For a comprehensive analysis, see [project_report.md](project_report.md).


## 🧩 Key Takeaways

1. **Data Characteristics Matter**: When data exhibits simple, dominant patterns (like high sparsity), complex models may not outperform simple heuristics.

2. **Evaluation Strategy is Critical**: Models trained only on trading-dense data performed well in validation but failed to generalize to the full sparse test set.

3. **Hybrid Models Need Better Classifiers**: The AutoGluon two-stage approach was limited by its Stage 1 classifier. Errors in predicting `is_trading` compounded in the final forecast.

4. **Foundation Models Show Promise**: Chronos2's performance suggests that larger-scale fine-tuning on more assets could yield competitive results.

5. **Aligned Loss Functions**: Training with sMAPE loss (instead of L1) might have improved LSTM and other models' performance on the evaluation metric.

## 👩‍💻 Author

Mert Alp Aydin

Deep Learning Course — Frankfurt School of Finance & Management

Autumn 2025


## 📝 License

This project is licensed under the MIT License.
