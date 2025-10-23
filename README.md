
# ⚡ Electricity Market Forecasting

A deep learning project to forecast electricity market prices and volumes from high-frequency (15-minute) trading data.

This repository was developed for the **Deep Learning Course (Autumn 2024)** assignment. It implements an end-to-end pipeline, including a custom data-streaming loader, robust baseline modeling (LightGBM, N-BEATS), and fine-tuning a state-of-the-art **`PatchTSMixer`** foundational model on an 8GB GPU.


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
electricity-market-forecasting/
│
├── data/                  # Raw and preprocessed data (Parquet/Zarr)
│
├── notebooks/             # Executable workflow and reports
│   ├── 01_EDA.ipynb       # Data loading, exploration, and visualization
│   ├── 02_Baseline_Models.ipynb # Training Naive, LightGBM, LSTM, N-BEATS
│   ├── 03_PatchTSMixer_Training.ipynb # Main training & tuning for the SOTA model
│   └── 04_Evaluation.ipynb    # Final model evaluation and results generation
│
├── src/                   # Core Python library (imported by notebooks)
│   ├── data_loader.py     # Custom PyTorch IterableDataset for streaming
│   ├── preprocess.py      # Cleaning, feature engineering, scaling
│   ├── models/
│   │   └── patchtsmixer_module.py # PyTorch Lightning wrapper for PatchTSMixer
│   └── utils.py           # Helper functions (metrics, logging)
│
├── models/                # Saved model checkpoints and scalers
├── results/               # Final predictions (CSVs) and metric reports
├── reports/               # Final PDF/HTML reports exported from notebooks
│
├── requirements.txt
├── README.md
└── LICENSE

```


## ⚙️ Environment Setup

1.  **Clone the repo**
    
    Bash
    
    ```
    git clone git clone https://github.com/mertalpaydin/Electricity-Market-Forecasting.git

    cd electricity-market-forecasting
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
    

## 💿 Data Handling Strategy (Out-of-Core)

To train on the full dataset with limited RAM/VRAM, this project uses a **disk-based streaming pipeline**.

-   **Preprocessing:** Data is preprocessed once and saved to disk in an efficient format (e.g., Parquet or Zarr).
    
-   **Streaming:** A custom `IterableDataset` (`src/data_loader.py`) is used to:
    
    1.  Read data chunks lazily from disk.
        
    2.  Generate `(input_window, target_window)` tuples on the fly.
        
    3.  Feed batches directly to the PyTorch Lightning trainer.
        
-   **Hybrid Strategy:**
    
    -   **Tuning:** A representative subsample (stratified by liquidity) is loaded into memory for fast hyperparameter tuning with **Optuna**.
        
    -   **Training:** The final model is trained on the _entire_ dataset using the streaming `IterableDataset`.
        

## 🚀 Modeling Workflow

### **Phase 1 — Data Exploration & Feature Engineering**

-   Load HLCV data, visualize asset lifecycles, and compute liquidity.
    
-   Engineer features:
    
    -   `is_trading` (binary flag)
        
    -   Time-based: `hour_of_day`, `day_of_week`, `time_to_delivery`
        
    -   Rolling stats: 4-step & 8-step moving averages (computed on non-zero data)
        
    -   Relational: `Close` price of adjacent delivery contracts
        

### **Phase 2 — Baseline Modeling**

-   **Naive Forecast:** Last observed value.
    
-   **LightGBM:** GPU-accelerated model using `asset-id` as a categorical feature.
    
-   **RNN/LSTM:** Standard deep learning sequence baseline.
    
-   **N-BEATS:** Powerful deep learning baseline that decomposes the time series.
    

### **Phase 3 — Advanced Modeling: Fine-tuning `PatchTSMixer`**

-   **Model:** Uses the `ibm/patchtsmixer-etth1-pretrain` foundational model from Hugging Face.
    
-   **Wrapper:** A custom PyTorch Lightning module (`src/models/patchtsmixer_module.py`) is used to integrate the model with the streaming data loader and training loop.
    
-   **Strategy:** A **gradual unfreezing** strategy is applied to fine-tune the model without catastrophic forgetting:
    
    1.  **Stage 1:** Train only the new prediction head.
        
    2.  **Stage 2:** Unfreeze and train the top 1-2 mixer layers.
        
    3.  **Stage 3:** Fine-tune the entire model with a low learning rate.
        

### **Phase 4 — Hyperparameter Tuning**

-   Use **Optuna** to minimize the **validation masked sMAPE**.
    
-   Tune `PatchTSMixer` (`d_model`, `n_layers`, `lr`) and all baselines on the stratified subsample.
    

### **Phase 5 & 6 — Iteration & Final Evaluation**

-   Experiment with lookback windows (e.g., 24, 96, 288 steps).
    
-   Build a **meta-model ensemble** stacking the best models (LightGBM + PatchTSMixer).
    
-   Train the final, best model on the **full `2021–2023` data** and perform a single evaluation on the **`2024` test set**.
    

## 🧩 Models Implemented

**Model**

**Description**

**Framework**

**Naive**

Last observed non-zero value

NumPy

**LightGBM**

Tabular time-window model

`Darts (LightGBM)`

**LSTM**

Recurrent seq2seq baseline

`Darts (PyTorch Lightning)`

**N-BEATS**

Decomposable deep learning model

`Darts (PyTorch Lightning)`

**PatchTSMixer**

**(Primary Model)** Fine-tuned foundational Transformer

`Hugging Face + PyTorch Lightning`

**Ensemble**

Stacked meta-model (Ridge)

`scikit-learn`


## 📈 Evaluation Metric

Masked sMAPE

The standard sMAPE formula is applied, but the loss is only calculated for time steps where the true value $y_i$ is not zero (i.e., when trading was active).

$$\text{sMAPE} = \frac{100\%}{N_{\text{active}}} \sum_{i \in \text{Active}} \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$


## ⚡ GPU and VRAM (8GB) Strategy

-   **Mixed Precision:** All deep models are trained with `precision="16-mixed"` to halve VRAM usage.
    
-   **VRAM Tactics:** If `CUDA out of memory` errors occur, the first step is to **reduce `batch_size`** (e.g., 16 $\rightarrow$ 8 $\rightarrow$ 4). The second is to reduce model complexity (e.g., `d_model`).
    
-   **Data Streaming:** The `IterableDataset` ensures that only the current batch resides in memory, not the full dataset.
    

## 📚 Results & Discussion (Template)

To be filled
    


## 🧩 Key Takeaways

to be filled


## 👩‍💻 Author

Mert Alp Aydin

Deep Learning Course — Frankfurt School of Finance & Management

Autumn 2024


## 📝 License

This project is licensed under the MIT License.
