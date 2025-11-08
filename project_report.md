# Project Report: Electricity Price Forecasting

## 1. Project Overview

This project aims to predict future electricity prices using various time-series forecasting models. The goal is to build a robust model that can capture the complex patterns and volatility inherent in energy markets. The project explores several modeling techniques, starting from baseline models and progressing to more sophisticated deep learning architectures and ensemble methods. The final evaluation is performed on a hold-out test set, simulating a real-world forecasting scenario.

## 2. Codebase & Execution Workflow

The project is organized into a modular structure to facilitate development, testing, and deployment.

### Key Directories:
-   `src/`: Contains the core Python source code for data loading, preprocessing, and the data module.
-   `notebooks/`: Jupyter notebooks for EDA, model development, and evaluation.
-   `data/`: Raw and processed data files.
-   `results/`: Stores model outputs, such as performance metrics and hyperparameter configurations.

### Core Modules:
-   `src/data_loader.py`: Defines the `ElectricityPriceIterableDataset`, which is responsible for loading and iterating over the dataset in a memory-efficient way. It handles the logic for creating sequences for the time-series models.
-   `src/datamodule.py`: Implements the `ElectricityDataModule`, a PyTorch Lightning DataModule that encapsulates all data-related steps, from loading and preprocessing to creating data loaders for training, validation, and testing.
-   `src/preprocess.py`: Contains functions for data preprocessing, such as scaling and feature engineering.
-   `src/prepare_continuous_data.py`: A script to process the raw data of each asset into smaller chunks of trading periods where only a small portion of trailing and following non-trading periods are added. 

### Execution Workflow:
The project is executed through a series of Jupyter notebooks, each corresponding to a specific phase of the project:
1.  `01_EDA.ipynb`: Initial data exploration.
2.  `02_Hyperparameter_Tuning.ipynb`: LSTM model tuning.
3.  `03_Baseline_Models.ipynb`: Implementation and evaluation of Naive model vs the tuned LSTM.
4.  `04_Ensemble_Model.ipynb`: Development of an AutoGluon hybrid ensemble model of a classifier and predictor.
5.  `05_Chronos2_Training.ipynb`: Fine-tuning the Chronos2 foundation model.
6.  `06_Final_Evaluation_SUPER_OPTIMIZED.ipynb`: The final script that loads the test data and evaluates the selected baseline model and  the trained deep learning model to determine the best performer. A Colab-friendly version is also available.

## 3. The Data Processing Pipeline

The transformation of raw data into a model-ready format is a multi-stage process, orchestrated primarily by `src/preprocess.py` with contributions from `src/data_loader.py` and `src/prepare_continuous_data.py`. The pipeline is designed to handle the large initial dataset, engineer a rich set of features, and then create smaller, more manageable data slices for efficient model training.

Here is the correct sequence of operations:

**Stage 1: Initial CSV to Parquet Conversion**

*   **What:** The process begins with massive CSV files (`TRAIN_Reco_2021_2022_2023.csv` and `TEST_Reco_2024.csv`). The `split_csv_to_parquet` function (from `data_loader.py`, called in `preprocess.py`) is used to convert these into a more efficient, per-asset Parquet file format.
*   **Why:** A single, huge CSV is slow to read and query. Splitting the data by individual asset and converting to Parquet provides two major benefits:
    1.  **Performance:** Parquet is a columnar storage format, making reads for specific columns (like `price` or `volume`) significantly faster.
    2.  **Granularity:** It organizes the data logically, allowing scripts to process one asset at a time, which is more memory-efficient.

**Stage 2: Feature Engineering**

*   **What:** Once the data is in the per-asset Parquet format, the `process_and_split_data` function in `preprocess.py` executes a comprehensive feature engineering workflow on the combined (train + test) data. This includes:
    1.  **Time-based Features:** Standard calendar features (`hour_of_day`, `day_of_week`, `month`, etc.) are created to help models learn seasonal patterns.
    2.  **`is_trading` Flag:** A crucial binary flag is created by checking if `high`, `low`, `close`, or `volume` are greater than zero. This explicitly identifies active trading periods.
    3.  **`time_to_delivery`:** A sophisticated feature is engineered to calculate the time remaining until the next delivery window for each specific contract.
    4.  **Cross-Contract Features:** The script analyzes relationships between "neighboring" contracts (e.g., contracts delivering at adjacent times). It creates features like the last known price/volume of neighboring contracts (`close_lag_adj_k`, `volume_adj_k`) and the price of the nearest actively trading contract (`nearest_liquid_contract_close`).
*   **Why:** This rich feature set provides the models with deep contextual information. Instead of just seeing its own price history, each asset is now aware of its position in the daily/weekly cycle and the behavior of related financial instruments, which is critical for accurate forecasting. All features are engineered on the combined dataset *before* splitting to ensure consistency and prevent look-ahead bias in lagged features.

**Stage 3: Data Splitting and Scaling**

*   **What:** After features are added, the data is split by year into `train` (2021-2022), `val` (2023), and `test` (2024) sets. Then, `StandardScaler` objects are fit *only* on the trading data (`is_trading == 1`) of the training set.
*   **Why:**
    *   **Splitting:** A time-based split is essential for a valid time-series evaluation, ensuring the model is tested on data it has never seen before.
    *   **Selective Scaling:** Scaling is vital for neural networks. By fitting the scaler *only* on active trading data, we prevent the vast number of zero-value, non-trading periods from skewing the scaler's mean and variance. This results in a much more meaningful normalization of the actual price and volume data.

**Stage 4: Sub-sampling for Model Training (The "Continuous" Data)**

*   **What:** This is the final step, performed by `prepare_continuous_data.py`. This script takes the feature-engineered and scaled `train` and `val` sets and extracts only the most valuable sequences for training. It identifies continuous blocks of active trading (`is_trading == 1`) that are long enough for the model's input/output window and saves them as smaller, separate Parquet files (e.g., `asset_name_slice_0.parquet`). A small amount of padding (non-trading data) is kept around these slices to provide context.
*   **Why:** This is a crucial optimization. Instead of forcing the model to iterate through hours or days of zero-activity data, this step creates a dense, information-rich dataset focused exclusively on the periods where price action occurs. This dramatically speeds up training and allows the model to focus on learning the complex dynamics of active trading periods.

This rigorous, multi-stage pipeline ensures that the data is clean, feature-rich, and structured for efficient and effective model training.

### 4. Hyperparameter Tuning for the LSTM Model

To optimize the performance of the Long Short-Term Memory (LSTM) model, a systematic hyperparameter tuning process was conducted prior to the training, as documented in `02_Hyperparameter_Tuning.ipynb`. 

The tuning process focused on the following key hyperparameters:
*   **`hidden_size`**: The number of features in the hidden state of the LSTM.
*   **`num_layers`**: The number of recurrent layers.
*   **`dropout`**: The dropout rate for regularization.
*   **`learning_rate`**: The initial learning rate for the optimizer.

The objective of the tuning was to minimize the validation loss. After running the optimization study for a limited number of epochs, the best performing hyperparameters were identified and saved in `results/best_params_lstm.json`.

The optimal parameters were found to be:
*   **`hidden_dim`**: 128
*   **`n_rnn_layers`**: 2
*   **`dropout`**: 0.1
*   **`learning_rate`**: 0.001
*   **`batch_size`**: 128

These parameters were subsequently used to define the LSTM architecture for training and evaluation, ensuring that the model was configured for optimal performance based on the validation data. Hyperparameter tuning was done on the previously created continuous dataset.

### 5. Baseline and LSTM Modeling

As detailed in `03_Baseline_Models.ipynb`, this phase focused on establishing a baseline performance metric and training the main LSTM model using the optimal hyperparameters identified previously.

**1. Baseline Model**

A crucial step in evaluating any complex model is to compare it against a simple, common-sense benchmark. In this notebook, a **Naive Last-Value** model was implemented for this purpose.

*   **Methodology:** This model uses a simple heuristic: it predicts that the price (and other target values) at the next timestep will be the same as the price at the most recent timestep.
*   **Evaluation:** This baseline was evaluated on the validation set using the smoothed Symmetric Mean Absolute Percentage Error (sMAPE).
*   **Result:** The Naive Last-Value model yielded an overall sMAPE of **48.85%**. This score serves as the critical benchmark. Any subsequent model, including the LSTM, must achieve a significantly lower sMAPE to be considered effective and justify its complexity. It's important to note that this score is calculated on the previously curated dataset that mostly consists of trading data.

**2. LSTM Model Training**

The primary effort in this phase was the training of the LSTM model.

*   **Architecture:** The model was constructed using the hyperparameters discovered during the tuning phase (`hidden_dim=128`, `n_rnn_layers=2`, `dropout=0.1`, `learning_rate=0.001`, loaded from `best_params_lstm.json`).
*   **Training:** The model was trained on the `train_continuous` dataset, which contains the information-dense slices of trading activity with `L1 loss`. The training process was monitored against the `val_continuous` set to prevent overfitting. To ensure the process did not run indefinitely, a `TimeLimitCallback` was implemented to stop training after a maximum of 1 hour. The best model checkpoint was saved based on the validation loss.
*   **Results:** The trained LSTM model significantly underperformed the naive baseline, achieving a sMAPE of **91.34%** compared to the **48.85%** benchmark. This poor performance suggests that the LSTM struggled to capture the complex patterns in the continuous trading data, possibly due to model capacity limitations or the choice of L1 loss as the training objective. Due to this underperformance, the LSTM model was not included in the final evaluation on the test set. The loss history during training was plotted and saved to `results/lstm_loss_history.png`.

### 6. AutoGluon Hybrid Ensemble Model

This phase, documented in `04_Ensemble_Model.ipynb`, implements a sophisticated two-stage hybrid forecasting approach using the AutoGluon library. The core idea is to decompose the complex electricity price prediction problem into two sub-problems: first, predicting whether trading will occur, and second, predicting the actual price and volume during trading periods.

**1. Two-Stage Approach Rationale**

The decision to use a two-stage hybrid model stems directly from the insights gained during the data processing and EDA phases, particularly the high sparsity of trading periods.
*   **Stage 1 (Classification):** A model predicts the binary `is_trading` flag. This addresses the challenge of forecasting when price movements are relevant.
*   **Stage 2 (Regression):** A separate model predicts the actual `high`, `low`, `close`, and `volume` values, but *only* for the periods identified as trading by the first stage. This allows the regressor to focus solely on the dynamics of active markets, avoiding the noise of zero-value non-trading periods.

**2. AutoGluon for Automated Machine Learning**

AutoGluon's `TimeSeriesPredictor` was chosen for its ability to automate model selection, hyperparameter tuning, and ensemble creation, allowing for rapid experimentation with various state-of-the-art time series models.

**3. Stage 1: Trading vs. Non-Trading Classifier**

*   **Objective:** Predict the `is_trading` flag (binary: 0 or 1).
*   **Data:** Trained on a subset of 100 assets from the full `train` and `val` datasets (not the `_continuous` versions), which include both trading and non-trading periods.
*   **Target & Metric:** The target was `is_trading`, and the evaluation metric used was Mean Absolute Scaled Error (MASE), which AutoGluon adapted for this binary classification task.
*   **Models Explored:** AutoGluon automatically considered an ensemble of models including `SeasonalNaive`, `TemporalFusionTransformer`, `PatchTST`, and `DeepAR`, each with various configurations (zero-shot, fine-tuned, deep fine-tuned).
*   **Best Performer:** The `DeepAR` model emerged as the best individual model for this classification task, achieving a validation MASE score of -0.1273.

**4. Stage 2: sMAPE Regressor on Trading Data**

*   **Objective:** Predict `high`, `low`, `close`, and `volume` for periods where `is_trading` is predicted to be 1.
*   **Data:** Trained on the `train_continuous` and `val_continuous` datasets, which exclusively contain trading periods (with some padding). The target columns (`high`, `low`, `close`, `volume`) were melted into a single `target` column for AutoGluon's multi-output forecasting capability. A larger subset of 2500 assets was used for this stage.
*   **Target & Metric:** The target was the melted `target` column, and the evaluation metric was sMAPE, directly optimizing for the competition's primary metric.
*   **Models Explored:** Similar to the classifier, AutoGluon explored an ensemble of models including `SeasonalNaive`, `TemporalFusionTransformer`, `PatchTST`, `DeepAR`, `AutoETS`, and `DynamicOptimizedTheta`.
*   **Best Performer:** The `TemporalFusionTransformer_FineTuned` model achieved the best performance for the regression task, with a validation sMAPE score of 41.85% sMAPE which is better than the naive baseline set on the same dataset.

This hybrid approach leverages the strengths of different models and targets the specific challenges posed by the dataset's sparsity, aiming for a more robust and accurate overall prediction system.

### 7. Fine-Tuning Chronos2 Foundation Model

This phase, detailed in `05_Chronos2_Training.ipynb`, explores the application of the Chronos2 foundation model, a pre-trained large language model for time series from Amazon Science, to the electricity price forecasting task. The goal is to leverage the model's pre-trained knowledge and adapt it through fine-tuning for superior performance.

**1. Chronos2 Model and Data Preparation**

*   **Model:** The `amazon/chronos-2` pre-trained model was used as the base.
*   **Data:** A subset of 10 assets from the `train` and `val` datasets (the full, not `_continuous`, versions) was used. The four target columns (`high`, `low`, `close`, `volume`) were stacked to form a multivariate target, allowing Chronos2 to forecast all four simultaneously. Both past and future covariates (time-based and cross-contract features) were incorporated.

**2. Progressive Fine-Tuning Strategy**

A multi-stage, progressive unfreezing strategy was employed to fine-tune the large Chronos2 model efficiently and effectively:

*   **Zero-Shot Evaluation (Baseline):**
    *   The pre-trained Chronos2 model was first evaluated on the validation set without any fine-tuning to establish a baseline performance.
    *   **Result:** The zero-shot sMAPE was **147.69%**, indicating that the generic pre-trained model, without adaptation, performs poorly on this specific task.

*   **Warmup Phase 1: Head-Only Fine-Tuning:**
    *   Only the `output_patch_embedding` (the model's output head responsible for quantile prediction) was unfrozen, while the rest of the model's layers remained frozen. This phase aimed to quickly adapt the output layer to the specific target variables and their scale.
    *   **Training:** Trained for 10 epochs with a low learning rate (2e-05), using gradient accumulation and clipping.

*   **Warmup Phase 2: Partial Encoder Unfreezing:**
    *   In addition to the output head, the last two encoder blocks of the Chronos2 transformer were unfrozen. This allowed for deeper adaptation of the model's feature extraction capabilities to the electricity price data.
    *   **Training:** Trained for another 10 epochs with a slightly reduced learning rate (1e-05).
    *   **Result:** This initial fine-tuning dramatically improved performance, reducing the sMAPE to **34.87%**, indicating that the initial head-only fine-tuning captured most of the immediate gains. 

*   **Burst Training with Early Stopping:**
    *   Finally, all layers of the Chronos2 model were unfrozen to allow for full fine-tuning. Training proceeded in "bursts" (200 steps per burst) with a learning rate decay schedule and early stopping (patience of 5 bursts without improvement). This aimed to extract the maximum performance from the model.
    *   **Result:** Despite further training, the sMAPE did not improve beyond **34.87%**, suggesting that the model had converged or that further fine-tuning on this dataset size did not yield additional benefits. The best checkpoint was saved from Warmup Phase 2.

**3. Final Evaluation on Test Set Sample**

*   The best fine-tuned model (from Warmup Phase 2) was evaluated on a small sample of 5 random files from the unseen `test` set.
*   **Result:** The sMAPE on this test set sample was **35.61%**. Although this sMAPE was lower than the naive baseline on the continuous data, it was achieved on the full dataset where the naive baseline had not yet been tested.

### 8. Final Model Evaluation

The final phase of the project, documented in `06_Final_Evaluation_Colab.ipynb`, involves a head-to-head comparison of the most promising models on the unseen test set. This notebook evaluates the performance of the Naive Baseline against the sophisticated AutoGluon Hybrid Model.

**1. Evaluation Setup**

*   **Dataset:** The evaluation was performed on a randomly selected subset of 25 assets from the `test` directory (data from 2024).
*   **Methodology:** A custom `ElectricityDataModule` was used to create a dataloader that serves batches of historical data (`past`) and the corresponding future data to be predicted (`future`). For the AutoGluon model, a more complex dataloader was used to also provide pre-computed past and future covariates for each window.

**2. Model Performance on the Test Set**

The two models were evaluated using the sMAPE metric on the test set.

*   **Naive Baseline (Last-Value):**
    *   This model, which simply predicts the next 10 values will be the same as the last known value, was evaluated first.
    *   **Result:** It achieved an overall sMAPE of **7.16%**. This surprisingly strong result is likely due to the high frequency of non-trading periods (considering approximately 48% sMAPE on continuous data it achieved), where the price is consistently zero, a pattern the last-value forecast captures perfectly.

*   **AutoGluon Hybrid Model:**
    *   The two-stage model was evaluated next. For each time window, the `DeepAR` classifier predicted the `is_trading` status, and the `TemporalFusionTransformer_FineTuned` regressor predicted the price/volume values. The final prediction was generated by multiplying the regressor's output by a binary mask from the classifier's prediction.
    *   **Result:** The hybrid model achieved an overall sMAPE of **21.36%**.

**3. Final Results and Conclusion**

The final ranking on the test set was definitive and somewhat unexpected:

1.  **Naive Baseline**: 7.16% sMAPE
2.  **AutoGluon Hybrid**: 21.36% sMAPE

The **Naive Baseline** was declared the best-performing model in this evaluation. While the AutoGluon model showed strong performance during its validation phase on the `_continuous` (trading-only) data, its performance suffered on the complete test set, which includes long periods of non-trading. The complexity of the hybrid model did not translate into better performance in the final real-world scenario, where the simple heuristic of the naive model proved more robust. This highlights a crucial lesson in time-series modeling: a complex model is not always better, especially when the underlying data has simple, dominant patterns.

### 9. Inferences & Future Suggestions

The project provided valuable insights into the challenges of forecasting electricity prices and the surprising effectiveness of simple models in certain scenarios. Based on the results, several avenues for future work could yield improved performance and a more robust forecasting system.

1.  **Refine the Hybrid Model Strategy:**
    *   The AutoGluon hybrid model was outperformed by the naive baseline, likely because errors in the Stage 1 classifier (predicting `is_trading`) were compounded in the final prediction considering that during training the regressor model had achieved a better sMAPE than naive baseline on the same dataset. A key area for improvement is the classifier. Instead of MASE, using a more appropriate binary classification metric like **LogLoss** or **F1-score** during training might yield a more accurate trading-period detector.
    *   Furthermore, the threshold for the classifier's output (currently 0.5) could be tuned as a hyperparameter to optimize the trade-off between precision and recall for identifying trading periods. Also, models outside of Autogluon library such as tree based models can be tested as a classifier. 

2.  **Enhance the Chronos Fine-Tuning Process:**
    *   The Chronos fine-tuning showed massive improvement over its zero-shot baseline but did not improve further during the full "burst training". This could be due to the small subset of data used (10 assets). Fine-tuning on a much larger and more diverse set of assets could allow the model to learn more generalizable patterns and benefit more from unfreezing its deeper layers.
    *   Experimenting with different learning rates and schedulers during the full fine-tuning phase might also unlock further performance gains.

3.  **Feature Engineering and Selection:**
    *   The project used a rich set of pre-engineered features. A more systematic feature selection process could be implemented to identify the most impactful covariates and remove noisy or redundant ones. This could simplify the models and potentially improve their generalization.
    *   Exploring additional external data sources, such as weather forecasts (temperature, wind speed for renewables), natural gas prices, or major grid events, could provide valuable predictive signals.

4.  **LSTM Limitations:**
    *   LSTM loss did not improve much after 15 epochs which suggests a capacity limitation. Also, LSTM was trained with L1 Loss. It could be tested with sMAPE as the evaluation metric to better align the training objective with the final evaluation criteria.  


By focusing on these areas, it is possible to build upon the findings of this project to develop a more accurate and reliable forecasting system that combines the strengths of both simple heuristics and complex deep learning models.