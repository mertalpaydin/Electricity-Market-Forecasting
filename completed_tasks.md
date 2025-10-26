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
- **Initial Data Loading and EDA (in `notebooks/01_EDA.ipynb`):**
    - Loaded sample data and visualized contract lifecycles (Price and Volume).
    - Analyzed basic statistics and confirmed no NaN values are present.
    - Visualized trading liquidity across sample assets.
    - Generated within-asset and cross-asset correlation heatmaps.
- **Documentation:**
    - Added comprehensive docstrings and comments to all functions and classes in `src/data_loader.py`, `src/datamodule.py`, and `src/preprocess.py`.


# TO BE Completed Before all Other Tasks

There is a mistake in the cross contract feature implementation at `src/preprocess.py`. It only records next contracts not the previous ones. Delete current cross contract features implementation and add below ones.

## Cross-Contract Feature Specification (computed **only when `is_trading == True`**)

For every asset (contract) and timestamp `t`, compute the following features using **historical data available up to `t` only**.  
Adjacent contracts refer to contracts with delivery times immediately before or after the current one in the delivery order (e.g., ±k steps).

### 1. `close_lag_adj_k`

-   Definition: last observed (before time t) non-zero **Close** price of the contract `k` steps away (where `k ∈ {-6,…,+6}` and `k ≠ 0`).
    
-   Behavior:
    
    -   Use the _most recent observed_ non-zero value before time `t`.
        
    -   If no valid value exists in the ±6 range, leave as N/A.
        

### 2. `close_delta_adj_k`

-   Definition: difference between the current contract’s close and that of an adjacent one.
    
**close_delta_adj_k = close − close_lag_adj_k**

-   Purpose: captures relative price spread between neighboring delivery periods.
    

### 3. `volume_adj_k`

-   Definition: last observed non-zero **Volume** for each adjacent contract (`k = -6 … +6`).
    
-   Use the same lookup logic as `close_lag_adj_k` (last known non-zero volume).
    

### 4. `nearest_liquid_contract_close`

-   Definition: **Close** price of the _nearest contract (in delivery order)_ that currently has `volume > 0`.
    
-   Behavior:
    
    -   Search ±6 neighbors first; if none are liquid, expand the search window outward until a liquid neighbor is found. If both previous and next neighbours are liquid use previous.
        
    -   If no contract is liquid at this time, use NaN.
        

### 5. `cross_contract_mean`

-   Definition: mean of the most recent **Close** prices across all available ±6 adjacent contracts.
    
**cross_contract_mean = mean {close_lag_adj_k  ∣  k=−6,…,+6}**

-   Ignore missing or zero values when taking the mean.
    
----------

## New Time-Based Features

`time_to_delivery`

Time remaining (in minutes or hours) until the delivery start time of this contract.

Use the timestamp parsed from the asset name (e.g., “Tue11Q4”) to infer delivery window.


### What `time_to_delivery` means

In the **electricity market**, each _contract_ (asset) corresponds to a specific **delivery period** — for example:

> Contract `Tue11Q4` means delivery on **Tuesday 11:45–12:00**.

Traders buy/sell this contract ahead of time — sometimes **hours or even days** before that delivery window.  
So at any given _trading timestamp_ `t`, the contract still has some remaining time until its scheduled delivery.

**`time_to_delivery`** is simply that remaining time — how far away the delivery start time is from the current trading time.

----------

### How to calculate it

For each record (contract, timestamp):

**time_to_delivery = (delivery_datetime − current_timestamp)**

Delivery datetime is the nearest date implied by the contract name. For instance if Contract is `Tue11Q4` and time step if time step is 26/08/2023 09:00 which is a Saturday then the delivery time is the closest Tuesday 11:45 in the future which is 29/08/2023 11:45. Whenever timesteps becomes larger than the delivery time, automatically jump to the next Tuesday. 

Clipped at zero (no negative values).  

----------

## Practical Notes

-   All cross-contract features must be **forward-looking safe** — use only data from times ≤ `t`.
    
-   If `is_trading == False` for the target contract at time `t`, **do not** compute cross-contract features (set them to NaN or zero).
    
-   Compute and store these features during preprocessing and save them in per-asset parquet files to avoid recomputation at training time.