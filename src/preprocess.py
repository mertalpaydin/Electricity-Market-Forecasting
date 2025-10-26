import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict

# Configure logging to provide informative output during the preprocessing pipeline.
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Mapping of day abbreviations to day-of-the-week numbers (Monday=0).
DAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

def _get_delivery_details(asset_name: str) -> Tuple[int, int, int]:
    """
    Parses an asset name (e.g., "Tue11Q4") to extract its delivery schedule.

    Args:
        asset_name (str): The name of the asset.

    Returns:
        A tuple containing:
        - The delivery day of the week (0-6 for Mon-Sun).
        - The delivery hour (0-23).
        - The delivery minute (0-59).
    """
    day_str = asset_name[:3]
    hour = int(asset_name[3:5])
    quarter = int(asset_name[6:])
    minute = (quarter - 1) * 15  # Convert quarter to minutes (Q1=0, Q2=15, etc.).
    delivery_day_of_week = DAY_MAP[day_str]
    return delivery_day_of_week, hour, minute

def add_time_features(df: pd.DataFrame, asset_name: str, timestamp_col: str = "ExecutionTime") -> pd.DataFrame:
    """
    Engineers time-based features for a given asset's DataFrame.

    This function adds standard calendar features and a custom `time_to_delivery`
    feature, which calculates the time remaining until the contract's next
    delivery window.

    Args:
        df (pd.DataFrame): The input DataFrame for a single asset.
        asset_name (str): The name of the asset, used to calculate delivery time.
        timestamp_col (str): The name of the timestamp column.

    Returns:
        pd.DataFrame: The DataFrame with added time-based features.
    """
    dt = df[timestamp_col].dt
    df["hour_of_day"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["month"] = dt.month
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    # The 'is_trading' flag is crucial for masking metrics and fitting scalers correctly.
    numeric_cols = ['high', 'low', 'close', 'volume']
    df['is_trading'] = (df[numeric_cols].sum(axis=1) > 0).astype(int)

    # --- Calculate time_to_delivery (in hours) ---
    delivery_day, hour, minute = _get_delivery_details(asset_name)
    
    # Vectorized calculation for the next delivery datetime for each row.
    days_ahead = (delivery_day - dt.dayofweek + 7) % 7
    delivery_date = dt.normalize() + pd.to_timedelta(days_ahead, unit='d')
    delivery_datetime = delivery_date + pd.DateOffset(hours=hour, minutes=minute)

    # If the calculated delivery time is on or before the current timestamp,
    # it means the delivery for this week has passed, so we target next week's delivery.
    mask = delivery_datetime <= df[timestamp_col]
    delivery_datetime[mask] += pd.Timedelta(days=7)
    
    # Calculate the difference in hours and clip at zero (no negative time).
    df['time_to_delivery'] = (delivery_datetime - df[timestamp_col]).dt.total_seconds() / 3600.0
    df['time_to_delivery'] = df['time_to_delivery'].clip(lower=0)
    
    return df

def add_cross_contract_features(all_dfs: Dict[str, pd.DataFrame], num_surrounding: int = 6) -> Dict[str, pd.DataFrame]:
    """
    Engineers complex cross-contract features across all assets.

    This function creates features for each asset based on the state of its
    neighboring contracts (in delivery order). It is designed to be
    forward-looking safe, using only data available up to each timestamp.

    The features created are:
    - `close_lag_adj_k`: Last observed non-zero close price of neighbor k.
    - `close_delta_adj_k`: Difference between current close and neighbor's lagged close.
    - `volume_adj_k`: Last observed non-zero volume of neighbor k.
    - `nearest_liquid_contract_close`: Close price of the nearest neighbor with active trading.
    - `cross_contract_mean`: Mean of the last observed close prices of all neighbors.

    Args:
        all_dfs (Dict[str, pd.DataFrame]): A dictionary of DataFrames, one for each asset.
        num_surrounding (int): The number of adjacent contracts to consider on each side.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of DataFrames with the added features.
    """
    logging.info("Engineering cross-contract features...")
    sorted_assets = sorted(all_dfs.keys())
    asset_map = {name: i for i, name in enumerate(sorted_assets)}

    # Create unified DataFrames for 'close' and 'volume' across all assets.
    # This aligns all data by timestamp, making cross-asset lookups possible.
    close_df = pd.concat(
        [df[['ExecutionTime', 'close']].set_index('ExecutionTime').rename(columns={'close': name}) for name, df in all_dfs.items()],
        axis=1
    )
    volume_df = pd.concat(
        [df[['ExecutionTime', 'volume']].set_index('ExecutionTime').rename(columns={'volume': name}) for name, df in all_dfs.items()],
        axis=1
    )
    
    # Sort by time to ensure chronological order for forward-fills.
    close_df.sort_index(inplace=True)
    volume_df.sort_index(inplace=True)

    # Create "last known non-zero" DataFrames. This is a crucial step.
    # `replace(0, nan).ffill()` propagates the last valid price/volume forward,
    # ensuring we use the most recent historical data without looking ahead.
    last_close_df = close_df.replace(0, np.nan).ffill()
    last_volume_df = volume_df.replace(0, np.nan).ffill()

    processed_dfs = {}
    for asset_name in tqdm(sorted_assets, desc="Adding cross-contract features"):
        df = all_dfs[asset_name].copy()
        df.set_index('ExecutionTime', inplace=True)
        
        asset_idx = asset_map[asset_name]
        
        # Identify the k-nearest neighbors for the current asset.
        neighbors = {}
        for k in range(1, num_surrounding + 1):
            if asset_idx - k >= 0:
                neighbors[f"adj_{k}"] = sorted_assets[asset_idx - k]
            if asset_idx + k < len(sorted_assets):
                neighbors[f"adj_{k}"] = sorted_assets[asset_idx + k]

        # --- Feature Calculation ---
        # These features are calculated for all timestamps first, then masked later.
        for k_str, neighbor_name in neighbors.items():
            # `close_lag_adj_k`: Last known close price of the k-th neighbor.
            close_lag_col = f"close_lag_{k_str}"
            df[close_lag_col] = last_close_df[neighbor_name].reindex(df.index, method='ffill')

            # `close_delta_adj_k`: Spread between current price and neighbor's price.
            df[f"close_delta_{k_str}"] = df['close'] - df[close_lag_col]

            # `volume_adj_k`: Last known volume of the k-th neighbor.
            df[f"volume_{k_str}"] = last_volume_df[neighbor_name].reindex(df.index, method='ffill')

        # `nearest_liquid_contract_close`: Find the closest neighbor that is currently trading.
        liquid_neighbors_close = pd.Series(index=df.index, dtype=float)
        all_neighbor_closes = []

        # Iterate outwards from the current asset to find the nearest liquid one.
        for k in range(1, num_surrounding + 1):
            prev_neighbor = sorted_assets[asset_idx - k] if asset_idx - k >= 0 else None
            next_neighbor = sorted_assets[asset_idx + k] if asset_idx + k < len(sorted_assets) else None

            # Check previous neighbor first, as per spec.
            if prev_neighbor:
                is_liquid = volume_df[prev_neighbor].reindex(df.index) > 0
                liquid_closes = close_df[prev_neighbor][is_liquid]
                liquid_neighbors_close.fillna(liquid_closes, inplace=True) # Fill only where still NaN
                all_neighbor_closes.append(last_close_df[prev_neighbor].reindex(df.index, method='ffill'))

            # Check next neighbor.
            if next_neighbor:
                is_liquid = volume_df[next_neighbor].reindex(df.index) > 0
                liquid_closes = close_df[next_neighbor][is_liquid]
                liquid_neighbors_close.fillna(liquid_closes, inplace=True)
                all_neighbor_closes.append(last_close_df[next_neighbor].reindex(df.index, method='ffill'))

        df['nearest_liquid_contract_close'] = liquid_neighbors_close
        
        # `cross_contract_mean`: Mean of all neighbors' last known close prices.
        if all_neighbor_closes:
            cross_contract_df = pd.concat(all_neighbor_closes, axis=1)
            df['cross_contract_mean'] = cross_contract_df.mean(axis=1, skipna=True)
        else:
            df['cross_contract_mean'] = np.nan

        # --- Masking ---
        # Set all engineered features to NaN where the primary asset was not trading.
        # This prevents the model from learning from artificial feature values.
        feature_cols = [col for col in df.columns if 'adj' in col or 'cross_contract' in col or 'nearest_liquid' in col]
        df.loc[df['is_trading'] == 0, feature_cols] = np.nan
        
        processed_dfs[asset_name] = df.reset_index()

    return processed_dfs

def split_data_by_year(
    parquet_root: str,
    train_dir: str,
    val_dir: str,
    timestamp_col: str = "ExecutionTime",
):
    """
    Orchestrates the feature engineering and data splitting pipeline.

    This function performs the following steps:
    1. Loads all per-asset parquet files.
    2. Applies basic time-based features to each asset.
    3. Applies complex cross-contract features across all assets.
    4. Splits the feature-engineered data into training (2021-2022) and
       validation (2023) sets.
    5. Saves the final datasets to the specified directories.

    Args:
        parquet_root (str): The directory containing the source per-asset Parquet files.
        train_dir (str): The output directory for the training data.
        val_dir (str): The output directory for the validation data.
        timestamp_col (str): The name of the timestamp column.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    asset_files = glob.glob(os.path.join(parquet_root, "*.parquet"))
    if not asset_files:
        logging.error(f"No parquet files found in {parquet_root}. Aborting.")
        return

    # --- Step 1: Load all data and apply basic time features ---
    all_dfs = {}
    for asset_path in tqdm(asset_files, desc="Loading data and time features"):
        asset_name = os.path.basename(asset_path).replace('.parquet', '')
        df = pd.read_parquet(asset_path)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = add_time_features(df, asset_name, timestamp_col)
        all_dfs[asset_name] = df

    # --- Step 2: Engineer Cross-Contract Features ---
    featured_dfs = add_cross_contract_features(all_dfs)

    # --- Step 3: Split and Save Data ---
    for asset_name, df in tqdm(featured_dfs.items(), desc="Splitting and saving data"):
        # Fill any remaining NaNs with 0. This typically applies to feature columns
        # during non-trading periods, making them neutral for the model.
        df.fillna(0, inplace=True)
        
        train_df = df[df[timestamp_col].dt.year.isin([2021, 2022])]
        val_df = df[df[timestamp_col].dt.year == 2023]

        if not train_df.empty:
            train_df.to_parquet(os.path.join(train_dir, f"{asset_name}.parquet"), index=False)
        if not val_df.empty:
            val_df.to_parquet(os.path.join(val_dir, f"{asset_name}.parquet"), index=False)

    logging.info("Data splitting and feature engineering complete.")

def fit_and_save_scalers(
    train_dir: str,
    scalers_dir: str,
    target_cols: list = ['high', 'low', 'close', 'volume']
):
    """
    Fits a StandardScaler for each asset and saves it to disk.

    The scaler is fitted ONLY on the non-zero trading data from the training set.
    This prevents the large number of zero values (from non-trading periods)
    from biasing the scaler's mean and variance, leading to better normalization.

    Args:
        train_dir (str): The directory containing the training Parquet files.
        scalers_dir (str): The directory where the fitted scalers will be saved.
        target_cols (list): The names of the columns to be scaled.
    """
    os.makedirs(scalers_dir, exist_ok=True)
    train_files = glob.glob(os.path.join(train_dir, "*.parquet"))

    if not train_files:
        logging.error(f"No training files found in {train_dir}. Aborting.")
        return

    for asset_path in tqdm(train_files, desc="Fitting Scalers"):
        asset_name = os.path.basename(asset_path).replace(".parquet", "")
        df = pd.read_parquet(asset_path)
        
        # Filter for trading data only before fitting the scaler.
        trading_data = df[df['is_trading'] == 1]
        
        if not trading_data.empty:
            scaler = StandardScaler()
            scaler.fit(trading_data[target_cols])
            joblib.dump(scaler, os.path.join(scalers_dir, f"{asset_name}_scaler.joblib"))

    logging.info(f"Scalers saved in: {scalers_dir}")

if __name__ == "__main__":
    # This main block orchestrates the entire preprocessing pipeline, making the
    # script runnable from the command line.
    
    # --- Define Paths ---
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    source_parquet_dir = os.path.join(base_dir, "data", "per_asset_parquet")
    train_output_dir = os.path.join(base_dir, "data", "train")
    val_output_dir = os.path.join(base_dir, "data", "val")
    scalers_output_dir = os.path.join(base_dir, "data", "scalers")

    # --- Clean up previous runs ---
    # This ensures a fresh start and prevents data from old runs from contaminating the new one.
    for dir_path in [train_output_dir, val_output_dir, scalers_output_dir]:
        if os.path.exists(dir_path):
            logging.info(f"Cleaning contents of directory: {dir_path}")
            for f in glob.glob(os.path.join(dir_path, '*')):
                os.remove(f)

    # --- Run Preprocessing Pipeline ---
    # 1. Split data, engineer features, and save train/val sets.
    logging.info("--- Starting Data Splitting and Feature Engineering ---")
    split_data_by_year(
        parquet_root=source_parquet_dir,
        train_dir=train_output_dir,
        val_dir=val_output_dir
    )
    logging.info("--- Data Splitting and Feature Engineering Finished ---")

    # 2. Fit and save scalers on the newly created training set.
    logging.info("--- Starting Scaler Fitting ---")
    fit_and_save_scalers(
        train_dir=train_output_dir,
        scalers_dir=scalers_output_dir
    )
    logging.info("--- Scaler Fitting Finished ---")