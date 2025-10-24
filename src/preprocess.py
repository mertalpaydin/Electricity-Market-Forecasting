import os
import glob
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def feature_engineer(df: pd.DataFrame, timestamp_col: str = "ExecutionTime") -> pd.DataFrame:
    """
    Engineers time-based features and an 'is_trading' flag for the given DataFrame.

    The new features are:
    - hour_of_day
    - day_of_week
    - week_of_year
    - month
    - is_weekend
    - is_trading: A binary flag that is 1 if any of the numeric target columns
      (high, low, close, volume) are non-zero, and 0 otherwise.

    Args:
        df (pd.DataFrame): The input DataFrame, containing a timestamp column.
        timestamp_col (str): The name of the timestamp column.

    Returns:
        pd.DataFrame: The DataFrame with the new features added.
    """
    dt = df[timestamp_col].dt
    df["hour_of_day"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["month"] = dt.month
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    numeric_cols = ['high', 'low', 'close', 'volume']
    df['is_trading'] = (df[numeric_cols].sum(axis=1) > 0).astype(int)

    # --- Rolling Window Features (calculated only on trading data) ---
    # Create a temporary DataFrame with NaNs where there is no trading
    trading_only_df = df[numeric_cols].where(df['is_trading'] == 1)

    windows = [2, 4, 6, 8]
    for col in numeric_cols:
        for w in windows:
            # Moving Average
            df[f'{col}_ma_{w}'] = trading_only_df[col].rolling(window=w, min_periods=1).mean()

        # Standard Deviation
        df[f'{col}_std_12'] = trading_only_df[col].rolling(window=12, min_periods=1).std()

    # Forward-fill the NaNs created by the rolling operations and where is_trading was 0
    roll_cols = [f'{c}_ma_{w}' for c in numeric_cols for w in windows] + \
                [f'{c}_std_12' for c in numeric_cols]
    df[roll_cols] = df[roll_cols].ffill().fillna(0)

    return df

def split_data_by_year(
    parquet_root: str,
    train_dir: str,
    val_dir: str,
    timestamp_col: str = "ExecutionTime"
):
    """
    Splits the per-asset Parquet files into training and validation sets based on
    the year, after applying feature engineering.

    - Training set: 2021-2022
    - Validation set: 2023

    Args:
        parquet_root (str): The directory containing the source Parquet files
            from `data_loader.py`.
        train_dir (str): The output directory for the training data.
        val_dir (str): The output directory for the validation data.
        timestamp_col (str): The name of the timestamp column used for splitting.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    asset_files = glob.glob(os.path.join(parquet_root, "*.parquet"))
    if not asset_files:
        logging.error(f"No parquet files found in {parquet_root}. Aborting.")
        return

    logging.info(f"Found {len(asset_files)} asset files to split and engineer features for.")

    for asset_path in tqdm(asset_files, desc="Splitting and Engineering Features"):
        asset_name = os.path.basename(asset_path)
        try:
            df = pd.read_parquet(asset_path)
            if timestamp_col not in df.columns:
                logging.warning(f"Timestamp column '{timestamp_col}' not found in {asset_name}. Skipping.")
                continue

            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            # 1. Feature Engineering
            df = feature_engineer(df, timestamp_col)

            # 2. Split data
            train_df = df[df[timestamp_col].dt.year.isin([2021, 2022])]
            val_df = df[df[timestamp_col].dt.year == 2023]

            # 3. Save splits
            if not train_df.empty:
                train_output_path = os.path.join(train_dir, asset_name)
                train_df.to_parquet(train_output_path, index=False)

            if not val_df.empty:
                val_output_path = os.path.join(val_dir, asset_name)
                val_df.to_parquet(val_output_path, index=False)

        except Exception as e:
            logging.error(f"Failed to process {asset_name}: {e}")

    logging.info("Data splitting and feature engineering complete.")
    logging.info(f"Train data saved in: {train_dir}")
    logging.info(f"Validation data saved in: {val_dir}")

def fit_and_save_scalers(
    train_dir: str,
    scalers_dir: str,
    target_cols: list = ['high', 'low', 'close', 'volume']
):
    """
    Fits a `StandardScaler` for each asset on its training data and saves it to disk.

    The scaler is fitted only on the time steps where `is_trading` is 1, to avoid
    the large number of zeros from biasing the scaler's mean and variance.

    Args:
        train_dir (str): The directory containing the training Parquet files.
        scalers_dir (str): The directory where the fitted scalers will be saved.
        target_cols (list): The names of the columns to be scaled.
    """
    os.makedirs(scalers_dir, exist_ok=True)
    train_files = glob.glob(os.path.join(train_dir, "*.parquet"))

    if not train_files:
        logging.error(f"No training files found in {train_dir} to fit scalers. Aborting.")
        return

    logging.info(f"Fitting scalers for {len(train_files)} assets.")

    for asset_path in tqdm(train_files, desc="Fitting Scalers"):
        asset_name = os.path.basename(asset_path).replace(".parquet", "")
        try:
            df = pd.read_parquet(asset_path)
            
            trading_data = df[df['is_trading'] == 1]
            
            if not trading_data.empty:
                scaler = StandardScaler()
                scaler.fit(trading_data[target_cols])
                
                scaler_path = os.path.join(scalers_dir, f"{asset_name}_scaler.joblib")
                joblib.dump(scaler, scaler_path)

        except Exception as e:
            logging.error(f"Failed to fit scaler for {asset_name}: {e}")

    logging.info(f"Scalers saved in: {scalers_dir}")


if __name__ == "__main__":
    # This script orchestrates the entire preprocessing pipeline.
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    source_parquet_dir = os.path.join(base_dir, "data", "per_asset_parquet")
    train_output_dir = os.path.join(base_dir, "data", "train")
    val_output_dir = os.path.join(base_dir, "data", "val")
    scalers_output_dir = os.path.join(base_dir, "data", "scalers")

    # Clean up directories from previous runs by deleting their contents.
    for dir_path in [train_output_dir, val_output_dir, scalers_output_dir]:
        if os.path.exists(dir_path):
            logging.info(f"Cleaning contents of directory: {dir_path}")
            files_to_delete = glob.glob(os.path.join(dir_path, '*.*'))
            if not files_to_delete:
                continue  # Directory is empty

            logging.info(f"  Found {len(files_to_delete)} files to delete.")
            for f in files_to_delete:
                try:
                    os.remove(f)
                except (OSError, PermissionError) as e:
                    logging.warning(f"  Could not delete file {f}: {e}. Skipping.")

    # Step 1: Split data by year and apply feature engineering.
    logging.info("--- Starting Data Splitting and Feature Engineering ---")
    split_data_by_year(
        parquet_root=source_parquet_dir,
        train_dir=train_output_dir,
        val_dir=val_output_dir
    )
    logging.info("--- Data Splitting and Feature Engineering Finished ---")

    # Step 2: Fit and save scalers on the newly created training set.
    logging.info("--- Starting Scaler Fitting ---")
    fit_and_save_scalers(
        train_dir=train_output_dir,
        scalers_dir=scalers_output_dir
    )
    logging.info("--- Scaler Fitting Finished ---")
