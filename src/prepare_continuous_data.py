import os
import pandas as pd
from tqdm import tqdm

def find_all_valid_sequences(df, min_length=58):
    """
    Finds all continuous sequences of trading periods (is_trading == 1)
    that are longer than a minimum length.
    """
    df['block'] = (df['is_trading'] != df['is_trading'].shift()).cumsum()
    trading_blocks = df[df['is_trading'] == 1]['block'].value_counts()
    
    valid_blocks = trading_blocks[trading_blocks >= min_length]
    
    if valid_blocks.empty:
        return []

    sequences = []
    for block_id in valid_blocks.index:
        sequence_indices = df[df['block'] == block_id].index
        sequences.append((sequence_indices[0], sequence_indices[-1]))
        
    return sequences

def extract_and_pad_sequence(df, start_idx, end_idx, padding=10):
    """
    Extracts a sequence and adds padding around it.
    """
    start_pad = max(0, start_idx - padding)
    end_pad = min(len(df), end_idx + 1 + padding)
    return df.iloc[start_pad:end_pad].copy()

def process_asset_files(input_dir, output_dir, min_length=58, padding=10):
    """
    Processes all parquet files, extracts all valid trading sequences,
    and saves each as a new numbered slice.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    print(f"Processing {len(files)} files from {input_dir}...")

    total_slices = 0
    for file in tqdm(files):
        input_path = os.path.join(input_dir, file)
        asset_name = file.replace('.parquet', '')
        
        df = pd.read_parquet(input_path)
        
        valid_sequences = find_all_valid_sequences(df, min_length)
        
        if not valid_sequences:
            continue

        for i, (start_idx, end_idx) in enumerate(valid_sequences):
            sliced_df = extract_and_pad_sequence(df, start_idx, end_idx, padding)
            
            output_filename = f"{asset_name}_slice_{i}.parquet"
            output_path = os.path.join(output_dir, output_filename)
            
            sliced_df.to_parquet(output_path)
            total_slices += 1
            
    print(f"Finished processing. Created {total_slices} slices in {output_dir}")

if __name__ == '__main__':
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    TRAIN_INPUT_DIR = os.path.join(DATA_DIR, "train")
    VAL_INPUT_DIR = os.path.join(DATA_DIR, "val")

    TRAIN_OUTPUT_DIR = os.path.join(DATA_DIR, "train_continuous")
    VAL_OUTPUT_DIR = os.path.join(DATA_DIR, "val_continuous")

    INPUT_CHUNK_LENGTH = 48
    OUTPUT_CHUNK_LENGTH = 10
    MIN_SEQUENCE_LENGTH = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH 

    # --- Run Processing ---
    process_asset_files(
        input_dir=TRAIN_INPUT_DIR, 
        output_dir=TRAIN_OUTPUT_DIR, 
        min_length=MIN_SEQUENCE_LENGTH
    )
    
    process_asset_files(
        input_dir=VAL_INPUT_DIR, 
        output_dir=VAL_OUTPUT_DIR, 
        min_length=MIN_SEQUENCE_LENGTH
    )

    print("\nMulti-slice data preparation complete.")
