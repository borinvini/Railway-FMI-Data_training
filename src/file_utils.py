import os
import glob
import re

import pandas as pd
from config.const import DATA_FILE_PREFIX, DATA_FILE_PREFIX_FOR_TRAINING, FOLDER_NAME, INPUT_FOLDER, OUTPUT_FOLDER


def check_csv_files():
    """
    Check if there are CSV files in the data/input folder and print their names.
    Returns a list of CSV file paths found.
    """
    # Using only Approach 1: Based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_folder = os.path.join(project_root, INPUT_FOLDER)
    
    print(f"Looking for CSV files in: {input_folder}")
    
    # Check for CSV files in the data/input folder
    csv_files = []
    if os.path.exists(input_folder):
        csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    # Report findings
    if csv_files:
        print(f"\nSuccess! Found {len(csv_files)} CSV files in the input folder:")
        for file in csv_files:
            print(f"- {os.path.basename(file)}")
    else:
        print("\nNo CSV files found in the input folder.")
    
    return csv_files

def extract_date_range(csv_files):
    """
    Extract the date range (YYYY_MM) from the CSV filenames.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        tuple: (earliest_date, latest_date) where each date is a tuple of (year, month)
               Returns None if no valid dates found
    """
    if not csv_files:
        print("No files to extract date range from.")
        return None
    
    # Extract all YYYY_MM patterns from filenames
    dates = []
    pattern = f'{DATA_FILE_PREFIX}(\\d{{4}})_(\\d{{2}})\\.csv'  # Use the constant here
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        match = re.match(pattern, filename)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            dates.append((year, month))
    
    if not dates:
        print(f"No files match the expected naming pattern ({DATA_FILE_PREFIX}YYYY_MM.csv).")
        return None
    
    # Sort dates to find earliest and latest
    dates.sort()
    earliest = dates[0]
    latest = dates[-1]
    
    print(f"\nDate range found in files:")
    print(f"Earliest: {earliest[0]}_{earliest[1]:02d}")
    print(f"Latest: {latest[0]}_{latest[1]:02d}")
    
    return earliest, latest

def load_and_preview_csv(csv_filename):
    """
    Load a CSV file and print its columns with their data types.
    
    Args:
        csv_filename: Name of the CSV file (without path)
        
    Returns:
        pandas.DataFrame: Loaded data or None if file couldn't be loaded
    """
    # Construct the full path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, INPUT_FOLDER, csv_filename)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None
    
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Print basic information
        print(f"\nSuccessfully loaded: {csv_filename}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Print column names and their data types
        print("\nColumns and data types:")
        for col, dtype in df.dtypes.items():
            print(f"- {col}: {dtype}")
        
        return df
    
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None
    
def generate_output_path(input_file_path):
    """
    Generate an output file path based on the input file name.
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file.
        
    Returns:
    --------
    tuple
        (output_file_path, dataframe) - The generated output path and the loaded dataframe.
    """
    # Get just the filename without the path
    basename = os.path.basename(input_file_path)
    
    # Use regex to find year and month in the filename
    # This pattern looks for 4 digits (year) followed by optional characters and then 2 digits (month)
    match = re.search(r'(\d{4}).*?(\d{2})', basename)
    
    if match:
        year, month = match.groups()
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year}_{month}.csv"
    else:
        # Fallback if date pattern not found
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}processed.csv"
        print(f"Warning: Could not extract date from filename. Using default: {output_filename}")
    
    # Create data directories if they don't exist
    os.makedirs(FOLDER_NAME, exist_ok=True)
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Join the output directory path with the output filename
    output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)
    print(f"Will save to {OUTPUT_FOLDER} directory: {output_file_path}")
    
    # Load the CSV file
    print(f"Loading CSV file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    return output_file_path, df