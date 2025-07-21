import os
import glob
import re

import pandas as pd
from config.const import (
    DATA_FILE_PREFIX, 
    DATA_FILE_PREFIX_FOR_TRAINING, 
    FOLDER_NAME, 
    INPUT_FOLDER, 
    OUTPUT_FOLDER,
    PREPROCESSED_OUTPUT_FOLDER,
    ALL_PREPROCESSED_OUTPUT_FOLDER,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER
)


def ensure_folder_structure():
    """
    Explicitly create the basic folder structure required by the application.
    
    This function guarantees that all necessary directories exist before
    the main processing begins, preventing any file operation errors.
    
    Returns:
    --------
    dict
        Summary of the folder creation operation with created/existing directories
    """
    print("Ensuring basic folder structure exists...")
    
    # List of all directories that need to be created
    required_directories = [
        FOLDER_NAME,                                    # data/
        INPUT_FOLDER,                                   # data/input/
        OUTPUT_FOLDER,                                  # data/output/
        os.path.join(FOLDER_NAME, "output", "log")      # data/output/log/
    ]
    
    created_dirs = []
    already_existed = []
    
    for directory in required_directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                created_dirs.append(directory)
                print(f"  ✓ Created directory: {directory}")
            except Exception as e:
                print(f"  ✗ Failed to create directory {directory}: {e}")
                raise
        else:
            already_existed.append(directory)
            print(f"  ✓ Directory already exists: {directory}")
    
    # Summary
    print(f"\nFolder structure verification complete:")
    print(f"  - Created: {len(created_dirs)} new directories")
    print(f"  - Existing: {len(already_existed)} directories")
    print(f"  - Total verified: {len(required_directories)} directories")
    
    if created_dirs:
        print(f"\nNewly created directories:")
        for dir_path in created_dirs:
            print(f"  - {dir_path}")
    
    return {
        "created": created_dirs,
        "existed": already_existed,
        "total_verified": len(required_directories)
    }


def check_csv_files():
    """
    Check if there are CSV files in the data/input folder and print their names.
    Returns a list of CSV file paths found.
    
    Note: The input folder is guaranteed to exist by the main initialization process.
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
        print(f"\n✓ Success! Found {len(csv_files)} CSV files in the input folder:")
        for file in csv_files:
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"  - {os.path.basename(file)} ({file_size:.1f} MB)")
    else:
        print(f"\n⚠️  No CSV files found in the input folder.")
        print(f"\nTo get started:")
        print(f"  1. Place your CSV files in: {input_folder}")
        print(f"  2. Expected file format: {DATA_FILE_PREFIX}YYYY_MM.csv")
        print(f"  3. Example: {DATA_FILE_PREFIX}2023_12.csv")
        print(f"  4. Then run: python main.py --target trainDelayed")
    
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
        
    Note: The input folder is guaranteed to exist by the main initialization process.
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
        
    Note: Basic folder structure is guaranteed to exist by the main initialization process,
          but we still ensure specific output directories exist for safety.
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
    
    # Ensure specific output directories exist (basic structure guaranteed by main initialization)
    # We still check these for safety in case this function is called independently
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Join the output directory path with the output filename
    output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)
    print(f"Will save to {OUTPUT_FOLDER} directory: {output_file_path}")
    
    # Load the CSV file
    print(f"Loading CSV file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    return output_file_path, df


def verify_folder_structure():
    """
    Verify that the basic folder structure exists and provide a status report.
    
    This function can be called independently to check if the required directories exist.
    
    Returns:
    --------
    dict
        Status report of folder structure verification
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # List of critical directories to check
    critical_directories = [
        (FOLDER_NAME, "Main data directory"),
        (INPUT_FOLDER, "Input CSV files directory"),
        (OUTPUT_FOLDER, "Output results directory")
    ]
    
    verification_results = {
        "all_exist": True,
        "missing_directories": [],
        "existing_directories": [],
        "details": {}
    }
    
    print("Verifying folder structure:")
    
    for dir_path, description in critical_directories:
        full_path = os.path.join(project_root, dir_path)
        exists = os.path.exists(full_path)
        
        verification_results["details"][dir_path] = {
            "exists": exists,
            "full_path": full_path,
            "description": description
        }
        
        if exists:
            verification_results["existing_directories"].append(dir_path)
            print(f"  ✓ {description}: {dir_path}")
        else:
            verification_results["all_exist"] = False
            verification_results["missing_directories"].append(dir_path)
            print(f"  ✗ {description}: {dir_path} (MISSING)")
    
    if verification_results["all_exist"]:
        print("✓ All critical directories exist!")
    else:
        print(f"⚠️  Missing {len(verification_results['missing_directories'])} critical directories!")
        print("Run the main script to auto-create missing directories.")
    
    return verification_results