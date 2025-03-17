import glob
import os
import pandas as pd
import ast

from config.const import DATA_FILE_PREFIX_FOR_TRAINING, OUTPUT_FOLDER
from src.file_utils import generate_output_path

def preprocess_csv_file(input_file_path):
    """
    Preprocess a CSV file by extracting nested data from timeTableRows,
    keeping only essential columns, and expanding weather conditions.
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file.
        
    Returns:
    --------
    pandas.DataFrame
        The processed DataFrame that was saved to the output file.
    """
    try:
        # Generate output path and load the dataframe
        output_file_path, df = generate_output_path(input_file_path)
        
        # Extract nested data from the "timeTableRows" column
        cross_records = []
        for index, row in df.iterrows():
            ttr = row.get("timeTableRows", None)
            if ttr is None or pd.isnull(ttr):
                continue  # Skip empty values
                
            # If the value is a string, attempt to convert it
            if isinstance(ttr, str):
                try:
                    # Replace unquoted 'nan' with 'None' so ast.literal_eval can parse it
                    ttr_fixed = ttr.replace("nan", "None")
                    stops = ast.literal_eval(ttr_fixed)
                except Exception as e:
                    print(f"Error parsing timeTableRows in row {index}: {e}")
                    continue
            else:
                stops = ttr  # Assume it's already a Python object
                
            # Ensure stops is a list; if not, wrap it in a list
            if not isinstance(stops, list):
                stops = [stops]
                
            for stop in stops:
                cross_records.append(stop)
        
        print(f"Extracted {len(cross_records)} records from timeTableRows")
        
        # Create a DataFrame from the extracted dictionaries
        cross_df = pd.DataFrame(cross_records)
        
        # Rename 'weather_observations' to 'weather_conditions' if it exists
        if "weather_observations" in cross_df.columns:
            cross_df = cross_df.rename(columns={"weather_observations": "weather_conditions"})
            print("Renamed 'weather_observations' to 'weather_conditions'")
        
        # Keep only the columns: "differenceInMinutes", "cancelled", and "weather_conditions"
        expected_cols = ["differenceInMinutes", "cancelled", "weather_conditions"]
        available_cols = [col for col in expected_cols if col in cross_df.columns]
        
        if set(expected_cols).issubset(cross_df.columns):
            cross_df = cross_df[expected_cols]
            print(f"Kept only columns: {expected_cols}")
        else:
            print(f"Warning: Not all expected columns are available. Available columns:")
            print(list(cross_df.columns))
            print(f"Keeping available columns from expected list: {available_cols}")
            cross_df = cross_df[available_cols]
        
        # Expand the 'weather_conditions' dictionaries into separate columns
        if "weather_conditions" in cross_df.columns:
            weather_df = cross_df["weather_conditions"].apply(pd.Series)
            
            # Drop unwanted keys if they exist
            weather_df = weather_df.drop(columns=["closest_ems", "Present weather (auto)"], errors="ignore")
            
            # Join the expanded weather conditions back to the main DataFrame
            cross_df = cross_df.drop("weather_conditions", axis=1).join(weather_df)
            print("Expanded weather_conditions into separate columns")
        
        # Reorder columns: differenceInMinutes, cancelled, then the expanded weather columns
        base_cols = [col for col in ["differenceInMinutes", "cancelled"] if col in cross_df.columns]
        other_cols = [col for col in cross_df.columns if col not in base_cols]
        cols_order = base_cols + other_cols
        cross_df = cross_df[cols_order]
        print("Reordered columns")
        
        # Save the final DataFrame to a CSV file
        cross_df.to_csv(output_file_path, index=False)
        print(f"Successfully saved processed data to {output_file_path}")
        
        return cross_df
        
    except Exception as e:
        print(f"Error processing file {input_file_path}: {e}")
        return None
    
def handle_missing_values(file_year_month=None, dataframe=None):
    """
    Handle missing values in preprocessed CSV files.
    
    Either uses the provided dataframe or loads a preprocessed CSV file 
    from the output folder and handles missing values:
    - Drop rows where all weather condition columns have missing values
    - Keep track of how many rows were dropped
    
    Parameters:
    -----------
    file_year_month : str, optional
        Year and month in format "YYYY_MM" to specify which file to process/save.
        Required if no dataframe is provided.
    dataframe : pandas.DataFrame, optional
        The preprocessed dataframe to handle missing values in.
        If provided, the file won't be loaded from disk.
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe with missing values handled.
    """
    try:
        # Determine the script directory and project root for file operations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, OUTPUT_FOLDER)
        
        # If no dataframe is provided, load it from file
        if dataframe is None:
            if not file_year_month:
                print("Error: Either dataframe or file_year_month must be provided")
                return None
                
            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                print(f"Error: Output directory {output_dir} not found")
                return None
            
            # Look for a specific file with the given year and month
            filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{file_year_month}.csv"
            file_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"Error: File {filename} not found in {output_dir}")
                return None
            
            # Load the CSV file
            print(f"Loading file: {filename}")
            df = pd.read_csv(file_path)
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        else:
            # Use the provided dataframe
            df = dataframe
            print(f"Using provided dataframe with {len(df)} rows and {len(df.columns)} columns")
            
            # Determine file path for saving
            if file_year_month:
                filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{file_year_month}.csv"
                file_path = os.path.join(output_dir, filename)
            else:
                print("Warning: file_year_month not provided, will not save results to disk")
                file_path = None
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Count rows before cleaning
        original_row_count = len(df)
        
        # Define the important weather conditions to check
        important_conditions = [
            'Air temperature', 
            'Relative humidity', 
            'Dew-point temperature', 
            'Precipitation amount', 
            'Precipitation intensity', 
            'Snow depth', 
            'Horizontal visibility'
        ]
        
        # Filter the list to only include columns that actually exist in the dataframe
        available_important_cols = [col for col in important_conditions if col in df.columns]
        
        if not available_important_cols:
            print("Warning: None of the specified important weather conditions found in the dataframe")
            return df
        
        print(f"Found {len(available_important_cols)} important weather condition columns: {available_important_cols}")
        
        # Drop rows where ALL of the important weather conditions are missing
        # (Keep rows with at least one of the specified conditions)
        df_cleaned = df.dropna(subset=available_important_cols, how='all')
        
        # Count how many rows were dropped
        dropped_rows = original_row_count - len(df_cleaned)
        
        # Report the results
        print(f"Missing values handling complete:")
        print(f"- Original row count: {original_row_count}")
        print(f"- Rows dropped (missing all important weather conditions): {dropped_rows}")
        print(f"- Remaining rows: {len(df_cleaned)}")
        
        # Calculate percentage of data retained
        if original_row_count > 0:
            retention_percentage = (len(df_cleaned) / original_row_count) * 100
            print(f"- Data retention: {retention_percentage:.2f}%")
            
        # Additional statistics on the important columns
        for col in available_important_cols:
            non_null_count = df_cleaned[col].count()
            null_count = len(df_cleaned) - non_null_count
            null_percentage = (null_count / len(df_cleaned) * 100) if len(df_cleaned) > 0 else 0
            print(f"  - {col}: {non_null_count} non-null values ({100-null_percentage:.2f}% complete)")
        
        # Save the cleaned dataframe back to file if path is available
        if file_path:
            df_cleaned.to_csv(file_path, index=False)
            print(f"Cleaned data saved to {file_path}")
        
        return df_cleaned
        
    except Exception as e:
        print(f"Error in handle_missing_values: {e}")
        return None