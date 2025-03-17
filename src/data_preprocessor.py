import glob
import os
import pandas as pd
import re
import ast

from config.const import DATA_FILE_PREFIX_FOR_TRAINING, OUTPUT_FOLDER
from src.file_utils import generate_output_path

class TrainingPipeline:
    def __init__(self):
        """
        Initialize the TrainingPipeline class with default values.
        """
        # Get script directory and project root for file operations
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.output_dir = os.path.join(self.project_root, OUTPUT_FOLDER)
        
        # Define important weather conditions to check
        self.important_conditions = [
            'Air temperature', 
            'Relative humidity', 
            'Dew-point temperature', 
            'Precipitation amount', 
            'Precipitation intensity', 
            'Snow depth', 
            'Horizontal visibility'
        ]

    def run_pipeline(self, csv_files):
        """
        Run the full processing pipeline on the provided CSV files.
        
        This method coordinates the execution of the preprocessing and 
        missing value handling steps for each input file.
        
        Parameters:
        -----------
        csv_files : list
            List of CSV file paths to process.
            
        Returns:
        --------
        dict
            A summary of the processing results containing:
            - total_files: Total number of files processed
            - successful_preprocessing: Number of files successfully preprocessed
            - successful_cleaning: Number of files with successful missing value handling
            - failed_files: Number of files that failed processing
            - processed_dataframes: Dictionary of processed dataframes (if save_dataframes=True)
        """
        if not csv_files:
            print("\nNo CSV files to process.")
            return {
                "total_files": 0,
                "successful_preprocessing": 0,
                "successful_cleaning": 0,
                "failed_files": 0
            }
        
        print(f"\nStarting pipeline processing for {len(csv_files)} CSV files.")
        
        # Initialize counters for summary
        successful_preprocessing = 0
        successful_cleaning = 0
        successful_deduplication = 0
        successful_saves = 0
        failed_files = 0
        
        # Process each file
        for i, input_file_path in enumerate(csv_files):
            filename = os.path.basename(input_file_path)
            print(f"\n[{i+1}/{len(csv_files)}] Processing file: {filename}")
            
            # Step 1: Preprocess the CSV file
            print(f"Starting preprocessing for {filename}...")
            processed_df = self.preprocess_csv_file(input_file_path)
            
            if processed_df is None:
                print(f"Error occurred during preprocessing of {filename}")
                failed_files += 1
                continue
                
            print(f"Preprocessing completed successfully for {filename}")
            print(f"Processed data contains {len(processed_df)} rows and {len(processed_df.columns)} columns")
            successful_preprocessing += 1
            
            # Step 2: Handle missing values
            # Extract year and month from the filename
            pattern = r'(\d{4})_(\d{2})'
            match = re.search(pattern, filename)
            
            if match:
                year_month = f"{match.group(1)}_{match.group(2)}"
                print(f"\nHandling missing values for {year_month}...")
                
                # Call the handle_missing_values method (now without saving)
                cleaned_df = self.handle_missing_values(dataframe=processed_df)
                
                if cleaned_df is not None:
                    print(f"Successfully cleaned missing values for {year_month}")
                    successful_cleaning += 1
                    
                    # Step 3: Remove duplicates
                    print(f"Removing duplicates for {year_month}...")
                    deduplicated_df = self.remove_duplicates(dataframe=cleaned_df)
                    
                    if deduplicated_df is not None:
                        print(f"Successfully removed duplicates for {year_month}")
                        successful_deduplication += 1
                        
                        # Step 4: Save the deduplicated dataframe immediately to free up memory
                        print(f"Saving processed dataframe for {year_month}...")
                        save_success = self.save_df_to_csv(year_month, deduplicated_df)
                        
                        if save_success:
                            successful_saves += 1
                            print(f"Successfully saved dataframe for {year_month}")
                        else:
                            print(f"Failed to save dataframe for {year_month}")
                            
                        # Clear the deduplicated dataframe from memory
                        del deduplicated_df
                    else:
                        print(f"Failed to remove duplicates for {year_month}")
                        # Try to save the cleaned dataframe anyway
                        print(f"Saving cleaned dataframe for {year_month}...")
                        save_success = self.save_df_to_csv(year_month, cleaned_df)
                        if save_success:
                            successful_saves += 1
                    
                    # Clear the cleaned dataframe from memory
                    del cleaned_df
                else:
                    print(f"Failed to clean missing values for {year_month}")
            else:
                print(f"Could not extract year_month from filename: {filename}")
            
            # Clear the processed dataframe from memory
            del processed_df
        
        # Generate and return summary
        summary = {
            "total_files": len(csv_files),
            "successful_preprocessing": successful_preprocessing,
            "successful_cleaning": successful_cleaning,
            "successful_deduplication": successful_deduplication,
            "successful_saves": successful_saves,
            "failed_files": failed_files
        }
        
        # Print summary
        print("\n" + "="*50)
        print("Processing Summary:")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successfully preprocessed: {summary['successful_preprocessing']}")
        print(f"Successfully cleaned missing values: {summary['successful_cleaning']}")
        print(f"Successfully deduplicated: {summary['successful_deduplication']}")
        print(f"Successfully saved to CSV: {summary['successful_saves']}")
        print(f"Failed to process: {summary['failed_files']}")
        print("="*50)
        
        return summary

    def preprocess_csv_file(self, input_file_path):
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
            # Load the dataframe from the input file path
            # Note: We're still using generate_output_path to load the dataframe but won't save it
            _, df = generate_output_path(input_file_path)
            
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
            
            # Return the processed DataFrame without saving to file
            return cross_df
            
        except Exception as e:
            print(f"Error processing file {input_file_path}: {e}")
            return None
        
    def handle_missing_values(self, dataframe=None):
        """
        Handle missing values in preprocessed dataframes.
        
        Processes the provided dataframe and handles missing values:
        - Drop rows where all weather condition columns have missing values
        - Drop rows where differenceInMinutes or cancelled are None
        - Keep track of how many rows were dropped
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The preprocessed dataframe to handle missing values in.
            
        Returns:
        --------
        pandas.DataFrame
            The cleaned dataframe with missing values handled.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe
        print(f"Processing dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Count rows before cleaning
        original_row_count = len(df)
        
        # Step 1: Check required columns (differenceInMinutes and cancelled)
        required_cols = ["differenceInMinutes", "cancelled"]
        available_required_cols = [col for col in required_cols if col in df.columns]
        
        if available_required_cols:
            print(f"Checking for missing values in required columns: {available_required_cols}")
            # Store the count before dropping rows
            before_required_drop = len(df)
            # Drop rows where any of the required columns are None/NaN
            df = df.dropna(subset=available_required_cols)
            # Calculate dropped rows
            dropped_required = before_required_drop - len(df)
            print(f"- Dropped {dropped_required} rows with missing values in required columns")
        else:
            print("Warning: Required columns (differenceInMinutes, cancelled) not found in dataframe")
            dropped_required = 0
        
        # Step 2: Handle weather condition columns
        # Filter the list to only include columns that actually exist in the dataframe
        available_important_cols = [col for col in self.important_conditions if col in df.columns]
        
        if not available_important_cols:
            print("Warning: None of the specified important weather conditions found in the dataframe")
            return df
        
        print(f"Found {len(available_important_cols)} important weather condition columns: {available_important_cols}")
        
        # Store count before dropping weather condition rows
        before_weather_drop = len(df)
        
        # Drop rows where ALL of the important weather conditions are missing
        # (Keep rows with at least one of the specified conditions)
        df_cleaned = df.dropna(subset=available_important_cols, how='all')
        
        # Count how many rows were dropped due to weather conditions
        dropped_weather = before_weather_drop - len(df_cleaned)
        
        # Count total rows dropped
        total_dropped = original_row_count - len(df_cleaned)
        
        # Report the results
        print(f"Missing values handling complete:")
        print(f"- Original row count: {original_row_count}")
        print(f"- Rows dropped due to missing required columns: {dropped_required}")
        print(f"- Rows dropped due to missing all weather conditions: {dropped_weather}")
        print(f"- Total rows dropped: {total_dropped}")
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
        
        return df_cleaned
    
    def remove_duplicates(self, dataframe=None):
        """
        Remove duplicate rows from the processed dataframe.
        
        Duplicate rows can affect model training by:
        - Introducing bias towards duplicated data points
        - Increasing training time unnecessarily
        - Preventing simpler models
        - Causing overfitting
        - Affecting overall model accuracy
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to deduplicate.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with duplicates removed.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe
        print(f"Checking for duplicates in dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Count rows before deduplication
        original_row_count = len(df)
        
        # Remove duplicate rows
        df_deduplicated = df.drop_duplicates()
        
        # Count how many rows were removed
        removed_duplicates = original_row_count - len(df_deduplicated)
        
        # Report the results
        print(f"Duplicate removal complete:")
        print(f"- Original row count: {original_row_count}")
        print(f"- Duplicate rows removed: {removed_duplicates}")
        print(f"- Remaining rows: {len(df_deduplicated)}")
        
        # Calculate percentage of data retained
        if original_row_count > 0:
            retention_percentage = (len(df_deduplicated) / original_row_count) * 100
            print(f"- Data retention: {retention_percentage:.2f}%")
            print(f"- Duplicate percentage: {100 - retention_percentage:.2f}%")
        
        return df_deduplicated
    
    def save_df_to_csv(self, year_month, dataframe):
        """
        Save a processed dataframe to a CSV file.
        
        Parameters:
        -----------
        year_month : str
            Year and month in format "YYYY_MM" for the filename.
        dataframe : pandas.DataFrame
            The dataframe to save.
            
        Returns:
        --------
        bool
            True if saving was successful, False otherwise.
        """
        try:
            if dataframe is None or dataframe.empty:
                print(f"Warning: Cannot save empty dataframe for {year_month}")
                return False
                
            # Create the output filename
            filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year_month}.csv"
            file_path = os.path.join(self.output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save the dataframe
            dataframe.to_csv(file_path, index=False)
            print(f"Successfully saved dataframe to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving dataframe for {year_month}: {e}")
            return False