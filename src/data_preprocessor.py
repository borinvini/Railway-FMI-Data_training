import glob
import os
import pandas as pd
import re
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from config.const import (
    DATA_FILE_PREFIX_FOR_TRAINING, 
    OUTPUT_FOLDER,
    PREPROCESSED_OUTPUT_FOLDER,
    DECISION_TREE_OUTPUT_FOLDER
)
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
        self.preprocessed_dir = os.path.join(self.project_root, PREPROCESSED_OUTPUT_FOLDER)
        self.decision_tree_dir = os.path.join(self.project_root, DECISION_TREE_OUTPUT_FOLDER)
        
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

    def run_pipeline(self, csv_files, target_feature='differenceInMinutes'):
        """
        Run the full processing pipeline on the provided CSV files.
        
        This method coordinates the execution of the preprocessing and 
        missing value handling steps for each input file.
        
        Parameters:
        -----------
        csv_files : list
            List of CSV file paths to process.
        target_feature : str, optional
            The feature to keep (one of 'differenceInMinutes', 'trainDelayed', or 'cancelled').
            Defaults to 'differenceInMinutes'.
                
        Returns:
        --------
        dict
            A summary of the processing results.
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
        counters = {
            "successful_preprocessing": 0,
            "successful_cleaning": 0,
            "successful_deduplication": 0,
            "successful_scaling": 0,
            "successful_feature_addition": 0,
            "successful_target_selection": 0,
            "successful_saves": 0,
            "successful_splits": 0,
            "successful_decision_tree": 0,  # New counter for decision tree success
            "failed_decision_tree": 0,      # New counter for decision tree failures
            "failed_files": 0
        }
        
        # Process each file
        for i, input_file_path in enumerate(csv_files):
            filename = os.path.basename(input_file_path)
            print(f"\n[{i+1}/{len(csv_files)}] Processing file: {filename}")
            
            # Extract year and month from the filename
            pattern = r'(\d{4})_(\d{2})'
            match = re.search(pattern, filename)
            
            if not match:
                print(f"Could not extract year_month from filename: {filename}")
                continue
                
            year_month = f"{match.group(1)}_{match.group(2)}"
            
            # Define all possible pipeline stages
            stages = [
                "preprocess",
                "clean_missing_values",
                "remove_duplicates",
                "scale_numeric",
                "add_train_delayed",
                "select_target",
                "save_csv",
                "split_dataset",
                "train_decision_tree"  # New stage for decision tree
            ]
            
            # Initialize pipeline state
            state = {
                "current_stage": stages[0],
                "df": None,
                "year_month": year_month,
                "success": True
            }
            
            # Run the pipeline stages
            while state["current_stage"] and state["success"]:
                match state["current_stage"]:
                    case "preprocess":
                        print(f"Starting preprocessing for {filename}...")
                        state["df"] = self.preprocess_csv_file(input_file_path)
                        
                        if state["df"] is None:
                            print(f"Error occurred during preprocessing of {filename}")
                            counters["failed_files"] += 1
                            state["success"] = False
                        else:
                            print(f"Preprocessing completed successfully for {filename}")
                            print(f"Processed data contains {len(state['df'])} rows and {len(state['df'].columns)} columns")
                            counters["successful_preprocessing"] += 1
                            state["current_stage"] = "clean_missing_values"
                    
                    case "clean_missing_values":
                        print(f"\nHandling missing values for {year_month}...")
                        cleaned_df = self.handle_missing_values(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if cleaned_df is None:
                            print(f"Failed to clean missing values for {year_month}")
                            state["success"] = False
                        else:
                            print(f"Successfully cleaned missing values for {year_month}")
                            counters["successful_cleaning"] += 1
                            state["df"] = cleaned_df
                            state["current_stage"] = "remove_duplicates"
                    
                    case "remove_duplicates":
                        print(f"Removing duplicates for {year_month}...")
                        deduplicated_df = self.remove_duplicates(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if deduplicated_df is None:
                            print(f"Failed to remove duplicates for {year_month}")
                            state["success"] = False
                        else:
                            print(f"Successfully removed duplicates for {year_month}")
                            counters["successful_deduplication"] += 1
                            state["df"] = deduplicated_df
                            state["current_stage"] = "scale_numeric"
                    
                    case "scale_numeric":
                        print(f"Scaling numeric columns for {year_month}...")
                        scaled_df = self.scale_numeric_columns(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if scaled_df is None:
                            print(f"Failed to scale numeric columns for {year_month}")
                            state["success"] = False
                        else:
                            print(f"Successfully scaled numeric columns for {year_month}")
                            counters["successful_scaling"] += 1
                            state["df"] = scaled_df
                            state["current_stage"] = "add_train_delayed"
                    
                    case "add_train_delayed":
                        print(f"Adding trainDelayed feature for {year_month}...")
                        featured_df = self.add_train_delayed_feature(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if featured_df is None:
                            print(f"Failed to add trainDelayed feature for {year_month}")
                            state["success"] = False
                        else:
                            print(f"Successfully added trainDelayed feature for {year_month}")
                            counters["successful_feature_addition"] += 1
                            state["df"] = featured_df
                            state["current_stage"] = "select_target"
                    
                    case "select_target":
                        print(f"Selecting target feature '{target_feature}' for {year_month}...")
                        target_df = self.select_target_feature(dataframe=state["df"], target_feature=target_feature)
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if target_df is None:
                            print(f"Failed to select target feature for {year_month}")
                            state["success"] = False
                        else:
                            print(f"Successfully selected target feature for {year_month}")
                            counters["successful_target_selection"] += 1
                            state["df"] = target_df
                            state["current_stage"] = "save_csv"
                    
                    case "save_csv":
                        print(f"Saving processed dataframe for {year_month}...")
                        save_success = self.save_df_to_csv(year_month, state["df"])
                        
                        if not save_success:
                            print(f"Failed to save dataframe for {year_month}")
                            state["success"] = False
                        else:
                            print(f"Successfully saved dataframe for {year_month}")
                            counters["successful_saves"] += 1
                            state["current_stage"] = "split_dataset"
                    
                    case "split_dataset":
                        print(f"Splitting dataset for {year_month}...")
                        split_result = self.split_dataset(year_month)
                        
                        if not split_result.get("success", False):
                            print(f"Failed to split dataset for {year_month}: {split_result.get('error', 'Unknown error')}")
                            state["success"] = False
                        else:
                            print(f"Successfully split dataset for {year_month}")
                            counters["successful_splits"] += 1
                            state["current_stage"] = "train_decision_tree"  # Move to decision tree stage
                    
                    case "train_decision_tree":
                        print(f"Training decision tree model for {year_month}...")
                        dt_result = self.train_decision_tree(year_month)
                        
                        if not dt_result.get("success", False):
                            print(f"Failed to train decision tree for {year_month}: {dt_result.get('error', 'Unknown error')}")
                            counters["failed_decision_tree"] += 1
                        else:
                            print(f"Successfully trained decision tree for {year_month}")
                            counters["successful_decision_tree"] += 1
                        
                        # This is the last stage, so we're done
                        state["current_stage"] = None
                        
                        # Clear the dataframe from memory if it exists
                        if "df" in state and state["df"] is not None:
                            del state["df"]
                    
                    case _:
                        # Should never reach here
                        print(f"Unknown pipeline stage: {state['current_stage']}")
                        state["current_stage"] = None
        
        # Generate and return summary
        summary = {
            "total_files": len(csv_files),
            **counters
        }
        
        # Print summary
        print("\n" + "="*50)
        print("Processing Summary:")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successfully preprocessed: {summary['successful_preprocessing']}")
        print(f"Successfully cleaned missing values: {summary['successful_cleaning']}")
        print(f"Successfully deduplicated: {summary['successful_deduplication']}")
        print(f"Successfully scaled numeric columns: {summary['successful_scaling']}")
        print(f"Successfully added trainDelayed feature: {summary['successful_feature_addition']}")
        print(f"Successfully selected target feature: {summary['successful_target_selection']}")
        print(f"Successfully saved to CSV: {summary['successful_saves']}")
        print(f"Successfully split into train/test sets: {summary['successful_splits']}")
        print(f"Successfully trained decision tree models: {summary['successful_decision_tree']}")
        print(f"Failed to train decision tree models: {summary['failed_decision_tree']}")
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
    
    def scale_numeric_columns(self, dataframe=None):
        """
        Scale numeric columns in the dataframe using StandardScaler.
        
        This standardizes numeric features by removing the mean and scaling to unit variance.
        Important for machine learning models that assume features are on similar scales.
        
        Note: The 'differenceInMinutes' column is excluded from scaling as it's likely 
        the target variable we want to predict.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with numeric columns scaled (except for differenceInMinutes).
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Scaling numeric columns in dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        try:
            # Identify all numeric columns
            all_numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(all_numeric_columns) == 0:
                print("Warning: No numeric columns found in the dataframe")
                return df
            
            # Exclude differenceInMinutes from scaling if it exists
            columns_to_scale = [col for col in all_numeric_columns if col != 'differenceInMinutes']
            
            # Report which columns will be scaled and which ones are excluded
            excluded_columns = set(all_numeric_columns) - set(columns_to_scale)
            print(f"Found {len(all_numeric_columns)} numeric columns.")
            print(f"Excluding from scaling: {list(excluded_columns)}")
            print(f"Columns to scale: {columns_to_scale}")
            
            if not columns_to_scale:
                print("No columns to scale after exclusions. Returning original dataframe.")
                return df
            
            # Initialize the scaler
            scaler = StandardScaler()
            
            # Scale only the selected numeric columns
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
            
            print(f"Successfully scaled {len(columns_to_scale)} numeric columns")
            
            return df
            
        except Exception as e:
            print(f"Error scaling numeric columns: {e}")
            return dataframe  # Return original dataframe on error
        
    def add_train_delayed_feature(self, dataframe=None):
        """
        Add a binary column 'trainDelayed' based on differenceInMinutes.
        
        The column will be:
        - True when differenceInMinutes < 0 (train is delayed)
        - False when differenceInMinutes >= 0 (train is on time or early)
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with the added 'trainDelayed' column.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Adding 'trainDelayed' column to dataframe with {len(df)} rows")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        try:
            # Check if differenceInMinutes column exists
            if 'differenceInMinutes' not in df.columns:
                print("Error: 'differenceInMinutes' column not found in dataframe")
                return df
            
            # Create trainDelayed column based on differenceInMinutes
            df['trainDelayed'] = df['differenceInMinutes'] > 0
            
            # Reorder columns to place trainDelayed after differenceInMinutes
            cols = list(df.columns)
            # Remove trainDelayed from its current position
            cols.remove('trainDelayed')
            # Find the position of differenceInMinutes
            diff_idx = cols.index('differenceInMinutes')
            # Insert trainDelayed after differenceInMinutes
            cols.insert(diff_idx + 1, 'trainDelayed')
            # Rearrange the dataframe columns
            df = df[cols]
            
            print(f"Successfully added 'trainDelayed' column")
            print(f"Number of delayed trains: {df['trainDelayed'].sum()} ({df['trainDelayed'].mean() * 100:.2f}%)")
            
            return df
            
        except Exception as e:
            print(f"Error adding 'trainDelayed' column: {e}")
            return dataframe  # Return original dataframe on error
        

    def select_target_feature(self, dataframe=None, target_feature=None):
        """
        Select one of three features (differenceInMinutes, trainDelayed, cancelled) as the target
        and drop the other two.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
        target_feature : str
            The feature to keep (one of 'differenceInMinutes', 'trainDelayed', or 'cancelled').
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with only the selected target feature retained.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        # Check if target_feature is valid
        valid_targets = ['differenceInMinutes', 'trainDelayed', 'cancelled']
        if target_feature not in valid_targets:
            print(f"Error: target_feature must be one of {valid_targets}")
            return dataframe
            
        df = dataframe.copy()
        print(f"Selecting '{target_feature}' as target feature and dropping others")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        try:
            # Check which of the features exist in the dataframe
            features_to_drop = [f for f in valid_targets if f != target_feature and f in df.columns]
            
            if target_feature not in df.columns:
                print(f"Error: Target feature '{target_feature}' not found in dataframe")
                return df
                
            # Drop the other features
            df = df.drop(columns=features_to_drop)
            
            print(f"Successfully selected '{target_feature}' as target feature")
            print(f"Dropped columns: {features_to_drop}")
            
            return df
            
        except Exception as e:
            print(f"Error selecting target feature: {e}")
            return dataframe  # Return original dataframe on error
    
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
            file_path = os.path.join(self.preprocessed_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(self.preprocessed_dir, exist_ok=True)
            
            # Save the dataframe
            dataframe.to_csv(file_path, index=False)
            print(f"Successfully saved dataframe to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving dataframe for {year_month}: {e}")
            return False
        

    def split_dataset(self, year_month, test_size=0.3, random_state=42):
        """
        Split a processed dataset into training and testing sets and save them separately.
        
        Parameters:
        -----------
        year_month : str
            Year and month in format "YYYY_MM" for the filename.
        test_size : float, optional
            Proportion of the dataset to include in the test split. Defaults to 0.3.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
            
        Returns:
        --------
        dict
            A summary of the split results.
        """
        try:
            # Construct file path for the saved CSV
            filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year_month}.csv"
            file_path = os.path.join(self.preprocessed_dir, filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found")
                return {
                    "success": False,
                    "error": f"File {file_path} not found"
                }
            
            # Load the dataset
            print(f"Loading dataset from {file_path}")
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"Error: Empty dataset in {file_path}")
                return {
                    "success": False,
                    "error": f"Empty dataset in {file_path}"
                }
            
            # Identify target column (should be one of these three based on previous processing)
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled']
            target_column = None
            
            for option in target_options:
                if option in df.columns:
                    target_column = option
                    break
            
            if not target_column:
                print(f"Error: No target column found in dataset")
                return {
                    "success": False,
                    "error": "No target column found in dataset"
                }
            
            print(f"Identified target column: {target_column}")
            
            # Split features and target
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            # Check if there are any features left after dropping the target
            if X.empty:
                print(f"Error: No feature columns found in dataset")
                return {
                    "success": False,
                    "error": "No feature columns found in dataset"
                }
            
            # For stratified split, ensure target is categorical
            # If target is continuous (like differenceInMinutes), we can't use stratify
            use_stratify = False
            if target_column in ['trainDelayed', 'cancelled']:
                use_stratify = True
                print(f"Using stratified split on {target_column}")
            else:
                print(f"Target {target_column} is continuous, not using stratification")
            
            # Perform train-test split
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=random_state
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            
            # Recombine features and target for saving
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            # Create filenames for train and test sets
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year_month}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year_month}_test.csv"
            
            train_path = os.path.join(self.preprocessed_dir, train_filename)
            test_path = os.path.join(self.preprocessed_dir, test_filename)
            
            # Ensure output directory exists
            os.makedirs(self.preprocessed_dir, exist_ok=True)
            
            # Save the datasets
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            print(f"Successfully saved train dataset to {train_path}")
            print(f"Successfully saved test dataset to {test_path}")
            
            # Print distribution statistics
            print("\nDistribution Statistics:")
            
            # For categorical targets, show the distribution in percentages
            if target_column in ['trainDelayed', 'cancelled']:
                print("\nOriginal Distribution (%):")
                print(df[target_column].value_counts(normalize=True) * 100)
                
                print("\nTraining Set Distribution (%):")
                print(y_train.value_counts(normalize=True) * 100)
                
                print("\nTest Set Distribution (%):")
                print(y_test.value_counts(normalize=True) * 100)
            else:
                # For continuous targets like differenceInMinutes, still show basic stats
                # but also add bins for better visualization of distribution
                print("\nOriginal Distribution:")
                print(f"Mean: {df[target_column].mean():.2f}, Std: {df[target_column].std():.2f}")
                
                print("\nTraining Set Distribution:")
                print(f"Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
                
                print("\nTest Set Distribution:")
                print(f"Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
            
            # Return summary
            return {
                "success": True,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_path": train_path,
                "test_path": test_path,
                "target_column": target_column,
                "stratified": use_stratify
            }
            
        except Exception as e:
            print(f"Error splitting dataset for {year_month}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def train_decision_tree(self, year_month, max_depth=None, random_state=42):
        """
        Train a Decision Tree classifier on the preprocessed and split data.
        
        Parameters:
        -----------
        year_month : str
            Year and month in format "YYYY_MM" for the filename.
        max_depth : int, optional
            Maximum depth of the decision tree. None means unlimited.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
            
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics.
        """
        try:
            # Construct file paths for the train and test sets
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year_month}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year_month}_test.csv"
            
            train_path = os.path.join(self.preprocessed_dir, train_filename)
            test_path = os.path.join(self.preprocessed_dir, test_filename)
            
            # Check if files exist
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                error_msg = f"Files not found: {train_path} or {test_path}"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Load datasets
            print(f"Loading training data from {train_path}")
            train_df = pd.read_csv(train_path)
            
            print(f"Loading test data from {test_path}")
            test_df = pd.read_csv(test_path)
            
            # Identify target column (should be one of these three based on previous processing)
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled']
            target_column = None
            
            for option in target_options:
                if option in train_df.columns:
                    target_column = option
                    break
            
            if not target_column:
                print(f"Error: No target column found in dataset")
                return {
                    "success": False,
                    "error": "No target column found in dataset"
                }
            
            print(f"Identified target column: {target_column}")
            
            # Split features and target
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            
            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]
            
            # Check if we have classification or regression problem
            is_classification = True
            if target_column == 'differenceInMinutes':
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                # Train Decision Tree Classifier
                dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
                print(f"Training Decision Tree classifier with max_depth={max_depth}")
                dt.fit(X_train, y_train)
                
                # Predict
                y_pred = dt.predict(X_test)
                
                # Evaluate model
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                print(f"\nDecision Tree Classifier Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': dt.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # Save the model
                try:
                    import joblib
                    
                    # Ensure decision tree output directory exists
                    os.makedirs(self.decision_tree_dir, exist_ok=True)
                    
                    model_filename = f"decision_tree_{year_month}.joblib"
                    model_path = os.path.join(self.decision_tree_dir, model_filename)
                    joblib.dump(dt, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{year_month}.csv"
                    importance_path = os.path.join(self.decision_tree_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "model_path": model_path,
                        "feature_importance_path": importance_path
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "model_saved": False
                    }
            else:
                # For regression problems we would need a different approach
                # For now, just note that we don't handle regression
                print(f"Regression with Decision Trees not implemented for target {target_column}")
                return {
                    "success": False,
                    "error": f"Regression with Decision Trees not implemented for target {target_column}"
                }
        
        except Exception as e:
            print(f"Error training decision tree for {year_month}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }