import glob
import os
import pandas as pd
import re
import ast
import xgboost as xgb
import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.combine import SMOTETomek
from src.file_utils import generate_output_path
from sklearn.model_selection import RandomizedSearchCV

from config.const import (
    DATA_FILE_PREFIX_FOR_TRAINING,
    IMPORTANT_FEATURES_OUTPUT_FOLDER,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER, 
    OUTPUT_FOLDER,
    PREPROCESSED_OUTPUT_FOLDER,
    DECISION_TREE_OUTPUT_FOLDER,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    IMPORTANCE_THRESHOLD,
    XGBOOST_OUTPUT_FOLDER,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    DEFAULT_TARGET_FEATURE
)


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
        self.randomized_search_dir = os.path.join(self.project_root, RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER)
        self.important_features_randomized_search_dir = os.path.join(self.project_root, IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
        self.important_features_dir = os.path.join(self.project_root, IMPORTANT_FEATURES_OUTPUT_FOLDER)
        self.xgboost_dir = os.path.join(self.project_root, XGBOOST_OUTPUT_FOLDER)
        self.xgboost_rs_dir = os.path.join(self.project_root, XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)

        # Add this line to make the constant available as an instance attribute
        self.DATA_FILE_PREFIX_FOR_TRAINING = DATA_FILE_PREFIX_FOR_TRAINING

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

    def run_pipeline_data_by_month(self, csv_files, target_feature=DEFAULT_TARGET_FEATURE):
        """
        Run the full processing pipeline on the provided CSV files, grouping by month across years.
        
        This method coordinates the execution of the preprocessing and 
        missing value handling steps for each month's combined data across years.
        
        Parameters:
        -----------
        csv_files : list
            List of CSV file paths to process.
        target_feature : str, optional
            The feature to keep (one of 'differenceInMinutes', 'trainDelayed', or 'cancelled').
            Defaults to DEFAULT_TARGET_FEATURE from the constants.
        """
        if not csv_files:
            print("\nNo CSV files to process.")
            return {
                "total_files": 0,
                "successful_preprocessing": 0,
                "successful_cleaning": 0,
                "failed_files": 0
            }
        
        print(f"\nStarting pipeline processing for {len(csv_files)} CSV files, grouped by month.")
        
        # Group files by month
        files_by_month = {}
        pattern = r'(\d{4})_(\d{2})\.csv$'
        
        for input_file_path in csv_files:
            filename = os.path.basename(input_file_path)
            match = re.search(pattern, filename)
            
            if match:
                year, month = match.groups()
                # Use month as the key for grouping
                if month not in files_by_month:
                    files_by_month[month] = []
                files_by_month[month].append((year, input_file_path))
        
        print(f"Found data for {len(files_by_month)} distinct months: {sorted(files_by_month.keys())}")
        
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
            "successful_decision_tree": 0,
            "failed_decision_tree": 0,
            "failed_files": 0
        }
        
        # Process each month's data
        for i, (month_num, file_info) in enumerate(sorted(files_by_month.items())):
            years = [year for year, _ in file_info]
            file_paths = [path for _, path in file_info]
            
            print(f"\n[{i+1}/{len(files_by_month)}] Processing month: {month_num} (Years: {', '.join(sorted(years))})")
            
            # Load and combine data from all files for this month
            combined_df = None
            for file_path in file_paths:
                try:
                    # Preprocess the file
                    processed_df = self.preprocess_csv_file(file_path)
                    
                    if processed_df is not None:
                        # Extract year from filename for tracking
                        filename = os.path.basename(file_path)
                        year_match = re.search(r'(\d{4})_', filename)
                        if year_match:
                            year = year_match.group(1)
                            processed_df['data_year'] = year
                        
                        # Combine with existing data
                        if combined_df is None:
                            combined_df = processed_df
                        else:
                            combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
                        
                        print(f"Added data from {filename} ({len(processed_df)} rows)")
                    else:
                        print(f"Warning: Failed to preprocess {file_path}")
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            if combined_df is None or combined_df.empty:
                print(f"No valid data was loaded for month {month_num}")
                counters["failed_files"] += 1
                continue
            
            print(f"Combined data for month {month_num} has {len(combined_df)} rows and {len(combined_df.columns)} columns")
            
            # Sort years for consistent naming
            years.sort()
            year_range = f"{years[0]}-{years[-1]}" if len(years) > 1 else years[0]
            
            # Generate a month identifier for filenames
            month_id = f"{year_range}_{month_num}"
            
            # Define all possible pipeline stages
            stages = [
                "clean_missing_values",
                "remove_duplicates",
                "scale_numeric",
                "add_train_delayed",
                "select_target",
                "save_csv",
                "split_dataset",
                "train_decision_tree",
                "train_with_important_features",
                "train_randomized_search_cv",
                "train_randomized_search_with_important_features",
                "train_xgboost",  
                "train_xgboost_with_randomized_search_cv" 
            ]
            
            # Initialize pipeline state
            state = {
                "current_stage": stages[0],
                "df": combined_df,
                "month_id": month_id,
                "success": True
            }
            
            # Count successful preprocessing
            counters["successful_preprocessing"] += 1
            
            # Run the pipeline stages
            while state["current_stage"] and state["success"]:
                match state["current_stage"]:
                    case "clean_missing_values":
                        print(f"\nHandling missing values for {month_id}...")
                        cleaned_df = self.handle_missing_values(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if cleaned_df is None:
                            print(f"Failed to clean missing values for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully cleaned missing values for {month_id}")
                            counters["successful_cleaning"] += 1
                            state["df"] = cleaned_df
                            state["current_stage"] = "remove_duplicates"
                    
                    case "remove_duplicates":
                        print(f"Removing duplicates for {month_id}...")
                        deduplicated_df = self.remove_duplicates(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if deduplicated_df is None:
                            print(f"Failed to remove duplicates for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully removed duplicates for {month_id}")
                            counters["successful_deduplication"] += 1
                            state["df"] = deduplicated_df
                            state["current_stage"] = "scale_numeric"
                    
                    case "scale_numeric":
                        print(f"Scaling numeric columns for {month_id}...")
                        scaled_df = self.scale_numeric_columns(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if scaled_df is None:
                            print(f"Failed to scale numeric columns for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully scaled numeric columns for {month_id}")
                            counters["successful_scaling"] += 1
                            state["df"] = scaled_df
                            state["current_stage"] = "add_train_delayed"
                    
                    case "add_train_delayed":
                        print(f"Adding trainDelayed feature for {month_id}...")
                        featured_df = self.add_train_delayed_feature(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if featured_df is None:
                            print(f"Failed to add trainDelayed feature for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully added trainDelayed feature for {month_id}")
                            counters["successful_feature_addition"] += 1
                            state["df"] = featured_df
                            state["current_stage"] = "select_target"
                    
                    case "select_target":
                        print(f"Selecting target feature '{target_feature}' for {month_id}...")
                        target_df = self.select_target_feature(dataframe=state["df"], target_feature=target_feature)
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if target_df is None:
                            print(f"Failed to select target feature for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully selected target feature for {month_id}")
                            counters["successful_target_selection"] += 1
                            state["df"] = target_df
                            state["current_stage"] = "save_csv"
                    
                    case "save_csv":
                        print(f"Saving processed dataframe for {month_id}...")
                        save_success = self.save_month_df_to_csv(month_id, state["df"])
                        
                        if not save_success:
                            print(f"Failed to save dataframe for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully saved dataframe for {month_id}")
                            counters["successful_saves"] += 1
                            state["current_stage"] = "split_dataset"
                    
                    case "split_dataset":
                        print(f"Splitting dataset for {month_id}...")
                        split_result = self.split_month_dataset(month_id)
                        
                        if not split_result.get("success", False):
                            print(f"Failed to split dataset for {month_id}: {split_result.get('error', 'Unknown error')}")
                            state["success"] = False
                        else:
                            print(f"Successfully split dataset for {month_id}")
                            counters["successful_splits"] += 1
                            state["current_stage"] = "train_decision_tree"
                    
                    case "train_decision_tree":
                        print(f"Training decision tree model for {month_id}...")
                        dt_result = self.train_month_decision_tree(month_id)
                        
                        if not dt_result.get("success", False):
                            print(f"Failed to train decision tree for {month_id}: {dt_result.get('error', 'Unknown error')}")
                            counters["failed_decision_tree"] += 1
                        else:
                            print(f"Successfully trained decision tree for {month_id}")
                            counters["successful_decision_tree"] += 1
                        
                        # Move to the next stage regardless of success
                        state["current_stage"] = "train_with_important_features"

                    case "train_with_important_features":
                        print(f"Training decision tree with important features for {month_id}...")
                        important_result = self.train_with_important_features(month_id, importance_threshold=IMPORTANCE_THRESHOLD)
                        
                        if not important_result.get("success", False):
                            print(f"Failed to train decision tree with important features for {month_id}: {important_result.get('error', 'Unknown error')}")
                            counters["failed_important_features"] = counters.get("failed_important_features", 0) + 1
                        else:
                            print(f"Successfully trained decision tree with important features for {month_id}")
                            counters["successful_important_features"] = counters.get("successful_important_features", 0) + 1

                        # This is no longer the last stage, so point to the next stage
                        state["current_stage"] = "train_randomized_search_cv"

                    case "train_randomized_search_cv":
                        print(f"Training decision tree with RandomizedSearchCV for {month_id}...")
                        random_search_result = self.train_randomized_search_cv(month_id)
                        
                        if not random_search_result.get("success", False):
                            print(f"Failed to train decision tree with RandomizedSearchCV for {month_id}: {random_search_result.get('error', 'Unknown error')}")
                            counters["failed_randomized_search"] = counters.get("failed_randomized_search", 0) + 1
                        else:
                            print(f"Successfully trained decision tree with RandomizedSearchCV for {month_id}")
                            counters["successful_randomized_search"] = counters.get("successful_randomized_search", 0) + 1

                        # Point to the new final stage
                        state["current_stage"] = "train_randomized_search_with_important_features"

                    case "train_randomized_search_with_important_features":
                        print(f"Training decision tree with RandomizedSearchCV on important features for {month_id}...")
                        combined_result = self.train_randomized_search_with_important_features(month_id)
                        
                        if not combined_result.get("success", False):
                            print(f"Failed to train with combined approach for {month_id}: {combined_result.get('error', 'Unknown error')}")
                            counters["failed_combined_approach"] = counters.get("failed_combined_approach", 0) + 1
                        else:
                            print(f"Successfully trained with combined approach for {month_id}")
                            counters["successful_combined_approach"] = counters.get("successful_combined_approach", 0) + 1

                        # Continue to the XGBoost stage
                        state["current_stage"] = "train_xgboost"

                    case "train_xgboost":
                        print(f"Training XGBoost model for {month_id}...")
                        xgb_result = self.train_xgboost(month_id)
                        
                        if not xgb_result.get("success", False):
                            print(f"Failed to train XGBoost for {month_id}: {xgb_result.get('error', 'Unknown error')}")
                            counters["failed_xgboost"] = counters.get("failed_xgboost", 0) + 1
                        else:
                            print(f"Successfully trained XGBoost for {month_id}")
                            counters["successful_xgboost"] = counters.get("successful_xgboost", 0) + 1

                        # Move to the next stage regardless of success
                        state["current_stage"] = "train_xgboost_with_randomized_search_cv"

                    case "train_xgboost_with_randomized_search_cv":
                        print(f"Training XGBoost model with RandomizedSearchCV for {month_id}...")
                        xgb_rs_result = self.train_xgboost_with_randomized_search_cv(month_id)
                        
                        if not xgb_rs_result.get("success", False):
                            print(f"Failed to train XGBoost with RandomizedSearchCV for {month_id}: {xgb_rs_result.get('error', 'Unknown error')}")
                            counters["failed_xgboost_rs"] = counters.get("failed_xgboost_rs", 0) + 1
                        else:
                            print(f"Successfully trained XGBoost with RandomizedSearchCV for {month_id}")
                            counters["successful_xgboost_rs"] = counters.get("successful_xgboost_rs", 0) + 1

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
            "total_months": len(files_by_month),
            "total_files": len(csv_files),
            **counters
        }
        
        # Print summary
        print("\n" + "="*50)
        print("Processing Summary:")
        print(f"Total months processed: {summary['total_months']}")
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
        print(f"Successfully trained decision tree models with important features: {summary.get('successful_important_features', 0)}")
        print(f"Successfully trained decision tree models with RandomizedSearchCV: {summary.get('successful_randomized_search', 0)}")
        print(f"Successfully trained with RandomizedSearchCV on important features: {summary.get('successful_combined_approach', 0)}")
        print(f"Successfully trained XGBoost models: {summary.get('successful_xgboost', 0)}")
        print(f"Successfully trained XGBoost models with RandomizedSearchCV: {summary.get('successful_xgboost_rs', 0)}")
        print(f"Failed to train decision tree models: {summary['failed_decision_tree']}")
        print(f"Failed to train decision tree models with important features: {summary.get('failed_important_features', 0)}")
        print(f"Failed to train decision tree models with RandomizedSearchCV: {summary.get('failed_randomized_search', 0)}")
        print(f"Failed to train with RandomizedSearchCV on important features: {summary.get('failed_combined_approach', 0)}")
        print(f"Failed to train XGBoost models: {summary.get('failed_xgboost', 0)}")
        print(f"Failed to train XGBoost models with RandomizedSearchCV: {summary.get('failed_xgboost_rs', 0)}")
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
        
    def save_month_df_to_csv(self, month_id, dataframe):
        """
        Save a processed month's dataframe to a CSV file.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        dataframe : pandas.DataFrame
            The dataframe to save.
            
        Returns:
        --------
        bool
            True if saving was successful, False otherwise.
        """
        try:
            if dataframe is None or dataframe.empty:
                print(f"Warning: Cannot save empty dataframe for {month_id}")
                return False
                
            # Create the output filename
            filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}.csv"
            file_path = os.path.join(self.preprocessed_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(self.preprocessed_dir, exist_ok=True)
            
            # Save the dataframe
            dataframe.to_csv(file_path, index=False)
            print(f"Successfully saved dataframe to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving dataframe for {month_id}: {e}")
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
        
    def split_month_dataset(self, month_id, test_size=0.3, random_state=42):
        """
        Split a processed month's dataset into training and testing sets and save them separately.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
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
            filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}.csv"
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
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_test.csv"
            
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
                # For continuous targets like differenceInMinutes, show basic stats
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
            print(f"Error splitting dataset for {month_id}: {str(e)}")
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
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(y_test, y_pred, report, year_month)
                
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
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "metrics": metrics_result["metrics"],
                        "metrics_path": metrics_result["metrics_path"],
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
        
        
    def train_month_decision_tree(self, month_id, max_depth=None, random_state=42):
        """
        Train a Decision Tree classifier on the preprocessed and split month data.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
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
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_test.csv"
            
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

            # Drop the data_year column if it exists
            if 'data_year' in X_train.columns:
                print(f"Dropping 'data_year' column from training features")
                X_train = X_train.drop('data_year', axis=1)
                
            if 'data_year' in X_test.columns:
                print(f"Dropping 'data_year' column from test features")
                X_test = X_test.drop('data_year', axis=1)
            
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
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(y_test, y_pred, report, month_id)
                
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
                    
                    model_filename = f"decision_tree_{month_id}.joblib"
                    model_path = os.path.join(self.decision_tree_dir, model_filename)
                    joblib.dump(dt, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{month_id}.csv"
                    importance_path = os.path.join(self.decision_tree_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "metrics": metrics_result["metrics"],
                        "metrics_path": metrics_result["metrics_path"],
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
            print(f"Error training decision tree for {month_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        

    def train_with_important_features(self, month_id, importance_threshold=IMPORTANCE_THRESHOLD, max_depth=None, random_state=42):
        """
        Train a Decision Tree classifier on only the important features.
        
        This method first trains a model on all features, identifies the most important features
        based on the threshold, and then trains a new model using only those features.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        importance_threshold : float, optional
            Threshold for selecting important features. Features with importance scores 
            above this threshold will be kept. Defaults to IMPORTANCE_THRESHOLD.
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
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_test.csv"
            
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

            if 'data_year' in X_train.columns:
                print(f"Dropping 'data_year' column from training features")
                X_train = X_train.drop('data_year', axis=1)
            
            if 'data_year' in X_test.columns:
                print(f"Dropping 'data_year' column from test features")
                X_test = X_test.drop('data_year', axis=1)
            
            # Check if we have classification or regression problem
            is_classification = True
            if target_column == 'differenceInMinutes':
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                # First, train a model on all features to get feature importance
                # Use class_weight="balanced" to handle imbalanced classes
                print(f"Training initial Decision Tree classifier with all features...")
                dt = DecisionTreeClassifier(class_weight="balanced", max_depth=max_depth, random_state=random_state)
                dt.fit(X_train, y_train)
                
                # Calculate feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': dt.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # Select important features based on threshold
                important_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
                
                if not important_features:
                    print(f"Warning: No features found with importance > {importance_threshold}. Using top 5 features instead.")
                    important_features = feature_importance.head(5)['Feature'].tolist()
                
                print(f"\nSelected {len(important_features)} important features:")
                print(important_features)
                
                # Train a new model with only the important features
                print(f"\nTraining new Decision Tree classifier with only important features...")
                dt_selected = DecisionTreeClassifier(random_state=random_state)  # Note: not using class_weight here as in example
                dt_selected.fit(X_train[important_features], y_train)
                
                # Predict
                y_pred = dt_selected.predict(X_test[important_features])
                
                # Evaluate model
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                print(f"\nDecision Tree Classifier Results (Important Features Only):")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(y_test, y_pred, report, f"{month_id}_important_features", output_dir=self.important_features_dir)
                
                # Feature importance for the new model
                selected_feature_importance = pd.DataFrame({
                    'Feature': important_features,
                    'Importance': dt_selected.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance for Selected Features:")
                print(selected_feature_importance)
                
                # Save the model and feature list
                try:
                    import joblib
                    
                    # Ensure decision tree output directory exists
                    os.makedirs(self.important_features_dir, exist_ok=True)
                    
                    # Save the model
                    model_filename = f"decision_tree_{month_id}_important_features.joblib"
                    model_path = os.path.join(self.important_features_dir, model_filename)
                    joblib.dump(dt_selected, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{month_id}_important_features.csv"
                    importance_path = os.path.join(self.important_features_dir, importance_filename)
                    selected_feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    # Save the list of important features
                    features_filename = f"important_features_{month_id}.txt"
                    features_path = os.path.join(self.important_features_dir, features_filename)
                    with open(features_path, 'w') as f:
                        for feature in important_features:
                            f.write(f"{feature}\n")
                    print(f"Important features list saved to {features_path}")
                    
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "important_features": important_features,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "metrics": metrics_result["metrics"],
                        "metrics_path": metrics_result["metrics_path"],
                        "model_saved": False,
                        "important_features": important_features
                    }
            else:
                # For regression problems we would need a different approach
                print(f"Regression with Decision Trees not implemented for target {target_column}")
                return {
                    "success": False,
                    "error": f"Regression with Decision Trees not implemented for target {target_column}"
                }
        
        except Exception as e:
            print(f"Error training decision tree with important features for {month_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        

    def train_randomized_search_cv(self, month_id, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train a Decision Tree classifier with hyperparameter tuning using RandomizedSearchCV.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        param_distributions : dict, optional
            Dictionary with parameters names as keys and distributions or lists of parameters to try.
            Defaults to RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS from constants.
        n_iter : int, optional
            Number of parameter settings that are sampled. Defaults to RANDOM_SEARCH_ITERATIONS.
        cv : int, optional
            Number of cross-validation folds. Defaults to RANDOM_SEARCH_CV_FOLDS.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
                
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics.
        """
        try:
            # Use default values from constants if not provided
            if param_distributions is None:
                from config.const import RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS
                param_distributions = RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS
            
            if n_iter is None:
                from config.const import RANDOM_SEARCH_ITERATIONS
                n_iter = RANDOM_SEARCH_ITERATIONS
                
            if cv is None:
                from config.const import RANDOM_SEARCH_CV_FOLDS
                cv = RANDOM_SEARCH_CV_FOLDS
            
            # Construct file paths for the train and test sets
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_test.csv"
            
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

            if 'data_year' in X_train.columns:
                print(f"Dropping 'data_year' column from training features")
                X_train = X_train.drop('data_year', axis=1)
                
            if 'data_year' in X_test.columns:
                print(f"Dropping 'data_year' column from test features")
                X_test = X_test.drop('data_year', axis=1)
            
            # Check if we have classification or regression problem
            is_classification = True
            if target_column == 'differenceInMinutes':
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                from sklearn.model_selection import RandomizedSearchCV
                
                # Initialize base classifier
                dt = DecisionTreeClassifier(random_state=random_state)
                
                print(f"Starting RandomizedSearchCV with {n_iter} iterations and {cv}-fold cross-validation...")
                
                # Run RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    dt, param_distributions, n_iter=n_iter, cv=cv, 
                    scoring='accuracy', random_state=random_state, n_jobs=-1
                )
                
                random_search.fit(X_train, y_train)
                
                best_params = random_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
                
                # Train model with best parameters
                best_dt = DecisionTreeClassifier(**best_params, random_state=random_state)
                best_dt.fit(X_train, y_train)
                
                # Predict
                y_pred = best_dt.predict(X_test)
                
                # Evaluate model
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                print(f"\nDecision Tree Classifier Results (Tuned with RandomizedSearchCV):")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(y_test, y_pred, report, f"{month_id}_randomized_search", output_dir=self.randomized_search_dir)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': best_dt.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # Save the model and params
                try:
                    import joblib
                    
                    # Ensure randomized search output directory exists
                    os.makedirs(self.randomized_search_dir, exist_ok=True)
                    
                    # Save the model
                    model_filename = f"decision_tree_{month_id}_randomized_search.joblib"
                    model_path = os.path.join(self.randomized_search_dir, model_filename)
                    joblib.dump(best_dt, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{month_id}_randomized_search.csv"
                    importance_path = os.path.join(self.randomized_search_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    # Save best parameters
                    params_filename = f"best_params_{month_id}_randomized_search.txt"
                    params_path = os.path.join(self.randomized_search_dir, params_filename)
                    with open(params_path, 'w') as f:
                        for param, value in best_params.items():
                            f.write(f"{param}: {value}\n")
                    print(f"Best parameters saved to {params_path}")
                    
                    # Also compare to baseline model
                    print("\nComparison with baseline model:")
                    baseline_metrics_file = os.path.join(self.decision_tree_dir, f"model_metrics_{month_id}.csv")
                    if os.path.exists(baseline_metrics_file):
                        baseline_metrics = pd.read_csv(baseline_metrics_file)
                        baseline_accuracy = baseline_metrics['accuracy'].values[0]
                        print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
                        print(f"RandomizedSearchCV model accuracy: {accuracy:.4f}")
                        improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
                        print(f"Improvement: {improvement:.2f}%")
                    
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "best_params": best_params,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "best_params": best_params,
                        "metrics": metrics_result["metrics"],
                        "metrics_path": metrics_result["metrics_path"],
                        "model_saved": False
                    }
            else:
                # For regression problems we would need a different approach
                print(f"Regression with Decision Trees not implemented for target {target_column}")
                return {
                    "success": False,
                    "error": f"Regression with Decision Trees not implemented for target {target_column}"
                }
        
        except Exception as e:
            print(f"Error in RandomizedSearchCV for {month_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        
    def train_randomized_search_with_important_features(self, month_id, importance_threshold=IMPORTANCE_THRESHOLD, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train a Decision Tree classifier with hyperparameter tuning using RandomizedSearchCV,
        but only using features that exceed the importance threshold.
        
        This method combines feature selection and hyperparameter optimization:
        1. First trains a model to identify important features
        2. Then uses RandomizedSearchCV to find optimal hyperparameters on those features only
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        importance_threshold : float, optional
            Threshold for selecting important features. Features with importance scores
            above this threshold will be kept. Defaults to IMPORTANCE_THRESHOLD.
        param_distributions : dict, optional
            Dictionary with parameters names as keys and distributions or lists of parameters to try.
            Defaults to RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS from constants.
        n_iter : int, optional
            Number of parameter settings that are sampled. Defaults to RANDOM_SEARCH_ITERATIONS.
        cv : int, optional
            Number of cross-validation folds. Defaults to RANDOM_SEARCH_CV_FOLDS.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
                
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics.
        """
        try:
            # Use default values from constants if not provided
            if param_distributions is None:
                from config.const import RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS
                param_distributions = RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS
            
            if n_iter is None:
                from config.const import RANDOM_SEARCH_ITERATIONS
                n_iter = RANDOM_SEARCH_ITERATIONS
                
            if cv is None:
                from config.const import RANDOM_SEARCH_CV_FOLDS
                cv = RANDOM_SEARCH_CV_FOLDS
            
            # Construct file paths for the train and test sets
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_test.csv"
            
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

            if 'data_year' in X_train.columns:
                print(f"Dropping 'data_year' column from training features")
                X_train = X_train.drop('data_year', axis=1)
                
            if 'data_year' in X_test.columns:
                print(f"Dropping 'data_year' column from test features")
                X_test = X_test.drop('data_year', axis=1)
            
            # Check if we have classification or regression problem
            is_classification = True
            if target_column == 'differenceInMinutes':
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                from sklearn.model_selection import RandomizedSearchCV
                
                # STEP 1: FIRST TRAIN A MODEL TO IDENTIFY IMPORTANT FEATURES
                print(f"Training initial Decision Tree classifier to identify important features...")
                dt_initial = DecisionTreeClassifier(random_state=random_state)
                dt_initial.fit(X_train, y_train)
                
                # Calculate feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': dt_initial.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # Select important features based on threshold
                important_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
                
                if not important_features:
                    print(f"Warning: No features found with importance > {importance_threshold}. Using top 5 features instead.")
                    important_features = feature_importance.head(5)['Feature'].tolist()
                
                print(f"\nSelected {len(important_features)} important features with threshold {importance_threshold}:")
                print(important_features)
                
                # STEP 2: RUN RANDOMIZED SEARCH CV ON IMPORTANT FEATURES ONLY
                print(f"\nStarting RandomizedSearchCV on important features with {n_iter} iterations and {cv}-fold cross-validation...")
                
                # Initialize base classifier for RandomizedSearchCV
                dt = DecisionTreeClassifier(random_state=random_state)
                
                # Run RandomizedSearchCV on the subset of important features
                random_search = RandomizedSearchCV(
                    dt, param_distributions, n_iter=n_iter, cv=cv, 
                    scoring='accuracy', random_state=random_state, n_jobs=-1
                )
                
                # Fit only on the important features
                random_search.fit(X_train[important_features], y_train)
                
                best_params = random_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
                
                # Train model with best parameters on important features
                best_dt = DecisionTreeClassifier(**best_params, random_state=random_state)
                best_dt.fit(X_train[important_features], y_train)
                
                # Predict using only important features
                y_pred = best_dt.predict(X_test[important_features])
                
                # Evaluate model
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                print(f"\nDecision Tree Results (Important Features + RandomizedSearchCV):")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Create a specific output directory for this combined approach
                combined_output_dir = os.path.join(self.project_root, "data/output/important_features_randomized_search")
                os.makedirs(combined_output_dir, exist_ok=True)
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(
                    y_test, y_pred, report, 
                    f"{month_id}_important_randomized", 
                    output_dir=combined_output_dir
                )
                
                # Feature importance for the selected features in the final model
                selected_feature_importance = pd.DataFrame({
                    'Feature': important_features,
                    'Importance': best_dt.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance in Final Model:")
                print(selected_feature_importance)
                
                # Save the model and related information
                try:
                    import joblib
                    
                    # Save the model
                    model_filename = f"decision_tree_{month_id}_important_randomized.joblib"
                    model_path = os.path.join(combined_output_dir, model_filename)
                    joblib.dump(best_dt, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{month_id}_important_randomized.csv"
                    importance_path = os.path.join(combined_output_dir, importance_filename)
                    selected_feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    # Save best parameters
                    params_filename = f"best_params_{month_id}_important_randomized.txt"
                    params_path = os.path.join(combined_output_dir, params_filename)
                    with open(params_path, 'w') as f:
                        f.write(f"Important features ({len(important_features)}):\n")
                        for feature in important_features:
                            f.write(f"- {feature}\n")
                        f.write("\nBest parameters:\n")
                        for param, value in best_params.items():
                            f.write(f"{param}: {value}\n")
                    print(f"Parameters and features saved to {params_path}")
                    
                    # Also compare to baseline models
                    print("\nComparison with other models:")
                    
                    # Compare with basic decision tree
                    baseline_metrics_file = os.path.join(self.decision_tree_dir, f"model_metrics_{month_id}.csv")
                    if os.path.exists(baseline_metrics_file):
                        baseline_metrics = pd.read_csv(baseline_metrics_file)
                        baseline_accuracy = baseline_metrics['accuracy'].values[0]
                        print(f"Basic Decision Tree accuracy: {baseline_accuracy:.4f}")
                        comparison1 = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
                        print(f"Improvement over basic model: {comparison1:.2f}%")
                    
                    # Compare with important features only
                    important_metrics_file = os.path.join(self.important_features_dir, f"model_metrics_{month_id}_important_features.csv")
                    if os.path.exists(important_metrics_file):
                        important_metrics = pd.read_csv(important_metrics_file)
                        important_accuracy = important_metrics['accuracy'].values[0]
                        print(f"Important Features Only accuracy: {important_accuracy:.4f}")
                        comparison2 = ((accuracy - important_accuracy) / important_accuracy) * 100
                        print(f"Improvement over important features model: {comparison2:.2f}%")
                    
                    # Compare with randomized search only
                    random_metrics_file = os.path.join(self.randomized_search_dir, f"model_metrics_{month_id}_randomized_search.csv")
                    if os.path.exists(random_metrics_file):
                        random_metrics = pd.read_csv(random_metrics_file)
                        random_accuracy = random_metrics['accuracy'].values[0]
                        print(f"RandomizedSearchCV Only accuracy: {random_accuracy:.4f}")
                        comparison3 = ((accuracy - random_accuracy) / random_accuracy) * 100
                        print(f"Improvement over randomized search model: {comparison3:.2f}%")
                    
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "best_params": best_params,
                        "important_features": important_features,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "best_params": best_params,
                        "important_features": important_features,
                        "metrics": metrics_result["metrics"],
                        "metrics_path": metrics_result["metrics_path"],
                        "model_saved": False
                    }
            else:
                # For regression problems we would need a different approach
                print(f"Regression with Decision Trees not implemented for target {target_column}")
                return {
                    "success": False,
                    "error": f"Regression with Decision Trees not implemented for target {target_column}"
                }
        
        except Exception as e:
            print(f"Error in RandomizedSearchCV with Important Features for {month_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def train_xgboost(self, month_id, params=None, random_state=42):
        """
        Train an XGBoost model (classifier or regressor) on the preprocessed and split month data.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        params : dict, optional
            Parameters for XGBoost model. If None, default parameters are used.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
            
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics.
        """
        try:
            # Use default parameters if none provided
            if params is None:
                from config.const import XGBOOST_DEFAULT_PARAMS
                params = XGBOOST_DEFAULT_PARAMS.copy()  # Create a copy to avoid modifying the original
            else:
                params = params.copy()  # Create a copy to avoid modifying the original
            
            # Add random state to params
            params['random_state'] = random_state
            
            # Construct file paths for the train and test sets
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_test.csv"
            
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

            # Drop the data_year column if it exists
            for col in ['data_year']:
                if col in X_train.columns:
                    print(f"Dropping '{col}' column from training features")
                    X_train = X_train.drop(col, axis=1)
                    
                if col in X_test.columns:
                    print(f"Dropping '{col}' column from test features")
                    X_test = X_test.drop(col, axis=1)
            
            # Create output directory
            xgboost_dir = os.path.join(self.project_root, XGBOOST_OUTPUT_FOLDER)
            os.makedirs(xgboost_dir, exist_ok=True)
            
            # Determine if it's a regression or classification problem
            from config.const import DEFAULT_TARGET_FEATURE
            is_regression = (target_column == DEFAULT_TARGET_FEATURE)
            
            if is_regression:
                # REGRESSION CASE
                print(f"Target '{target_column}' indicates a regression problem")
                print("Training XGBoost Regressor")
                
                # Set regression-specific parameters
                params['objective'] = 'reg:squarederror'
                
                # Train the model
                xgb_model = xgb.XGBRegressor(**params)
                xgb_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = xgb_model.predict(X_test)
                
                # Calculate regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Print metrics
                print(f"\nXGBoost Regressor Results:")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"R²: {r2:.4f}")
                
                # Create metrics dictionary
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Save metrics
                metrics_filename = f"model_metrics_{month_id}.csv"
                metrics_path = os.path.join(xgboost_dir, metrics_filename)
                pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
                print(f"Model metrics saved to {metrics_path}")
                
                # Set metrics_result for consistent return
                metrics_result = {
                    "metrics": metrics,
                    "metrics_path": metrics_path
                }
                
                # Create summary result
                result = {
                    "success": True,
                    "model_type": "regression",
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
                
            else:
                # CLASSIFICATION CASE
                print(f"Target '{target_column}' indicates a classification problem")
                print("Training XGBoost Classifier")
                
                # Set classification parameters
                if target_column == 'trainDelayed':
                    params['objective'] = 'binary:logistic'
                    print("Using binary classification")
                else:
                    # For multi-class problems like 'cancelled' with >2 classes
                    num_classes = len(y_train.unique())
                    params['objective'] = 'multi:softprob'
                    params['num_class'] = num_classes
                    print(f"Using multi-class classification with {num_classes} classes")
                
                # Train the model
                xgb_model = xgb.XGBClassifier(**params)
                xgb_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = xgb_model.predict(X_test)
                
                # Calculate classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Print metrics
                print(f"\nXGBoost Classifier Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(
                    y_test, y_pred, report, month_id, 
                    output_dir=xgboost_dir
                )
                
                # Create summary result
                result = {
                    "success": True,
                    "model_type": "classification",
                    "accuracy": accuracy,
                    "report": report
                }
            
            # Feature importance is common for both regression and classification
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': xgb_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            print("\nFeature Importance (top 10):")
            print(feature_importance.head(10))
            
            # Save the model and feature importance
            try:
                import joblib
                
                # Save the model
                model_filename = f"xgboost_{month_id}.joblib"
                model_path = os.path.join(xgboost_dir, model_filename)
                joblib.dump(xgb_model, model_path)
                print(f"Model saved to {model_path}")
                
                # Save feature importance
                importance_filename = f"feature_importance_{month_id}.csv"
                importance_path = os.path.join(xgboost_dir, importance_filename)
                feature_importance.to_csv(importance_path, index=False)
                print(f"Feature importance saved to {importance_path}")
                
                # Add paths to result
                result.update({
                    "metrics": metrics_result["metrics"],
                    "model_path": model_path,
                    "feature_importance_path": importance_path,
                    "metrics_path": metrics_result["metrics_path"]
                })
                
                return result
                    
            except Exception as e:
                print(f"Warning: Could not save model: {str(e)}")
                result.update({
                    "metrics": metrics_result["metrics"],
                    "metrics_path": metrics_result["metrics_path"],
                    "model_saved": False
                })
                return result
        
        except Exception as e:
            print(f"Error training XGBoost for {month_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def train_xgboost_with_randomized_search_cv(self, month_id, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train an XGBoost model (classifier or regressor) with hyperparameter tuning using manual CV.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        param_distributions : dict, optional
            Dictionary with parameters names as keys and distributions or lists of parameters to try.
        n_iter : int, optional
            Number of parameter settings that are sampled.
        cv : int, optional
            Number of cross-validation folds.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
                    
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics.
        """
        try:
            # Use default values from constants if not provided
            if param_distributions is None:
                from config.const import XGBOOST_PARAM_DISTRIBUTIONS
                param_distributions = XGBOOST_PARAM_DISTRIBUTIONS
            
            if n_iter is None:
                from config.const import RANDOM_SEARCH_ITERATIONS
                n_iter = RANDOM_SEARCH_ITERATIONS
                
            if cv is None:
                from config.const import RANDOM_SEARCH_CV_FOLDS
                cv = RANDOM_SEARCH_CV_FOLDS
            
            # MEMORY OPTIMIZATION: Limit parameters for better memory usage
            n_iter = min(n_iter, 20)  # Reduce number of iterations
            cv = min(cv, 3)  # Reduce CV folds
            
            # Construct file paths for the train and test sets
            train_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_train.csv"
            test_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}_test.csv"
            
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
            
            # Identify target column
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

            if 'data_year' in X_train.columns:
                print(f"Dropping 'data_year' column from training features")
                X_train = X_train.drop('data_year', axis=1)
                
            if 'data_year' in X_test.columns:
                print(f"Dropping 'data_year' column from test features")
                X_test = X_test.drop('data_year', axis=1)
            
            # DATA VALIDATION: Check for non-numeric columns
            non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
                X_train = X_train.select_dtypes(include=['number'])
                X_test = X_test.select_dtypes(include=['number'])
            
            # Check if we have classification or regression problem
            is_classification = True
            if target_column == 'differenceInMinutes':
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            # --------- MANUAL HYPERPARAMETER TUNING APPROACH ---------
            from sklearn.model_selection import ParameterSampler, KFold, StratifiedKFold
            import numpy as np
            
            print(f"Starting manual hyperparameter tuning with {n_iter} iterations and {cv}-fold cross-validation...")
            
            # Generate parameter combinations
            param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
            
            # Setup cross-validation
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            
            best_score = float('-inf')
            best_params = None
            
            # Memory tracking
            process = psutil.Process()
            before_mem = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage before training: {before_mem:.2f} MB")
            
            # Try each parameter combination
            for i, params in enumerate(param_list):
                print(f"Testing parameter combination {i+1}/{len(param_list)}")
                
                # Make a copy of params to modify
                current_params = params.copy()
                
                # Set the objective based on problem type
                if is_classification:
                    if target_column == 'trainDelayed':  # Binary classification
                        current_params['objective'] = 'binary:logistic'
                    else:  # Multi-class
                        current_params['objective'] = 'multi:softprob'
                        current_params['num_class'] = len(np.unique(y_train))
                else:
                    current_params['objective'] = 'reg:squarederror'
                
                # Add random_state for reproducibility
                current_params['random_state'] = random_state
                
                # Perform cross-validation
                cv_scores = []
                for train_idx, val_idx in cv_splitter.split(X_train, y_train if is_classification else np.zeros(len(y_train))):
                    # Get train and validation sets for this fold
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    # Initialize and train model
                    if is_classification:
                        model = xgb.XGBClassifier(**current_params)
                    else:
                        model = xgb.XGBRegressor(**current_params)
                    
                    model.fit(X_fold_train, y_fold_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_fold_val)
                    
                    # Calculate score based on problem type
                    if is_classification:
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_fold_val, y_pred)
                    else:
                        from sklearn.metrics import mean_squared_error
                        score = -mean_squared_error(y_fold_val, y_pred)  # Negative for higher=better
                    
                    cv_scores.append(score)
                
                # Calculate average score across folds
                avg_score = np.mean(cv_scores)
                print(f"  Average CV score: {avg_score:.4f}")
                
                # Update best if better
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = current_params.copy()
                    print(f"  New best score: {best_score:.4f}")
            
            after_mem = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage after training: {after_mem:.2f} MB (Δ: {after_mem - before_mem:.2f} MB)")
            
            print(f"Best Hyperparameters: {best_params}")
            print(f"Best CV Score: {best_score:.4f}")
            
            # Train final model with best parameters
            if is_classification:
                xgb_model = xgb.XGBClassifier(**best_params)
            else:
                xgb_model = xgb.XGBRegressor(**best_params)
            
            xgb_model.fit(X_train, y_train)
            
            # Create XGBoost RandomizedSearch output directory
            xgboost_rs_dir = os.path.join(self.project_root, XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
            os.makedirs(xgboost_rs_dir, exist_ok=True)
            
            # Evaluate on test set
            if is_classification:
                y_pred = xgb_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                print(f"\nXGBoost Classifier Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(
                    y_test, y_pred, report, f"{month_id}_rs", 
                    output_dir=xgboost_rs_dir
                )
            else:
                # Handle regression evaluation
                y_pred = xgb_model.predict(X_test)
                
                # Use the regression metrics function
                metrics_result = self.extract_and_save_regression_metrics(
                    y_test, y_pred, f"{month_id}_rs", 
                    output_dir=xgboost_rs_dir
                )
                
                # Keep these lines for printing to console
                mse = metrics_result["metrics"]["mse"]
                rmse = metrics_result["metrics"]["rmse"]
                mae = metrics_result["metrics"]["mae"]
                r2 = metrics_result["metrics"]["r2"]
                
                print(f"\nXGBoost Regressor Results:")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"R²: {r2:.4f}")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': xgb_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            print("\nFeature Importance (top 10):")
            print(feature_importance.head(10))
            
            # Save the model and feature importance
            try:
                import joblib
                
                # Create output directory
                os.makedirs(xgboost_rs_dir, exist_ok=True)
                
                # Save the model
                model_filename = f"xgboost_{month_id}_rs.joblib"
                model_path = os.path.join(xgboost_rs_dir, model_filename)
                joblib.dump(xgb_model, model_path)
                print(f"Model saved to {model_path}")
                
                # Save feature importance
                importance_filename = f"feature_importance_{month_id}_rs.csv"
                importance_path = os.path.join(xgboost_rs_dir, importance_filename)
                feature_importance.to_csv(importance_path, index=False)
                print(f"Feature importance saved to {importance_path}")
                
                # Save best parameters
                params_filename = f"best_params_{month_id}_rs.txt"
                params_path = os.path.join(xgboost_rs_dir, params_filename)
                with open(params_path, 'w') as f:
                    for param, value in best_params.items():
                        f.write(f"{param}: {value}\n")
                print(f"Best parameters saved to {params_path}")
                
                if is_classification:
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "best_params": best_params,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                else:
                    return {
                        "success": True,
                        "model_type": "regression",
                        "rmse": rmse,
                        "r2": r2,
                        "best_params": best_params,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                    
            except Exception as e:
                print(f"Warning: Could not save model: {str(e)}")
                if is_classification:
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy if 'accuracy' in locals() else None,
                        "metrics": metrics_result["metrics"] if 'metrics_result' in locals() else None,
                        "metrics_path": metrics_result["metrics_path"] if 'metrics_result' in locals() else None,
                        "model_saved": False,
                        "best_params": best_params
                    }
                else:
                    return {
                        "success": True,
                        "model_type": "regression",
                        "rmse": rmse if 'rmse' in locals() else None,
                        "r2": r2 if 'r2' in locals() else None,
                        "metrics": metrics_result["metrics"] if 'metrics_result' in locals() else None,
                        "metrics_path": metrics_result["metrics_path"] if 'metrics_result' in locals() else None,
                        "model_saved": False,
                        "best_params": best_params
                    }
                
        except Exception as e:
            import traceback
            print(f"Error in XGBoost hyperparameter tuning for {month_id}: {str(e)}")
            print("\nDetailed traceback:")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
        

    def extract_and_save_metrics(self, y_test, y_pred, report, month_id, output_dir=None):
        """
        Extract key metrics from model evaluation and save them to a CSV file.
        
        Parameters:
        -----------
        y_test : array-like
            True labels from the test dataset.
        y_pred : array-like
            Predicted labels from the model.
        report : dict
            Classification report dictionary from sklearn.
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        output_dir : str, optional
            Directory to save the metrics file. Defaults to decision_tree_dir.
            
        Returns:
        --------
        dict
            Dictionary containing the metrics and the path to the saved metrics file.
        """
        # Create metrics dictionary
        metrics = {}
        
        # Use decision_tree_dir as default if output_dir is None
        if output_dir is None:
            output_dir = self.decision_tree_dir
        
        # Basic accuracy
        from sklearn.metrics import accuracy_score
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Extract F1 scores from the classification report
        metrics['weighted_avg_f1'] = report['weighted avg']['f1-score']
        metrics['macro_avg_f1'] = report['macro avg']['f1-score']
        
        # For classification, also extract F1 scores for each class
        for class_label in report:
            if class_label not in ['weighted avg', 'macro avg', 'accuracy']:
                metrics[f'class_{class_label}_f1'] = report[class_label]['f1-score']
                
                # Also store precision and recall for completeness
                metrics[f'class_{class_label}_precision'] = report[class_label]['precision']
                metrics[f'class_{class_label}_recall'] = report[class_label]['recall']
        
        # Print metrics
        print("\nModel Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        # Save metrics to a file
        metrics_filename = f"model_metrics_{month_id}.csv"
        metrics_path = os.path.join(output_dir, metrics_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # Save to CSV
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Model metrics saved to {metrics_path}")
        
        return {
            "metrics": metrics,
            "metrics_path": metrics_path
        }
    
    def extract_and_save_regression_metrics(self, y_test, y_pred, month_id, output_dir=None):
        """
        Extract key metrics from regression model evaluation and save them to a CSV file.
        
        Parameters:
        -----------
        y_test : array-like
            True target values from the test dataset.
        y_pred : array-like
            Predicted target values from the model.
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        output_dir : str, optional
            Directory to save the metrics file. Defaults to xgboost_dir.
                
        Returns:
        --------
        dict
            Dictionary containing the metrics and the path to the saved metrics file.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import pandas as pd
        import os
        
        # Create metrics dictionary
        metrics = {}
        
        # Use xgboost_dir as default if output_dir is None
        if output_dir is None:
            output_dir = self.xgboost_dir
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        metrics['mse'] = mse
        metrics['rmse'] = rmse
        metrics['mae'] = mae
        metrics['r2'] = r2
        
        # Print metrics
        print("\nRegression Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        # Save metrics to a file
        metrics_filename = f"model_metrics_{month_id}.csv"
        metrics_path = os.path.join(output_dir, metrics_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # Save to CSV
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Model metrics saved to {metrics_path}")
        
        return {
            "metrics": metrics,
            "metrics_path": metrics_path
        }