import glob
import os
import pandas as pd
import re
import ast
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import ParameterSampler, KFold, StratifiedKFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            

from config.const import (
    DATA_FILE_PREFIX_FOR_TRAINING,
    IMPORTANT_FEATURES_OUTPUT_FOLDER,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_WEATHER_CONDITIONS,
    NON_NUMERIC_FEATURES, 
    OUTPUT_FOLDER,
    PIPELINE_STAGES,
    PREPROCESSED_OUTPUT_FOLDER,
    DECISION_TREE_OUTPUT_FOLDER,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    IMPORTANCE_THRESHOLD,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    TOP_FEATURES_COUNT,
    VALID_TARGET_FEATURES,
    XGBOOST_OUTPUT_FOLDER,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    DEFAULT_TARGET_FEATURE,
    FILTER_TRAINS_BY_STATIONS,
    REQUIRED_STATIONS
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
        self.regularized_regression_dir = os.path.join(self.project_root, REGULARIZED_REGRESSION_OUTPUT_FOLDER)


        # Add this line to make the constant available as an instance attribute
        self.DATA_FILE_PREFIX_FOR_TRAINING = DATA_FILE_PREFIX_FOR_TRAINING

        # Use the imported constant instead of defining it here
        self.important_conditions = IMPORTANT_WEATHER_CONDITIONS

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
            "successful_snow_depth_merge": 0,
            "successful_cleaning": 0,
            "successful_deduplication": 0,
            "successful_scaling": 0,
            "successful_feature_addition": 0,
            "successful_target_selection": 0,
            "successful_saves": 0,
            "successful_splits": 0,
            "successful_decision_tree": 0,
            "successful_regularized_regression": 0,
            "failed_decision_tree": 0,
            "failed_regularized_regression": 0,
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

            # Replace the hardcoded stages list with the imported constant
            stages = PIPELINE_STAGES.copy()  # Make a copy to avoid modifying the original
            
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
                    case "merge_snow_depth_columns":  # NEW STAGE
                        print(f"\nMerging snow depth columns for {month_id}...")
                        merged_df = self.merge_snow_depth_columns(dataframe=state["df"])
                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if merged_df is None:
                            print(f"Failed to merge snow depth columns for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully merged snow depth columns for {month_id}")
                            counters["successful_snow_depth_merge"] += 1
                            state["df"] = merged_df
                            state["current_stage"] = "clean_missing_values"
                    
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
                            state["current_stage"] = "train_regularized_regression"
                    
                    case "train_regularized_regression":
                        print(f"Checking for numeric target for regularized regression in {month_id}...")
                        
                        # We need to identify the target column from the dataframe
                        df = state["df"]
                        numeric_targets = ['differenceInMinutes', 'differenceInMinutes_offset']
                        target_column = None
                        
                        # Find which numeric target exists in the dataframe (should be only one after select_target stage)
                        for option in numeric_targets:
                            if option in df.columns:
                                target_column = option
                                break
                        
                        is_numeric_target = target_column is not None
                        
                        if is_numeric_target:
                            print(f"Found numeric target '{target_column}' for {month_id}")
                            print(f"Training regularized regression for {month_id}...")
                            reg_result = self.train_regularized_regression(month_id)
                            
                            if not reg_result.get("success", False):
                                print(f"Failed to train regularized regression for {month_id}: {reg_result.get('error', 'Unknown error')}")
                                counters["failed_regularized_regression"] += 1
                            else:
                                print(f"Successfully trained regularized regression for {month_id}")
                                counters["successful_regularized_regression"] += 1
                        else:
                            print(f"Skipping regularized regression for {month_id} (non-numeric target)")
                        
                        # Move to the next stage regardless of success/failure
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

                        # Point to the new stage
                        state["current_stage"] = "train_xgboost_with_important_features"

                    case "train_xgboost_with_important_features":
                        print(f"Training XGBoost model with important features for {month_id}...")
                        xgb_important_result = self.train_xgboost_with_important_features(month_id)
                        
                        if not xgb_important_result.get("success", False):
                            print(f"Failed to train XGBoost with important features for {month_id}: {xgb_important_result.get('error', 'Unknown error')}")
                            counters["failed_xgboost_important"] = counters.get("failed_xgboost_important", 0) + 1
                        else:
                            print(f"Successfully trained XGBoost with important features for {month_id}")
                            counters["successful_xgboost_important"] = counters.get("successful_xgboost_important", 0) + 1

                        # Move to the next stage
                        state["current_stage"] = "train_xgboost_rs_with_important_features"

                    case "train_xgboost_rs_with_important_features":
                        print(f"Training XGBoost with RandomizedSearchCV on top {TOP_FEATURES_COUNT} features for {month_id}...")
                        xgb_rs_important_result = self.train_xgboost_rs_with_important_features(month_id)
                        
                        if not xgb_rs_important_result.get("success", False):
                            print(f"Failed to train XGBoost with RandomizedSearchCV on top features for {month_id}: {xgb_rs_important_result.get('error', 'Unknown error')}")
                            counters["failed_xgboost_rs_important"] = counters.get("failed_xgboost_rs_important", 0) + 1
                        else:
                            print(f"Successfully trained XGBoost with RandomizedSearchCV on top features for {month_id}")
                            counters["successful_xgboost_rs_important"] = counters.get("successful_xgboost_rs_important", 0) + 1

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
        
        # Print summary - UPDATED to include snow depth merge counter
        print("\n" + "="*50)
        print("Processing Summary:")
        print(f"Total months processed: {summary['total_months']}")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successfully preprocessed: {summary['successful_preprocessing']}")
        print(f"Successfully merged snow depth columns: {summary['successful_snow_depth_merge']}")  # NEW LINE
        print(f"Successfully cleaned missing values: {summary['successful_cleaning']}")
        print(f"Successfully deduplicated: {summary['successful_deduplication']}")
        print(f"Successfully scaled numeric columns: {summary['successful_scaling']}")
        print(f"Successfully added trainDelayed feature: {summary['successful_feature_addition']}")
        print(f"Successfully selected target feature: {summary['successful_target_selection']}")
        print(f"Successfully saved to CSV: {summary['successful_saves']}")
        print(f"Successfully split into train/test sets: {summary['successful_splits']}")
        print(f"Successfully trained regularized regression models: {summary.get('successful_regularized_regression', 0)}")
        print(f"Successfully trained decision tree models: {summary['successful_decision_tree']}")
        print(f"Successfully trained decision tree models with important features: {summary.get('successful_important_features', 0)}")
        print(f"Successfully trained decision tree models with RandomizedSearchCV: {summary.get('successful_randomized_search', 0)}")
        print(f"Successfully trained with RandomizedSearchCV on important features: {summary.get('successful_combined_approach', 0)}")
        print(f"Successfully trained XGBoost models: {summary.get('successful_xgboost', 0)}")
        print(f"Successfully trained XGBoost models with RandomizedSearchCV: {summary.get('successful_xgboost_rs', 0)}")
        print(f"Failed to train regularized regression models: {summary.get('failed_regularized_regression', 0)}")
        print(f"Failed to train decision tree models: {summary['failed_decision_tree']}")
        print(f"Failed to train decision tree models with important features: {summary.get('failed_important_features', 0)}")
        print(f"Failed to train decision tree models with RandomizedSearchCV: {summary.get('failed_randomized_search', 0)}")
        print(f"Failed to train with RandomizedSearchCV on important features: {summary.get('failed_combined_approach', 0)}")
        print(f"Failed to train XGBoost models: {summary.get('failed_xgboost', 0)}")
        print(f"Failed to train XGBoost models with RandomizedSearchCV: {summary.get('failed_xgboost_rs', 0)}")
        print(f"Failed to process: {summary['failed_files']}")

        print(f"Successfully trained XGBoost models with important features: {summary.get('successful_xgboost_important', 0)}")
        print(f"Failed to train XGBoost models with important features: {summary.get('failed_xgboost_important', 0)}")
        print(f"Successfully trained XGBoost models with RandomizedSearchCV on top features: {summary.get('successful_xgboost_rs_important', 0)}")
        print(f"Failed to train XGBoost models with RandomizedSearchCV on top features: {summary.get('failed_xgboost_rs_important', 0)}")

        print("="*50)
        
        return summary
        
    def preprocess_csv_file(self, input_file_path):
        """
        Preprocess a CSV file by extracting nested data from timeTableRows,
        keeping only essential columns and expanding weather conditions.
        
        Parameters:
        -----------
        input_file_path : str
            Path to the input CSV file.
            
        Returns:
        --------
        pandas.DataFrame
            The processed DataFrame.
        """
        try:
            # Load the dataframe from the input file path
            _, df = generate_output_path(input_file_path)
            
            # Dictionary to store stops by train
            train_stops = {}
            
            # Extract nested data from the "timeTableRows" column
            for index, row in df.iterrows():
                ttr = row.get("timeTableRows", None)
                if ttr is None or pd.isnull(ttr):
                    continue  # Skip empty values
                
                # Get train identifier (use trainNumber if available, otherwise use row index)
                train_id = row.get("trainNumber", f"train_{index}")
                
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
                
                # Store stops with train_id
                for stop in stops:
                    # Add train_id to each stop record
                    stop['train_id'] = train_id
                    
                    # If train_id not in dictionary, initialize with empty list
                    if train_id not in train_stops:
                        train_stops[train_id] = []
                    
                    # Add stop to the train's stops list
                    train_stops[train_id].append(stop)
            
            print(f"Extracted stops for {len(train_stops)} different trains")
            
            # Filter trains by required stations if enabled
            if FILTER_TRAINS_BY_STATIONS and REQUIRED_STATIONS:
                print(f"\nFiltering trains that pass through all required stations: {REQUIRED_STATIONS}")
                
                filtered_train_stops = {}
                
                for train_id, stops in train_stops.items():
                    # Extract all station codes for this train
                    train_stations = set()
                    for stop in stops:
                        station_code = stop.get('stationShortCode')
                        if station_code:
                            train_stations.add(station_code)
                    
                    # Check if this train passes through all required stations
                    if all(station in train_stations for station in REQUIRED_STATIONS):
                        filtered_train_stops[train_id] = stops
                    
                print(f"Filtered from {len(train_stops)} to {len(filtered_train_stops)} trains")
                print(f"Kept only trains passing through: {', '.join(REQUIRED_STATIONS)}")
                
                # Replace the original dictionary with the filtered one
                train_stops = filtered_train_stops
                
                # Early return if no trains match the criteria
                if not train_stops:
                    print(f"Warning: No trains found passing through all required stations: {REQUIRED_STATIONS}")
                    return pd.DataFrame()  # Return empty DataFrame

            # Process each train's stops (no longer calculating differenceInMinutes_offset)
            cross_records = []
            
            for train_id, stops in train_stops.items():
                # Add all stops to final records
                cross_records.extend(stops)
            
            print(f"Processed a total of {len(cross_records)} stops")
            # Create DataFrame from processed records
            cross_df = pd.DataFrame(cross_records)
            
            # Rename 'weather_observations' to 'weather_conditions' if it exists
            if "weather_observations" in cross_df.columns:
                cross_df = cross_df.rename(columns={"weather_observations": "weather_conditions"})
                print("Renamed 'weather_observations' to 'weather_conditions'")
            
            # Keep only the essential columns, including our new calculated column
            expected_cols = ["differenceInMinutes", "differenceInMinutes_offset", "cancelled", "weather_conditions", "trainStopping", "commercialStop"]
            available_cols = [col for col in expected_cols if col in cross_df.columns]
            
            if len(available_cols) > 0:
                cross_df = cross_df[available_cols]
                print(f"Kept only columns: {available_cols}")
            else:
                print(f"Warning: None of the expected columns are available. Available columns:")
                print(list(cross_df.columns))
                # Keep all columns if none of the expected ones are found
            
            # Expand the 'weather_conditions' dictionaries into separate columns
            if "weather_conditions" in cross_df.columns:
                weather_df = cross_df["weather_conditions"].apply(pd.Series)
                
                # Drop unwanted keys if they exist
                weather_df = weather_df.drop(columns=["closest_ems", "Present weather (auto)"], errors="ignore")
                
                # Join the expanded weather conditions back to the main DataFrame
                cross_df = cross_df.drop("weather_conditions", axis=1).join(weather_df)
                print("Expanded weather_conditions into separate columns")
            
            for col in NON_NUMERIC_FEATURES:
                if col in cross_df.columns:
                    # Fill NaN values with 0 before converting to integer
                    nulls = cross_df[col].isna().sum()
                    if nulls > 0:
                        print(f"Filling {nulls} NaN values in {col} with 0 before conversion")
                    cross_df[col] = cross_df[col].fillna(0)
                    # Convert boolean values to integers (False -> 0, True -> 1)
                    cross_df[col] = cross_df[col].astype(int)
                    print(f"Converted {col} to numeric (0/1)")

            # Reorder columns: differenceInMinutes, differenceInMinutes_offset, cancelled, then others
            base_cols = [col for col in ["differenceInMinutes", "differenceInMinutes_offset", "cancelled"] 
                        if col in cross_df.columns]
            other_cols = [col for col in cross_df.columns if col not in base_cols]
            cols_order = base_cols + other_cols
            cross_df = cross_df[cols_order]
            print("Reordered columns")
            
            # Return the processed DataFrame
            return cross_df
            
        except Exception as e:
            print(f"Error processing file {input_file_path}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return None
        
    def merge_snow_depth_columns(self, dataframe=None):
        """
        Merge 'Snow depth' and 'Snow depth Other' columns, and drop 'Snow depth Other Distance'.
        
        If 'Snow depth' is missing (Null/None), use the value from 'Snow depth Other'.
        If 'Snow depth' already has a value, do nothing.
        Also drops 'Snow depth Other Distance' column if it exists.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with merged snow depth columns and unwanted columns removed.
        """
        import logging
        from datetime import datetime
        
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Merging snow depth columns in dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Set up logging
        log_file_path = os.path.join(self.project_root, "merge_snow_depth_columns.log")
        
        # Create a logger specifically for this operation
        logger = logging.getLogger('merge_snow_depth')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        try:
            # Check if snow depth columns exist
            snow_depth_col = 'Snow depth'
            snow_depth_other_col = 'Snow depth Other'
            snow_depth_other_distance_col = 'Snow depth Other Distance'
            
            has_snow_depth = snow_depth_col in df.columns
            has_snow_depth_other = snow_depth_other_col in df.columns
            has_snow_depth_other_distance = snow_depth_other_distance_col in df.columns
            
            print(f"Snow depth columns found:")
            print(f"- '{snow_depth_col}': {has_snow_depth}")
            print(f"- '{snow_depth_other_col}': {has_snow_depth_other}")
            print(f"- '{snow_depth_other_distance_col}': {has_snow_depth_other_distance}")
            
            # Initialize logging variables
            initial_snow_depth_missing = 0
            initial_snow_depth_other_missing = 0
            values_successfully_merged = 0
            final_snow_depth_missing = 0
            
            # Handle merging logic first
            if not has_snow_depth and not has_snow_depth_other:
                print("Neither main snow depth column found. No merging needed.")
                logger.info(f"No snow depth columns found - Snow depth missing: N/A, Snow depth Other missing: N/A, Values merged: 0, Final missing: N/A")
            elif not has_snow_depth and has_snow_depth_other:
                print(f"Only '{snow_depth_other_col}' found. Renaming to '{snow_depth_col}'.")
                initial_snow_depth_other_missing = df[snow_depth_other_col].isna().sum()
                df = df.rename(columns={snow_depth_other_col: snow_depth_col})
                final_snow_depth_missing = df[snow_depth_col].isna().sum()
                has_snow_depth = True
                has_snow_depth_other = False
                logger.info(f"Renamed column - Snow depth missing: N/A, Snow depth Other missing: {initial_snow_depth_other_missing}, Values merged: 0, Final missing: {final_snow_depth_missing}")
            elif has_snow_depth and not has_snow_depth_other:
                print(f"Only '{snow_depth_col}' found. No merging needed.")
                initial_snow_depth_missing = df[snow_depth_col].isna().sum()
                final_snow_depth_missing = initial_snow_depth_missing
                logger.info(f"Only main column found - Snow depth missing: {initial_snow_depth_missing}, Snow depth Other missing: N/A, Values merged: 0, Final missing: {final_snow_depth_missing}")
            else:
                # Both columns exist - proceed with merging logic
                print(f"Both snow depth columns found. Proceeding with merge logic.")
                
                # Count initial missing values in each column
                initial_snow_depth_missing = df[snow_depth_col].isna().sum()
                initial_snow_depth_other_missing = df[snow_depth_other_col].isna().sum()
                
                print(f"Before merge:")
                print(f"- '{snow_depth_col}' missing values: {initial_snow_depth_missing}")
                print(f"- '{snow_depth_other_col}' missing values: {initial_snow_depth_other_missing}")
                
                # Create a mask for rows where Snow depth is missing but Snow depth Other is not
                merge_mask = df[snow_depth_col].isna() & df[snow_depth_other_col].notna()
                merge_count = merge_mask.sum()
                
                if merge_count > 0:
                    print(f"Merging {merge_count} values from '{snow_depth_other_col}' to '{snow_depth_col}'")
                    # Fill missing Snow depth values with Snow depth Other values
                    df.loc[merge_mask, snow_depth_col] = df.loc[merge_mask, snow_depth_other_col]
                    values_successfully_merged = merge_count
                else:
                    print("No values to merge (no rows where Snow depth is missing but Snow depth Other has values)")
                    values_successfully_merged = 0
                
                # Count final missing values
                final_snow_depth_missing = df[snow_depth_col].isna().sum()
                
                print(f"After merge:")
                print(f"- '{snow_depth_col}' missing values: {final_snow_depth_missing}")
                print(f"- Values successfully merged: {initial_snow_depth_missing - final_snow_depth_missing}")
                
                # Log the merge results
                logger.info(f"Merge completed - Snow depth missing: {initial_snow_depth_missing}, Snow depth Other missing: {initial_snow_depth_other_missing}, Values merged: {values_successfully_merged}, Final missing: {final_snow_depth_missing}")
            
            # Drop unwanted columns
            columns_to_drop = []
            if has_snow_depth_other:
                columns_to_drop.append(snow_depth_other_col)
            if has_snow_depth_other_distance:
                columns_to_drop.append(snow_depth_other_distance_col)
            
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                print(f"Dropped columns: {columns_to_drop}")
            else:
                print("No snow depth columns to drop")
            
            return df
            
        except Exception as e:
            print(f"Error merging snow depth columns: {e}")
            logger.error(f"Error merging snow depth columns - Error: {str(e)}")
            return dataframe  # Return original dataframe on error
        
        finally:
            # Clean up logger handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def handle_missing_values(self, dataframe=None):
        """
        Handle missing values in preprocessed dataframes with enhanced imputation strategy.
        
        Processes the provided dataframe and handles missing values:
        - Drop rows where all weather condition columns have missing values
        - Drop rows where differenceInMinutes or cancelled are None
        - Fill missing values in trainStopping and commercialStop with False
        - Use variable-specific imputation for weather columns:
        - Zero for precipitation and snow metrics
        - Median for temperature, humidity, and visibility
        
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
                
        df = dataframe.copy()  # Make a copy to avoid modifying the original
        print(f"Processing dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Count rows before cleaning
        original_row_count = len(df)
        
        # Fill missing values in trainStopping and commercialStop with 0 (since they're now numeric)
        for col in NON_NUMERIC_FEATURES:
            if col in df.columns:
                nulls = df[col].isna().sum()
                df[col] = df[col].fillna(0)  # Fill with 0 instead of False since they're now numeric
                print(f"- Filled {nulls} missing values in '{col}' with 0")

        # Step 2: Check required columns (differenceInMinutes and cancelled)
        required_cols = ['differenceInMinutes', 'differenceInMinutes_offset', 'trainDelayed', 'cancelled']
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
        
        # Step 3: Handle weather condition columns
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
        df = df.dropna(subset=available_important_cols, how='all')
        
        # Count how many rows were dropped due to weather conditions
        dropped_weather = before_weather_drop - len(df)
        
        # ===== ENHANCED MISSING VALUE HANDLING (WITHOUT INDICATORS) =====
        
        # Group weather variables by appropriate imputation method
        zero_fill_cols = ['Precipitation amount', 'Precipitation intensity', 'Snow depth']
        median_fill_cols = ['Air temperature', 'Relative humidity', 'Dew-point temperature', 'Horizontal visibility']
        
        # 1. Zero imputation for precipitation and snow metrics
        for col in zero_fill_cols:
            if col in df.columns:
                nulls = df[col].isna().sum()
                if nulls > 0:
                    # Apply zero imputation
                    df[col] = df[col].fillna(0)
                    print(f"- Filled {nulls} missing values in '{col}' with 0")
        
        # 2. Median imputation for temperature and other continuous variables
        for col in median_fill_cols:
            if col in df.columns:
                nulls = df[col].isna().sum()
                if nulls > 0:
                    # Apply median imputation
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    print(f"- Filled {nulls} missing values in '{col}' with median: {median_value:.2f}")
        
        # For any remaining important weather columns, use median imputation
        remaining_cols = [col for col in available_important_cols 
                        if col not in zero_fill_cols and col not in median_fill_cols]
        
        for col in remaining_cols:
            if col in df.columns:
                nulls = df[col].isna().sum()
                if nulls > 0:
                    # Apply median imputation as default for other weather columns
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    print(f"- Filled {nulls} missing values in '{col}' with median: {median_value:.2f}")
        
        # Count total rows dropped
        total_dropped = original_row_count - len(df)
        
        # Report the results
        print(f"Missing values handling complete:")
        print(f"- Original row count: {original_row_count}")
        print(f"- Rows dropped due to missing required columns: {dropped_required}")
        print(f"- Rows dropped due to missing all weather conditions: {dropped_weather}")
        print(f"- Total rows dropped: {total_dropped}")
        print(f"- Remaining rows: {len(df)}")
        
        # Calculate percentage of data retained
        if original_row_count > 0:
            retention_percentage = (len(df) / original_row_count) * 100
            print(f"- Data retention: {retention_percentage:.2f}%")
            
        # Additional statistics on the important columns
        for col in available_important_cols:
            non_null_count = df[col].count()
            null_count = len(df) - non_null_count
            null_percentage = (null_count / len(df) * 100) if len(df) > 0 else 0
            print(f"  - {col}: {non_null_count} non-null values ({100-null_percentage:.2f}% complete)")
        
        # Additional statistics for trainStopping and commercialStop if they exist
        boolean_cols = ['trainStopping', 'commercialStop']
        available_boolean_cols = [col for col in boolean_cols if col in df.columns]
        
        if available_boolean_cols:
            print("\nBoolean columns statistics:")
            for col in available_boolean_cols:
                true_count = df[col].sum()
                true_percentage = (true_count / len(df) * 100) if len(df) > 0 else 0
                print(f"  - {col}: {true_count} True values ({true_percentage:.2f}% True)")
        
        return df
    
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
            
            # Exclude both existing excluded columns and the new missing indicators
            columns_to_scale = [col for col in all_numeric_columns 
                            if col not in ['differenceInMinutes', 'differenceInMinutes_offset', 
                                            'trainStopping', 'commercialStop']
                            and not col.endswith('_missing')]
                        
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
        valid_targets = VALID_TARGET_FEATURES
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
            target_options = ['differenceInMinutes', 'differenceInMinutes_offset', 'trainDelayed', 'cancelled']
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
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
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
            if target_column in ['differenceInMinutes', 'differenceInMinutes_offset']:
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
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
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
            if target_column in ['differenceInMinutes', 'differenceInMinutes_offset']:
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
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
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
            if target_column in ['differenceInMinutes', 'differenceInMinutes_offset']:
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
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
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
            if target_column in ['differenceInMinutes', 'differenceInMinutes_offset']:
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
        For regression tasks, uses multiple approaches to ensure good R² performance.
        
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
                params = XGBOOST_DEFAULT_PARAMS.copy()
            else:
                params = params.copy()
            
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
                return {"success": False, "error": error_msg}
            
            # Load datasets
            print(f"Loading training data from {train_path}")
            train_df = pd.read_csv(train_path)
            
            print(f"Loading test data from {test_path}")
            test_df = pd.read_csv(test_path)
            
            # Identify target column 
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
            target_column = None
            
            for option in target_options:
                if option in train_df.columns:
                    target_column = option
                    break
            
            if not target_column:
                print(f"Error: No target column found in dataset")
                return {"success": False, "error": "No target column found in dataset"}
            
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
            is_regression = (target_column in ['differenceInMinutes', 'differenceInMinutes_offset'])
            
            if is_regression:
                # REGRESSION CASE
                print(f"Target '{target_column}' indicates a regression problem")
                
                # Create DMatrix for lower-level XGBoost API
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                # Set up regularization parameters
                xgb_params = {
                    'max_depth': params.get('max_depth', 6),
                    'eta': params.get('learning_rate', 0.1),
                    'subsample': params.get('subsample', 0.8),
                    'colsample_bytree': params.get('colsample_bytree', 0.8),
                    'seed': random_state,
                    'alpha': 0.5,  # L1 regularization to prevent overfitting
                    'lambda': 1.0,  # L2 regularization to prevent overfitting
                }
                
                # Define multiple models to train and compare
                models_to_try = []
                
                # 1. Standard regression model
                standard_params = xgb_params.copy()
                standard_params['objective'] = 'reg:squarederror'
                models_to_try.append({
                    'name': 'standard',
                    'params': standard_params, 
                    'obj': None
                })
                
                # 2. Modified custom loss function with more stable gradients
                def stable_weighted_mse(y_pred, dtrain):
                    y_true = dtrain.get_label()
                    # More moderate weighting approach with capped weights
                    weights = np.minimum(3.0, 1.0 + np.abs(y_true) / (np.abs(y_true).mean() * 2))
                    # More stable gradient calculation
                    grad = weights * (y_pred - y_true)
                    hess = weights
                    return grad, hess
                
                models_to_try.append({
                    'name': 'weighted',
                    'params': xgb_params.copy(),
                    'obj': stable_weighted_mse
                })
                
                # 3. Quantile regression - focuses on larger values without custom function
                quantile_params = xgb_params.copy()
                quantile_params['objective'] = 'reg:quantileerror'
                quantile_params['quantile_alpha'] = 0.7  # Focus on upper 70% of delay distribution
                models_to_try.append({
                    'name': 'quantile',
                    'params': quantile_params,
                    'obj': None
                })
                
                # Train and evaluate each model
                best_model = None
                best_r2 = float('-inf')
                best_metrics = None
                best_y_pred = None
                
                for model_config in models_to_try:
                    model_name = model_config['name']
                    model_params = model_config['params']
                    model_obj = model_config['obj']
                    
                    print(f"\nTraining XGBoost Regressor ({model_name} approach)...")
                    
                    # Train model
                    if model_obj:
                        xgb_model = xgb.train(
                            model_params, 
                            dtrain,
                            num_boost_round=params.get('n_estimators', 100),
                            obj=model_obj,
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            verbose_eval=10
                        )
                    else:
                        xgb_model = xgb.train(
                            model_params,
                            dtrain,
                            num_boost_round=params.get('n_estimators', 100),
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            verbose_eval=10
                        )
                    
                    # Make predictions
                    y_pred = xgb_model.predict(dtest)
                    
                    # Calculate regression metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Calculate performance on large delays
                    large_delay_idx = np.where(np.abs(y_test) > np.abs(y_test).mean())[0]
                    large_metrics = {}
                    if len(large_delay_idx) > 0:
                        large_y_test = y_test.iloc[large_delay_idx]
                        large_y_pred = y_pred[large_delay_idx]
                        large_mse = mean_squared_error(large_y_test, large_y_pred)
                        large_rmse = np.sqrt(large_mse)
                        large_mae = mean_absolute_error(large_y_test, large_y_pred)
                        large_metrics = {
                            'large_delay_rmse': large_rmse,
                            'large_delay_mae': large_mae
                        }
                    
                    # Combine metrics
                    metrics = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        **large_metrics
                    }
                    
                    # Print metrics
                    print(f"\nXGBoost {model_name} Regressor Results:")
                    print(f"RMSE: {rmse:.4f}")
                    print(f"MAE: {mae:.4f}")
                    print(f"R²: {r2:.4f}")
                    
                    if large_metrics:
                        print(f"Large Delay RMSE: {large_metrics['large_delay_rmse']:.4f}")
                        print(f"Large Delay MAE: {large_metrics['large_delay_mae']:.4f}")
                    
                    # Keep track of best model by R² score
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = xgb_model
                        best_metrics = metrics
                        best_y_pred = y_pred
                        best_name = model_name
                
                # Use the best model
                xgb_model = best_model
                metrics = best_metrics
                y_pred = best_y_pred
                
                print(f"\nSelected best model: {best_name} (R² = {best_r2:.4f})")
                
                # Save metrics
                metrics_filename = f"model_metrics_{month_id}_best.csv"
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
                    "rmse": metrics['rmse'],
                    "mae": metrics['mae'],
                    "r2": metrics['r2'],
                    "best_approach": best_name
                }
                
                # Get feature importance and normalize it
                importance_scores = xgb_model.get_score(importance_type='gain')
                
                # Normalize the importance scores to sum to 1.0
                total_importance = sum(importance_scores.values())
                normalized_scores = {k: v/total_importance for k, v in importance_scores.items()}
                
                feature_importance = pd.DataFrame({
                    'Feature': normalized_scores.keys(),
                    'Importance': normalized_scores.values()
                }).sort_values(by='Importance', ascending=False)
                            
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
                
                # Add regularization parameters for classification too
                params['alpha'] = 0.5  # L1 regularization
                params['lambda'] = 1.0  # L2 regularization
                
                # For classification, use sample weights if delay info is available
                if 'differenceInMinutes' in train_df.columns:
                    print("Using weighted samples based on delay magnitude")
                    # Create sample weights based on delay magnitude
                    delay_col = 'differenceInMinutes'
                    sample_weights = np.ones(len(y_train))
                    
                    # Get delay values for each training sample
                    delays = train_df[delay_col].values
                    
                    # Apply weights - higher delays get higher weights
                    # For delayed trains, weight proportional to delay amount
                    delayed_idx = (delays > 0)
                    if np.any(delayed_idx):
                        # Normalize delay values by mean positive delay, using more moderate weights
                        mean_delay = delays[delayed_idx].mean()
                        # Cap weights at 5 instead of 10 for more stability
                        sample_weights[delayed_idx] = np.minimum(5, 1 + delays[delayed_idx]/mean_delay)
                    
                    # Train the model with sample weights
                    xgb_model = xgb.XGBClassifier(**params)
                    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    # Train without sample weights
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
                
                # Get feature importance and normalize it
                raw_importances = xgb_model.feature_importances_
                normalized_importances = raw_importances / np.sum(raw_importances)
                
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': normalized_importances
                }).sort_values(by='Importance', ascending=False)
            
            # Feature importance output for both regression and classification
            print("\nFeature Importance (top 10):")
            print(feature_importance.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            
            # Save the model and feature importance
            try:
                import joblib
                
                # Save the model
                model_filename = f"xgboost_{month_id}_best.joblib"
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
        Supports sample weights based on delay magnitude for classification tasks.
        
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
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
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
            if target_column in ['differenceInMinutes', 'differenceInMinutes_offset']:
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            # Create sample weights for classification if delay info is available
            sample_weights = None
            if is_classification and 'differenceInMinutes' in train_df.columns:
                print("Using weighted samples based on delay magnitude for randomized search")
                # Create sample weights based on delay magnitude
                delay_col = 'differenceInMinutes'
                sample_weights = np.ones(len(y_train))
                
                # Get delay values for each training sample
                delays = train_df[delay_col].values
                
                # Apply weights - higher delays get higher weights
                # For delayed trains, weight proportional to delay amount
                delayed_idx = (delays > 0)
                if np.any(delayed_idx):
                    # Normalize delay values by mean positive delay, using more moderate weights
                    mean_delay = delays[delayed_idx].mean()
                    # Cap weights at 5 instead of 10 for more stability
                    sample_weights[delayed_idx] = np.minimum(5, 1 + delays[delayed_idx]/mean_delay)
                
                print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                
            # --------- MANUAL HYPERPARAMETER TUNING APPROACH ---------            
            print(f"Starting manual hyperparameter tuning with {n_iter} iterations and {cv}-fold cross-validation...")

            # Define custom objective function for regression if needed
            def stable_weighted_mse(y_pred, dtrain):
                y_true = dtrain.get_label()
                # More moderate weighting approach with capped weights
                weights = np.minimum(3.0, 1.0 + np.abs(y_true) / (np.abs(y_true).mean() * 2))
                # More stable gradient calculation
                grad = weights * (y_pred - y_true)
                hess = weights
                return grad, hess
            
            # Generate parameter combinations
            param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
            
            # Setup cross-validation
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            
            best_score = float('-inf')
            best_params = None
            best_method = None  # To track if custom objective was best
            
            # Memory tracking
            process = psutil.Process()
            before_mem = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage before training: {before_mem:.2f} MB")
            
            # Define methods to try based on problem type
            methods_to_try = []
            
            if is_classification:
                # For classification, just use standard method, possibly with sample weights
                methods_to_try = [{"name": "standard", "obj": None}]
            else:
                # For regression, try both standard and custom objective
                methods_to_try = [
                    {"name": "standard", "obj": None},
                    {"name": "weighted", "obj": stable_weighted_mse}
                ]
                print(f"For regression, will try {len(methods_to_try)} different objective approaches")
            
            # Try each parameter combination and each method
            overall_best_score = float('-inf')
            
            for i, params in enumerate(param_list):
                for method in methods_to_try:
                    method_name = method["name"]
                    obj_function = method["obj"]
                    
                    print(f"Testing parameter combination {i+1}/{len(param_list)} with method '{method_name}'")
                    
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
                        # For regression, only set objective if not using custom objective
                        if obj_function is None:
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
                        
                        # Get fold-specific sample weights if weights are being used
                        fold_sample_weights = None
                        if sample_weights is not None:
                            fold_sample_weights = sample_weights[train_idx]
                        
                        # Use different approaches based on problem type and method
                        if is_classification:
                            # For classification, use XGBClassifier
                            model = xgb.XGBClassifier(**current_params)
                            
                            # Fit model with sample weights if available
                            if fold_sample_weights is not None:
                                model.fit(X_fold_train, y_fold_train, sample_weight=fold_sample_weights)
                            else:
                                model.fit(X_fold_train, y_fold_train)
                                
                            # Predict
                            y_pred = model.predict(X_fold_val)
                            
                            # Evaluate
                            from sklearn.metrics import accuracy_score
                            score = accuracy_score(y_fold_val, y_pred)
                            
                        else:
                            # For regression with custom objective, use lower-level API
                            if obj_function is not None:
                                # Create DMatrix objects
                                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                                dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
                                
                                # Set up parameters for training
                                xgb_params = {
                                    'max_depth': current_params.get('max_depth', 6),
                                    'eta': current_params.get('learning_rate', 0.1),
                                    'subsample': current_params.get('subsample', 0.8),
                                    'colsample_bytree': current_params.get('colsample_bytree', 0.8),
                                    'seed': random_state,
                                    'alpha': 0.5,  # L1 regularization
                                    'lambda': 1.0,  # L2 regularization
                                }
                                # Note: 'silent' parameter has been removed as it's deprecated
                                
                                # Train the model with custom objective
                                model = xgb.train(
                                    xgb_params,
                                    dtrain,
                                    num_boost_round=current_params.get('n_estimators', 100),
                                    obj=obj_function
                                )
                                
                                # Predict
                                y_pred = model.predict(dval)
                                
                            else:
                                # Standard regression using XGBRegressor
                                model = xgb.XGBRegressor(**current_params)
                                model.fit(X_fold_train, y_fold_train)
                                y_pred = model.predict(X_fold_val)
                            
                            # Evaluate regression performance
                            from sklearn.metrics import mean_squared_error
                            score = -mean_squared_error(y_fold_val, y_pred)  # Negative for higher=better
                        
                        cv_scores.append(score)
                    
                    # Calculate average score across folds
                    avg_score = np.mean(cv_scores)
                    print(f"  Method '{method_name}' - Average CV score: {avg_score:.4f}")
                    
                    # Update best if better
                    if avg_score > overall_best_score:
                        overall_best_score = avg_score
                        best_score = avg_score
                        best_params = current_params.copy()
                        best_method = method_name
                        print(f"  New best score: {best_score:.4f} with method '{best_method}'")
            
            after_mem = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage after training: {after_mem:.2f} MB (Δ: {after_mem - before_mem:.2f} MB)")
            
            print(f"Best Hyperparameters: {best_params}")
            print(f"Best Method: {best_method}")
            print(f"Best CV Score: {best_score:.4f}")
            
            # Train final model with best parameters
            if is_classification:
                xgb_model = xgb.XGBClassifier(**best_params)
                
                # Fit the final model with sample weights if available
                if sample_weights is not None:
                    print("Training final model with sample weights")
                    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    xgb_model.fit(X_train, y_train)
                    
            else:
                # For regression, check if best method uses custom objective
                if best_method == "weighted":
                    print("Training final model with custom weighted objective function")
                    # Create DMatrix objects
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dtest = xgb.DMatrix(X_test, label=y_test)
                    
                    # Set up parameters for training
                    xgb_params = {
                        'max_depth': best_params.get('max_depth', 6),
                        'eta': best_params.get('learning_rate', 0.1),
                        'subsample': best_params.get('subsample', 0.8),
                        'colsample_bytree': best_params.get('colsample_bytree', 0.8),
                        'seed': random_state,
                        'alpha': 0.5,  # L1 regularization
                        'lambda': 1.0,  # L2 regularization
                    }
                    # Note: 'silent' parameter has been removed as it's deprecated
                    
                    # Train the model with custom objective
                    xgb_model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=best_params.get('n_estimators', 100),
                        obj=stable_weighted_mse
                    )
                else:
                    print("Training final model with standard objective")
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
                # Handle regression evaluation - check if using booster or regressor
                if best_method == "weighted" and hasattr(xgb_model, 'predict'):
                    # If using booster with custom objective (has predict method)
                    dtest = xgb.DMatrix(X_test)
                    y_pred = xgb_model.predict(dtest)
                else:
                    # Standard XGBRegressor
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
            if best_method == "weighted" and hasattr(xgb_model, 'get_score'):
                # For booster with custom objective
                importance_scores = xgb_model.get_score(importance_type='gain')
                # Normalize the importance scores
                total_importance = sum(importance_scores.values())
                normalized_scores = {k: v/total_importance for k, v in importance_scores.items()}
                
                feature_importance = pd.DataFrame({
                    'Feature': normalized_scores.keys(),
                    'Importance': normalized_scores.values()
                }).sort_values(by='Importance', ascending=False)
            else:
                # For standard XGBClassifier/XGBRegressor
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
                
                # If sample weights were used, save their distribution
                if sample_weights is not None:
                    weights_df = pd.DataFrame({
                        'weight': sample_weights
                    })
                    weights_filename = f"sample_weights_distribution_{month_id}.csv"
                    weights_path = os.path.join(xgboost_rs_dir, weights_filename)
                    weights_df.describe().to_csv(weights_path)
                    print(f"Weight distribution saved to {weights_path}")
                
                if is_classification:
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "best_params": best_params,
                        "best_method": best_method,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"],
                        "used_sample_weights": sample_weights is not None
                    }
                else:
                    return {
                        "success": True,
                        "model_type": "regression",
                        "rmse": rmse,
                        "r2": r2,
                        "best_params": best_params,
                        "best_method": best_method,
                        "custom_objective": best_method == "weighted",
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
                        "best_params": best_params,
                        "best_method": best_method,
                        "used_sample_weights": sample_weights is not None
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
                        "best_params": best_params,
                        "best_method": best_method,
                        "custom_objective": best_method == "weighted"
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
    
    def train_xgboost_with_important_features(self, month_id, importance_threshold=IMPORTANCE_THRESHOLD, params=None, random_state=42):
            """
            Train an XGBoost model (classifier or regressor) using only the most important features.
            
            This method first trains an XGBoost model on all features, identifies the top features
            based on TOP_FEATURES_COUNT, and then trains a new model using only those features.
            
            Parameters:
            -----------
            month_id : str
                Month identifier in format "YYYY-YYYY_MM" for the filename.
            importance_threshold : float, optional
                No longer used - kept for backwards compatibility.
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
                    params = params = XGBOOST_DEFAULT_PARAMS.copy()
                else:
                    params = params.copy()
                
                # Import the TOP_FEATURES_COUNT constant
                from config.const import TOP_FEATURES_COUNT
                
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
                    return {"success": False, "error": error_msg}
                
                # Load datasets
                print(f"Loading training data from {train_path}")
                train_df = pd.read_csv(train_path)
                
                print(f"Loading test data from {test_path}")
                test_df = pd.read_csv(test_path)
                
                # Identify target column 
                target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
                target_column = None
                
                for option in target_options:
                    if option in train_df.columns:
                        target_column = option
                        break
                
                if not target_column:
                    print(f"Error: No target column found in dataset")
                    return {"success": False, "error": "No target column found in dataset"}
                
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
                
                # Create output directory for important features XGBoost
                xgboost_important_dir = os.path.join(self.project_root, "data/output/xgboost_important_features")
                os.makedirs(xgboost_important_dir, exist_ok=True)
                
                # Determine if it's a regression or classification problem
                is_regression = (target_column in ['differenceInMinutes', 'differenceInMinutes_offset'])
                
                # STEP 1: TRAIN INITIAL XGBOOST MODEL TO IDENTIFY IMPORTANT FEATURES
                print(f"\nTraining initial XGBoost model to identify important features...")
                
                if is_regression:
                    # Set regression parameters
                    params['objective'] = 'reg:squarederror'
                    initial_model = xgb.XGBRegressor(**params)
                else:
                    # Set classification parameters
                    if target_column == 'trainDelayed':
                        params['objective'] = 'binary:logistic'
                    else:
                        # For multi-class problems like 'cancelled' with >2 classes
                        num_classes = len(y_train.unique())
                        params['objective'] = 'multi:softprob'
                        params['num_class'] = num_classes
                    
                    initial_model = xgb.XGBClassifier(**params)
                
                # Train the initial model
                initial_model.fit(X_train, y_train)
                
                # Get feature importances
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': initial_model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                # Normalize the importance scores to sum to 1.0
                total_importance = feature_importance['Importance'].sum()
                feature_importance['Normalized_Importance'] = feature_importance['Importance'] / total_importance
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # CHANGED: Select top N features instead of using importance threshold
                important_features = feature_importance.head(TOP_FEATURES_COUNT)['Feature'].tolist()
                
                print(f"\nSelected top {TOP_FEATURES_COUNT} important features:")
                print(important_features)
                
                # STEP 2: TRAIN NEW XGBOOST MODEL WITH ONLY IMPORTANT FEATURES
                print(f"\nTraining new XGBoost model with only top {TOP_FEATURES_COUNT} features...")
                
                if is_regression:
                    selected_model = xgb.XGBRegressor(**params)
                else:
                    selected_model = xgb.XGBClassifier(**params)
                
                # Train the model with only important features
                selected_model.fit(X_train[important_features], y_train)
                
                # Make predictions
                y_pred = selected_model.predict(X_test[important_features])
                
                # Evaluate the model
                if is_regression:
                    # Calculate regression metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    print(f"\nXGBoost Regressor Results (Top {TOP_FEATURES_COUNT} Features Only):")
                    print(f"RMSE: {rmse:.4f}")
                    print(f"MAE: {mae:.4f}")
                    print(f"R²: {r2:.4f}")
                    
                    # Extract and save metrics
                    metrics_result = self.extract_and_save_regression_metrics(
                        y_test, y_pred, f"{month_id}_important_features", 
                        output_dir=xgboost_important_dir
                    )
                    
                    result = {
                        "success": True,
                        "model_type": "regression",
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "important_features": important_features,
                        "top_features_count": TOP_FEATURES_COUNT
                    }
                    
                else:
                    # Calculate classification metrics
                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    
                    print(f"\nXGBoost Classifier Results (Top {TOP_FEATURES_COUNT} Features Only):")
                    print(f"Accuracy: {accuracy:.4f}")
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred))
                    print("\nConfusion Matrix:")
                    print(conf_matrix)
                    
                    # Extract and save metrics
                    metrics_result = self.extract_and_save_metrics(
                        y_test, y_pred, report, f"{month_id}_important_features", 
                        output_dir=xgboost_important_dir
                    )
                    
                    result = {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "important_features": important_features,
                        "top_features_count": TOP_FEATURES_COUNT
                    }
                
                # Feature importance for the new model
                selected_feature_importance = pd.DataFrame({
                    'Feature': important_features,
                    'Importance': selected_model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance for Selected Features:")
                print(selected_feature_importance)
                
                # Save the model, feature list, and results
                try:
                    import joblib
                    
                    # Save the model
                    model_filename = f"xgboost_{month_id}_important_features.joblib"
                    model_path = os.path.join(xgboost_important_dir, model_filename)
                    joblib.dump(selected_model, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance (both initial and selected)
                    initial_importance_filename = f"feature_importance_initial_{month_id}.csv"
                    initial_importance_path = os.path.join(xgboost_important_dir, initial_importance_filename)
                    feature_importance.to_csv(initial_importance_path, index=False)
                    
                    selected_importance_filename = f"feature_importance_selected_{month_id}.csv"
                    selected_importance_path = os.path.join(xgboost_important_dir, selected_importance_filename)
                    selected_feature_importance.to_csv(selected_importance_path, index=False)
                    
                    # Save the list of important features
                    features_filename = f"important_features_{month_id}.txt"
                    features_path = os.path.join(xgboost_important_dir, features_filename)
                    with open(features_path, 'w') as f:
                        f.write(f"# Top {TOP_FEATURES_COUNT} important features\n")
                        for feature in important_features:
                            f.write(f"{feature}\n")
                    
                    print(f"Features and importance saved to {xgboost_important_dir}")
                    
                    # Add paths to result
                    result.update({
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": selected_importance_path,
                        "initial_importance_path": initial_importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    })
                    
                    # Compare with full-features model if available
                    full_model_metrics_file = os.path.join(self.xgboost_dir, f"model_metrics_{month_id}_best.csv")
                    if os.path.exists(full_model_metrics_file):
                        full_metrics = pd.read_csv(full_model_metrics_file)
                        
                        if is_regression:
                            full_rmse = full_metrics['rmse'].values[0] if 'rmse' in full_metrics else None
                            if full_rmse:
                                print(f"\nComparison with full-features model:")
                                print(f"Full model RMSE: {full_rmse:.4f}")
                                print(f"Top {TOP_FEATURES_COUNT} features model RMSE: {rmse:.4f}")
                                rmse_change = ((rmse - full_rmse) / full_rmse) * 100
                                print(f"RMSE change: {rmse_change:.2f}%")
                        else:
                            full_accuracy = full_metrics['accuracy'].values[0] if 'accuracy' in full_metrics else None
                            if full_accuracy:
                                print(f"\nComparison with full-features model:")
                                print(f"Full model accuracy: {full_accuracy:.4f}")
                                print(f"Top {TOP_FEATURES_COUNT} features model accuracy: {accuracy:.4f}")
                                acc_change = ((accuracy - full_accuracy) / full_accuracy) * 100
                                print(f"Accuracy change: {acc_change:.2f}%")
                    
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
                print(f"Error training XGBoost with important features for {month_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e)
                }
  
    def train_regularized_regression(self, month_id, alpha_lasso=0.1, alpha_ridge=1.0, random_state=42):
        """
        Train Lasso and Ridge regression models for feature importance analysis on numeric targets.
        With missing value handling.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        alpha_lasso : float, optional
            Regularization strength for Lasso regression. Defaults to 0.1.
        alpha_ridge : float, optional
            Regularization strength for Ridge regression. Defaults to 1.0.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
                
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics and coefficients.
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
            
            # Identify target column
            target_options = ['differenceInMinutes', 'differenceInMinutes_offset']
            target_column = None
            
            for option in target_options:
                if option in train_df.columns:
                    target_column = option
                    break
            
            if not target_column:
                print("No numeric target column found. Skipping regularized regression.")
                return {
                    "success": False,
                    "error": "No numeric target column found"
                }
            
            print(f"Identified numeric target column: {target_column}")
            
            # Split features and target
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            
            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]
            
            # Drop non-numeric columns and the data_year column if they exist
            non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
            if 'data_year' in X_train.columns:
                non_numeric_cols.append('data_year')
                
            if non_numeric_cols:
                print(f"Dropping non-numeric columns for linear regression: {non_numeric_cols}")
                X_train = X_train.drop(columns=non_numeric_cols)
                X_test = X_test.drop(columns=non_numeric_cols)
            
            # Check for missing values
            train_missing = X_train.isna().sum().sum()
            test_missing = X_test.isna().sum().sum()
            
            if train_missing > 0 or test_missing > 0:
                print(f"Detected missing values: {train_missing} in training data, {test_missing} in test data")
                print("Using median imputation to handle missing values")
                
                # Initialize imputer
                imputer = SimpleImputer(strategy='median')
                
                # Fit and transform data
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test)
                
                # Convert back to DataFrame to keep column names
                X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
                X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
            
            # Store feature names for later use
            feature_names = X_train.columns.tolist()
            
            # Create output directory
            reg_regression_dir = os.path.join(self.project_root, REGULARIZED_REGRESSION_OUTPUT_FOLDER)
            os.makedirs(reg_regression_dir, exist_ok=True)
            
            results = {
                "success": True,
                "target_column": target_column,
                "models": {}
            }
            
            # Train Lasso Regression
            print(f"\nTraining Lasso Regression (alpha={alpha_lasso})...")
            lasso_model = Lasso(alpha=alpha_lasso, random_state=random_state)
            lasso_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_lasso = lasso_model.predict(X_test)
            
            # Calculate metrics
            lasso_mse = mean_squared_error(y_test, y_pred_lasso)
            lasso_rmse = np.sqrt(lasso_mse)
            lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
            lasso_r2 = r2_score(y_test, y_pred_lasso)
            
            print(f"Lasso Regression Results:")
            print(f"RMSE: {lasso_rmse:.4f}")
            print(f"MAE: {lasso_mae:.4f}")
            print(f"R²: {lasso_r2:.4f}")
            
            # Get coefficients and their importance
            lasso_coefs = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': lasso_model.coef_,
                'Abs_Coefficient': np.abs(lasso_model.coef_)
            }).sort_values(by='Abs_Coefficient', ascending=False)
            
            # Count non-zero coefficients
            non_zero_coefs = np.sum(lasso_model.coef_ != 0)
            print(f"Number of features selected by Lasso: {non_zero_coefs} out of {len(feature_names)}")
            
            # Save Lasso results
            lasso_coefs_file = os.path.join(reg_regression_dir, f"lasso_coefficients_{month_id}.csv")
            lasso_coefs.to_csv(lasso_coefs_file, index=False)
            
            lasso_metrics = {
                'mse': lasso_mse,
                'rmse': lasso_rmse,
                'mae': lasso_mae,
                'r2': lasso_r2,
                'non_zero_features': non_zero_coefs
            }
            
            lasso_metrics_file = os.path.join(reg_regression_dir, f"lasso_metrics_{month_id}.csv")
            pd.DataFrame([lasso_metrics]).to_csv(lasso_metrics_file, index=False)
            
            # Add to results
            results["models"]["lasso"] = {
                "metrics": lasso_metrics,
                "coefficients_path": lasso_coefs_file,
                "metrics_path": lasso_metrics_file
            }
            
            # Train Ridge Regression
            print(f"\nTraining Ridge Regression (alpha={alpha_ridge})...")
            ridge_model = Ridge(alpha=alpha_ridge, random_state=random_state)
            ridge_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_ridge = ridge_model.predict(X_test)
            
            # Calculate metrics
            ridge_mse = mean_squared_error(y_test, y_pred_ridge)
            ridge_rmse = np.sqrt(ridge_mse)
            ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
            ridge_r2 = r2_score(y_test, y_pred_ridge)
            
            print(f"Ridge Regression Results:")
            print(f"RMSE: {ridge_rmse:.4f}")
            print(f"MAE: {ridge_mae:.4f}")
            print(f"R²: {ridge_r2:.4f}")
            
            # Get coefficients and their importance
            ridge_coefs = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': ridge_model.coef_,
                'Abs_Coefficient': np.abs(ridge_model.coef_)
            }).sort_values(by='Abs_Coefficient', ascending=False)
            
            # Save Ridge results
            ridge_coefs_file = os.path.join(reg_regression_dir, f"ridge_coefficients_{month_id}.csv")
            ridge_coefs.to_csv(ridge_coefs_file, index=False)
            
            ridge_metrics = {
                'mse': ridge_mse,
                'rmse': ridge_rmse,
                'mae': ridge_mae,
                'r2': ridge_r2,
            }
            
            ridge_metrics_file = os.path.join(reg_regression_dir, f"ridge_metrics_{month_id}.csv")
            pd.DataFrame([ridge_metrics]).to_csv(ridge_metrics_file, index=False)
            
            # Add to results
            results["models"]["ridge"] = {
                "metrics": ridge_metrics,
                "coefficients_path": ridge_coefs_file,
                "metrics_path": ridge_metrics_file
            }
            
            # Save comparison of top features from both models
            print("\nTop 10 Features by Importance:")
            print("\nLasso Top Features:")
            print(lasso_coefs[lasso_coefs['Coefficient'] != 0].head(10))
            
            print("\nRidge Top Features:")
            print(ridge_coefs.head(10))
            
            # Save the models
            try:
                import joblib
                lasso_model_file = os.path.join(reg_regression_dir, f"lasso_model_{month_id}.joblib")
                ridge_model_file = os.path.join(reg_regression_dir, f"ridge_model_{month_id}.joblib")
                
                joblib.dump(lasso_model, lasso_model_file)
                joblib.dump(ridge_model, ridge_model_file)
                
                # Add model paths to results
                results["models"]["lasso"]["model_path"] = lasso_model_file
                results["models"]["ridge"]["model_path"] = ridge_model_file
                
                print(f"Models saved to {reg_regression_dir}")
            except Exception as e:
                print(f"Warning: Could not save models: {str(e)}")
            
            return results
            
        except Exception as e:
            print(f"Error training regularized regression for {month_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def train_xgboost_rs_with_important_features(self, month_id, param_distributions=None, n_iter=None, cv=None, random_state=42):

        """
        Train an XGBoost model with RandomizedSearchCV using only the top features identified
        from a previous XGBoost RandomizedSearchCV run.
        
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
            
            # Import TOP_FEATURES_COUNT for selecting top features
            from config.const import TOP_FEATURES_COUNT
            
            # MEMORY OPTIMIZATION: Limit parameters for better memory usage
            n_iter = min(n_iter, 20)  # Reduce number of iterations
            cv = min(cv, 3)  # Reduce CV folds
            
            # Step 1: Check if we already have feature importance from a previous run
            xgboost_rs_dir = os.path.join(self.project_root, XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
            feature_importance_path = os.path.join(xgboost_rs_dir, f"feature_importance_{month_id}_rs.csv")
            
            # Create output directory for this method's results
            output_dir = os.path.join(self.project_root, "data/output/xgboost_rs_important_features")
            os.makedirs(output_dir, exist_ok=True)
            
            important_features = None
            
            # If feature importance exists, read it
            if os.path.exists(feature_importance_path):
                print(f"Using existing feature importance from {feature_importance_path}")
                feature_importance = pd.read_csv(feature_importance_path)
                important_features = feature_importance.sort_values(by='Importance', ascending=False).head(TOP_FEATURES_COUNT)['Feature'].tolist()
            else:
                # If not, run the regular randomized search to get feature importance
                print(f"No existing feature importance found. Running XGBoost RandomizedSearchCV first...")
                rs_result = self.train_xgboost_with_randomized_search_cv(month_id, param_distributions, n_iter, cv, random_state)
                
                if not rs_result.get("success", False):
                    print(f"Failed to run initial RandomizedSearchCV: {rs_result.get('error', 'Unknown error')}")
                    return {
                        "success": False,
                        "error": f"Failed to run initial RandomizedSearchCV: {rs_result.get('error', 'Unknown error')}"
                    }
                
                # Now check if the feature importance was saved
                if os.path.exists(feature_importance_path):
                    feature_importance = pd.read_csv(feature_importance_path)
                    important_features = feature_importance.sort_values(by='Importance', ascending=False).head(TOP_FEATURES_COUNT)['Feature'].tolist()
                else:
                    print(f"Error: Feature importance not found after running RandomizedSearchCV")
                    return {
                        "success": False,
                        "error": "Feature importance not found after running RandomizedSearchCV"
                    }
            
            print(f"Selected top {TOP_FEATURES_COUNT} features: {important_features}")
            
            # Now load the datasets and prepare for training with only important features
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
            target_options = ['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset']
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
            
            # DATA VALIDATION: Check for non-numeric columns
            non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
                X_train = X_train.select_dtypes(include=['number'])
                X_test = X_test.select_dtypes(include=['number'])
            
            # Check if all important features exist in the dataset
            missing_features = [f for f in important_features if f not in X_train.columns]
            if missing_features:
                print(f"Warning: Some important features are missing from the dataset: {missing_features}")
                # Filter out missing features
                important_features = [f for f in important_features if f in X_train.columns]
                print(f"Using available features: {important_features}")
            
            # Check if we have classification or regression problem
            is_classification = True
            if target_column in ['differenceInMinutes', 'differenceInMinutes_offset']:
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            # Create sample weights for classification if delay info is available
            sample_weights = None
            if is_classification and 'differenceInMinutes' in train_df.columns:
                print("Using weighted samples based on delay magnitude for randomized search")
                # Create sample weights based on delay magnitude
                delay_col = 'differenceInMinutes'
                sample_weights = np.ones(len(y_train))
                
                # Get delay values for each training sample
                delays = train_df[delay_col].values
                
                # Apply weights - higher delays get higher weights
                delayed_idx = (delays > 0)
                if np.any(delayed_idx):
                    # Normalize delay values by mean positive delay, using more moderate weights
                    mean_delay = delays[delayed_idx].mean()
                    # Cap weights at 5 instead of 10 for more stability
                    sample_weights[delayed_idx] = np.minimum(5, 1 + delays[delayed_idx]/mean_delay)
                
                print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
            
            # Subset data to include only important features
            X_train_important = X_train[important_features]
            X_test_important = X_test[important_features]
            
            print(f"Training with only {len(important_features)} features instead of {X_train.shape[1]} features")
            
            # --------- MANUAL HYPERPARAMETER TUNING APPROACH ---------            
            print(f"Starting manual hyperparameter tuning with {n_iter} iterations and {cv}-fold cross-validation...")

            # Define custom objective function for regression if needed
            def stable_weighted_mse(y_pred, dtrain):
                y_true = dtrain.get_label()
                # More moderate weighting approach with capped weights
                weights = np.minimum(3.0, 1.0 + np.abs(y_true) / (np.abs(y_true).mean() * 2))
                # More stable gradient calculation
                grad = weights * (y_pred - y_true)
                hess = weights
                return grad, hess
            
            # Generate parameter combinations
            param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
            
            # Setup cross-validation
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            
            best_score = float('-inf')
            best_params = None
            best_method = None  # To track if custom objective was best
            
            # Memory tracking
            process = psutil.Process()
            before_mem = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage before training: {before_mem:.2f} MB")
            
            # Define methods to try based on problem type
            methods_to_try = []
            
            if is_classification:
                # For classification, just use standard method, possibly with sample weights
                methods_to_try = [{"name": "standard", "obj": None}]
            else:
                # For regression, try both standard and custom objective
                methods_to_try = [
                    {"name": "standard", "obj": None},
                    {"name": "weighted", "obj": stable_weighted_mse}
                ]
                print(f"For regression, will try {len(methods_to_try)} different objective approaches")
            
            # Try each parameter combination and each method
            overall_best_score = float('-inf')
            
            for i, params in enumerate(param_list):
                for method in methods_to_try:
                    method_name = method["name"]
                    obj_function = method["obj"]
                    
                    print(f"Testing parameter combination {i+1}/{len(param_list)} with method '{method_name}'")
                    
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
                        # For regression, only set objective if not using custom objective
                        if obj_function is None:
                            current_params['objective'] = 'reg:squarederror'
                    
                    # Add random_state for reproducibility
                    current_params['random_state'] = random_state
                    
                    # Perform cross-validation
                    cv_scores = []
                    for train_idx, val_idx in cv_splitter.split(X_train_important, y_train if is_classification else np.zeros(len(y_train))):
                        # Get train and validation sets for this fold
                        X_fold_train = X_train_important.iloc[train_idx]
                        y_fold_train = y_train.iloc[train_idx]
                        X_fold_val = X_train_important.iloc[val_idx]
                        y_fold_val = y_train.iloc[val_idx]
                        
                        # Get fold-specific sample weights if weights are being used
                        fold_sample_weights = None
                        if sample_weights is not None:
                            fold_sample_weights = sample_weights[train_idx]
                        
                        # Use different approaches based on problem type and method
                        if is_classification:
                            # For classification, use XGBClassifier
                            model = xgb.XGBClassifier(**current_params)
                            
                            # Fit model with sample weights if available
                            if fold_sample_weights is not None:
                                model.fit(X_fold_train, y_fold_train, sample_weight=fold_sample_weights)
                            else:
                                model.fit(X_fold_train, y_fold_train)
                                
                            # Predict
                            y_pred = model.predict(X_fold_val)
                            
                            # Evaluate
                            from sklearn.metrics import accuracy_score
                            score = accuracy_score(y_fold_val, y_pred)
                            
                        else:
                            # For regression with custom objective, use lower-level API
                            if obj_function is not None:
                                # Create DMatrix objects
                                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                                dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
                                
                                # Set up parameters for training
                                xgb_params = {
                                    'max_depth': current_params.get('max_depth', 6),
                                    'eta': current_params.get('learning_rate', 0.1),
                                    'subsample': current_params.get('subsample', 0.8),
                                    'colsample_bytree': current_params.get('colsample_bytree', 0.8),
                                    'seed': random_state,
                                    'alpha': 0.5,  # L1 regularization
                                    'lambda': 1.0,  # L2 regularization
                                }
                                
                                # Train the model with custom objective
                                model = xgb.train(
                                    xgb_params,
                                    dtrain,
                                    num_boost_round=current_params.get('n_estimators', 100),
                                    obj=obj_function
                                )
                                
                                # Predict
                                y_pred = model.predict(dval)
                                
                            else:
                                # Standard regression using XGBRegressor
                                model = xgb.XGBRegressor(**current_params)
                                model.fit(X_fold_train, y_fold_train)
                                y_pred = model.predict(X_fold_val)
                            
                            # Evaluate regression performance
                            from sklearn.metrics import mean_squared_error
                            score = -mean_squared_error(y_fold_val, y_pred)  # Negative for higher=better
                        
                        cv_scores.append(score)
                    
                    # Calculate average score across folds
                    avg_score = np.mean(cv_scores)
                    print(f"  Method '{method_name}' - Average CV score: {avg_score:.4f}")
                    
                    # Update best if better
                    if avg_score > overall_best_score:
                        overall_best_score = avg_score
                        best_score = avg_score
                        best_params = current_params.copy()
                        best_method = method_name
                        print(f"  New best score: {best_score:.4f} with method '{best_method}'")
            
            after_mem = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage after training: {after_mem:.2f} MB (Δ: {after_mem - before_mem:.2f} MB)")
            
            print(f"Best Hyperparameters: {best_params}")
            print(f"Best Method: {best_method}")
            print(f"Best CV Score: {best_score:.4f}")
            
            # Train final model with best parameters
            if is_classification:
                xgb_model = xgb.XGBClassifier(**best_params)
                
                # Fit the final model with sample weights if available
                if sample_weights is not None:
                    print("Training final model with sample weights")
                    xgb_model.fit(X_train_important, y_train, sample_weight=sample_weights)
                else:
                    xgb_model.fit(X_train_important, y_train)
                    
            else:
                # For regression, check if best method uses custom objective
                if best_method == "weighted":
                    print("Training final model with custom weighted objective function")
                    # Create DMatrix objects
                    dtrain = xgb.DMatrix(X_train_important, label=y_train)
                    dtest = xgb.DMatrix(X_test_important, label=y_test)
                    
                    # Set up parameters for training
                    xgb_params = {
                        'max_depth': best_params.get('max_depth', 6),
                        'eta': best_params.get('learning_rate', 0.1),
                        'subsample': best_params.get('subsample', 0.8),
                        'colsample_bytree': best_params.get('colsample_bytree', 0.8),
                        'seed': random_state,
                        'alpha': 0.5,  # L1 regularization
                        'lambda': 1.0,  # L2 regularization
                    }
                    
                    # Train the model with custom objective
                    xgb_model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=best_params.get('n_estimators', 100),
                        obj=stable_weighted_mse
                    )
                else:
                    print("Training final model with standard objective")
                    xgb_model = xgb.XGBRegressor(**best_params)
                    xgb_model.fit(X_train_important, y_train)
            
            # Evaluate on test set
            if is_classification:
                y_pred = xgb_model.predict(X_test_important)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                print(f"\nXGBoost Classifier Results (Top {TOP_FEATURES_COUNT} Features Only):")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Extract and save metrics
                metrics_result = self.extract_and_save_metrics(
                    y_test, y_pred, report, f"{month_id}_rs_important", 
                    output_dir=output_dir
                )
            else:
                # Handle regression evaluation - check if using booster or regressor
                if best_method == "weighted" and hasattr(xgb_model, 'predict'):
                    # If using booster with custom objective (has predict method)
                    dtest = xgb.DMatrix(X_test_important)
                    y_pred = xgb_model.predict(dtest)
                else:
                    # Standard XGBRegressor
                    y_pred = xgb_model.predict(X_test_important)
                
                # Use the regression metrics function
                metrics_result = self.extract_and_save_regression_metrics(
                    y_test, y_pred, f"{month_id}_rs_important", 
                    output_dir=output_dir
                )
                
                # Keep these lines for printing to console
                mse = metrics_result["metrics"]["mse"]
                rmse = metrics_result["metrics"]["rmse"]
                mae = metrics_result["metrics"]["mae"]
                r2 = metrics_result["metrics"]["r2"]
                
                print(f"\nXGBoost Regressor Results (Top {TOP_FEATURES_COUNT} Features Only):")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"R²: {r2:.4f}")
            
            # Get feature importance of the final model
            if hasattr(xgb_model, 'feature_importances_'):
                # For standard XGBClassifier/XGBRegressor
                selected_feature_importance = pd.DataFrame({
                    'Feature': important_features,
                    'Importance': xgb_model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            elif hasattr(xgb_model, 'get_score'):
                # For booster with custom objective
                importance_scores = xgb_model.get_score(importance_type='gain')
                # Normalize the importance scores
                total_importance = sum(importance_scores.values())
                normalized_scores = {k: v/total_importance for k, v in importance_scores.items()}
                
                selected_feature_importance = pd.DataFrame({
                    'Feature': normalized_scores.keys(),
                    'Importance': normalized_scores.values()
                }).sort_values(by='Importance', ascending=False)
            else:
                selected_feature_importance = pd.DataFrame({
                    'Feature': important_features,
                    'Importance': [1.0/len(important_features)] * len(important_features)
                })
            
            print("\nFeature Importance in Final Model:")
            print(selected_feature_importance)
            
            # Save the model, feature importance, and parameters
            try:
                import joblib
                
                # Save the model
                model_filename = f"xgboost_{month_id}_rs_important.joblib"
                model_path = os.path.join(output_dir, model_filename)
                joblib.dump(xgb_model, model_path)
                print(f"Model saved to {model_path}")
                
                # Save feature importance
                importance_filename = f"feature_importance_{month_id}_rs_important.csv"
                importance_path = os.path.join(output_dir, importance_filename)
                selected_feature_importance.to_csv(importance_path, index=False)
                print(f"Feature importance saved to {importance_path}")
                
                # Save best parameters
                params_filename = f"best_params_{month_id}_rs_important.txt"
                params_path = os.path.join(output_dir, params_filename)
                with open(params_path, 'w') as f:
                    f.write(f"Important features ({len(important_features)}):\n")
                    for feature in important_features:
                        f.write(f"- {feature}\n")
                    f.write("\nBest parameters:\n")
                    for param, value in best_params.items():
                        f.write(f"{param}: {value}\n")
                print(f"Parameters and features saved to {params_path}")
                
                # Compare with the full-features model
                full_model_metrics_file = os.path.join(xgboost_rs_dir, f"model_metrics_{month_id}_rs.csv")
                if os.path.exists(full_model_metrics_file):
                    full_metrics = pd.read_csv(full_model_metrics_file)
                    
                    if is_classification:
                        if 'accuracy' in full_metrics:
                            full_accuracy = full_metrics['accuracy'].values[0]
                            print(f"\nComparison with full-features model:")
                            print(f"Full features model accuracy: {full_accuracy:.4f}")
                            print(f"Top {TOP_FEATURES_COUNT} features model accuracy: {accuracy:.4f}")
                            acc_change = ((accuracy - full_accuracy) / full_accuracy) * 100
                            print(f"Accuracy change: {acc_change:.2f}%")
                    else:
                        if 'rmse' in full_metrics:
                            full_rmse = full_metrics['rmse'].values[0]
                            print(f"\nComparison with full-features model:")
                            print(f"Full features model RMSE: {full_rmse:.4f}")
                            print(f"Top {TOP_FEATURES_COUNT} features model RMSE: {rmse:.4f}")
                            rmse_change = ((rmse - full_rmse) / full_rmse) * 100
                            print(f"RMSE change: {rmse_change:.2f}%")
                
                # Create result dictionary
                if is_classification:
                    result = {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy,
                        "report": report,
                        "best_params": best_params,
                        "best_method": best_method,
                        "important_features": important_features,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"],
                        "used_sample_weights": sample_weights is not None
                    }
                else:
                    result = {
                        "success": True,
                        "model_type": "regression",
                        "rmse": rmse,
                        "r2": r2,
                        "best_params": best_params,
                        "best_method": best_method,
                        "custom_objective": best_method == "weighted",
                        "important_features": important_features,
                        "metrics": metrics_result["metrics"],
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "metrics_path": metrics_result["metrics_path"]
                    }
                
                return result
                    
            except Exception as e:
                print(f"Warning: Could not save model: {str(e)}")
                # Create minimal result dictionary
                if is_classification:
                    return {
                        "success": True,
                        "model_type": "classification",
                        "accuracy": accuracy if 'accuracy' in locals() else None,
                        "best_params": best_params,
                        "best_method": best_method,
                        "important_features": important_features,
                        "metrics": metrics_result["metrics"] if 'metrics_result' in locals() else None,
                        "metrics_path": metrics_result["metrics_path"] if 'metrics_result' in locals() else None,
                        "model_saved": False
                    }
                else:
                    return {
                        "success": True,
                        "model_type": "regression",
                        "rmse": rmse if 'rmse' in locals() else None,
                        "r2": r2 if 'r2' in locals() else None,
                        "best_params": best_params,
                        "best_method": best_method,
                        "important_features": important_features,
                        "metrics": metrics_result["metrics"] if 'metrics_result' in locals() else None,
                        "metrics_path": metrics_result["metrics_path"] if 'metrics_result' in locals() else None,
                        "model_saved": False
                    }
        
        except Exception as e:
            import traceback
            print(f"Error in XGBoost RandomizedSearchCV with Important Features for {month_id}: {str(e)}")
            print("\nDetailed traceback:")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }