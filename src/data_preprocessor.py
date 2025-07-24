from contextlib import contextmanager
import glob
import os
import pandas as pd
import re
import ast
import shap
import joblib
import logging
from sklearn.impute import SimpleImputer
import xgboost as xgb
import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours
from src.file_utils import generate_output_path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler, KFold, StratifiedKFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, 
    average_precision_score, cohen_kappa_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
            

from config.const import (
    ALL_PREPROCESSED_OUTPUT_FOLDER,
    CATEGORIAL_TARGET_FEATURES,
    DATA_FILE_PREFIX_FOR_TRAINING,
    DROP_TRAIN_FEATURES,
    ENN_N_NEIGHBORS,
    IMBALANCE_THRESHOLD,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_WEATHER_CONDITIONS,
    NON_NUMERIC_FEATURES, 
    OUTPUT_FOLDER,
    PREPROCESSING_STATE_MACHINE,
    PREPROCESSED_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    IMPORTANCE_THRESHOLD,
    REGRESSION_PROBLEM,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    RESAMPLING_METHOD,
    SCORE_METRIC,
    TOP_FEATURES_COUNT,
    TRAIN_DELAY_MINUTES,
    TRAIN_DELAYED_TARGET_COLUMN,
    VALID_PREDICTION_FEATURES,
    VALID_TARGET_FEATURES,
    WEATHER_COLS_TO_MERGE,
    WEIGHT_DELAY_COLUMN,
    XGBOOST_METHODS_CONFIG,
    XGBOOST_OBJECTIVE_FUNCTIONS,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    DEFAULT_TARGET_FEATURE,
    FILTER_TRAINS_BY_STATIONS,
    REQUIRED_STATIONS,
    MAX_SAMPLE_WEIGHT_CLASSIFICATION,
    MAX_SAMPLE_WEIGHT_REGRESSION
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
        self.all_preprocessed_dir = os.path.join(self.project_root, ALL_PREPROCESSED_OUTPUT_FOLDER)  # NEW LINE
        self.randomized_search_dir = os.path.join(self.project_root, RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER)
        self.random_forest_dir = os.path.join(self.project_root, RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
        self.important_features_randomized_search_dir = os.path.join(self.project_root, IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
        self.xgboost_rs_dir = os.path.join(self.project_root, XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
        self.regularized_regression_dir = os.path.join(self.project_root, REGULARIZED_REGRESSION_OUTPUT_FOLDER)


        # Create log directory
        self.log_dir = os.path.join(self.project_root, "data", "output", "log")
        os.makedirs(self.log_dir, exist_ok=True)

        # Add this line to make the constant available as an instance attribute
        self.DATA_FILE_PREFIX_FOR_TRAINING = DATA_FILE_PREFIX_FOR_TRAINING

        # Use the imported constant instead of defining it here
        self.important_conditions = IMPORTANT_WEATHER_CONDITIONS

    @contextmanager
    def get_logger(self, log_filename, logger_name=None, month_id=None):
        """
        Context manager for creating and managing loggers with automatic cleanup.
        
        Parameters:
        -----------
        log_filename : str
            Name of the log file (e.g., "merge_snow_depth.log")
        logger_name : str, optional
            Name of the logger. If None, uses log_filename without extension.
        month_id : str, optional
            Month identifier (e.g., "2023-2024_12") to log at the beginning of the file.
            
        Yields:
        -------
        logging.Logger
            Configured logger instance
        """
        # Create log file path
        log_file_path = os.path.join(self.log_dir, log_filename)
        
        # Create logger name if not provided
        if logger_name is None:
            logger_name = os.path.splitext(log_filename)[0]
        
        # Create a logger specifically for this operation
        logger = logging.getLogger(logger_name)
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
        
        # Log the month_id at the beginning if provided
        if month_id:
            logger.info(f"=== Processing Month: {month_id} ===")
        
        try:
            yield logger
        finally:
            # Clean up logger handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def run_pipeline(self, csv_files, target_feature=DEFAULT_TARGET_FEATURE):
        """
        Run pipeline - processes each CSV file individually based on state machine configuration.
        
        This method processes each input file through the configured pipeline steps,
        resulting in individual preprocessed files for each YYYY-MM period.
        
        Parameters:
        -----------
        csv_files : list
            List of CSV file paths to process.
        target_feature : str, optional
            The target feature (currently used for compatibility, may be used in future steps).
            
        Returns:
        --------
        dict
            Summary of the preprocessing results.
        """
        if not csv_files:
            print("\nNo CSV files to process.")
            return {
                "total_files": 0,
                "successful_preprocessing": 0,
                "failed_files": 0,
                "state_machine_used": PREPROCESSING_STATE_MACHINE
            }
        
        # Check if any pipeline steps are enabled
        enabled_steps = [step for step, enabled in PREPROCESSING_STATE_MACHINE.items() if enabled]
        if not enabled_steps:
            print("Warning: No pipeline steps are enabled in the state machine.")
            return {
                "total_files": len(csv_files),
                "successful_preprocessing": 0,
                "failed_files": len(csv_files),
                "state_machine_used": PREPROCESSING_STATE_MACHINE,
                "error": "No pipeline steps enabled"
            }
        
        print(f"\nStarting pipeline processing for {len(csv_files)} CSV files")
        print(f"State machine configuration: {PREPROCESSING_STATE_MACHINE}")
        print(f"Enabled pipeline steps: {enabled_steps}")
        
        # Initialize counters
        successful_preprocessing = 0
        failed_files = 0
        processed_files_info = []
        pipeline_execution_details = []
        
        # Process each file individually
        for i, input_file_path in enumerate(csv_files):
            filename = os.path.basename(input_file_path)
            
            # Extract year and month from filename
            match = re.search(r'(\d{4})_(\d{2})\.csv$', filename)
            
            if not match:
                print(f"\n[{i+1}/{len(csv_files)}] Warning: Could not extract date from filename {filename}. Skipping.")
                failed_files += 1
                pipeline_execution_details.append({
                    "filename": filename,
                    "success": False,
                    "error": "Could not extract date from filename",
                    "steps_executed": []
                })
                continue
            
            year, month = match.groups()
            file_id = f"{year}_{month}"
            
            print(f"\n[{i+1}/{len(csv_files)}] Processing file: {filename} (Year: {year}, Month: {month})")
            
            try:
                # Execute pipeline steps using the state machine
                pipeline_result = self.execute_pipeline_steps(
                    input_file_path=input_file_path,
                    file_id=file_id,
                    year=year,
                    state_machine=PREPROCESSING_STATE_MACHINE
                )
                
                # Record pipeline execution details
                execution_detail = {
                    "filename": filename,
                    "file_id": file_id,
                    "year": year,
                    "month": month,
                    "success": pipeline_result["success"],
                    "steps_executed": pipeline_result["steps_executed"],
                    "errors": pipeline_result["errors"]
                }
                
                if pipeline_result["success"]:
                    print(f"✓ Successfully processed {filename}")
                    successful_preprocessing += 1
                    
                    # Add file info to results
                    processed_files_info.append({
                        "original_file": filename,
                        "year": year,
                        "month": month,
                        "file_id": file_id,
                        "rows": pipeline_result["file_info"]["rows"],
                        "columns": pipeline_result["file_info"]["columns"],
                        "steps_executed": pipeline_result["steps_executed"]
                    })
                    
                    execution_detail.update(pipeline_result["file_info"])
                else:
                    print(f"✗ Failed to process {filename}")
                    print(f"  Errors: {'; '.join(pipeline_result['errors'])}")
                    failed_files += 1
                
                pipeline_execution_details.append(execution_detail)
                    
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
                failed_files += 1
                pipeline_execution_details.append({
                    "filename": filename,
                    "success": False,
                    "error": f"Unexpected error: {str(e)}",
                    "steps_executed": []
                })
        
        # Generate summary
        summary = {
            "total_files": len(csv_files),
            "successful_preprocessing": successful_preprocessing,
            "failed_files": failed_files,
            "processed_files_info": processed_files_info,
            "state_machine_used": PREPROCESSING_STATE_MACHINE,
            "enabled_steps": enabled_steps,
            "pipeline_execution_details": pipeline_execution_details
        }
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE PROCESSING SUMMARY:")
        print("="*60)
        print(f"State machine configuration: {PREPROCESSING_STATE_MACHINE}")
        print(f"Enabled steps: {enabled_steps}")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successfully processed and saved: {summary['successful_preprocessing']}")
        print(f"Failed to process: {summary['failed_files']}")
        
        if processed_files_info:
            print(f"\nSuccessfully processed files:")
            for info in processed_files_info:
                steps_str = " → ".join(info['steps_executed'])
                print(f"  {info['original_file']} -> preprocessed_data_{info['file_id']}.csv")
                print(f"    ({info['rows']:,} rows, {len(info['steps_executed'])} steps: {steps_str})")
        
        if failed_files > 0:
            print(f"\nFailed files:")
            for detail in pipeline_execution_details:
                if not detail["success"]:
                    error_msg = detail.get("error", "Unknown error")
                    print(f"  {detail['filename']}: {error_msg}")
        
        # Calculate and display success rate
        success_rate = (successful_preprocessing / len(csv_files) * 100) if csv_files else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
        print("="*60)
        
        return summary

    def execute_pipeline_steps(self, input_file_path, file_id, year, state_machine):
        """
        Execute pipeline steps based on the state machine configuration.
        
        This helper method processes a single file through the configured pipeline steps,
        maintaining data flow between steps and handling errors gracefully.
        
        Parameters:
        -----------
        input_file_path : str
            Path to the input CSV file
        file_id : str
            File identifier (e.g., "2023_12")
        year : str
            Year extracted from filename for reference
        state_machine : dict
            State machine configuration defining which steps to execute
            
        Returns:
        --------
        dict
            Results of pipeline execution including success status, data, and metadata
        """
        result = {
            "success": False,
            "data": None,
            "steps_executed": [],
            "errors": [],
            "file_info": {
                "file_id": file_id,
                "year": year,
                "rows": 0,
                "columns": 0
            }
        }
        
        print(f"  Executing pipeline steps based on state machine configuration...")
        
        # Step 1: Extract nested data if enabled
        if state_machine.get("extract_nested_data", False):
            try:
                print(f"    → extract_nested_data")
                processed_df = self.extract_nested_data(input_file_path)
                
                if processed_df is not None and not processed_df.empty:
                    # Add year information for reference
                    processed_df['data_year'] = year
                    result["data"] = processed_df
                    result["steps_executed"].append("extract_nested_data")
                    result["file_info"]["rows"] = len(processed_df)
                    result["file_info"]["columns"] = len(processed_df.columns)
                    print(f"      ✓ Extracted {len(processed_df)} rows, {len(processed_df.columns)} columns")
                else:
                    result["errors"].append("extract_nested_data returned empty data")
                    print(f"      ✗ Failed - empty result")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"extract_nested_data failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ extract_nested_data (disabled)")
        
        # Step 2: Save to CSV if enabled and we have data
        if state_machine.get("save_month_df_to_csv", False):
            if result["data"] is not None:
                try:
                    print(f"    → save_month_df_to_csv")
                    save_success = self.save_month_df_to_csv(file_id, result["data"])
                    
                    if save_success:
                        result["steps_executed"].append("save_month_df_to_csv")
                        print(f"      ✓ Saved preprocessed data for {file_id}")
                    else:
                        result["errors"].append("save_month_df_to_csv failed")
                        print(f"      ✗ Failed to save data for {file_id}")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"save_month_df_to_csv failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ save_month_df_to_csv (no data available)")
                result["errors"].append("save_month_df_to_csv skipped - no data available")
        else:
            print(f"    ⊝ save_month_df_to_csv (disabled)")
        
        # Mark as successful if no errors occurred
        result["success"] = len(result["errors"]) == 0
        return result

    def extract_nested_data(self, input_file_path):
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
            
            # Process each train's stops
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

            # Return the processed DataFrame
            return cross_df
            
        except Exception as e:
            print(f"Error processing file {input_file_path}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return None


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
            "successful_random_forest": 0,  # NEW
            "successful_regularized_regression": 0,
            "failed_decision_tree": 0,
            "failed_random_forest": 0,  # NEW
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
                    case "merge_weather_columns":  
                        print(f"\nMerging weather columns for {month_id}...")
                        merged_df = self.merge_weather_columns(dataframe=state["df"], month_id=month_id)
                                        
                        # Clear previous dataframe from memory
                        del state["df"]
                        
                        if merged_df is None:
                            print(f"Failed to merge weather columns for {month_id}")
                            state["success"] = False
                        else:
                            print(f"Successfully merged weather columns for {month_id}")
                            counters["successful_snow_depth_merge"] += 1
                            state["df"] = merged_df
                            state["current_stage"] = "clean_missing_values"
                    
                    case "clean_missing_values":
                        print(f"\nHandling missing values for {month_id}...")
                        cleaned_df = self.handle_missing_values(dataframe=state["df"], month_id=month_id)
                        
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
                        deduplicated_df = self.remove_duplicates(dataframe=state["df"], month_id=month_id)
                        
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
                        scaled_df = self.scale_numeric_columns(dataframe=state["df"], month_id=month_id)
                        
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
                        state["current_stage"] = "train_decision_tree_with_randomized_search_cv"

                    case "train_decision_tree_with_randomized_search_cv":
                        print(f"Training decision tree with RandomizedSearchCV for {month_id}...")
                        random_search_result = self.train_decision_tree_with_randomized_search_cv(month_id)
                        
                        if not random_search_result.get("success", False):
                            print(f"Failed to train decision tree with RandomizedSearchCV for {month_id}: {random_search_result.get('error', 'Unknown error')}")
                            counters["failed_randomized_search"] = counters.get("failed_randomized_search", 0) + 1
                        else:
                            print(f"Successfully trained decision tree with RandomizedSearchCV for {month_id}")
                            counters["successful_randomized_search"] = counters.get("successful_randomized_search", 0) + 1

                        # UPDATED: Go directly to XGBoost instead of important features
                        state["current_stage"] = "train_xgboost_with_randomized_search_cv"

                    case "train_random_forest_with_randomized_search_cv":
                        print(f"Training Random Forest with RandomizedSearchCV for {month_id}...")
                        random_forest_result = self.train_random_forest_with_randomized_search_cv(month_id)
                        
                        if not random_forest_result.get("success", False):
                            print(f"Failed to train Random Forest with RandomizedSearchCV for {month_id}: {random_forest_result.get('error', 'Unknown error')}")
                            counters["failed_random_forest"] = counters.get("failed_random_forest", 0) + 1
                        else:
                            print(f"Successfully trained Random Forest with RandomizedSearchCV for {month_id}")
                            counters["successful_random_forest"] = counters.get("successful_random_forest", 0) + 1

                        # Move to the next stage
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

                        # Move to the next stage
                        state["current_stage"] = "merge_all_preprocessed_files"

                    case "merge_all_preprocessed_files":
                        print(f"Merging all preprocessed files...")
                        merge_result = self.merge_all_preprocessed_files()
                        
                        if not merge_result.get("success", False):
                            print(f"Failed to merge preprocessed files: {merge_result.get('error', 'Unknown error')}")
                            counters["failed_merge"] = counters.get("failed_merge", 0) + 1
                        else:
                            print(f"Successfully merged all preprocessed files")
                            print(f"Output: {merge_result.get('output_path', 'Unknown')}")
                            print(f"Total rows: {merge_result.get('total_rows', 0):,}")
                            counters["successful_merge"] = counters.get("successful_merge", 0) + 1

                        # Move to the next stage
                        state["current_stage"] = "split_combined_dataset"

                    case "split_combined_dataset":
                        print(f"Splitting combined dataset...")
                        split_result = self.split_combined_dataset()
                        
                        if not split_result.get("success", False):
                            print(f"Failed to split combined dataset: {split_result.get('error', 'Unknown error')}")
                            counters["failed_combined_split"] = counters.get("failed_combined_split", 0) + 1
                        else:
                            print(f"Successfully split combined dataset")
                            print(f"  Data source: {split_result.get('data_source', 'Unknown')}")
                            print(f"  Total samples: {split_result.get('total_samples', 0):,}")
                            print(f"  Final train samples: {split_result.get('final_train_size', 0):,}")
                            print(f"  Test samples: {split_result.get('test_size', 0):,}")
                            print(f"  Resampling applied: {split_result.get('resampling_applied', False)}")
                            counters["successful_combined_split"] = counters.get("successful_combined_split", 0) + 1

                        # Move to the next stage
                        state["current_stage"] = "train_decision_tree_combined_data"

                    case "train_decision_tree_combined_data":
                        print(f"Training Decision Tree on combined data...")
                        combined_dt_result = self.train_decision_tree_combined_data()
                        
                        if not combined_dt_result.get("success", False):
                            print(f"Failed to train Decision Tree on combined data: {combined_dt_result.get('error', 'Unknown error')}")
                            counters["failed_combined_decision_tree"] = counters.get("failed_combined_decision_tree", 0) + 1
                        else:
                            print(f"Successfully trained Decision Tree on combined data")
                            print(f"  Data source: {combined_dt_result.get('data_source', 'Unknown')}")
                            print(f"  Total samples: {combined_dt_result.get('total_samples', 0):,}")
                            print(f"  Train samples: {combined_dt_result.get('train_samples', 0):,}")
                            print(f"  Test samples: {combined_dt_result.get('test_samples', 0):,}")
                            print(f"  Test {combined_dt_result.get('optimized_metric_name', 'accuracy')}: {combined_dt_result.get('optimized_metric', 0):.4f}")
                            counters["successful_combined_decision_tree"] = counters.get("successful_combined_decision_tree", 0) + 1

                        # This is the final stage
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
        
        # Print summary - UPDATED to remove mentions of removed methods
        print("\n" + "="*50)
        print("Processing Summary:")
        print(f"Total months processed: {summary['total_months']}")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successfully preprocessed: {summary['successful_preprocessing']}")
        print(f"Successfully merged snow depth columns: {summary['successful_snow_depth_merge']}")
        print(f"Successfully cleaned missing values: {summary['successful_cleaning']}")
        print(f"Successfully deduplicated: {summary['successful_deduplication']}")
        print(f"Successfully scaled numeric columns: {summary['successful_scaling']}")
        print(f"Successfully added trainDelayed feature: {summary['successful_feature_addition']}")
        print(f"Successfully selected target feature: {summary['successful_target_selection']}")
        print(f"Successfully saved to CSV: {summary['successful_saves']}")
        print(f"Successfully split into train/test sets: {summary['successful_splits']}")
        print(f"Successfully trained regularized regression models: {summary.get('successful_regularized_regression', 0)}")
        print(f"Successfully trained decision tree models with RandomizedSearchCV: {summary.get('successful_randomized_search', 0)}")
        print(f"Successfully trained Random Forest models with RandomizedSearchCV: {summary.get('successful_random_forest', 0)}")
        print(f"Successfully trained XGBoost models with RandomizedSearchCV: {summary.get('successful_xgboost_rs', 0)}")
        print(f"Successfully merged all preprocessed files: {summary.get('successful_merge', 0)}")  # NEW LINE
        print(f"Failed to train regularized regression models: {summary.get('failed_regularized_regression', 0)}")
        print(f"Failed to train decision tree models with RandomizedSearchCV: {summary.get('failed_randomized_search', 0)}")
        print(f"Failed to train Random Forest models with RandomizedSearchCV: {summary.get('failed_random_forest', 0)}")
        print(f"Failed to train XGBoost models with RandomizedSearchCV: {summary.get('failed_xgboost_rs', 0)}")
        print(f"Failed to merge preprocessed files: {summary.get('failed_merge', 0)}")  # NEW LINE
        print(f"Failed to process: {summary['failed_files']}")

        print(f"Successfully trained Decision Tree on combined data: {summary.get('successful_combined_decision_tree', 0)}")
        print(f"Failed to train Decision Tree on combined data: {summary.get('failed_combined_decision_tree', 0)}")

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
            expected_cols = VALID_TARGET_FEATURES + VALID_PREDICTION_FEATURES
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
        
    def merge_weather_columns(self, dataframe=None, month_id=None):
        """
        Merge weather feature columns that have 'Other' variants into their main columns.
        
        For each weather feature in WEATHER_COLS_TO_MERGE:
        - If main column (e.g., 'Snow depth') is missing, use value from 'Other' variant
        - Drop the 'Other' and 'Other Distance' columns after merging
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
        month_id : str, optional
            Month identifier for logging purposes.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with merged weather columns and unwanted columns removed.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Merging weather columns in dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Use the new logging method
        with self.get_logger("merge_weather_columns.log", "merge_weather", month_id) as logger:
            try:
                logger.info(f"Starting weather column merging for {len(WEATHER_COLS_TO_MERGE)} weather features")
                
                total_merges = 0
                total_drops = 0
                
                # Process each weather feature in the list
                for weather_feature in WEATHER_COLS_TO_MERGE:
                    main_col = weather_feature
                    other_col = f"{weather_feature} Other"
                    other_distance_col = f"{weather_feature} Other Distance"
                    
                    # Check which columns exist for this weather feature
                    has_main = main_col in df.columns
                    has_other = other_col in df.columns
                    has_other_distance = other_distance_col in df.columns
                    
                    print(f"\nProcessing '{weather_feature}':")
                    print(f"- Main column '{main_col}': {has_main}")
                    print(f"- Other column '{other_col}': {has_other}")
                    print(f"- Other Distance column '{other_distance_col}': {has_other_distance}")
                    
                    # Initialize merge statistics for this feature
                    initial_main_missing = 0
                    initial_other_missing = 0
                    values_merged = 0
                    final_main_missing = 0
                    
                    # Handle merging logic
                    if not has_main and not has_other:
                        print(f"  No columns found for '{weather_feature}'. Skipping.")
                        logger.info(f"{weather_feature}: No columns found")
                        
                    elif not has_main and has_other:
                        print(f"  Only 'Other' column found. Renaming '{other_col}' to '{main_col}'.")
                        initial_other_missing = df[other_col].isna().sum()
                        df = df.rename(columns={other_col: main_col})
                        final_main_missing = df[main_col].isna().sum()
                        has_main = True
                        has_other = False
                        logger.info(f"{weather_feature}: Renamed Other column to main - Missing: {final_main_missing}")
                        
                    elif has_main and not has_other:
                        print(f"  Only main column found. No merging needed.")
                        initial_main_missing = df[main_col].isna().sum()
                        final_main_missing = initial_main_missing
                        logger.info(f"{weather_feature}: Only main column - Missing: {final_main_missing}")
                        
                    else:
                        # Both columns exist - proceed with merging logic
                        print(f"  Both columns found. Proceeding with merge logic.")
                        
                        # Count initial missing values in each column
                        initial_main_missing = df[main_col].isna().sum()
                        initial_other_missing = df[other_col].isna().sum()
                        
                        print(f"  Before merge:")
                        print(f"  - '{main_col}' missing values: {initial_main_missing}")
                        print(f"  - '{other_col}' missing values: {initial_other_missing}")
                        
                        # Create a mask for rows where main column is missing but other column is not
                        merge_mask = df[main_col].isna() & df[other_col].notna()
                        merge_count = merge_mask.sum()
                        
                        if merge_count > 0:
                            print(f"  Merging {merge_count} values from '{other_col}' to '{main_col}'")
                            # Fill missing main column values with other column values
                            df.loc[merge_mask, main_col] = df.loc[merge_mask, other_col]
                            values_merged = merge_count
                            total_merges += merge_count
                        else:
                            print(f"  No values to merge (no rows where {main_col} is missing but {other_col} has values)")
                            values_merged = 0
                        
                        # Count final missing values
                        final_main_missing = df[main_col].isna().sum()
                        
                        print(f"  After merge:")
                        print(f"  - '{main_col}' missing values: {final_main_missing}")
                        print(f"  - Values successfully merged: {values_merged}")
                    
                    # Log the merge results for this feature
                    logger.info(f"{weather_feature} - Main missing: {initial_main_missing}, Other missing: {initial_other_missing}, Values merged: {values_merged}, Final missing: {final_main_missing}")
                    
                    # Drop unwanted columns for this feature
                    columns_to_drop = []
                    if has_other:
                        columns_to_drop.append(other_col)
                    if has_other_distance:
                        columns_to_drop.append(other_distance_col)
                    
                    if columns_to_drop:
                        df = df.drop(columns=columns_to_drop)
                        total_drops += len(columns_to_drop)
                        print(f"  Dropped columns: {columns_to_drop}")
                    else:
                        print(f"  No columns to drop for '{weather_feature}'")
                
                # Summary logging
                print(f"\nWeather column merging summary:")
                print(f"- Total features processed: {len(WEATHER_COLS_TO_MERGE)}")
                print(f"- Total values merged: {total_merges}")
                print(f"- Total columns dropped: {total_drops}")
                
                logger.info(f"Weather column merging completed - Features processed: {len(WEATHER_COLS_TO_MERGE)}, Total merges: {total_merges}, Columns dropped: {total_drops}")
                logger.info(f"Final dataframe shape: {df.shape}")
                
                return df
                
            except Exception as e:
                print(f"Error merging weather columns: {e}")
                logger.error(f"Error merging weather columns: {str(e)}")
                return dataframe  # Return original dataframe on error

    def handle_missing_values(self, dataframe=None, month_id=None):
        """
        Handle missing values in preprocessed dataframes with enhanced imputation strategy.
        Now includes initial data completeness analysis and weather column filtering.
        
        Processes the provided dataframe and handles missing values:
        - Drop weather columns that exceed the missing value threshold
        - Drop rows where all remaining weather condition columns have missing values
        - Drop rows where differenceInMinutes or cancelled are None
        - Fill missing values in trainStopping and commercialStop with False
        - Use variable-specific imputation for weather columns:
        - Zero for precipitation and snow metrics
        - Median for all other columns that still have missing values
        
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
        
        # Use the logging context manager
        with self.get_logger("handle_missing_values.log", "missing_values_handler", month_id) as logger:
            print(f"Processing dataframe with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Processing dataframe with {len(df)} rows and {len(df.columns)} columns")
            
            if df.empty:
                print("Warning: Empty dataframe")
                logger.warning("Empty dataframe provided")
                return df
            
            # NEW: DATA COMPLETENESS ANALYSIS
            print(f"\n--- DATA COMPLETENESS ANALYSIS ---")
            logger.info("=== Data Completeness Analysis ===")
            
            # Check completeness for important weather conditions
            available_important_cols = [col for col in self.important_conditions if col in df.columns]
            if available_important_cols:
                # Count rows where ALL important weather conditions are filled
                important_weather_complete = df[available_important_cols].notna().all(axis=1).sum()
                important_weather_complete_pct = (important_weather_complete / len(df)) * 100
                
                print(f"Rows with ALL important weather conditions filled: {important_weather_complete} / {len(df)} ({important_weather_complete_pct:.2f}%)")
                logger.info(f"Rows with ALL important weather conditions filled: {important_weather_complete} / {len(df)} ({important_weather_complete_pct:.2f}%)")
            else:
                print("No important weather conditions found in the dataframe")
                logger.info("No important weather conditions found in the dataframe")
            
            # Check overall completeness (no missing data anywhere)
            completely_filled_rows = df.notna().all(axis=1).sum()
            completely_filled_pct = (completely_filled_rows / len(df)) * 100
            
            print(f"Completely filled rows (no missing data): {completely_filled_rows} / {len(df)} ({completely_filled_pct:.2f}%)")
            logger.info(f"Completely filled rows (no missing data): {completely_filled_rows} / {len(df)} ({completely_filled_pct:.2f}%)")
            
            # Additional statistics: show missing data per column for important weather conditions
            if available_important_cols:
                print(f"\nMissing data breakdown for important weather conditions:")
                logger.info("Missing data breakdown for important weather conditions:")
                for col in available_important_cols:
                    missing_count = df[col].isna().sum()
                    missing_pct = (missing_count / len(df)) * 100
                    print(f"  - {col}: {missing_count} missing ({missing_pct:.2f}%)")
                    logger.info(f"  - {col}: {missing_count} missing ({missing_pct:.2f}%)")
            
            # Show total missing values across all columns
            total_missing = df.isna().sum().sum()
            total_cells = len(df) * len(df.columns)
            total_missing_pct = (total_missing / total_cells) * 100
            
            print(f"\nOverall missing data statistics:")
            print(f"Total missing values: {total_missing} / {total_cells} cells ({total_missing_pct:.2f}%)")
            logger.info(f"Total missing values: {total_missing} / {total_cells} cells ({total_missing_pct:.2f}%)")
            
            print("--- END COMPLETENESS ANALYSIS ---\n")
            logger.info("=== End Completeness Analysis ===")
            # END NEW COMPLETENESS ANALYSIS
            
            # Count rows before cleaning
            original_row_count = len(df)
            original_col_count = len(df.columns)
            
            # Fill missing values in trainStopping and commercialStop with 0 (since they're now numeric)
            for col in NON_NUMERIC_FEATURES:
                if col in df.columns:
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        # Calculate percentage of missing values
                        percentage = (nulls / len(df)) * 100
                        df[col] = df[col].fillna(0)  # Fill with 0 instead of False since they're now numeric
                        print(f"- Filled {nulls} missing values in '{col}' with 0 ({percentage:.2f}%)")
                        logger.info(f"Filled {nulls} missing values in '{col}' with 0 ({percentage:.2f}%)")
                    else:
                        print(f"- Filled {nulls} missing values in '{col}' with 0")
                        logger.info(f"Filled {nulls} missing values in '{col}' with 0")

            # Step 2: Check required columns (differenceInMinutes and cancelled)
            required_cols = ['differenceInMinutes', 'differenceInMinutes_offset', 'trainDelayed', 'cancelled']
            available_required_cols = [col for col in required_cols if col in df.columns]
            
            if available_required_cols:
                print(f"Checking for missing values in required columns: {available_required_cols}")
                logger.info(f"Checking for missing values in required columns: {available_required_cols}")
                # Store the count before dropping rows
                before_required_drop = len(df)
                # Drop rows where any of the required columns are None/NaN
                df = df.dropna(subset=available_required_cols)
                # Calculate dropped rows
                dropped_required = before_required_drop - len(df)
                dropped_percentage = (dropped_required / before_required_drop) * 100 if before_required_drop > 0 else 0
                print(f"- Dropped {dropped_required} rows with missing values in required columns ({dropped_percentage:.2f}%)")
                logger.info(f"Dropped {dropped_required} rows with missing values in required columns ({dropped_percentage:.2f}%)")
            else:
                print("Warning: Required columns (differenceInMinutes, cancelled) not found in dataframe")
                logger.warning("Required columns (differenceInMinutes, cancelled) not found in dataframe")
                dropped_required = 0
            
            # Step 3: Handle weather condition columns
            print(f"\n--- WEATHER COLUMN FILTERING ---")
            logger.info("=== Weather Column Filtering ===")
            
            # NEW: Drop weather columns that exceed the missing value threshold
            from config.const import WEATHER_MISSING_THRESHOLD
            
            # Identify all weather-related columns (not just important ones)
            all_weather_cols = [col for col in df.columns if any(weather_condition in col for weather_condition in self.important_conditions)]
            
            if all_weather_cols:
                print(f"Checking missing value threshold for {len(all_weather_cols)} weather-related columns...")
                logger.info(f"Checking missing value threshold for {len(all_weather_cols)} weather-related columns...")
                
                columns_to_drop = []
                columns_kept = []
                
                for col in all_weather_cols:
                    missing_count = df[col].isna().sum()
                    missing_pct = (missing_count / len(df)) * 100
                    
                    if missing_pct > WEATHER_MISSING_THRESHOLD:
                        columns_to_drop.append(col)
                        print(f"  - DROPPING '{col}': {missing_count} missing ({missing_pct:.2f}% > {WEATHER_MISSING_THRESHOLD}%)")
                        logger.info(f"DROPPING '{col}': {missing_pct:.2f}% missing > {WEATHER_MISSING_THRESHOLD}% threshold")
                    else:
                        columns_kept.append(col)
                        print(f"  - KEEPING '{col}': {missing_count} missing ({missing_pct:.2f}% <= {WEATHER_MISSING_THRESHOLD}%)")
                        logger.info(f"KEEPING '{col}': {missing_pct:.2f}% missing <= {WEATHER_MISSING_THRESHOLD}% threshold")
                
                # Drop columns that exceed the threshold
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop)
                    print(f"\nDropped {len(columns_to_drop)} weather columns exceeding {WEATHER_MISSING_THRESHOLD}% missing threshold")
                    logger.info(f"Dropped {len(columns_to_drop)} weather columns exceeding {WEATHER_MISSING_THRESHOLD}% missing threshold")
                    logger.info(f"Dropped columns: {columns_to_drop}")
                else:
                    print(f"\nNo weather columns exceeded the {WEATHER_MISSING_THRESHOLD}% missing threshold")
                    logger.info(f"No weather columns exceeded the {WEATHER_MISSING_THRESHOLD}% missing threshold")
                    
                print(f"Kept {len(columns_kept)} weather columns within threshold")
                logger.info(f"Kept {len(columns_kept)} weather columns within threshold")
            else:
                print("No weather-related columns found in dataframe")
                logger.info("No weather-related columns found in dataframe")
            
            print("--- END WEATHER COLUMN FILTERING ---\n")
            logger.info("=== End Weather Column Filtering ===")
            
            # Update the available important columns list after dropping columns
            available_important_cols = [col for col in self.important_conditions if col in df.columns]
            
            if not available_important_cols:
                print("Warning: None of the specified important weather conditions found in the dataframe after filtering")
                logger.warning("None of the specified important weather conditions found in the dataframe after filtering")
                return df
            
            print(f"Found {len(available_important_cols)} important weather condition columns after filtering: {available_important_cols}")
            logger.info(f"Found {len(available_important_cols)} important weather condition columns after filtering: {available_important_cols}")
            
            # Store count before dropping weather condition rows
            before_weather_drop = len(df)
            
            # Drop rows where ALL of the remaining important weather conditions are missing
            # (Keep rows with at least one of the specified conditions)
            df = df.dropna(subset=available_important_cols, how='all')
            
            # Count how many rows were dropped due to weather conditions
            dropped_weather = before_weather_drop - len(df)
            weather_dropped_percentage = (dropped_weather / before_weather_drop) * 100 if before_weather_drop > 0 else 0
            
            if dropped_weather > 0:
                print(f"- Dropped {dropped_weather} rows with missing all weather conditions ({weather_dropped_percentage:.2f}%)")
                logger.info(f"Dropped {dropped_weather} rows with missing all weather conditions ({weather_dropped_percentage:.2f}%)")
            else:
                logger.info(f"Dropped {dropped_weather} rows with missing all weather conditions ({weather_dropped_percentage:.2f}%)")
            
            # ===== ENHANCED MISSING VALUE HANDLING (WITHOUT INDICATORS) =====
            
            # Group weather variables by appropriate imputation method
            zero_fill_cols = ['Precipitation amount', 'Precipitation intensity', 'Snow depth']
            
            # 1. Zero imputation for precipitation and snow metrics
            for col in zero_fill_cols:
                if col in df.columns:
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        # Calculate percentage of missing values
                        percentage = (nulls / len(df)) * 100
                        # Apply zero imputation
                        df[col] = df[col].fillna(0)
                        print(f"- Filled {nulls} missing values in '{col}' with 0 ({percentage:.2f}%)")
                        logger.info(f"Filled {nulls} missing values in '{col}' with 0 ({percentage:.2f}%)")
            
            # 2. Median imputation for all remaining columns with missing values
            # Find all columns that still have missing values after zero imputation
            remaining_cols_with_na = [col for col in df.columns if df[col].isna().sum() > 0]
            
            for col in remaining_cols_with_na:
                nulls = df[col].isna().sum()
                if nulls > 0:
                    # Calculate percentage of missing values
                    percentage = (nulls / len(df)) * 100
                    # Apply median imputation to all remaining columns
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    print(f"- Filled {nulls} missing values in '{col}' with median: {median_value:.2f} ({percentage:.2f}%)")
                    logger.info(f"Filled {nulls} missing values in '{col}' with median: {median_value:.2f} ({percentage:.2f}%)")
            
            # Count total rows and columns dropped
            total_rows_dropped = original_row_count - len(df)
            total_cols_dropped = original_col_count - len(df.columns)
            total_dropped_percentage = (total_rows_dropped / original_row_count) * 100 if original_row_count > 0 else 0
            
            # Report the results
            print(f"\nMissing values handling complete:")
            print(f"- Original shape: {original_row_count} rows × {original_col_count} columns")
            print(f"- Columns dropped (weather threshold): {total_cols_dropped}")
            print(f"- Rows dropped due to missing required columns: {dropped_required}")
            print(f"- Rows dropped due to missing all weather conditions: {dropped_weather}")
            print(f"- Total rows dropped: {total_rows_dropped} ({total_dropped_percentage:.2f}%)")
            print(f"- Final shape: {len(df)} rows × {len(df.columns)} columns")
            
            # Log the summary
            logger.info(f"Missing values handling complete:")
            logger.info(f"Original shape: {original_row_count} rows × {original_col_count} columns")
            logger.info(f"Columns dropped (weather threshold): {total_cols_dropped}")
            logger.info(f"Rows dropped due to missing required columns: {dropped_required}")
            logger.info(f"Rows dropped due to missing all weather conditions: {dropped_weather}")
            logger.info(f"Total rows dropped: {total_rows_dropped} ({total_dropped_percentage:.2f}%)")
            logger.info(f"Final shape: {len(df)} rows × {len(df.columns)} columns")
            
            # Calculate percentage of data retained
            if original_row_count > 0:
                retention_percentage = (len(df) / original_row_count) * 100
                print(f"- Data retention: {retention_percentage:.2f}%")
                logger.info(f"Data retention: {retention_percentage:.2f}%")
                
            # Additional statistics on the remaining important columns
            for col in available_important_cols:
                non_null_count = df[col].count()
                null_count = len(df) - non_null_count
                null_percentage = (null_count / len(df) * 100) if len(df) > 0 else 0
                print(f"  - {col}: {non_null_count} non-null values ({100-null_percentage:.2f}% complete)")
                logger.info(f"{col}: {non_null_count} non-null values ({100-null_percentage:.2f}% complete)")
            
            # Additional statistics for trainStopping and commercialStop if they exist
            boolean_cols = ['trainStopping', 'commercialStop']
            available_boolean_cols = [col for col in boolean_cols if col in df.columns]
            
            if available_boolean_cols:
                print("\nBoolean columns statistics:")
                logger.info("Boolean columns statistics:")
                for col in available_boolean_cols:
                    true_count = df[col].sum()
                    true_percentage = (true_count / len(df) * 100) if len(df) > 0 else 0
                    print(f"  - {col}: {true_count} True values ({true_percentage:.2f}% True)")
                    logger.info(f"{col}: {true_count} True values ({true_percentage:.2f}% True)")
            
            return df
    
    def remove_duplicates(self, dataframe=None, month_id=None):
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
        
        # Use the logging context manager
        with self.get_logger("remove_duplicates.log", "remove_duplicates", month_id) as logger:
            logger.info(f"Processing dataframe with {len(df)} rows and {len(df.columns)} columns")
            
            # Count rows before deduplication
            original_row_count = len(df)
            
            # Remove duplicate rows
            df_deduplicated = df.drop_duplicates()
            
            # Count how many rows were removed
            removed_duplicates = original_row_count - len(df_deduplicated)
            
            # Calculate percentage of data retained
            if original_row_count > 0:
                retention_percentage = (len(df_deduplicated) / original_row_count) * 100
                duplicate_percentage = 100 - retention_percentage
            else:
                retention_percentage = 0
                duplicate_percentage = 0
            
            # Report the results to console
            print(f"Duplicate removal complete:")
            print(f"- Original row count: {original_row_count}")
            print(f"- Duplicate rows removed: {removed_duplicates}")
            print(f"- Remaining rows: {len(df_deduplicated)}")
            print(f"- Data retention: {retention_percentage:.2f}%")
            print(f"- Duplicate percentage: {duplicate_percentage:.2f}%")
            
            # Log the essential summary information
            logger.info(f"Original row count: {original_row_count}")
            logger.info(f"Duplicate rows removed: {removed_duplicates}")
            logger.info(f"Remaining rows: {len(df_deduplicated)}")
            logger.info(f"Data retention: {retention_percentage:.2f}%")
            logger.info(f"Duplicate percentage: {duplicate_percentage:.2f}%")
        
        return df_deduplicated
    
    def scale_numeric_columns(self, dataframe=None, month_id=None):
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
        
        # Use the logging context manager
        with self.get_logger("scale_numeric_columns.log", "scale_numeric", month_id) as logger:
            logger.info(f"Processing dataframe with {len(df)} rows and {len(df.columns)} columns")
            
            try:
                # Identify all numeric columns
                all_numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if len(all_numeric_columns) == 0:
                    print("Warning: No numeric columns found in the dataframe")
                    logger.warning("No numeric columns found in the dataframe")
                    return df
                
                # Exclude both existing excluded columns and the new missing indicators
                columns_to_scale = [col for col in all_numeric_columns 
                                if col not in VALID_TARGET_FEATURES
                                and not col.endswith('_missing')]
                            
                # Report which columns will be scaled and which ones are excluded
                excluded_columns = set(all_numeric_columns) - set(columns_to_scale)
                
                # Console output
                print(f"Found {len(all_numeric_columns)} numeric columns.")
                print(f"Excluding from scaling: {list(excluded_columns)}")
                print(f"Columns to scale: {columns_to_scale}")
                
                # Log the same information
                logger.info(f"Found {len(all_numeric_columns)} numeric columns.")
                logger.info(f"Excluding from scaling: {list(excluded_columns)}")
                logger.info(f"Columns to scale: {columns_to_scale}")
                
                if not columns_to_scale:
                    print("No columns to scale after exclusions. Returning original dataframe.")
                    logger.info("No columns to scale after exclusions. Returning original dataframe.")
                    return df
                
                # Initialize the scaler
                scaler = StandardScaler()
                
                # Scale only the selected numeric columns
                df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                
                # Console and log success message
                success_msg = f"Successfully scaled {len(columns_to_scale)} numeric columns"
                print(success_msg)
                logger.info(success_msg)
                
                return df
                
            except Exception as e:
                error_msg = f"Error scaling numeric columns: {e}"
                print(error_msg)
                logger.error(error_msg)
                return dataframe  # Return original dataframe on error
        
    def add_train_delayed_feature(self, dataframe=None):
        """
        Add a binary column 'trainDelayed' based on the configured target column.
        
        The column will be:
        - True when target column > 0 (train is delayed)
        - False when target column <= 0 (train is on time or early)
        
        The target column is determined by TRAIN_DELAYED_TARGET_COLUMN constant.
        
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
        print(f"Using '{TRAIN_DELAYED_TARGET_COLUMN}' as the basis for trainDelayed calculation")
    
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        try:
            # Check if the target column exists
            if TRAIN_DELAYED_TARGET_COLUMN not in df.columns:
                print(f"Error: '{TRAIN_DELAYED_TARGET_COLUMN}' column not found in dataframe")
                print(f"Available columns: {list(df.columns)}")
                return df
            
            # Create trainDelayed column based on the configured target column
            df['trainDelayed'] = df[TRAIN_DELAYED_TARGET_COLUMN] > TRAIN_DELAY_MINUTES
            
            # Reorder columns to place trainDelayed after the target column
            cols = list(df.columns)
            # Remove trainDelayed from its current position
            cols.remove('trainDelayed')
            # Find the position of the target column
            if TRAIN_DELAYED_TARGET_COLUMN in cols:
                target_idx = cols.index(TRAIN_DELAYED_TARGET_COLUMN)
                # Insert trainDelayed after the target column
                cols.insert(target_idx + 1, 'trainDelayed')
                # Rearrange the dataframe columns
                df = df[cols]
            else:
                # If target column not found in remaining columns, just append trainDelayed
                cols.append('trainDelayed')
                df = df[cols]
            
            print(f"Successfully added 'trainDelayed' column based on '{TRAIN_DELAYED_TARGET_COLUMN}'")
            print(f"Number of delayed trains: {df['trainDelayed'].sum()} ({df['trainDelayed'].mean() * 100:.2f}%)")
            
            return df
            
        except Exception as e:
            print(f"Error adding 'trainDelayed' column: {e}")
            return dataframe  # Return original dataframe on error
        
    def select_target_feature(self, dataframe=None, target_feature=None):
        """
        Select one of the target features and optionally drop train-specific features.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
        target_feature : str
            The feature to keep as target.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with only the selected target feature retained and 
            optional train features dropped based on configuration.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Selecting target feature '{target_feature}' from dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Validate target feature
        if target_feature is None:
            print("Error: target_feature must be specified")
            return df
        
        if target_feature not in VALID_TARGET_FEATURES:
            print(f"Error: Invalid target feature '{target_feature}'. Must be one of: {VALID_TARGET_FEATURES}")
            return df
        
        if target_feature not in df.columns:
            print(f"Error: Target feature '{target_feature}' not found in dataframe columns")
            return df
        
        # Drop other target features, keeping only the selected one
        other_targets = [col for col in VALID_TARGET_FEATURES if col != target_feature and col in df.columns]
        
        if other_targets:
            print(f"Dropping other target features: {other_targets}")
            df = df.drop(columns=other_targets)
        
        print(f"Kept target feature: '{target_feature}'")
        
        # NEW: Drop train features if configured
        if DROP_TRAIN_FEATURES:
            features_to_drop = [col for col in NON_NUMERIC_FEATURES if col in df.columns]
            if features_to_drop:
                df = df.drop(columns=features_to_drop)
                print(f"Dropped train features as configured: {features_to_drop}")
            else:
                print("No train features found to drop")
        
        print(f"Final dataframe shape: {df.shape}")
        return df
    
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
    
    def split_month_dataset(self, month_id, test_size=0.3, random_state=42):
        """
        Split a processed month's dataset into training and testing sets and save them separately.
        Automatically applies the configured resampling method for severely imbalanced categorical targets.
        
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
            target_options = VALID_TARGET_FEATURES
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
            # stratify parameterensures that the class distribution is preserved in both the training and test sets
            # If target is continuous (like differenceInMinutes), we can't use stratify
            use_stratify = False
            if target_column in CATEGORIAL_TARGET_FEATURES:
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
            
            # Store original training set size for comparison
            original_train_size = len(y_train)
            
            # ========== APPLY CONFIGURED RESAMPLING METHOD ==========
            print(f"\nApplying resampling method: {RESAMPLING_METHOD}")
            
            if RESAMPLING_METHOD == "SMOTE_TOMEK":
                X_train, y_train, resampling_applied, resampling_info = self.apply_smote_tomek_resampling(
                    X_train, y_train, target_column, IMBALANCE_THRESHOLD, random_state
                )
                resampling_method_used = "SMOTE-Tomek"
                
            elif RESAMPLING_METHOD == "EDITED_NEAREST_NEIGHBORS":
                X_train, y_train, resampling_applied, resampling_info = self.apply_edited_nearest_neighbors_resampling(
                    X_train, y_train, target_column, IMBALANCE_THRESHOLD, random_state, n_neighbors=ENN_N_NEIGHBORS
                )
                resampling_method_used = "EditedNearestNeighbors"
                
            elif RESAMPLING_METHOD == "NONE":
                print("No resampling method configured - using original data")
                resampling_applied = False
                resampling_method_used = "None"
                resampling_info = {
                    "original_size": original_train_size,
                    "final_size": original_train_size,
                    "samples_changed": 0,
                    "original_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "final_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "threshold_used": IMBALANCE_THRESHOLD,
                    "method": "none"
                }
                
            else:
                print(f"Warning: Unknown resampling method '{RESAMPLING_METHOD}'. Using original data.")
                resampling_applied = False
                resampling_method_used = "Unknown (fallback to None)"
                resampling_info = {
                    "original_size": original_train_size,
                    "final_size": original_train_size,
                    "samples_changed": 0,
                    "original_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "final_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "threshold_used": IMBALANCE_THRESHOLD,
                    "method": "fallback_none"
                }
            
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
            
            # Print and log distribution statistics
            print("\nFinal Distribution Statistics:")
            
            # Use the logging context manager for distribution statistics
            with self.get_logger("split_dataset_distribution.log", "split_distribution", month_id) as logger:
                logger.info(f"Dataset split completed - Train size: {len(train_df)}, Test size: {len(test_df)}")
                logger.info(f"Target column: {target_column}")
                logger.info(f"Stratified split used: {use_stratify}")
                logger.info(f"Resampling method configured: {RESAMPLING_METHOD}")
                logger.info(f"Resampling method used: {resampling_method_used}")
                logger.info(f"Resampling applied: {resampling_applied}")
                if resampling_applied:
                    if RESAMPLING_METHOD == "SMOTE_TOMEK":
                        logger.info(f"Original train size: {original_train_size}, Final train size: {len(train_df)}")
                        logger.info(f"Samples added: {resampling_info.get('samples_added', 0)}")
                    elif RESAMPLING_METHOD == "EDITED_NEAREST_NEIGHBORS":
                        logger.info(f"Original train size: {original_train_size}, Final train size: {len(train_df)}")
                        logger.info(f"Samples removed: {resampling_info.get('samples_removed', 0)}")
                        logger.info(f"N_neighbors used: {resampling_info.get('n_neighbors', ENN_N_NEIGHBORS)}")
                
                # For categorical targets, show the distribution in percentages
                if target_column in ['trainDelayed', 'cancelled']:
                    print("\nOriginal Distribution (%):")
                    original_dist = df[target_column].value_counts(normalize=True) * 100
                    print(original_dist)
                    
                    print("\nFinal Training Set Distribution (%):")
                    train_dist = y_train.value_counts(normalize=True) * 100
                    print(train_dist)
                    
                    print("\nTest Set Distribution (%):")
                    test_dist = y_test.value_counts(normalize=True) * 100
                    print(test_dist)
                    
                    # Log the categorical distributions
                    logger.info("=== Categorical Target Distribution Analysis ===")
                    logger.info("Original Distribution (%):")
                    for label, percentage in original_dist.items():
                        logger.info(f"  {label}: {percentage:.2f}%")
                    
                    logger.info("Final Training Set Distribution (%):")
                    for label, percentage in train_dist.items():
                        logger.info(f"  {label}: {percentage:.2f}%")
                    
                    logger.info("Test Set Distribution (%):")
                    for label, percentage in test_dist.items():
                        logger.info(f"  {label}: {percentage:.2f}%")
                    
                    # Calculate and log distribution differences
                    logger.info("Distribution Differences (Final Train vs Original):")
                    for label in original_dist.index:
                        if label in train_dist.index:
                            diff = train_dist[label] - original_dist[label]
                            logger.info(f"  {label}: {diff:+.2f} percentage points")
                    
                    # Log resampling-specific info with method identification
                    if resampling_applied:
                        if RESAMPLING_METHOD == "SMOTE_TOMEK":
                            logger.info("=== SMOTE-Tomek Resampling Applied ===")
                            logger.info(f"Reason: Minority class below {IMBALANCE_THRESHOLD}% threshold")
                            logger.info(f"Training samples added: {resampling_info['samples_added']}")
                        elif RESAMPLING_METHOD == "EDITED_NEAREST_NEIGHBORS":
                            logger.info("=== EditedNearestNeighbors Undersampling Applied ===")
                            logger.info(f"Reason: Minority class below {IMBALANCE_THRESHOLD}% threshold")
                            logger.info(f"Training samples removed: {resampling_info['samples_removed']}")
                            logger.info(f"N_neighbors parameter: {resampling_info['n_neighbors']}")
                        
                        # Log final distribution for any resampling method
                        for label, percentage in resampling_info["final_distribution"].items():
                            final_count = (percentage / 100) * len(train_df)
                            logger.info(f"  {label}: {final_count:.0f} samples ({percentage:.2f}%)")
                    
                else:
                    # For continuous targets like differenceInMinutes, show basic stats
                    original_mean = df[target_column].mean()
                    original_std = df[target_column].std()
                    train_mean = y_train.mean()
                    train_std = y_train.std()
                    test_mean = y_test.mean()
                    test_std = y_test.std()
                    
                    print("\nOriginal Distribution:")
                    print(f"Mean: {original_mean:.2f}, Std: {original_std:.2f}")
                    
                    print("\nTraining Set Distribution:")
                    print(f"Mean: {train_mean:.2f}, Std: {train_std:.2f}")
                    
                    print("\nTest Set Distribution:")
                    print(f"Mean: {test_mean:.2f}, Std: {test_std:.2f}")
                    
                    # Log the continuous distributions
                    logger.info("=== Continuous Target Distribution Analysis ===")
                    logger.info(f"Original Distribution - Mean: {original_mean:.4f}, Std: {original_std:.4f}")
                    logger.info(f"Training Set Distribution - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
                    logger.info(f"Test Set Distribution - Mean: {test_mean:.4f}, Std: {test_std:.4f}")
                    
                    # Calculate and log distribution differences
                    mean_diff_train = train_mean - original_mean
                    std_diff_train = train_std - original_std
                    mean_diff_test = test_mean - original_mean
                    std_diff_test = test_std - original_std
                    
                    logger.info("Distribution Differences:")
                    logger.info(f"  Train vs Original - Mean diff: {mean_diff_train:+.4f}, Std diff: {std_diff_train:+.4f}")
                    logger.info(f"  Test vs Original - Mean diff: {mean_diff_test:+.4f}, Std diff: {std_diff_test:+.4f}")
                    
                    # Additional statistics for continuous targets
                    logger.info("Additional Statistics:")
                    logger.info(f"  Original - Min: {df[target_column].min():.4f}, Max: {df[target_column].max():.4f}")
                    logger.info(f"  Training - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}")
                    logger.info(f"  Test - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}")
            
            # Return summary
            return {
                "success": True,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "original_train_size": original_train_size,
                "train_path": train_path,
                "test_path": test_path,
                "target_column": target_column,
                "stratified": use_stratify,
                "resampling_method_configured": RESAMPLING_METHOD,
                "resampling_method_used": resampling_method_used,
                "resampling_applied": resampling_applied,
                "imbalance_threshold": IMBALANCE_THRESHOLD,
                "resampling_info": resampling_info
            }
            
        except Exception as e:
            print(f"Error splitting dataset for {month_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def train_decision_tree_with_randomized_search_cv(self, month_id, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train a Decision Tree classifier with hyperparameter tuning using RandomizedSearchCV.
        Includes SHAP analysis for enhanced model interpretability.
        Updated to include sample weights based on delay magnitude.
        Now uses comprehensive evaluation method for consistent metrics.
        
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
                from config.const import DECISION_TREE_PARAM_DISTRIBUTIONS
                param_distributions = DECISION_TREE_PARAM_DISTRIBUTIONS
            
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
            target_options = VALID_TARGET_FEATURES
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
            if target_column in REGRESSION_PROBLEM:
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                # NEW: Create sample weights for classification if delay info is available
                sample_weights = None
                if WEIGHT_DELAY_COLUMN in train_df.columns:
                    print("Using weighted samples based on delay magnitude for randomized search")
                    # Create sample weights based on delay magnitude
                    delay_col = WEIGHT_DELAY_COLUMN
                    sample_weights = np.ones(len(y_train))
                    
                    # Get delay values for each training sample
                    delays = train_df[delay_col].values
                    
                    # Apply weights - higher delays get higher weights
                    delayed_idx = (delays > TRAIN_DELAY_MINUTES)
                    if np.any(delayed_idx):
                        # Normalize delay values by mean positive delay
                        mean_delay = delays[delayed_idx].mean()
                        # Use configured maximum weight
                        sample_weights[delayed_idx] = np.minimum(MAX_SAMPLE_WEIGHT_CLASSIFICATION, 1 + delays[delayed_idx]/mean_delay)
                    
                    print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                
                # Create proper CV strategy for classification
                cv_strategy = StratifiedKFold(
                    n_splits=cv,       
                    shuffle=True,
                    random_state=random_state
                )

                # Initialize base classifier
                dt = DecisionTreeClassifier(random_state=random_state)
                
                print(f"Starting RandomizedSearchCV with {n_iter} iterations and {cv}-fold cross-validation...")
            
                # Run RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    dt, param_distributions, n_iter=n_iter, cv=cv_strategy, 
                    scoring=SCORE_METRIC, random_state=random_state, n_jobs=-1
                )
                
                # Fit RandomizedSearchCV with sample weights if available
                if sample_weights is not None:
                    print("Training RandomizedSearchCV with sample weights")
                    # Note: RandomizedSearchCV will automatically handle sample weights for cross-validation
                    random_search.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    random_search.fit(X_train, y_train)
                
                best_params = random_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
                
                # Train model with best parameters
                best_dt = DecisionTreeClassifier(**best_params, random_state=random_state)
                
                # Fit the final model with sample weights if available
                if sample_weights is not None:
                    print("Training final model with sample weights")
                    best_dt.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    best_dt.fit(X_train, y_train)
                
                # === NEW COMPREHENSIVE EVALUATION METHOD ===
                evaluation_result = self.evaluate_model_comprehensive(
                    model=best_dt,
                    X_test=X_test,
                    y_test=y_test,
                    model_name="Decision Tree with RandomizedSearchCV",
                    month_id=month_id,
                    output_dir=self.randomized_search_dir,
                    target_column=target_column,
                    random_search_obj=random_search,  # Pass the RandomizedSearchCV object
                    is_classification=is_classification
                )
                
                if not evaluation_result["success"]:
                    return {
                        "success": False,
                        "error": f"Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
                    }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': best_dt.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # ========== SHAP ANALYSIS ==========
                print("\nPerforming SHAP analysis on the RandomizedSearchCV tuned model...")
                
                shap_result = self.analyze_model_with_shap(
                    model=best_dt,
                    X_test=X_test,
                    y_test=y_test,
                    model_type='classification',
                    month_id=month_id,
                    output_dir=self.randomized_search_dir,
                    target_column=target_column,
                    max_samples=1000,
                    random_state=random_state,
                    model_name="decision_tree_randomized_search",
                    baseline_data=train_df  # Use training data for better baseline calculation
                )
                
                if shap_result.get("success", False):
                    print("SHAP analysis completed successfully for RandomizedSearchCV model!")
                    
                    # Compare with standard importance if SHAP was successful
                    if "shap_importance_path" in shap_result:
                        print("\n" + "-"*60)
                        print("COMPARISON: Standard vs SHAP Feature Importance (RandomizedSearchCV Model)")
                        print("-"*60)
                        
                        try:
                            # Load SHAP importance for comparison
                            shap_importance = pd.read_csv(shap_result["shap_importance_path"])
                            
                            # Merge the two importance measures
                            comparison = feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 
                                            'SHAP_Percentage_Points', 'SHAP_Abs_Percentage_Points', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Top 10 features by Standard Importance vs SHAP Importance:")
                            for _, row in comparison.head(10).iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Abs_Percentage_Points'] if pd.notna(row['SHAP_Abs_Percentage_Points']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>5.2f}pp {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                        except Exception as e:
                            print(f"Could not perform comparison: {e}")
                    
                else:
                    print(f"SHAP analysis failed: {shap_result.get('error', 'Unknown error')}")
                
                print("="*60)
                
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
                    
                    # NEW: Save sample weights information if used
                    if sample_weights is not None:
                        weights_filename = f"sample_weights_info_{month_id}.txt"
                        weights_path = os.path.join(self.randomized_search_dir, weights_filename)
                        with open(weights_path, 'w') as f:
                            f.write(f"Sample Weights Information - {month_id}\n")
                            f.write("="*40 + "\n")
                            f.write(f"Used sample weights: Yes\n")
                            f.write(f"Weight range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]\n")
                            f.write(f"Mean weight: {sample_weights.mean():.2f}\n")
                            f.write(f"Standard deviation: {sample_weights.std():.2f}\n")
                            f.write(f"Number of weighted samples: {(sample_weights > 1.0).sum()}\n")
                            f.write(f"Max weight constant used: {MAX_SAMPLE_WEIGHT_CLASSIFICATION}\n")
                        print(f"Sample weights info saved to {weights_path}")
                    
                    # Also compare to baseline model
                    print("\nComparison with baseline model:")
                    baseline_metrics_file = os.path.join(self.decision_tree_dir, f"model_metrics_{month_id}.csv")
                    if os.path.exists(baseline_metrics_file):
                        baseline_metrics = pd.read_csv(baseline_metrics_file)
                        if 'accuracy' in baseline_metrics.columns:
                            baseline_accuracy = baseline_metrics['accuracy'].values[0]
                            current_accuracy = evaluation_result.get('accuracy', 0)
                            print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
                            print(f"RandomizedSearchCV model accuracy: {current_accuracy:.4f}")
                            improvement = ((current_accuracy - baseline_accuracy) / baseline_accuracy) * 100
                            print(f"Improvement: {improvement:.2f}%")
                    
                    return {
                        "success": True,
                        **evaluation_result,  # Include all evaluation results
                        "best_params": best_params,
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "shap_analysis": shap_result,  # Include SHAP results
                        "used_sample_weights": sample_weights is not None
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        **evaluation_result,  # Include evaluation results even if save failed
                        "best_params": best_params,
                        "model_saved": False,
                        "shap_analysis": shap_result,  # Include SHAP results even if model save failed
                        "used_sample_weights": sample_weights is not None
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

    def train_decision_tree_rs_with_important_features(self, month_id, importance_threshold=IMPORTANCE_THRESHOLD, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train a Decision Tree classifier with hyperparameter tuning using RandomizedSearchCV,
        but only using features that exceed the importance threshold.
        Now includes SHAP analysis for enhanced model interpretability.
        Updated to include sample weights based on delay magnitude.
        
        This method combines feature selection and hyperparameter optimization:
        1. First trains a model to identify important features
        2. Then uses RandomizedSearchCV to find optimal hyperparameters on those features only
        3. Performs SHAP analysis for model interpretability
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        importance_threshold : float, optional
            Threshold for selecting important features. Features with importance scores
            above this threshold will be kept. Defaults to IMPORTANCE_THRESHOLD.
        param_distributions : dict, optional
            Dictionary with parameters names as keys and distributions or lists of parameters to try.
            Defaults to DECISION_TREE_PARAM_DISTRIBUTIONS from constants.
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
        pass
        """try:
            # Use default values from constants if not provided
            if param_distributions is None:
                from config.const import DECISION_TREE_PARAM_DISTRIBUTIONS
                param_distributions = DECISION_TREE_PARAM_DISTRIBUTIONS
            
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
            if target_column in REGRESSION_PROBLEM:
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                # NEW: Create sample weights for classification if delay info is available
                sample_weights = None
                if 'differenceInMinutes' in train_df.columns:
                    print("Using weighted samples based on delay magnitude for feature selection and randomized search")
                    # Create sample weights based on delay magnitude
                    delay_col = 'differenceInMinutes'
                    sample_weights = np.ones(len(y_train))
                    
                    # Get delay values for each training sample
                    delays = train_df[delay_col].values
                    
                    # Apply weights - higher delays get higher weights
                    delayed_idx = (delays > 0)
                    if np.any(delayed_idx):
                        # Normalize delay values by mean positive delay
                        mean_delay = delays[delayed_idx].mean()
                        # Use configured maximum weight
                        sample_weights[delayed_idx] = np.minimum(MAX_SAMPLE_WEIGHT_CLASSIFICATION, 1 + delays[delayed_idx]/mean_delay)
                    
                    print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                
                # STEP 1: FIRST TRAIN A MODEL TO IDENTIFY IMPORTANT FEATURES
                print(f"Training initial Decision Tree classifier to identify important features...")
                dt_initial = DecisionTreeClassifier(random_state=random_state)
                
                # Fit with sample weights if available
                if sample_weights is not None:
                    print("Training initial feature importance model with sample weights")
                    dt_initial.fit(X_train, y_train, sample_weight=sample_weights)
                else:
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
                
                # Fit only on the important features with sample weights if available
                if sample_weights is not None:
                    print("Training RandomizedSearchCV with sample weights on important features")
                    random_search.fit(X_train[important_features], y_train, sample_weight=sample_weights)
                else:
                    random_search.fit(X_train[important_features], y_train)
                
                best_params = random_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
                
                # Train model with best parameters on important features
                best_dt = DecisionTreeClassifier(**best_params, random_state=random_state)
                
                # Fit the final model with sample weights if available
                if sample_weights is not None:
                    print("Training final model with sample weights on important features")
                    best_dt.fit(X_train[important_features], y_train, sample_weight=sample_weights)
                else:
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
                combined_output_dir = os.path.join(self.project_root, "data/output/decision_tree_important_features_randomized_search")
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
                
                # ========== SHAP ANALYSIS ==========
                print("\nPerforming SHAP analysis on the RandomizedSearchCV + Important Features model...")
                
                shap_result = self.analyze_model_with_shap(
                    model=best_dt,
                    X_test=X_test[important_features],  # Use only important features
                    y_test=y_test,
                    model_type='classification',
                    month_id=month_id,
                    output_dir=combined_output_dir,
                    target_column=target_column,
                    max_samples=1000,
                    random_state=random_state,
                    model_name="decision_tree_rs_important_features",
                    baseline_data=train_df  # Use training data for better baseline calculation
                )
                
                if shap_result.get("success", False):
                    print("SHAP analysis completed successfully for RandomizedSearchCV + Important Features model!")
                    
                    # Compare with standard importance if SHAP was successful
                    if "shap_importance_path" in shap_result:
                        print("\n" + "-"*60)
                        print("COMPARISON: Standard vs SHAP Feature Importance (RS + Important Features)")
                        print("-"*60)
                        
                        try:
                            # Load SHAP importance for comparison
                            shap_importance = pd.read_csv(shap_result["shap_importance_path"])
                            
                            # Merge the two importance measures
                            comparison = selected_feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 
                                            'SHAP_Percentage_Points', 'SHAP_Abs_Percentage_Points', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Important features - Standard Importance vs SHAP Importance:")
                            for _, row in comparison.iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Abs_Percentage_Points'] if pd.notna(row['SHAP_Abs_Percentage_Points']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>5.2f}pp {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                        except Exception as e:
                            print(f"Could not perform comparison: {e}")
                    
                else:
                    print(f"SHAP analysis failed: {shap_result.get('error', 'Unknown error')}")
                
                print("="*60)
                
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
                    
                    # NEW: Save sample weights information if used
                    if sample_weights is not None:
                        weights_filename = f"sample_weights_info_{month_id}.txt"
                        weights_path = os.path.join(combined_output_dir, weights_filename)
                        with open(weights_path, 'w') as f:
                            f.write(f"Sample Weights Information - {month_id}\n")
                            f.write("="*40 + "\n")
                            f.write(f"Used sample weights: Yes\n")
                            f.write(f"Weight range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]\n")
                            f.write(f"Mean weight: {sample_weights.mean():.2f}\n")
                            f.write(f"Standard deviation: {sample_weights.std():.2f}\n")
                            f.write(f"Number of weighted samples: {(sample_weights > 1.0).sum()}\n")
                            f.write(f"Max weight constant used: {MAX_SAMPLE_WEIGHT_CLASSIFICATION}\n")
                            f.write(f"\nApplied to:\n")
                            f.write(f"- Initial feature importance model\n")
                            f.write(f"- RandomizedSearchCV cross-validation\n")
                            f.write(f"- Final model training\n")
                        print(f"Sample weights info saved to {weights_path}")
                    
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
                        "metrics_path": metrics_result["metrics_path"],
                        "shap_analysis": shap_result,  # Include SHAP results
                        "used_sample_weights": sample_weights is not None
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
                        "model_saved": False,
                        "shap_analysis": shap_result,  # Include SHAP results even if model save failed
                        "used_sample_weights": sample_weights is not None
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
            }"""

    def train_random_forest_with_randomized_search_cv(self, month_id, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train a Random Forest classifier/regressor with hyperparameter tuning using RandomizedSearchCV.
        Includes SHAP analysis for enhanced model interpretability.
        Updated to include sample weights based on delay magnitude.
        Uses comprehensive evaluation method for consistent metrics.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY-YYYY_MM" for the filename.
        param_distributions : dict, optional
            Dictionary with parameters names as keys and distributions or lists of parameters to try.
            Defaults to RANDOM_FOREST_PARAM_DISTRIBUTIONS from constants.
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
                from config.const import RANDOM_FOREST_PARAM_DISTRIBUTIONS
                param_distributions = RANDOM_FOREST_PARAM_DISTRIBUTIONS
            
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
            target_options = VALID_TARGET_FEATURES
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
            if target_column in REGRESSION_PROBLEM:
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                
                # NEW: Create sample weights for classification if delay info is available
                sample_weights = None
                if WEIGHT_DELAY_COLUMN in train_df.columns:
                    print("Using weighted samples based on delay magnitude for randomized search")
                    # Create sample weights based on delay magnitude
                    delay_col = WEIGHT_DELAY_COLUMN
                    sample_weights = np.ones(len(y_train))
                    
                    # Get delay values for each training sample
                    delays = train_df[delay_col].values
                    
                    # Apply weights - higher delays get higher weights
                    delayed_idx = (delays > TRAIN_DELAY_MINUTES)
                    if np.any(delayed_idx):
                        # Normalize delay values by mean positive delay
                        mean_delay = delays[delayed_idx].mean()
                        # Use configured maximum weight
                        sample_weights[delayed_idx] = np.minimum(MAX_SAMPLE_WEIGHT_CLASSIFICATION, 1 + delays[delayed_idx]/mean_delay)
                    
                    print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                
                # Create proper CV strategy for classification
                cv_strategy = StratifiedKFold(
                    n_splits=cv,       
                    shuffle=True,
                    random_state=random_state
                )

                # Initialize base classifier
                rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
                
                print(f"Starting RandomizedSearchCV with {n_iter} iterations and {cv}-fold cross-validation...")
            
                # Run RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    rf, param_distributions, n_iter=n_iter, cv=cv_strategy, 
                    scoring=SCORE_METRIC, random_state=random_state, n_jobs=-1
                )
                
                # Fit RandomizedSearchCV with sample weights if available
                if sample_weights is not None:
                    print("Training RandomizedSearchCV with sample weights")
                    # Note: RandomizedSearchCV will automatically handle sample weights for cross-validation
                    random_search.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    random_search.fit(X_train, y_train)
                
                best_params = random_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
                
                # Train model with best parameters
                best_rf = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
                
                # Fit the final model with sample weights if available
                if sample_weights is not None:
                    print("Training final model with sample weights")
                    best_rf.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    best_rf.fit(X_train, y_train)
                
                # Create Random Forest output directory
                random_forest_dir = os.path.join(self.project_root, RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
                os.makedirs(random_forest_dir, exist_ok=True)
                
                # === USE COMPREHENSIVE EVALUATION METHOD ===
                evaluation_result = self.evaluate_model_comprehensive(
                    model=best_rf,
                    X_test=X_test,
                    y_test=y_test,
                    model_name="Random Forest with RandomizedSearchCV",
                    month_id=month_id,
                    output_dir=random_forest_dir,
                    target_column=target_column,
                    random_search_obj=random_search,  # Pass the RandomizedSearchCV object
                    is_classification=is_classification
                )
                
                if not evaluation_result["success"]:
                    return {
                        "success": False,
                        "error": f"Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
                    }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': best_rf.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # ========== SHAP ANALYSIS ==========
                print("\nPerforming SHAP analysis on the Random Forest RandomizedSearchCV tuned model...")
                
                shap_result = self.analyze_model_with_shap(
                    model=best_rf,
                    X_test=X_test,
                    y_test=y_test,
                    model_type='classification',
                    month_id=month_id,
                    output_dir=random_forest_dir,
                    target_column=target_column,
                    max_samples=1000,
                    random_state=random_state,
                    model_name="random_forest_randomized_search",
                    baseline_data=train_df  # Use training data for better baseline calculation
                )
                
                if shap_result.get("success", False):
                    print("SHAP analysis completed successfully for Random Forest RandomizedSearchCV model!")
                    
                    # Compare with standard importance if SHAP was successful
                    if "shap_importance_path" in shap_result:
                        print("\n" + "-"*60)
                        print("COMPARISON: Standard vs SHAP Feature Importance (Random Forest RandomizedSearchCV)")
                        print("-"*60)
                        
                        try:
                            # Load SHAP importance for comparison
                            shap_importance = pd.read_csv(shap_result["shap_importance_path"])
                            
                            # Merge the two importance measures
                            comparison = feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 
                                            'SHAP_Percentage_Points', 'SHAP_Abs_Percentage_Points', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Top 10 features by Standard Importance vs SHAP Importance:")
                            for _, row in comparison.head(10).iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Abs_Percentage_Points'] if pd.notna(row['SHAP_Abs_Percentage_Points']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>5.2f}pp {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                        except Exception as e:
                            print(f"Could not perform comparison: {e}")
                    
                else:
                    print(f"SHAP analysis failed: {shap_result.get('error', 'Unknown error')}")
                
                print("="*60)
                
                # Save the model and params
                try:
                    import joblib
                    
                    # Save the model
                    model_filename = f"random_forest_{month_id}_randomized_search.joblib"
                    model_path = os.path.join(random_forest_dir, model_filename)
                    joblib.dump(best_rf, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{month_id}_randomized_search.csv"
                    importance_path = os.path.join(random_forest_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    # Save best parameters
                    params_filename = f"best_params_{month_id}_randomized_search.txt"
                    params_path = os.path.join(random_forest_dir, params_filename)
                    with open(params_path, 'w') as f:
                        for param, value in best_params.items():
                            f.write(f"{param}: {value}\n")
                    print(f"Best parameters saved to {params_path}")
                    
                    # NEW: Save sample weights information if used
                    if sample_weights is not None:
                        weights_filename = f"sample_weights_info_{month_id}.txt"
                        weights_path = os.path.join(random_forest_dir, weights_filename)
                        with open(weights_path, 'w') as f:
                            f.write(f"Sample Weights Information - {month_id}\n")
                            f.write("="*40 + "\n")
                            f.write(f"Used sample weights: Yes\n")
                            f.write(f"Weight range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]\n")
                            f.write(f"Mean weight: {sample_weights.mean():.2f}\n")
                            f.write(f"Standard deviation: {sample_weights.std():.2f}\n")
                            f.write(f"Number of weighted samples: {(sample_weights > 1.0).sum()}\n")
                            f.write(f"Max weight constant used: {MAX_SAMPLE_WEIGHT_CLASSIFICATION}\n")
                        print(f"Sample weights info saved to {weights_path}")
                    
                    return {
                        "success": True,
                        **evaluation_result,  # Include all evaluation results
                        "best_params": best_params,
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "shap_analysis": shap_result,  # Include SHAP results
                        "used_sample_weights": sample_weights is not None
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        **evaluation_result,  # Include evaluation results even if save failed
                        "best_params": best_params,
                        "model_saved": False,
                        "shap_analysis": shap_result,  # Include SHAP results even if model save failed
                        "used_sample_weights": sample_weights is not None
                    }
            else:
                # For regression problems
                from sklearn.ensemble import RandomForestRegressor
                
                # Create sample weights for regression if delay info is available
                sample_weights = None
                if WEIGHT_DELAY_COLUMN in train_df.columns:
                    print("Using weighted samples based on delay magnitude for randomized search")
                    # Create sample weights based on delay magnitude
                    delay_col = WEIGHT_DELAY_COLUMN
                    sample_weights = np.ones(len(y_train))
                    
                    # Get delay values for each training sample
                    delays = train_df[delay_col].values
                    
                    # Apply weights - higher delays get higher weights (for regression)
                    delayed_idx = (delays > TRAIN_DELAY_MINUTES)
                    if np.any(delayed_idx):
                        # Normalize delay values by mean positive delay
                        mean_delay = delays[delayed_idx].mean()
                        # Use configured maximum weight for regression
                        sample_weights[delayed_idx] = np.minimum(MAX_SAMPLE_WEIGHT_REGRESSION, 1 + delays[delayed_idx]/mean_delay)
                    
                    print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                
                # Create proper CV strategy for regression
                cv_strategy = KFold(
                    n_splits=cv,
                    shuffle=True,
                    random_state=random_state
                )

                # Initialize base regressor
                rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
                
                print(f"Starting RandomizedSearchCV with {n_iter} iterations and {cv}-fold cross-validation...")
            
                # Run RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    rf, param_distributions, n_iter=n_iter, cv=cv_strategy, 
                    scoring='neg_mean_squared_error', random_state=random_state, n_jobs=-1
                )
                
                # Fit RandomizedSearchCV with sample weights if available
                if sample_weights is not None:
                    print("Training RandomizedSearchCV with sample weights")
                    random_search.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    random_search.fit(X_train, y_train)
                
                best_params = random_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
                
                # Train model with best parameters
                best_rf = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
                
                # Fit the final model with sample weights if available
                if sample_weights is not None:
                    print("Training final model with sample weights")
                    best_rf.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    best_rf.fit(X_train, y_train)
                
                # Create Random Forest output directory
                random_forest_dir = os.path.join(self.project_root, RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
                os.makedirs(random_forest_dir, exist_ok=True)
                
                # === USE COMPREHENSIVE EVALUATION METHOD ===
                evaluation_result = self.evaluate_model_comprehensive(
                    model=best_rf,
                    X_test=X_test,
                    y_test=y_test,
                    model_name="Random Forest with RandomizedSearchCV",
                    month_id=month_id,
                    output_dir=random_forest_dir,
                    target_column=target_column,
                    random_search_obj=random_search,  # Pass the RandomizedSearchCV object
                    is_classification=is_classification
                )
                
                if not evaluation_result["success"]:
                    return {
                        "success": False,
                        "error": f"Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
                    }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': best_rf.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # ========== SHAP ANALYSIS ==========
                print("\nPerforming SHAP analysis on the Random Forest RandomizedSearchCV tuned model...")
                
                shap_result = self.analyze_model_with_shap(
                    model=best_rf,
                    X_test=X_test,
                    y_test=y_test,
                    model_type='regression',
                    month_id=month_id,
                    output_dir=random_forest_dir,
                    target_column=target_column,
                    max_samples=1000,
                    random_state=random_state,
                    model_name="random_forest_randomized_search",
                    baseline_data=train_df  # Use training data for better baseline calculation
                )
                
                if shap_result.get("success", False):
                    print("SHAP analysis completed successfully for Random Forest RandomizedSearchCV model!")
                    
                    # Compare with standard importance if SHAP was successful
                    if "shap_importance_path" in shap_result:
                        print("\n" + "-"*60)
                        print("COMPARISON: Standard vs SHAP Feature Importance (Random Forest RandomizedSearchCV)")
                        print("-"*60)
                        
                        try:
                            # Load SHAP importance for comparison
                            shap_importance = pd.read_csv(shap_result["shap_importance_path"])
                            
                            # Merge the two importance measures
                            comparison = feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Top 10 features by Standard Importance vs SHAP Importance:")
                            for _, row in comparison.head(10).iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Importance_Abs'] if pd.notna(row['SHAP_Importance_Abs']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>8.4f} {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                        except Exception as e:
                            print(f"Could not perform comparison: {e}")
                    
                else:
                    print(f"SHAP analysis failed: {shap_result.get('error', 'Unknown error')}")
                
                print("="*60)
                
                # Save the model and params
                try:
                    import joblib
                    
                    # Save the model
                    model_filename = f"random_forest_{month_id}_randomized_search.joblib"
                    model_path = os.path.join(random_forest_dir, model_filename)
                    joblib.dump(best_rf, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{month_id}_randomized_search.csv"
                    importance_path = os.path.join(random_forest_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    # Save best parameters
                    params_filename = f"best_params_{month_id}_randomized_search.txt"
                    params_path = os.path.join(random_forest_dir, params_filename)
                    with open(params_path, 'w') as f:
                        for param, value in best_params.items():
                            f.write(f"{param}: {value}\n")
                    print(f"Best parameters saved to {params_path}")
                    
                    # NEW: Save sample weights information if used
                    if sample_weights is not None:
                        weights_filename = f"sample_weights_info_{month_id}.txt"
                        weights_path = os.path.join(random_forest_dir, weights_filename)
                        with open(weights_path, 'w') as f:
                            f.write(f"Sample Weights Information - {month_id}\n")
                            f.write("="*40 + "\n")
                            f.write(f"Used sample weights: Yes\n")
                            f.write(f"Weight range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]\n")
                            f.write(f"Mean weight: {sample_weights.mean():.2f}\n")
                            f.write(f"Standard deviation: {sample_weights.std():.2f}\n")
                            f.write(f"Number of weighted samples: {(sample_weights > 1.0).sum()}\n")
                            f.write(f"Max weight constant used: {MAX_SAMPLE_WEIGHT_REGRESSION}\n")
                        print(f"Sample weights info saved to {weights_path}")
                    
                    return {
                        "success": True,
                        **evaluation_result,  # Include all evaluation results
                        "best_params": best_params,
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "shap_analysis": shap_result,  # Include SHAP results
                        "used_sample_weights": sample_weights is not None
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        **evaluation_result,  # Include evaluation results even if save failed
                        "best_params": best_params,
                        "model_saved": False,
                        "shap_analysis": shap_result,  # Include SHAP results even if model save failed
                        "used_sample_weights": sample_weights is not None
                    }
        
        except Exception as e:
            print(f"Error in RandomizedSearchCV for Random Forest {month_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def train_xgboost_with_randomized_search_cv(self, month_id, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train an XGBoost model (classifier or regressor) with hyperparameter tuning using manual CV.
        Supports sample weights based on delay magnitude for classification tasks.
        Now includes SHAP analysis for enhanced model interpretability.
        Uses the comprehensive evaluation method for consistent metrics across all models.
        
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
            #n_iter = min(n_iter, 20)  # Reduce number of iterations
            #cv = min(cv, 5)  # Reduce CV folds
            
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
            target_options = VALID_TARGET_FEATURES
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
            if target_column in REGRESSION_PROBLEM:
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            # Calculate scale_pos_weight for binary classification
            if is_classification and len(np.unique(y_train)) == 2:
                pos_samples = np.sum(y_train == True)  # or == 1
                neg_samples = len(y_train) - pos_samples
                scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
                print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
            else:
                scale_pos_weight = 1.0

            # Create sample weights for classification if delay info is available
            sample_weights = None
            if WEIGHT_DELAY_COLUMN in train_df.columns:
                print("Using weighted samples based on delay magnitude for randomized search")
                # Create sample weights based on delay magnitude
                delay_col = WEIGHT_DELAY_COLUMN
                sample_weights = np.ones(len(y_train))
                
                # Get delay values for each training sample
                delays = train_df[delay_col].values
                
                # Apply weights - higher delays get higher weights
                delayed_idx = (delays > TRAIN_DELAY_MINUTES)
                if np.any(delayed_idx):
                    # Normalize delay values by mean positive delay, using more moderate weights
                    mean_delay = delays[delayed_idx].mean()
                    # UPDATED: Use constant from config instead of hardcoded value
                    sample_weights[delayed_idx] = np.minimum(MAX_SAMPLE_WEIGHT_CLASSIFICATION, 1 + delays[delayed_idx]/mean_delay)
                
                print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
            
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
            
            # === UPDATED: Use methods configuration from const.py ===
            problem_type = "classification" if is_classification else "regression"
            methods_config = XGBOOST_METHODS_CONFIG[problem_type]
            
            # Resolve objective functions from configuration
            methods_to_try = []
            for method_config in methods_config:
                method_dict = {
                    "name": method_config["name"],
                    "obj": None
                }
                
                # Resolve objective function if specified
                if method_config["obj"] is not None:
                    if isinstance(method_config["obj"], str):
                        # String reference to function in XGBOOST_OBJECTIVE_FUNCTIONS
                        method_dict["obj"] = XGBOOST_OBJECTIVE_FUNCTIONS[method_config["obj"]]
                    else:
                        # Direct function reference
                        method_dict["obj"] = method_config["obj"]
                
                methods_to_try.append(method_dict)
            
            print(f"For {problem_type}, will try {len(methods_to_try)} different objective approaches:")
            for method in methods_to_try:
                obj_name = method["obj"].__name__ if method["obj"] is not None else "default"
                print(f"  - {method['name']}: {obj_name}")
            
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
                        if target_column == 'trainDelayed' or target_column == 'cancelled':  # Binary classification
                            current_params['objective'] = 'binary:logistic'
                            current_params['eval_metric'] = 'auc'
                            current_params['scale_pos_weight'] = scale_pos_weight
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
                            if SCORE_METRIC == 'roc_auc':
                                # Need probabilities for ROC AUC
                                if hasattr(model, 'predict_proba'):
                                    y_pred_proba = model.predict_proba(X_fold_val)
                                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                                        y_pred_proba = y_pred_proba[:, 1]  # Positive class probability
                                    score = roc_auc_score(y_fold_val, y_pred_proba)
                                else:
                                    # Fallback to accuracy if probabilities not available
                                    score = accuracy_score(y_fold_val, y_pred)
                            elif SCORE_METRIC == 'balanced_accuracy':
                                score = balanced_accuracy_score(y_fold_val, y_pred)
                            elif SCORE_METRIC == 'f1':
                                score = f1_score(y_fold_val, y_pred)
                            elif SCORE_METRIC == 'f1_weighted':
                                score = f1_score(y_fold_val, y_pred, average='weighted')
                            elif SCORE_METRIC == 'average_precision':
                                if hasattr(model, 'predict_proba'):
                                    y_pred_proba = model.predict_proba(X_fold_val)
                                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                                        y_pred_proba = y_pred_proba[:, 1]
                                    score = average_precision_score(y_fold_val, y_pred_proba)
                                else:
                                    score = accuracy_score(y_fold_val, y_pred)
                            else:
                                # Default to accuracy
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
                    
                    # Get the objective function for the best method
                    best_objective_func = None
                    for method in methods_to_try:
                        if method["name"] == best_method:
                            best_objective_func = method["obj"]
                            break
                    
                    # Train the model with custom objective
                    xgb_model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=best_params.get('n_estimators', 100),
                        obj=best_objective_func
                    )
                else:
                    print("Training final model with standard objective")
                    xgb_model = xgb.XGBRegressor(**best_params)
                    xgb_model.fit(X_train, y_train)
            
            # Create XGBoost RandomizedSearch output directory
            xgboost_rs_dir = os.path.join(self.project_root, XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
            os.makedirs(xgboost_rs_dir, exist_ok=True)
            
            # === NEW: USE COMPREHENSIVE EVALUATION METHOD ===
            print(f"\nUsing comprehensive evaluation method for XGBoost model...")
            evaluation_result = self.evaluate_model_comprehensive(
                model=xgb_model,
                X_test=X_test,
                y_test=y_test,
                model_name="XGBoost with RandomizedSearchCV",
                month_id=month_id,
                output_dir=xgboost_rs_dir,
                target_column=target_column,
                best_cv_score=best_score,  # Pass the best CV score
                random_search_obj=None,  # No RandomizedSearchCV object since we do manual search
                is_classification=is_classification
            )
            
            if not evaluation_result["success"]:
                return {
                    "success": False,
                    "error": f"Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
                }

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
            
            # ========== SHAP ANALYSIS ==========
            print("\nPerforming SHAP analysis on the XGBoost RandomizedSearchCV model...")
            
            shap_result = self.analyze_model_with_shap(
                model=xgb_model,
                X_test=X_test,
                y_test=y_test,
                model_type='classification' if is_classification else 'regression',
                month_id=month_id,
                output_dir=xgboost_rs_dir,
                target_column=target_column,
                max_samples=1000,
                random_state=random_state,
                model_name="xgboost_randomized_search",
                baseline_data=train_df  # Use training data for better baseline calculation
            )
            
            if shap_result.get("success", False):
                print("SHAP analysis completed successfully for XGBoost RandomizedSearchCV model!")
                
                # Compare with standard importance if SHAP was successful
                if "shap_importance_path" in shap_result:
                    print("\n" + "-"*60)
                    print("COMPARISON: Standard vs SHAP Feature Importance (XGBoost RandomizedSearchCV)")
                    print("-"*60)
                    
                    try:
                        # Load SHAP importance for comparison
                        shap_importance = pd.read_csv(shap_result["shap_importance_path"])
                        
                        # Merge the two importance measures
                        if is_classification:
                            comparison = feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 
                                            'SHAP_Percentage_Points', 'SHAP_Abs_Percentage_Points', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Top 10 features by Standard Importance vs SHAP Importance:")
                            for _, row in comparison.head(10).iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Abs_Percentage_Points'] if pd.notna(row['SHAP_Abs_Percentage_Points']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>5.2f}pp {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                        else:
                            comparison = feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Top 10 features by Standard Importance vs SHAP Importance:")
                            for _, row in comparison.head(10).iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Importance_Abs'] if pd.notna(row['SHAP_Importance_Abs']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>8.4f} {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                    except Exception as e:
                        print(f"Could not perform comparison: {e}")
                
            else:
                print(f"SHAP analysis failed: {shap_result.get('error', 'Unknown error')}")
            
            print("="*60)
            
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
                    f.write(f"Best Method: {best_method}\n")
                    f.write(f"Best CV Score: {best_score:.4f}\n\n")
                    f.write("Best Parameters:\n")
                    for param, value in best_params.items():
                        f.write(f"{param}: {value}\n")
                print(f"Best parameters saved to {params_path}")
                
                # NEW: Save sample weights information if used
                if sample_weights is not None:
                    weights_filename = f"sample_weights_info_{month_id}.txt"
                    weights_path = os.path.join(xgboost_rs_dir, weights_filename)
                    with open(weights_path, 'w') as f:
                        f.write(f"Sample Weights Information - {month_id}\n")
                        f.write("="*40 + "\n")
                        f.write(f"Used sample weights: Yes\n")
                        f.write(f"Weight range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]\n")
                        f.write(f"Mean weight: {sample_weights.mean():.2f}\n")
                        f.write(f"Standard deviation: {sample_weights.std():.2f}\n")
                        f.write(f"Number of weighted samples: {(sample_weights > 1.0).sum()}\n")
                        f.write(f"Max weight constant used: {MAX_SAMPLE_WEIGHT_CLASSIFICATION}\n")
                    print(f"Sample weights info saved to {weights_path}")
                
                # Return comprehensive results similar to decision tree method
                return {
                    "success": True,
                    **evaluation_result,  # Include all evaluation results
                    "best_params": best_params,
                    "best_method": best_method,
                    "best_cv_score": best_score,
                    "custom_objective": best_method == "weighted",
                    "model_path": model_path,
                    "feature_importance_path": importance_path,
                    "shap_analysis": shap_result,  # Include SHAP results
                    "used_sample_weights": sample_weights is not None
                }
                    
            except Exception as e:
                print(f"Warning: Could not save model: {str(e)}")
                return {
                    "success": True,
                    **evaluation_result,  # Include evaluation results even if save failed
                    "best_params": best_params,
                    "best_method": best_method,
                    "best_cv_score": best_score,
                    "custom_objective": best_method == "weighted",
                    "model_saved": False,
                    "shap_analysis": shap_result,  # Include SHAP results even if model save failed
                    "used_sample_weights": sample_weights is not None
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

    def train_xgboost_rs_with_important_features(self, month_id, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train an XGBoost model with RandomizedSearchCV using only the top features identified
        from a previous XGBoost RandomizedSearchCV run.
        Now includes SHAP analysis for enhanced model interpretability.
        
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
        pass
        """try:
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
            target_options = VALID_TARGET_FEATURES
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
            if target_column in REGRESSION_PROBLEM:
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
                    # UPDATED: Use constant from config instead of hardcoded value
                    sample_weights[delayed_idx] = np.minimum(MAX_SAMPLE_WEIGHT_CLASSIFICATION, 1 + delays[delayed_idx]/mean_delay)
                
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
                # UPDATED: Use constant from config instead of hardcoded value
                weights = np.minimum(MAX_SAMPLE_WEIGHT_REGRESSION, 1.0 + np.abs(y_true) / (np.abs(y_true).mean() * 2))
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
            
            # ========== NEW: ADD SHAP ANALYSIS ==========
            print("\nPerforming SHAP analysis on the XGBoost RandomizedSearchCV + Important Features model...")
            
            shap_result = self.analyze_model_with_shap(
                model=xgb_model,
                X_test=X_test_important,  # Use only important features
                y_test=y_test,
                model_type='regression' if not is_classification else 'classification',
                month_id=month_id,
                output_dir=output_dir,
                target_column=target_column,
                max_samples=1000,
                random_state=random_state,
                model_name="xgboost_rs_important_features",
                baseline_data=train_df  # Use training data for better baseline calculation
            )
            
            if shap_result.get("success", False):
                print("SHAP analysis completed successfully for XGBoost RS + Important Features model!")
                
                # Compare with standard importance if SHAP was successful
                if "shap_importance_path" in shap_result:
                    print("\n" + "-"*60)
                    print("COMPARISON: Standard vs SHAP Feature Importance (XGBoost RS + Important Features)")
                    print("-"*60)
                    
                    try:
                        # Load SHAP importance for comparison
                        shap_importance = pd.read_csv(shap_result["shap_importance_path"])
                        
                        # Merge the two importance measures
                        if is_classification:
                            comparison = selected_feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 
                                            'SHAP_Percentage_Points', 'SHAP_Abs_Percentage_Points', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Important features - Standard Importance vs SHAP Importance:")
                            for _, row in comparison.iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Abs_Percentage_Points'] if pd.notna(row['SHAP_Abs_Percentage_Points']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>5.2f}pp {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                        else:
                            comparison = selected_feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Important features - Standard Importance vs SHAP Importance:")
                            for _, row in comparison.iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Importance_Abs'] if pd.notna(row['SHAP_Importance_Abs']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>8.4f} {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                    except Exception as e:
                        print(f"Could not perform comparison: {e}")
                
            else:
                print(f"SHAP analysis failed: {shap_result.get('error', 'Unknown error')}")
            
            print("="*60)
            
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
                        "used_sample_weights": sample_weights is not None,
                        "shap_analysis": shap_result  # Include SHAP results
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
                        "metrics_path": metrics_result["metrics_path"],
                        "shap_analysis": shap_result  # Include SHAP results
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
                        "model_saved": False,
                        "shap_analysis": shap_result  # Include SHAP results even if model save failed
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
                        "model_saved": False,
                        "shap_analysis": shap_result  # Include SHAP results even if model save failed
                    }
        
        except Exception as e:
            import traceback
            print(f"Error in XGBoost RandomizedSearchCV with Important Features for {month_id}: {str(e)}")
            print("\nDetailed traceback:")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }"""

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
        pass
        """try:
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
            }"""
 
    def extract_and_save_metrics(self, y_test, y_pred, report, month_id, output_dir=None, y_pred_proba=None):
        """
        Extract key metrics from model evaluation and save them to a CSV file.
        Enhanced for imbalanced classification problems.
        
        Parameters:
        -----------
        y_test : array-like
            True target values
        y_pred : array-like
            Predicted target values
        report : dict
            Classification report dictionary from sklearn
        month_id : str
            Month identifier for file naming
        output_dir : str, optional
            Output directory path
        y_pred_proba : array-like, optional
            Predicted probabilities for positive class (for ROC AUC, PR AUC)
        """
        
        # Create metrics dictionary
        metrics = {}
        
        # Use decision_tree_dir as default if output_dir is None
        if output_dir is None:
            output_dir = self.decision_tree_dir
        
        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
        
        # Cohen's Kappa (accounts for class imbalance)
        metrics['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
        
        # F1 scores
        if len(np.unique(y_test)) == 2:
            metrics['f1'] = f1_score(y_test, y_pred)
        else:
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
        
        metrics['weighted_avg_f1'] = report['weighted avg']['f1-score']
        metrics['macro_avg_f1'] = report['macro avg']['f1-score']
        
        # ROC AUC and PR AUC for binary classification
        if len(np.unique(y_test)) == 2:
            if y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
                except Exception as e:
                    print(f"Warning: Could not calculate ROC/PR AUC: {e}")
                    metrics['roc_auc'] = None
                    metrics['pr_auc'] = None
            else:
                # Try to calculate ROC AUC with predictions as fallback
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
                except:
                    metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        
        # For classification, also extract metrics for each class
        for class_label in report:
            if class_label not in ['weighted avg', 'macro avg', 'accuracy']:
                metrics[f'class_{class_label}_f1'] = report[class_label]['f1-score']
                metrics[f'class_{class_label}_precision'] = report[class_label]['precision']
                metrics[f'class_{class_label}_recall'] = report[class_label]['recall']
                metrics[f'class_{class_label}_support'] = report[class_label]['support']
        
        # Add class distribution info
        unique, counts = np.unique(y_test, return_counts=True)
        for class_val, count in zip(unique, counts):
            metrics[f'class_{class_val}_test_count'] = count
            metrics[f'class_{class_val}_test_percentage'] = (count / len(y_test)) * 100
        
        # Add the optimized metric value
        if SCORE_METRIC == 'roc_auc' and metrics.get('roc_auc') is not None:
            metrics['optimized_metric_value'] = metrics['roc_auc']
        elif SCORE_METRIC == 'balanced_accuracy':
            metrics['optimized_metric_value'] = metrics['balanced_accuracy']
        elif SCORE_METRIC == 'f1':
            metrics['optimized_metric_value'] = metrics['f1']
        elif SCORE_METRIC == 'f1_weighted':
            metrics['optimized_metric_value'] = metrics['weighted_avg_f1']
        elif SCORE_METRIC == 'average_precision' and metrics.get('pr_auc') is not None:
            metrics['optimized_metric_value'] = metrics['pr_auc']
        else:
            metrics['optimized_metric_value'] = metrics['accuracy']
        
        metrics['optimized_metric_name'] = SCORE_METRIC
        
        # Print metrics with focus on the optimized metric
        print(f"\nDetailed Model Metrics:")
        print(f"Optimized Metric ({SCORE_METRIC}): {metrics['optimized_metric_value']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        if metrics.get('roc_auc'):
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        if metrics.get('pr_auc'):
            print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        print(f"\nClass Performance:")
        for class_label in ['False', 'True']:  # Assuming binary classification
            if f'class_{class_label}_f1' in metrics:
                print(f"Class {class_label} - F1: {metrics[f'class_{class_label}_f1']:.4f}, "
                    f"Precision: {metrics[f'class_{class_label}_precision']:.4f}, "
                    f"Recall: {metrics[f'class_{class_label}_recall']:.4f}")
        
        # Save metrics to a file
        metrics_filename = f"model_metrics_{month_id}.csv"
        metrics_path = os.path.join(output_dir, metrics_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # Save to CSV
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Enhanced model metrics saved to {metrics_path}")
        
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
            True target values
        y_pred : array-like
            Predicted target values
        month_id : str
            Month identifier for file naming
        output_dir : str, optional
            Output directory path
            
        Returns:
        --------
        dict
            Dictionary containing metrics and file path
        """
        # Import required metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        import pandas as pd
        import os
        
        # Create metrics dictionary
        metrics = {}
        
        # Use a default output directory if none provided
        if output_dir is None:
            output_dir = self.output_dir
        
        # Calculate regression metrics
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        
        # Calculate residual statistics
        residuals = y_test - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['median_residual'] = np.median(residuals)
        
        # Calculate target and prediction ranges
        metrics['target_min'] = float(np.min(y_test))
        metrics['target_max'] = float(np.max(y_test))
        metrics['target_mean'] = float(np.mean(y_test))
        metrics['target_std'] = float(np.std(y_test))
        
        metrics['pred_min'] = float(np.min(y_pred))
        metrics['pred_max'] = float(np.max(y_pred))
        metrics['pred_mean'] = float(np.mean(y_pred))
        metrics['pred_std'] = float(np.std(y_pred))
        
        # Calculate additional useful metrics
        metrics['explained_variance'] = 1 - (np.var(residuals) / np.var(y_test))
        metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
        
        # Add sample size
        metrics['n_samples'] = len(y_test)
        
        # Print key metrics
        print(f"\nRegression Model Metrics:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"Mean Residual: {metrics['mean_residual']:.4f}")
        print(f"Std Residual: {metrics['std_residual']:.4f}")
        
        # Save metrics to a file
        metrics_filename = f"model_metrics_{month_id}.csv"
        metrics_path = os.path.join(output_dir, metrics_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # Save to CSV
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Regression model metrics saved to {metrics_path}")
        
        return {
            "metrics": metrics,
            "metrics_path": metrics_path
        }

    def evaluate_model_comprehensive(self, model, X_test, y_test, model_name, month_id, 
                                    output_dir, target_column, best_cv_score=None, 
                                    random_search_obj=None, is_classification=True):
        """
        Comprehensive model evaluation method that can be used across all training techniques.
        
        Handles both classification and regression problems, calculates relevant metrics
        based on the configured SCORE_METRIC, and provides consistent output formatting.
        
        Parameters:
        -----------
        model : sklearn model or compatible
            The trained model to evaluate.
        X_test : pandas.DataFrame
            Test features.
        y_test : pandas.Series
            Test target values.
        model_name : str
            Name of the model/technique (e.g., "Decision Tree RandomizedSearchCV").
        month_id : str
            Month identifier for file naming.
        output_dir : str
            Directory to save evaluation results.
        target_column : str
            Name of the target column being predicted.
        best_cv_score : float, optional
            Best cross-validation score from hyperparameter tuning.
        random_search_obj : RandomizedSearchCV object, optional
            The RandomizedSearchCV object to extract best_score_ from.
        is_classification : bool, optional
            Whether this is a classification problem. Defaults to True.
            
        Returns:
        --------
        dict
            Comprehensive evaluation results dictionary.
        """
       
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE MODEL EVALUATION: {model_name}")
        print(f"{'='*80}")
        
        # Extract best CV score if RandomizedSearchCV object provided
        if best_cv_score is None and random_search_obj is not None:
            best_cv_score = random_search_obj.best_score_
        
        if is_classification:
            # === CLASSIFICATION EVALUATION ===
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # Get probabilities for metrics that need them
            try:
                y_pred_proba = model.predict_proba(X_test)
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # Probability of positive class
            except Exception as e:
                print(f"Note: Could not get prediction probabilities: {e}")
                y_pred_proba = None
            
            # Calculate all relevant metrics
            metrics = {}
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
            metrics['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
            
            # F1 scores
            if len(np.unique(y_test)) == 2:
                metrics['f1'] = f1_score(y_test, y_pred)
            else:
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics['weighted_avg_f1'] = report['weighted avg']['f1-score']
            metrics['macro_avg_f1'] = report['macro avg']['f1-score']
            
            # ROC AUC and PR AUC for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
                except Exception as e:
                    print(f"Warning: Could not calculate ROC/PR AUC: {e}")
                    metrics['roc_auc'] = None
                    metrics['pr_auc'] = None
            else:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
            
            # Determine the optimized metric value
            optimized_metric_value = None
            optimized_metric_name = SCORE_METRIC
            
            if SCORE_METRIC == 'roc_auc' and metrics.get('roc_auc') is not None:
                optimized_metric_value = metrics['roc_auc']
            elif SCORE_METRIC == 'balanced_accuracy':
                optimized_metric_value = metrics['balanced_accuracy']
            elif SCORE_METRIC == 'f1':
                optimized_metric_value = metrics['f1']
            elif SCORE_METRIC == 'f1_weighted':
                optimized_metric_value = metrics['weighted_avg_f1']
            elif SCORE_METRIC == 'average_precision' and metrics.get('pr_auc') is not None:
                optimized_metric_value = metrics['pr_auc']
            else:
                # Fallback to accuracy
                optimized_metric_value = metrics['accuracy']
                optimized_metric_name = 'accuracy'
            
            # Print results
            print(f"Target Column: {target_column}")
            print(f"Model Type: Classification")
            if best_cv_score is not None:
                print(f"Best CV Score ({SCORE_METRIC}): {best_cv_score:.4f}")
            print(f"Test {optimized_metric_name}: {optimized_metric_value:.4f}")
            
            print(f"\nCore Classification Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
            
            if metrics.get('roc_auc') is not None:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            if metrics.get('pr_auc') is not None:
                print(f"  PR AUC: {metrics['pr_auc']:.4f}")
            
            print(f"\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred))
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(conf_matrix)
            
            # Add per-class metrics
            for class_label in report:
                if class_label not in ['weighted avg', 'macro avg', 'accuracy']:
                    metrics[f'class_{class_label}_f1'] = report[class_label]['f1-score']
                    metrics[f'class_{class_label}_precision'] = report[class_label]['precision']
                    metrics[f'class_{class_label}_recall'] = report[class_label]['recall']
                    metrics[f'class_{class_label}_support'] = report[class_label]['support']
            
            # Add class distribution
            unique, counts = np.unique(y_test, return_counts=True)
            for class_val, count in zip(unique, counts):
                metrics[f'class_{class_val}_test_count'] = count
                metrics[f'class_{class_val}_test_percentage'] = (count / len(y_test)) * 100
            
            # Store additional info
            metrics['optimized_metric_value'] = optimized_metric_value
            metrics['optimized_metric_name'] = optimized_metric_name
            metrics['best_cv_score'] = best_cv_score
            
            # Save enhanced metrics
            enhanced_metrics_result = self.extract_and_save_metrics(
                y_test, y_pred, report, month_id, 
                output_dir=output_dir, 
                y_pred_proba=y_pred_proba
            )
            
            return {
                "success": True,
                "model_type": "classification",
                "predictions": y_pred,
                "probabilities": y_pred_proba,
                "accuracy": metrics['accuracy'],
                "optimized_metric": optimized_metric_value,
                "optimized_metric_name": optimized_metric_name,
                "best_cv_score": best_cv_score,
                "balanced_accuracy": metrics['balanced_accuracy'],
                "f1_score": metrics['f1'],
                "roc_auc": metrics.get('roc_auc'),
                "pr_auc": metrics.get('pr_auc'),
                "cohen_kappa": metrics['cohen_kappa'],
                "confusion_matrix": conf_matrix,
                "classification_report": report,
                "all_metrics": metrics,
                "metrics_path": enhanced_metrics_result["metrics_path"]
            }
        
        else:
            # === REGRESSION EVALUATION ===
            
            # Handle different model types for regression prediction
            if hasattr(model, 'predict'):
                # Standard sklearn-style model
                y_pred = model.predict(X_test)
            elif hasattr(model, 'predict') and hasattr(X_test, 'values'):
                # XGBoost booster with DMatrix needed
                import xgboost as xgb
                dtest = xgb.DMatrix(X_test)
                y_pred = model.predict(dtest)
            else:
                raise ValueError("Cannot determine how to make predictions with this model")
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Determine optimized metric (for regression, usually R² or negative MSE)
            optimized_metric_value = None
            optimized_metric_name = SCORE_METRIC if SCORE_METRIC in ['r2', 'neg_mean_squared_error'] else 'r2'
            
            if optimized_metric_name == 'r2':
                optimized_metric_value = r2
            elif optimized_metric_name == 'neg_mean_squared_error':
                optimized_metric_value = -mse
            else:
                optimized_metric_value = r2
                optimized_metric_name = 'r2'
            
            # Print results
            print(f"Target Column: {target_column}")
            print(f"Model Type: Regression")
            if best_cv_score is not None:
                print(f"Best CV Score ({SCORE_METRIC}): {best_cv_score:.4f}")
            print(f"Test {optimized_metric_name}: {optimized_metric_value:.4f}")
            
            print(f"\nRegression Metrics:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MSE: {mse:.4f}")
            
            # Additional regression statistics
            residuals = y_test - y_pred
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            print(f"\nResidual Analysis:")
            print(f"  Mean Residual: {mean_residual:.4f}")
            print(f"  Std Residual: {std_residual:.4f}")
            print(f"  Target Range: [{y_test.min():.2f}, {y_test.max():.2f}]")
            print(f"  Prediction Range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
            
            # Save regression metrics
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_residual': mean_residual,
                'std_residual': std_residual,
                'optimized_metric_value': optimized_metric_value,
                'optimized_metric_name': optimized_metric_name,
                'best_cv_score': best_cv_score,
                'target_min': y_test.min(),
                'target_max': y_test.max(),
                'pred_min': y_pred.min(),
                'pred_max': y_pred.max()
            }
            
            regression_metrics_result = self.extract_and_save_regression_metrics(
                y_test, y_pred, month_id, output_dir=output_dir
            )

            print(f"{'='*80}")
            print(f"EVALUATION COMPLETE: {model_name}")
            print(f"{'='*80}\n")
            
            return {
                "success": True,
                "model_type": "regression",
                "predictions": y_pred,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mse": mse,
                "optimized_metric": optimized_metric_value,
                "optimized_metric_name": optimized_metric_name,
                "best_cv_score": best_cv_score,
                "residuals": residuals,
                "all_metrics": metrics,
                "metrics_path": regression_metrics_result["metrics_path"]
            }
        
    def analyze_model_with_shap(self, model, X_test, y_test, model_type, month_id, 
                            output_dir, target_column, max_samples=1000, 
                            random_state=42, model_name="model", baseline_data=None):
        """
        Perform SHAP analysis on a trained model for feature importance and interpretability.
        
        This method creates SHAP explanations for any compatible model type and provides
        detailed feature importance analysis with business interpretations.
        
        Parameters:
        -----------
        model : sklearn model or compatible
            The trained model to analyze (DecisionTree, XGBoost, etc.).
        X_test : pandas.DataFrame
            Test features for SHAP analysis.
        y_test : pandas.Series
            Test target values.
        model_type : str
            Type of problem: 'classification' or 'regression'.
        month_id : str
            Month identifier for file naming.
        output_dir : str
            Directory to save SHAP analysis results.
        target_column : str
            Name of the target column being predicted.
        max_samples : int, optional
            Maximum number of test samples to use for SHAP analysis. Defaults to 1000.
        random_state : int, optional
            Random seed for sampling. Defaults to 42.
        model_name : str, optional
            Name of the model for file naming. Defaults to "model".
        baseline_data : pandas.DataFrame, optional
            Training data for baseline calculations. If None, uses X_test for baseline.
            
        Returns:
        --------
        dict
            Dictionary containing SHAP analysis results and file paths.
        """
        try:
            print("\n" + "="*60)
            print("PERFORMING SHAP ANALYSIS FOR MODEL INTERPRETABILITY")
            print("="*60)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine the appropriate SHAP explainer based on model type
            print("Creating SHAP explainer...")
            
            # Handle different model types
            explainer = None
            if hasattr(model, 'tree_') or 'DecisionTree' in str(type(model)):
                # Decision Tree models
                explainer = shap.TreeExplainer(model)
                explainer_type = "TreeExplainer"
            elif hasattr(model, 'get_booster') or 'XGB' in str(type(model)) or 'xgboost' in str(type(model)).lower():
                # XGBoost models
                explainer = shap.TreeExplainer(model)
                explainer_type = "TreeExplainer"
            elif hasattr(model, 'estimators_') or 'RandomForest' in str(type(model)):
                # Random Forest and ensemble models
                explainer = shap.TreeExplainer(model)
                explainer_type = "TreeExplainer"
            else:
                # Linear models or others - use KernelExplainer as fallback
                if baseline_data is not None:
                    # Use a sample of training data as background for KernelExplainer
                    background_sample = baseline_data.sample(min(100, len(baseline_data)), random_state=random_state)
                else:
                    # Use a sample of test data as background
                    background_sample = X_test.sample(min(100, len(X_test)), random_state=random_state)
                
                explainer = shap.KernelExplainer(model.predict, background_sample)
                explainer_type = "KernelExplainer"
            
            print(f"Using {explainer_type} for SHAP analysis")
            
            # Sample test data if needed to avoid memory issues
            if len(X_test) > max_samples:
                print(f"Using sample of {max_samples} test instances for SHAP analysis")
                X_test_sample = X_test.sample(n=max_samples, random_state=random_state)
                y_test_sample = y_test.loc[X_test_sample.index]
            else:
                X_test_sample = X_test
                y_test_sample = y_test
            
            print(f"Calculating SHAP values for {len(X_test_sample)} samples...")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            
            # Debug: Check the format of SHAP values
            print(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"SHAP values list length: {len(shap_values)}")
                for i, sv in enumerate(shap_values):
                    print(f"  Element {i} shape: {sv.shape if hasattr(sv, 'shape') else 'No shape'}")
            elif hasattr(shap_values, 'shape'):
                print(f"SHAP values shape: {shap_values.shape}")
            
            # Initialize result dictionary
            result = {
                "success": True,
                "explainer_type": explainer_type,
                "model_name": model_name,
                "samples_analyzed": len(X_test_sample)
            }
            
            if model_type == 'classification':
                # CLASSIFICATION ANALYSIS
                unique_classes = np.unique(y_test)
                print(f"Unique classes in target: {unique_classes}")
                print(f"Number of classes: {len(unique_classes)}")
                
                # Handle different SHAP value formats for classification
                shap_values_positive = None
                
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # Standard binary classification format: list with 2 arrays
                    print("Using standard binary classification format (list with 2 arrays)")
                    shap_values_positive = shap_values[1]  # Values for positive class
                    
                elif isinstance(shap_values, list) and len(shap_values) == 1:
                    # Single array format
                    print("Using single array format for binary classification")
                    shap_values_positive = shap_values[0]
                    
                elif not isinstance(shap_values, list) and hasattr(shap_values, 'shape'):
                    # Direct numpy array format
                    if len(shap_values.shape) == 2:
                        print("Using direct array format for binary classification")
                        shap_values_positive = shap_values
                    elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
                        print("Using 3D array format, extracting positive class")
                        shap_values_positive = shap_values[:, :, 1]
                    else:
                        print(f"Unexpected array shape: {shap_values.shape}")
                        raise ValueError(f"Unexpected SHAP values array shape: {shap_values.shape}")
                else:
                    raise ValueError("Unable to parse SHAP values format")
                
                if shap_values_positive is not None:
                    print(f"Processing SHAP values for positive class, shape: {shap_values_positive.shape}")
                    
                    # Calculate mean absolute and signed SHAP values
                    mean_shap_abs = np.mean(np.abs(shap_values_positive), axis=0)
                    mean_shap_signed = np.mean(shap_values_positive, axis=0)
                    
                    # Create SHAP importance DataFrame
                    shap_importance = pd.DataFrame({
                        'Feature': X_test.columns,
                        'SHAP_Importance_Abs': mean_shap_abs,
                        'SHAP_Importance_Signed': mean_shap_signed,
                        'Increases_Positive_Probability': mean_shap_signed > 0
                    }).sort_values(by='SHAP_Importance_Abs', ascending=False)
                    
                    # Calculate baseline prediction
                    if baseline_data is not None and target_column in baseline_data.columns:
                        baseline_prob = baseline_data[target_column].mean()
                    else:
                        baseline_prob = y_test.mean()
                    
                    print(f"\nBaseline positive class probability: {baseline_prob:.1%}")
                    
                    # Calculate total SHAP contribution
                    total_shap_contribution = np.sum(np.abs(mean_shap_signed))
                    print(f"Total absolute SHAP contribution: {total_shap_contribution:.4f} ({total_shap_contribution*100:.2f} percentage points)")
                    
                    # Convert SHAP values to percentage points
                    shap_importance['SHAP_Percentage_Points'] = shap_importance['SHAP_Importance_Signed'] * 100
                    shap_importance['SHAP_Abs_Percentage_Points'] = shap_importance['SHAP_Importance_Abs'] * 100
                    
                    # Calculate relative importance
                    shap_importance['Relative_Contribution_Pct'] = (
                        shap_importance['SHAP_Importance_Abs'] / total_shap_contribution * 100
                    )
                    
                    # Get top features
                    positive_features = shap_importance[
                        shap_importance['Increases_Positive_Probability'] == True
                    ].head(10)
                    
                    negative_features = shap_importance[
                        shap_importance['Increases_Positive_Probability'] == False
                    ].head(5)
                    
                    # Print results
                    print("\nSHAP Feature Importance for POSITIVE CLASS:")
                    print("Features that INCREASE positive class probability:")
                    
                    for _, row in positive_features.iterrows():
                        print(f"  {row['Feature']:<25}: +{row['SHAP_Percentage_Points']:>6.2f} pp "
                            f"({row['Relative_Contribution_Pct']:>4.1f}% of total impact)")
                    
                    print("\nFeatures that DECREASE positive class probability:")
                    
                    for _, row in negative_features.iterrows():
                        print(f"  {row['Feature']:<25}: {row['SHAP_Percentage_Points']:>6.2f} pp "
                            f"({row['Relative_Contribution_Pct']:>4.1f}% of total impact)")
                    
                    # Save SHAP importance
                    shap_filename = f"SHAP_feature_importance_{month_id}.csv"
                    shap_path = os.path.join(output_dir, shap_filename)
                    shap_importance.to_csv(shap_path, index=False)
                    print(f"\nSHAP feature importance saved to {shap_path}")
                    
                    # Save detailed analysis
                    summary_filename = f"SHAP_summary_{month_id}.txt"
                    summary_path = os.path.join(output_dir, summary_filename)
                    
                    with open(summary_path, 'w') as f:
                        f.write(f"SHAP Analysis Summary - {month_id}\n")
                        f.write("="*50 + "\n\n")
                        f.write(f"Model: {model_name}\n")
                        f.write(f"Target: {target_column}\n")
                        f.write(f"Model Type: {model_type}\n")
                        f.write(f"Explainer: {explainer_type}\n")
                        f.write(f"Samples analyzed: {len(X_test_sample)}\n")
                        f.write(f"Baseline positive class probability: {baseline_prob:.1%}\n")
                        f.write(f"Total SHAP contribution: {total_shap_contribution:.4f}\n\n")
                        
                        f.write("TOP FEATURES THAT INCREASE POSITIVE CLASS PROBABILITY:\n")
                        f.write("-"*50 + "\n")
                        for _, row in positive_features.iterrows():
                            f.write(f"{row['Feature']:<25}: +{row['SHAP_Percentage_Points']:>6.2f} pp "
                                    f"({row['Relative_Contribution_Pct']:>4.1f}% of total)\n")
                        
                        f.write("\nTOP FEATURES THAT DECREASE POSITIVE CLASS PROBABILITY:\n")
                        f.write("-"*50 + "\n")
                        for _, row in negative_features.iterrows():
                            f.write(f"{row['Feature']:<25}: {row['SHAP_Percentage_Points']:>6.2f} pp "
                                    f"({row['Relative_Contribution_Pct']:>4.1f}% of total)\n")
                    
                    print(f"Detailed SHAP analysis saved to {summary_path}")
                    
                    # Update result
                    result.update({
                        "shap_importance_path": shap_path,
                        "summary_path": summary_path,
                        "baseline_probability": baseline_prob,
                        "total_shap_contribution": total_shap_contribution,
                        "top_positive_features": positive_features.head(5)['Feature'].tolist(),
                        "top_negative_features": negative_features.head(5)['Feature'].tolist()
                    })
            
            else:
                # REGRESSION ANALYSIS
                print("Processing SHAP values for regression model")
                
                # For regression, SHAP values should be a single array
                if isinstance(shap_values, list):
                    if len(shap_values) == 1:
                        shap_values_reg = shap_values[0]
                    else:
                        print(f"Warning: Unexpected list length for regression: {len(shap_values)}")
                        shap_values_reg = shap_values[0]
                else:
                    shap_values_reg = shap_values
                
                print(f"Processing SHAP values for regression, shape: {shap_values_reg.shape}")
                
                # Calculate mean absolute and signed SHAP values
                mean_shap_abs = np.mean(np.abs(shap_values_reg), axis=0)
                mean_shap_signed = np.mean(shap_values_reg, axis=0)
                
                # Create SHAP importance DataFrame
                shap_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'SHAP_Importance_Abs': mean_shap_abs,
                    'SHAP_Importance_Signed': mean_shap_signed,
                    'Increases_Target': mean_shap_signed > 0
                }).sort_values(by='SHAP_Importance_Abs', ascending=False)
                
                # Calculate baseline prediction
                if baseline_data is not None and target_column in baseline_data.columns:
                    baseline_value = baseline_data[target_column].mean()
                else:
                    baseline_value = y_test.mean()
                
                print(f"\nBaseline target value: {baseline_value:.2f}")
                
                # Calculate total SHAP contribution
                total_shap_contribution = np.sum(np.abs(mean_shap_signed))
                print(f"Total absolute SHAP contribution: {total_shap_contribution:.4f}")
                
                # Calculate relative importance
                shap_importance['Relative_Contribution_Pct'] = (
                    shap_importance['SHAP_Importance_Abs'] / total_shap_contribution * 100
                )
                
                # Get top features
                increasing_features = shap_importance[
                    shap_importance['Increases_Target'] == True
                ].head(10)
                
                decreasing_features = shap_importance[
                    shap_importance['Increases_Target'] == False
                ].head(5)
                
                # Print results
                print("\nSHAP Feature Importance for REGRESSION:")
                print("Features that INCREASE target value:")
                
                for _, row in increasing_features.iterrows():
                    print(f"  {row['Feature']:<25}: +{row['SHAP_Importance_Signed']:>8.4f} "
                        f"({row['Relative_Contribution_Pct']:>4.1f}% of total impact)")
                
                print("\nFeatures that DECREASE target value:")
                
                for _, row in decreasing_features.iterrows():
                    print(f"  {row['Feature']:<25}: {row['SHAP_Importance_Signed']:>8.4f} "
                        f"({row['Relative_Contribution_Pct']:>4.1f}% of total impact)")
                
                # Save SHAP importance
                shap_filename = f"SHAP_feature_importance_{month_id}.csv"
                shap_path = os.path.join(output_dir, shap_filename)
                shap_importance.to_csv(shap_path, index=False)
                print(f"\nSHAP feature importance saved to {shap_path}")
                
                # Save detailed analysis
                summary_filename = f"SHAP_summary_{month_id}.txt"
                summary_path = os.path.join(output_dir, summary_filename)
                
                with open(summary_path, 'w') as f:
                    f.write(f"SHAP Analysis Summary - {month_id}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Target: {target_column}\n")
                    f.write(f"Model Type: {model_type}\n")
                    f.write(f"Explainer: {explainer_type}\n")
                    f.write(f"Samples analyzed: {len(X_test_sample)}\n")
                    f.write(f"Baseline target value: {baseline_value:.2f}\n")
                    f.write(f"Total SHAP contribution: {total_shap_contribution:.4f}\n\n")
                    
                    f.write("TOP FEATURES THAT INCREASE TARGET VALUE:\n")
                    f.write("-"*50 + "\n")
                    for _, row in increasing_features.iterrows():
                        f.write(f"{row['Feature']:<25}: +{row['SHAP_Importance_Signed']:>8.4f} "
                                f"({row['Relative_Contribution_Pct']:>4.1f}% of total)\n")
                    
                    f.write("\nTOP FEATURES THAT DECREASE TARGET VALUE:\n")
                    f.write("-"*50 + "\n")
                    for _, row in decreasing_features.iterrows():
                        f.write(f"{row['Feature']:<25}: {row['SHAP_Importance_Signed']:>8.4f} "
                                f"({row['Relative_Contribution_Pct']:>4.1f}% of total)\n")
                
                print(f"Detailed SHAP analysis saved to {summary_path}")
                
                # Update result
                result.update({
                    "shap_importance_path": shap_path,
                    "summary_path": summary_path,
                    "baseline_value": baseline_value,
                    "total_shap_contribution": total_shap_contribution,
                    "top_increasing_features": increasing_features.head(5)['Feature'].tolist(),
                    "top_decreasing_features": decreasing_features.head(5)['Feature'].tolist()
                })
            
            print("="*60)
            print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*60)
            
            return result
            
        except Exception as e:
            print(f"Error in SHAP analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
        
    def apply_smote_tomek_resampling(self, X_train, y_train, target_column, imbalance_threshold, random_state=42):
        """
        Apply SMOTE-Tomek resampling to training data if class imbalance exceeds threshold.
        
        This method checks for class imbalance in categorical target variables and applies
        SMOTE-Tomek resampling if the minority class percentage falls below the specified threshold.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series
            Training target values.
        target_column : str
            Name of the target column.
        imbalance_threshold : float
            Threshold for minority class percentage. If minority class is below this percentage,
            SMOTE-Tomek will be applied.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
            
        Returns:
        --------
        tuple
            (X_train_resampled, y_train_resampled, smote_applied, resampling_info)
            - X_train_resampled: Potentially resampled training features
            - y_train_resampled: Potentially resampled training target
            - smote_applied: Boolean indicating if SMOTE-Tomek was applied
            - resampling_info: Dictionary with resampling statistics
        """
        # Store original training set size for comparison
        original_train_size = len(y_train)
        smote_applied = False
        resampling_info = {
            "original_size": original_train_size,
            "final_size": original_train_size,
            "samples_added": 0,
            "original_distribution": {},
            "final_distribution": {},
            "minority_class_pct": 0.0,
            "threshold_used": imbalance_threshold
        }
        
        # Only apply SMOTE-Tomek for categorical targets
        if target_column not in CATEGORIAL_TARGET_FEATURES:
            print(f"Target '{target_column}' is not categorical. Skipping SMOTE-Tomek resampling.")
            return X_train, y_train, smote_applied, resampling_info
        
        # Check class distribution in training set
        class_counts = y_train.value_counts()
        total_samples = len(y_train)
        class_percentages = (class_counts / total_samples * 100).sort_values(ascending=True)
        
        minority_class_pct = class_percentages.iloc[0]  # Smallest class percentage
        minority_class_label = class_percentages.index[0]
        
        # Store original distribution
        resampling_info["original_distribution"] = class_percentages.to_dict()
        resampling_info["minority_class_pct"] = minority_class_pct
        
        print(f"\nClass distribution in training set:")
        for class_label, percentage in class_percentages.items():
            print(f"  {class_label}: {percentage:.2f}%")
        
        # Apply SMOTE-Tomek if minority class is below threshold
        if minority_class_pct < imbalance_threshold:
            print(f"\nDetected significant class imbalance (minority class '{minority_class_label}': {minority_class_pct:.2f}%)")
            print(f"Applying SMOTE-Tomek to balance the training set (threshold: {imbalance_threshold}%)...")
            
            try:
                # Initialize SMOTE-Tomek
                smote_tomek = SMOTETomek(random_state=random_state)
                
                # Apply resampling to training data only (never touch test data)
                X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
                
                # Convert back to pandas DataFrame/Series to maintain structure and column names
                X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                y_train = pd.Series(y_train_resampled, name=y_train.name)
                
                # Calculate new distribution
                new_class_counts = y_train.value_counts()
                new_total_samples = len(y_train)
                new_class_percentages = (new_class_counts / new_total_samples * 100).sort_values(ascending=True)
                
                # Update resampling info
                resampling_info["final_size"] = new_total_samples
                resampling_info["samples_added"] = new_total_samples - original_train_size
                resampling_info["final_distribution"] = new_class_percentages.to_dict()
                
                print(f"After SMOTE-Tomek resampling:")
                print(f"  Training set size: {new_total_samples} (was {original_train_size}, change: +{new_total_samples - original_train_size})")
                for class_label, percentage in new_class_percentages.items():
                    count = new_class_counts[class_label]
                    original_count = class_counts.get(class_label, 0)
                    print(f"  {class_label}: {count} samples ({percentage:.2f}%, was {original_count})")
                
                smote_applied = True
                
            except Exception as e:
                print(f"Error applying SMOTE-Tomek: {str(e)}")
                print("Continuing with original imbalanced dataset...")
                smote_applied = False
                # Keep original data
                X_train = X_train
                y_train = y_train
        else:
            print(f"\nClass distribution acceptable (minority class '{minority_class_label}': {minority_class_pct:.2f}%)")
            print(f"Skipping SMOTE-Tomek resampling (threshold: {imbalance_threshold}%)")
        
        # Set final distribution in resampling info
        if not smote_applied:
            resampling_info["final_distribution"] = resampling_info["original_distribution"]
        
        return X_train, y_train, smote_applied, resampling_info

    def apply_edited_nearest_neighbors_resampling(self, X_train, y_train, target_column, imbalance_threshold, random_state=42, n_neighbors=3):

        """
        Apply EditedNearestNeighbors undersampling to training data if class imbalance exceeds threshold.
        
        This method checks for class imbalance in categorical target variables and applies
        EditedNearestNeighbors undersampling if the minority class percentage falls below the specified threshold.
        Unlike SMOTE-Tomek which oversamples, ENN removes instances that don't agree with their neighbors,
        typically resulting in a cleaner but smaller dataset.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series
            Training target values.
        target_column : str
            Name of the target column.
        imbalance_threshold : float
            Threshold for minority class percentage. If minority class is below this percentage,
            EditedNearestNeighbors will be applied.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
        n_neighbors : int, optional
            Number of neighbors to consider for EditedNearestNeighbors. Defaults to 3.
            
        Returns:
        --------
        tuple
            (X_train_resampled, y_train_resampled, enn_applied, resampling_info)
            - X_train_resampled: Potentially resampled training features
            - y_train_resampled: Potentially resampled training target
            - enn_applied: Boolean indicating if EditedNearestNeighbors was applied
            - resampling_info: Dictionary with resampling statistics
        """
        from imblearn.under_sampling import EditedNearestNeighbours
        
        # Store original training set size for comparison
        original_train_size = len(y_train)
        enn_applied = False
        resampling_info = {
            "original_size": original_train_size,
            "final_size": original_train_size,
            "samples_removed": 0,
            "original_distribution": {},
            "final_distribution": {},
            "minority_class_pct": 0.0,
            "threshold_used": imbalance_threshold,
            "n_neighbors": n_neighbors
        }
        
        # Only apply ENN for categorical targets
        if target_column not in CATEGORIAL_TARGET_FEATURES:
            print(f"Target '{target_column}' is not categorical. Skipping EditedNearestNeighbors resampling.")
            return X_train, y_train, enn_applied, resampling_info
        
        # Check class distribution in training set
        class_counts = y_train.value_counts()
        total_samples = len(y_train)
        class_percentages = (class_counts / total_samples * 100).sort_values(ascending=True)
        
        minority_class_pct = class_percentages.iloc[0]  # Smallest class percentage
        minority_class_label = class_percentages.index[0]
        
        # Store original distribution
        resampling_info["original_distribution"] = class_percentages.to_dict()
        resampling_info["minority_class_pct"] = minority_class_pct
        
        print(f"\nClass distribution in training set:")
        for class_label, percentage in class_percentages.items():
            print(f"  {class_label}: {percentage:.2f}%")
        
        # Apply EditedNearestNeighbors if minority class is below threshold
        if minority_class_pct < imbalance_threshold:
            print(f"\nDetected significant class imbalance (minority class '{minority_class_label}': {minority_class_pct:.2f}%)")
            print(f"Applying EditedNearestNeighbors to clean the training set (threshold: {imbalance_threshold}%, n_neighbors: {n_neighbors})...")
            
            try:
                # Initialize EditedNearestNeighbors
                enn = EditedNearestNeighbours(n_neighbors=n_neighbors, random_state=random_state)
                
                # Apply undersampling to training data only (never touch test data)
                X_train_resampled, y_train_resampled = enn.fit_resample(X_train, y_train)
                
                # Convert back to pandas DataFrame/Series to maintain structure and column names
                X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                y_train = pd.Series(y_train_resampled, name=y_train.name)
                
                # Calculate new distribution
                new_class_counts = y_train.value_counts()
                new_total_samples = len(y_train)
                new_class_percentages = (new_class_counts / new_total_samples * 100).sort_values(ascending=True)
                
                # Update resampling info
                resampling_info["final_size"] = new_total_samples
                resampling_info["samples_removed"] = original_train_size - new_total_samples
                resampling_info["final_distribution"] = new_class_percentages.to_dict()
                
                print(f"After EditedNearestNeighbors undersampling:")
                print(f"  Training set size: {new_total_samples} (was {original_train_size}, change: -{original_train_size - new_total_samples})")
                for class_label, percentage in new_class_percentages.items():
                    count = new_class_counts[class_label]
                    original_count = class_counts.get(class_label, 0)
                    removed_count = original_count - count
                    print(f"  {class_label}: {count} samples ({percentage:.2f}%, was {original_count}, removed {removed_count})")
                
                enn_applied = True
                
            except Exception as e:
                print(f"Error applying EditedNearestNeighbors: {str(e)}")
                print("Continuing with original imbalanced dataset...")
                enn_applied = False
                # Keep original data
                X_train = X_train
                y_train = y_train
        else:
            print(f"\nClass distribution acceptable (minority class '{minority_class_label}': {minority_class_pct:.2f}%)")
            print(f"Skipping EditedNearestNeighbors undersampling (threshold: {imbalance_threshold}%)")
        
        # Set final distribution in resampling info
        if not enn_applied:
            resampling_info["final_distribution"] = resampling_info["original_distribution"]
        
        return X_train, y_train, enn_applied, resampling_info


        """
        Apply class weight parameter to a model class if supported.
        
        Parameters:
        -----------
        model_class : class
            The sklearn model class (e.g., DecisionTreeClassifier).
        base_params : dict
            Base parameters for the model.
        class_weight_param : various
            Class weight parameter to apply.
            
        Returns:
        --------
        dict
            Updated parameters including class_weight if supported.
        """
        updated_params = base_params.copy()
        
        # Check if the model supports class_weight
        try:
            # Try creating a dummy instance to check if class_weight is supported
            dummy_model = model_class()
            if hasattr(dummy_model, 'class_weight') or 'class_weight' in dummy_model.get_params():
                updated_params['class_weight'] = class_weight_param
                print(f"Applied class_weight to {model_class.__name__}")
            else:
                print(f"Model {model_class.__name__} does not support class_weight parameter")
        except Exception as e:
            print(f"Could not determine class_weight support for {model_class.__name__}: {e}")
        
        return updated_params
    
    def merge_all_preprocessed_files(self):

        """
        Merge all preprocessed CSV files into a single file with a Month column.
        
        This method:
        1. Finds all preprocessed CSV files in the preprocessed output directory
        2. Extracts the month number from each filename
        3. Reads each file and adds a Month column with the extracted month number
        4. Merges all files into one DataFrame
        5. Saves the merged file to the all_preprocessed directory
        
        Returns:
        --------
        dict
            A summary of the merge operation results.
        """
        try:
            print(f"\nMerging all preprocessed files...")
            
            # Create output directory
            all_preprocessed_dir = os.path.join(self.project_root, ALL_PREPROCESSED_OUTPUT_FOLDER)
            os.makedirs(all_preprocessed_dir, exist_ok=True)
            
            # Find all preprocessed CSV files
            preprocessed_pattern = os.path.join(self.preprocessed_dir, f"{DATA_FILE_PREFIX_FOR_TRAINING}*.csv")
            preprocessed_files = glob.glob(preprocessed_pattern)
            
            if not preprocessed_files:
                print("No preprocessed files found to merge.")
                return {
                    "success": False,
                    "error": "No preprocessed files found"
                }
            
            print(f"Found {len(preprocessed_files)} preprocessed files to merge:")
            for file_path in preprocessed_files:
                print(f"  - {os.path.basename(file_path)}")
            
            # Initialize list to store DataFrames
            all_dataframes = []
            file_info = []
            
            # Process each file
            for file_path in preprocessed_files:
                try:
                    filename = os.path.basename(file_path)
                    print(f"\nProcessing {filename}...")
                    
                    # Extract month number from filename using regex
                    # Pattern matches: preprocessed_data_YYYY-YYYY_MM.csv or preprocessed_data_YYYY_MM.csv
                    month_match = re.search(r'(\d{2})\.csv$', filename)
                    
                    if not month_match:
                        print(f"Warning: Could not extract month from filename {filename}. Skipping.")
                        continue
                    
                    month_number = int(month_match.group(1))
                    print(f"  Extracted month: {month_number}")
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"  Warning: File {filename} is empty. Skipping.")
                        continue
                    
                    # Add Month column
                    df['Month'] = month_number
                    
                    # Add to list
                    all_dataframes.append(df)
                    file_info.append({
                        "filename": filename,
                        "month": month_number,
                        "rows": len(df),
                        "columns": len(df.columns)
                    })
                    
                    print(f"  Added {len(df)} rows with {len(df.columns)} columns")
                    
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
                    continue
            
            if not all_dataframes:
                print("No valid files were processed.")
                return {
                    "success": False,
                    "error": "No valid files were processed"
                }
            
            # Merge all DataFrames
            print(f"\nMerging {len(all_dataframes)} DataFrames...")
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Reorder columns to put Month first (after target columns if they exist)
            columns = list(merged_df.columns)
            
            # Remove Month from its current position
            columns.remove('Month')
            
            # Find target columns and insert Month after them
            target_cols = [col for col in ['differenceInMinutes', 'differenceInMinutes_offset', 'trainDelayed', 'cancelled'] 
                        if col in columns]
            
            if target_cols:
                # Insert Month after the last target column
                last_target_idx = max([columns.index(col) for col in target_cols])
                columns.insert(last_target_idx + 1, 'Month')
            else:
                # Insert Month at the beginning if no target columns found
                columns.insert(0, 'Month')
            
            # Reorder the DataFrame
            merged_df = merged_df[columns]
            
            # Generate output filename with current timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"all_preprocessed_data_{timestamp}.csv"
            output_path = os.path.join(all_preprocessed_dir, output_filename)
            
            # Save merged file
            merged_df.to_csv(output_path, index=False)
            
            # Print summary
            print(f"\n{'='*60}")
            print("MERGE OPERATION COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Output file: {output_path}")
            print(f"Total rows: {len(merged_df):,}")
            print(f"Total columns: {len(merged_df.columns)}")
            print(f"Files merged: {len(all_dataframes)}")
            
            # Show month distribution
            month_distribution = merged_df['Month'].value_counts().sort_index()
            print(f"\nMonth distribution:")
            for month, count in month_distribution.items():
                print(f"  Month {month:2d}: {count:,} rows")
            
            # Show file details
            print(f"\nFile details:")
            for info in file_info:
                print(f"  {info['filename']}: Month {info['month']}, {info['rows']:,} rows")
            
            print(f"\nColumn order:")
            for i, col in enumerate(merged_df.columns, 1):
                print(f"  {i:2d}. {col}")
            
            # Also save a summary file
            summary_filename = f"merge_summary_{timestamp}.txt"
            summary_path = os.path.join(all_preprocessed_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write(f"Preprocessed Files Merge Summary\n")
                f.write("="*40 + "\n\n")
                f.write(f"Merge timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output file: {output_filename}\n")
                f.write(f"Total rows: {len(merged_df):,}\n")
                f.write(f"Total columns: {len(merged_df.columns)}\n")
                f.write(f"Files merged: {len(all_dataframes)}\n\n")
                
                f.write("Month Distribution:\n")
                f.write("-" * 20 + "\n")
                for month, count in month_distribution.items():
                    f.write(f"Month {month:2d}: {count:,} rows\n")
                
                f.write("\nFile Details:\n")
                f.write("-" * 20 + "\n")
                for info in file_info:
                    f.write(f"{info['filename']}: Month {info['month']}, {info['rows']:,} rows\n")
                
                f.write("\nColumn Order:\n")
                f.write("-" * 20 + "\n")
                for i, col in enumerate(merged_df.columns, 1):
                    f.write(f"{i:2d}. {col}\n")
            
            print(f"Merge summary saved to: {summary_path}")
            
            return {
                "success": True,
                "output_path": output_path,
                "summary_path": summary_path,
                "total_rows": len(merged_df),
                "total_columns": len(merged_df.columns),
                "files_merged": len(all_dataframes),
                "month_distribution": month_distribution.to_dict(),
                "file_details": file_info
            }
            
        except Exception as e:
            print(f"Error merging preprocessed files: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def split_combined_dataset(self, test_size=0.3, random_state=42):
        """
        Split the combined preprocessed dataset into training and testing sets and save them separately.
        Uses simple random train/test split with optional stratification for classification problems.
        Automatically applies the configured resampling method for severely imbalanced categorical targets.
        
        Parameters:
        -----------
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
            from sklearn.model_selection import train_test_split
            
            print(f"\n{'='*60}")
            print("SPLITTING COMBINED PREPROCESSED DATASET")
            print(f"{'='*60}")
            
            # Find the most recent merged file
            all_preprocessed_dir = os.path.join(self.project_root, ALL_PREPROCESSED_OUTPUT_FOLDER)
            
            if not os.path.exists(all_preprocessed_dir):
                error_msg = f"All preprocessed directory not found: {all_preprocessed_dir}"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Find all merged files
            merged_files = glob.glob(os.path.join(all_preprocessed_dir, "all_preprocessed_data_*.csv"))
            
            if not merged_files:
                error_msg = f"No merged preprocessed files found in {all_preprocessed_dir}"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Use the most recent file
            latest_file = max(merged_files, key=os.path.getctime)
            print(f"Loading combined dataset from: {os.path.basename(latest_file)}")
            
            # Load the combined dataset
            combined_df = pd.read_csv(latest_file)
            
            if combined_df.empty:
                error_msg = "Combined dataset is empty"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"Loaded combined dataset with {len(combined_df):,} rows and {len(combined_df.columns)} columns")
            
            # Check if Month column exists
            if 'Month' not in combined_df.columns:
                error_msg = "Month column not found in combined dataset"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Show month distribution
            month_counts = combined_df['Month'].value_counts().sort_index()
            print(f"\nMonth distribution in combined data:")
            for month, count in month_counts.items():
                print(f"  Month {month:2d}: {count:,} rows")
            
            # Identify target column
            target_options = VALID_TARGET_FEATURES
            target_column = None
            
            for option in target_options:
                if option in combined_df.columns:
                    target_column = option
                    break
            
            if not target_column:
                error_msg = "No target column found in combined dataset"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"Identified target column: {target_column}")
            
            # Show available months for reference
            available_months = sorted(combined_df['Month'].unique())
            print(f"Available months: {available_months}")
            
            # Use simple random train/test split instead of month-based splitting
            test_size = 0.3  # 30% for test, 70% for training
            print(f"Using simple random train/test split with test_size={test_size}")
            
            # Check if we have classification or regression problem for stratification
            is_classification_problem = target_column in CATEGORIAL_TARGET_FEATURES
            
            if is_classification_problem:
                # Use stratified split for classification
                train_df, test_df = train_test_split(
                    combined_df, 
                    test_size=test_size, 
                    stratify=combined_df[target_column], 
                    random_state=random_state
                )
                print(f"Used stratified split for classification target '{target_column}'")
            else:
                # Use regular split for regression
                train_df, test_df = train_test_split(
                    combined_df, 
                    test_size=test_size, 
                    random_state=random_state
                )
                print(f"Used random split for regression target '{target_column}'")
            
            print(f"\nInitial data split:")
            print(f"  Training data: {len(train_df):,} rows")
            print(f"  Test data: {len(test_df):,} rows")
            print(f"  Train/Test ratio: {len(train_df)/len(test_df):.2f}")
            
            if len(train_df) == 0 or len(test_df) == 0:
                error_msg = "Invalid data split - train or test set is empty"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Prepare features and target for resampling
            feature_columns = [col for col in combined_df.columns if col not in [target_column, 'Month', 'data_year']]
            
            X_train = train_df[feature_columns]
            y_train = train_df[target_column]
            X_test = test_df[feature_columns]
            y_test = test_df[target_column]
            
            print(f"Feature columns: {len(feature_columns)}")
            
            # Store original training set size for comparison
            original_train_size = len(y_train)
            
            # ========== APPLY CONFIGURED RESAMPLING METHOD ==========
            print(f"\nApplying resampling method: {RESAMPLING_METHOD}")
            
            if RESAMPLING_METHOD == "SMOTE_TOMEK":
                X_train, y_train, resampling_applied, resampling_info = self.apply_smote_tomek_resampling(
                    X_train, y_train, target_column, IMBALANCE_THRESHOLD, random_state
                )
                resampling_method_used = "SMOTE-Tomek"
                
            elif RESAMPLING_METHOD == "EDITED_NEAREST_NEIGHBORS":
                X_train, y_train, resampling_applied, resampling_info = self.apply_edited_nearest_neighbors_resampling(
                    X_train, y_train, target_column, IMBALANCE_THRESHOLD, random_state, n_neighbors=ENN_N_NEIGHBORS
                )
                resampling_method_used = "EditedNearestNeighbors"
                
            elif RESAMPLING_METHOD == "NONE":
                print("No resampling method configured - using original data")
                resampling_applied = False
                resampling_method_used = "None"
                resampling_info = {
                    "original_size": original_train_size,
                    "final_size": original_train_size,
                    "samples_changed": 0,
                    "original_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "final_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "threshold_used": IMBALANCE_THRESHOLD,
                    "method": "none"
                }
                
            else:
                print(f"Warning: Unknown resampling method '{RESAMPLING_METHOD}'. Using original data.")
                resampling_applied = False
                resampling_method_used = "Unknown (fallback to None)"
                resampling_info = {
                    "original_size": original_train_size,
                    "final_size": original_train_size,
                    "samples_changed": 0,
                    "original_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "final_distribution": (y_train.value_counts(normalize=True) * 100).to_dict() if target_column in CATEGORIAL_TARGET_FEATURES else {},
                    "threshold_used": IMBALANCE_THRESHOLD,
                    "method": "fallback_none"
                }
            
            # Recombine features and target for saving (keeping Month column for reference)
            train_df_final = pd.concat([X_train, y_train], axis=1)
            test_df_final = pd.concat([X_test, y_test], axis=1)
            
            # Add Month column back for reference
            # For training data, we need to reconstruct the Month information
            if resampling_applied and len(train_df_final) != len(train_df):
                # If resampling changed the size, we need to handle Month column carefully
                most_common_month = train_df['Month'].mode()[0] if len(train_df) > 0 else available_months[0]
                
                # Create Month column for resampled training data
                train_months_reconstructed = []
                original_train_months = train_df['Month'].values
                
                # First, add original months for original samples
                train_months_reconstructed.extend(original_train_months)
                
                # For additional samples (if any), use the most common month
                additional_samples = len(train_df_final) - len(original_train_months)
                if additional_samples > 0:
                    train_months_reconstructed.extend([most_common_month] * additional_samples)
                    print(f"  Added Month {most_common_month} to {additional_samples} resampled training samples")
                
                train_df_final['Month'] = train_months_reconstructed[:len(train_df_final)]
            else:
                # No resampling or undersampling - use original months
                train_df_final['Month'] = train_df['Month'].values
            
            test_df_final['Month'] = test_df['Month'].values
            
            # Create filenames for train and test sets
            train_filename = f"combined_data_train.csv"
            test_filename = f"combined_data_test.csv"
            
            train_path = os.path.join(all_preprocessed_dir, train_filename)
            test_path = os.path.join(all_preprocessed_dir, test_filename)
            
            # Save the datasets
            train_df_final.to_csv(train_path, index=False)
            test_df_final.to_csv(test_path, index=False)
            
            print(f"\nDataset split completed:")
            print(f"  Training set saved to: {train_filename}")
            print(f"  Test set saved to: {test_filename}")
            print(f"  Final training set size: {len(train_df_final):,} rows")
            print(f"  Final test set size: {len(test_df_final):,} rows")
            
            # Print and log distribution statistics
            print("\nFinal Distribution Statistics:")
            
            # Use the logging context manager for distribution statistics
            with self.get_logger("split_combined_dataset_distribution.log", "split_combined_distribution", "combined") as logger:
                logger.info(f"Combined dataset split completed - Train size: {len(train_df_final)}, Test size: {len(test_df_final)}")
                logger.info(f"Target column: {target_column}")
                logger.info(f"Data source: {os.path.basename(latest_file)}")
                logger.info(f"Available months: {available_months}")
                logger.info(f"Split method: Random train/test split (test_size=0.3)")
                logger.info(f"Stratified split: {is_classification_problem}")
                logger.info(f"Resampling method configured: {RESAMPLING_METHOD}")
                logger.info(f"Resampling method used: {resampling_method_used}")
                logger.info(f"Resampling applied: {resampling_applied}")
                
                if resampling_applied:
                    if RESAMPLING_METHOD == "SMOTE_TOMEK":
                        logger.info(f"Original train size: {original_train_size}, Final train size: {len(train_df_final)}")
                        logger.info(f"Samples added: {resampling_info.get('samples_added', 0)}")
                    elif RESAMPLING_METHOD == "EDITED_NEAREST_NEIGHBORS":
                        logger.info(f"Original train size: {original_train_size}, Final train size: {len(train_df_final)}")
                        logger.info(f"Samples removed: {resampling_info.get('samples_removed', 0)}")
                        logger.info(f"N_neighbors used: {resampling_info.get('n_neighbors', ENN_N_NEIGHBORS)}")
                
                # For categorical targets, show the distribution in percentages
                if target_column in CATEGORIAL_TARGET_FEATURES:
                    print("\nOriginal Combined Distribution (%):")
                    original_dist = combined_df[target_column].value_counts(normalize=True) * 100
                    print(original_dist)
                    
                    print("\nFinal Training Set Distribution (%):")
                    train_dist = y_train.value_counts(normalize=True) * 100
                    print(train_dist)
                    
                    print("\nTest Set Distribution (%):")
                    test_dist = y_test.value_counts(normalize=True) * 100
                    print(test_dist)
                    
                    # Log the categorical distributions
                    logger.info("=== Categorical Target Distribution Analysis ===")
                    logger.info("Original Combined Distribution (%):")
                    for label, percentage in original_dist.items():
                        logger.info(f"  {label}: {percentage:.2f}%")
                    
                    logger.info("Final Training Set Distribution (%):")
                    for label, percentage in train_dist.items():
                        logger.info(f"  {label}: {percentage:.2f}%")
                    
                    logger.info("Test Set Distribution (%):")
                    for label, percentage in test_dist.items():
                        logger.info(f"  {label}: {percentage:.2f}%")
                    
                    # Log resampling-specific info
                    if resampling_applied:
                        if RESAMPLING_METHOD == "SMOTE_TOMEK":
                            logger.info("=== SMOTE-Tomek Resampling Applied ===")
                            logger.info(f"Reason: Minority class below {IMBALANCE_THRESHOLD}% threshold")
                            logger.info(f"Training samples added: {resampling_info['samples_added']}")
                        elif RESAMPLING_METHOD == "EDITED_NEAREST_NEIGHBORS":
                            logger.info("=== EditedNearestNeighbors Undersampling Applied ===")
                            logger.info(f"Reason: Minority class below {IMBALANCE_THRESHOLD}% threshold")
                            logger.info(f"Training samples removed: {resampling_info['samples_removed']}")
                            logger.info(f"N_neighbors parameter: {resampling_info['n_neighbors']}")
                        
                        # Log final distribution for any resampling method
                        for label, percentage in resampling_info["final_distribution"].items():
                            final_count = (percentage / 100) * len(train_df_final)
                            logger.info(f"  {label}: {final_count:.0f} samples ({percentage:.2f}%)")
                    
                else:
                    # For continuous targets like differenceInMinutes, show basic stats
                    original_mean = combined_df[target_column].mean()
                    original_std = combined_df[target_column].std()
                    train_mean = y_train.mean()
                    train_std = y_train.std()
                    test_mean = y_test.mean()
                    test_std = y_test.std()
                    
                    print("\nOriginal Combined Distribution:")
                    print(f"Mean: {original_mean:.2f}, Std: {original_std:.2f}")
                    
                    print("\nTraining Set Distribution:")
                    print(f"Mean: {train_mean:.2f}, Std: {train_std:.2f}")
                    
                    print("\nTest Set Distribution:")
                    print(f"Mean: {test_mean:.2f}, Std: {test_std:.2f}")
                    
                    # Log the continuous distributions
                    logger.info("=== Continuous Target Distribution Analysis ===")
                    logger.info(f"Original Combined Distribution - Mean: {original_mean:.4f}, Std: {original_std:.4f}")
                    logger.info(f"Training Set Distribution - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
                    logger.info(f"Test Set Distribution - Mean: {test_mean:.4f}, Std: {test_std:.4f}")
            
            print(f"{'='*60}")
            print("COMBINED DATASET SPLIT COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            
            # Return summary
            return {
                "success": True,
                "data_source": os.path.basename(latest_file),
                "total_samples": len(combined_df),
                "original_train_size": original_train_size,
                "final_train_size": len(train_df_final),
                "test_size": len(test_df_final),
                "train_path": train_path,
                "test_path": test_path,
                "target_column": target_column,
                "available_months": available_months,
                "split_method": "random",
                "test_size_ratio": 0.3,
                "stratified": is_classification_problem,
                "resampling_method_configured": RESAMPLING_METHOD,
                "resampling_method_used": resampling_method_used,
                "resampling_applied": resampling_applied,
                "imbalance_threshold": IMBALANCE_THRESHOLD,
                "resampling_info": resampling_info
            }
            
        except Exception as e:
            print(f"Error splitting combined dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def train_decision_tree_combined_data(self, param_distributions=None, n_iter=None, cv=None, random_state=42):
        """
        Train a Decision Tree classifier with hyperparameter tuning using RandomizedSearchCV on combined preprocessed data.
        Uses pre-split data from the split_combined_dataset method.
        Includes SHAP analysis for enhanced model interpretability.
        Updated to include sample weights based on delay magnitude.
        
        Parameters:
        -----------
        param_distributions : dict, optional
            Dictionary with parameters names as keys and distributions or lists of parameters to try.
            Defaults to DECISION_TREE_PARAM_DISTRIBUTIONS from constants.
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
                from config.const import DECISION_TREE_PARAM_DISTRIBUTIONS
                param_distributions = DECISION_TREE_PARAM_DISTRIBUTIONS
            
            if n_iter is None:
                from config.const import RANDOM_SEARCH_ITERATIONS
                n_iter = RANDOM_SEARCH_ITERATIONS
                
            if cv is None:
                from config.const import RANDOM_SEARCH_CV_FOLDS
                cv = RANDOM_SEARCH_CV_FOLDS
            
            print(f"\n{'='*80}")
            print("TRAINING DECISION TREE ON PRE-SPLIT COMBINED DATA")
            print(f"{'='*80}")
            
            # Load pre-split data from the split_combined_dataset method
            all_preprocessed_dir = os.path.join(self.project_root, ALL_PREPROCESSED_OUTPUT_FOLDER)
            
            train_path = os.path.join(all_preprocessed_dir, "combined_data_train.csv")
            test_path = os.path.join(all_preprocessed_dir, "combined_data_test.csv")
            
            # Check if pre-split files exist
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                error_msg = f"Pre-split combined data files not found: {train_path} or {test_path}"
                print(f"Error: {error_msg}")
                print("Make sure split_combined_dataset was called before this method.")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Load the pre-split datasets
            print(f"Loading pre-split training data from: {os.path.basename(train_path)}")
            train_df = pd.read_csv(train_path)
            
            print(f"Loading pre-split test data from: {os.path.basename(test_path)}")
            test_df = pd.read_csv(test_path)
            
            if train_df.empty or test_df.empty:
                error_msg = "Pre-split datasets are empty"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"Loaded pre-split data:")
            print(f"  Training set: {len(train_df):,} rows")
            print(f"  Test set: {len(test_df):,} rows")
            
            # Show month distribution if Month column exists
            if 'Month' in train_df.columns and 'Month' in test_df.columns:
                train_months = sorted(train_df['Month'].unique())
                test_months = sorted(test_df['Month'].unique())
                print(f"  Train months: {train_months}")
                print(f"  Test months: {test_months}")
                
                # Show detailed month distribution
                print("\nMonth distribution in training data:")
                train_month_counts = train_df['Month'].value_counts().sort_index()
                for month, count in train_month_counts.items():
                    print(f"  Month {month:2d}: {count:,} rows")
                
                print("\nMonth distribution in test data:")
                test_month_counts = test_df['Month'].value_counts().sort_index()
                for month, count in test_month_counts.items():
                    print(f"  Month {month:2d}: {count:,} rows")
            
            # Identify target column
            target_options = VALID_TARGET_FEATURES
            target_column = None
            
            for option in target_options:
                if option in train_df.columns:
                    target_column = option
                    break
            
            if not target_column:
                error_msg = "No target column found in pre-split datasets"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"Identified target column: {target_column}")
            
            # Prepare features and target
            feature_columns = [col for col in train_df.columns if col not in [target_column, 'Month', 'data_year']]
            
            X_train = train_df[feature_columns]
            y_train = train_df[target_column]
            X_test = test_df[feature_columns]
            y_test = test_df[target_column]
            
            print(f"Feature columns: {len(feature_columns)}")
            print(f"Features: {feature_columns[:10]}..." if len(feature_columns) > 10 else f"Features: {feature_columns}")
            
            # Check if we have classification or regression problem
            is_classification = True
            if target_column in REGRESSION_PROBLEM:
                is_classification = False
                print(f"Target '{target_column}' indicates a regression problem")
            else:
                print(f"Target '{target_column}' indicates a classification problem")
            
            if is_classification:
                # Create sample weights for classification if delay info is available
                sample_weights = None
                if WEIGHT_DELAY_COLUMN in train_df.columns:
                    print("Using weighted samples based on delay magnitude for randomized search")
                    # Create sample weights based on delay magnitude
                    delay_col = WEIGHT_DELAY_COLUMN
                    sample_weights = np.ones(len(y_train))
                    
                    # Get delay values for each training sample
                    delays = train_df[delay_col].values
                    
                    # Apply weights - higher delays get higher weights
                    delayed_idx = (delays > TRAIN_DELAY_MINUTES)
                    if np.any(delayed_idx):
                        # Normalize delay values by mean positive delay
                        mean_delay = delays[delayed_idx].mean()
                        # Use configured maximum weight
                        sample_weights[delayed_idx] = np.minimum(MAX_SAMPLE_WEIGHT_CLASSIFICATION, 1 + delays[delayed_idx]/mean_delay)
                    
                    print(f"Created sample weights with range [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                
                # Create proper CV strategy for classification
                cv_strategy = StratifiedKFold(
                    n_splits=cv,       
                    shuffle=True,
                    random_state=random_state
                )

                # Initialize base classifier
                dt = DecisionTreeClassifier(random_state=random_state)
                
                print(f"Starting RandomizedSearchCV with {n_iter} iterations and {cv}-fold cross-validation...")
            
                # Run RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    dt, param_distributions, n_iter=n_iter, cv=cv_strategy, 
                    scoring=SCORE_METRIC, random_state=random_state, n_jobs=-1
                )
                
                # Fit RandomizedSearchCV with sample weights if available
                if sample_weights is not None:
                    print("Training RandomizedSearchCV with sample weights")
                    random_search.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    random_search.fit(X_train, y_train)
                
                best_params = random_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
                
                # Train model with best parameters
                best_dt = DecisionTreeClassifier(**best_params, random_state=random_state)
                
                # Fit the final model with sample weights if available
                if sample_weights is not None:
                    print("Training final model with sample weights")
                    best_dt.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    best_dt.fit(X_train, y_train)
                
                # Create output directory for combined data results
                combined_output_dir = os.path.join(self.project_root, "data/output/decision_tree_combined_data")
                os.makedirs(combined_output_dir, exist_ok=True)
                
                # === COMPREHENSIVE EVALUATION ===
                evaluation_result = self.evaluate_model_comprehensive(
                    model=best_dt,
                    X_test=X_test,
                    y_test=y_test,
                    model_name="Decision Tree with RandomizedSearchCV (Combined Data)",
                    month_id="combined",
                    output_dir=combined_output_dir,
                    target_column=target_column,
                    random_search_obj=random_search,
                    is_classification=is_classification
                )
                
                if not evaluation_result["success"]:
                    return {
                        "success": False,
                        "error": f"Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
                    }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': best_dt.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                print("\nFeature Importance (top 10):")
                print(feature_importance.head(10))
                
                # ========== SHAP ANALYSIS ==========
                print("\nPerforming SHAP analysis on the Combined Data Decision Tree model...")
                
                shap_result = self.analyze_model_with_shap(
                    model=best_dt,
                    X_test=X_test,
                    y_test=y_test,
                    model_type='classification',
                    month_id="combined",
                    output_dir=combined_output_dir,
                    target_column=target_column,
                    max_samples=1000,
                    random_state=random_state,
                    model_name="decision_tree_combined_data",
                    baseline_data=train_df
                )
                
                if shap_result.get("success", False):
                    print("SHAP analysis completed successfully for Combined Data Decision Tree model!")
                    
                    # Compare with standard importance if SHAP was successful
                    if "shap_importance_path" in shap_result:
                        print("\n" + "-"*60)
                        print("COMPARISON: Standard vs SHAP Feature Importance (Combined Data Decision Tree)")
                        print("-"*60)
                        
                        try:
                            # Load SHAP importance for comparison
                            shap_importance = pd.read_csv(shap_result["shap_importance_path"])
                            
                            # Merge the two importance measures
                            comparison = feature_importance.merge(
                                shap_importance[['Feature', 'SHAP_Importance_Abs', 'SHAP_Importance_Signed', 
                                            'SHAP_Percentage_Points', 'SHAP_Abs_Percentage_Points', 'Relative_Contribution_Pct']], 
                                on='Feature', how='left'
                            )
                            
                            print("Top 10 features by Standard Importance vs SHAP Importance:")
                            for _, row in comparison.head(10).iterrows():
                                direction = "↑" if row['SHAP_Importance_Signed'] > 0 else "↓"
                                shap_abs = row['SHAP_Abs_Percentage_Points'] if pd.notna(row['SHAP_Abs_Percentage_Points']) else 0
                                rel_contrib = row['Relative_Contribution_Pct'] if pd.notna(row['Relative_Contribution_Pct']) else 0
                                print(f"{row['Feature']:<25}: Standard={row['Importance']:>6.4f}, "
                                    f"SHAP={shap_abs:>5.2f}pp {direction}, "
                                    f"({rel_contrib:>4.1f}% of impact)")
                        except Exception as e:
                            print(f"Could not perform comparison: {e}")
                    
                else:
                    print(f"SHAP analysis failed: {shap_result.get('error', 'Unknown error')}")
                
                print("="*60)
                
                # Save the model and related files
                try:
                    import joblib
                    
                    # Save the model
                    model_filename = f"decision_tree_combined_data.joblib"
                    model_path = os.path.join(combined_output_dir, model_filename)
                    joblib.dump(best_dt, model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_combined_data.csv"
                    importance_path = os.path.join(combined_output_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to {importance_path}")
                    
                    # Save best parameters and training info
                    params_filename = f"training_summary_combined_data.txt"
                    params_path = os.path.join(combined_output_dir, params_filename)
                    with open(params_path, 'w') as f:
                        f.write(f"Combined Data Decision Tree Training Summary\n")
                        f.write("="*50 + "\n\n")
                        f.write(f"Training file: {os.path.basename(train_path)}\n")
                        f.write(f"Test file: {os.path.basename(test_path)}\n")
                        f.write(f"Training samples: {len(train_df):,}\n")
                        f.write(f"Test samples: {len(test_df):,}\n")
                        f.write(f"Split method: Random train/test split\n")
                        if 'Month' in train_df.columns:
                            f.write(f"Available months in data: {sorted(set(train_df['Month'].unique()) | set(test_df['Month'].unique()))}\n")
                        f.write(f"Target column: {target_column}\n")
                        f.write(f"Number of features: {len(feature_columns)}\n")
                        f.write(f"Used sample weights: {sample_weights is not None}\n\n")
                        f.write("Best Hyperparameters:\n")
                        for param, value in best_params.items():
                            f.write(f"{param}: {value}\n")
                        f.write(f"\nBest CV Score ({SCORE_METRIC}): {random_search.best_score_:.4f}\n")
                        f.write(f"Test {evaluation_result.get('optimized_metric_name', 'accuracy')}: {evaluation_result.get('optimized_metric', 0):.4f}\n")
                    print(f"Training summary saved to {params_path}")
                    
                    # Save sample weights information if used
                    if sample_weights is not None:
                        weights_filename = f"sample_weights_info_combined.txt"
                        weights_path = os.path.join(combined_output_dir, weights_filename)
                        with open(weights_path, 'w') as f:
                            f.write(f"Sample Weights Information - Combined Data\n")
                            f.write("="*40 + "\n")
                            f.write(f"Used sample weights: Yes\n")
                            f.write(f"Weight range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]\n")
                            f.write(f"Mean weight: {sample_weights.mean():.2f}\n")
                            f.write(f"Standard deviation: {sample_weights.std():.2f}\n")
                            f.write(f"Number of weighted samples: {(sample_weights > 1.0).sum()}\n")
                            f.write(f"Max weight constant used: {MAX_SAMPLE_WEIGHT_CLASSIFICATION}\n")
                        print(f"Sample weights info saved to {weights_path}")
                    
                    print(f"\n{'='*80}")
                    print("COMBINED DATA DECISION TREE TRAINING COMPLETED SUCCESSFULLY")
                    print(f"{'='*80}")
                    
                    return {
                        "success": True,
                        **evaluation_result,  # Include all evaluation results
                        "best_params": best_params,
                        "model_path": model_path,
                        "feature_importance_path": importance_path,
                        "shap_analysis": shap_result,
                        "used_sample_weights": sample_weights is not None,
                        "data_source": f"{os.path.basename(train_path)} + {os.path.basename(test_path)}",
                        "total_samples": len(train_df) + len(test_df),
                        "train_samples": len(train_df),
                        "test_samples": len(test_df),
                        "split_method": "random",
                        "available_months": sorted(set(train_df['Month'].unique()) | set(test_df['Month'].unique())) if 'Month' in train_df.columns else []
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
                    return {
                        "success": True,
                        **evaluation_result,  # Include evaluation results even if save failed
                        "best_params": best_params,
                        "model_saved": False,
                        "shap_analysis": shap_result,
                        "used_sample_weights": sample_weights is not None,
                        "data_source": f"{os.path.basename(train_path)} + {os.path.basename(test_path)}",
                        "total_samples": len(train_df) + len(test_df),
                        "train_samples": len(train_df),
                        "test_samples": len(test_df),
                        "split_method": "random"
                    }
            else:
                # For regression problems we would need a different approach
                print(f"Regression with Decision Trees on combined data not implemented for target {target_column}")
                return {
                    "success": False,
                    "error": f"Regression with Decision Trees on combined data not implemented for target {target_column}"
                }
        
        except Exception as e:
            print(f"Error in Combined Data Decision Tree training: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }