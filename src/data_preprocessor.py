from contextlib import contextmanager
import os
import pandas as pd
import re
import ast
import joblib
import logging
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.file_utils import generate_output_path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
            

from config.const import (
    ALL_PREPROCESSED_OUTPUT_FOLDER,
    ALL_WEATHER_FEATURES,
    DATA_FILE_PREFIX_FOR_TRAINING,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_WEATHER_CONDITIONS,
    BOOLEAN_FEATURES, 
    CATEGORICAL_FEATURES,
    OUTPUT_FOLDER,
    POSSIBLE_INDICATORS,
    PREPROCESSING_STATE_MACHINE,
    PREPROCESSED_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    REGRESSION_PROBLEM,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    SCORE_METRIC,
    STRONG_INDICATORS,
    TRAIN_DELAY_MINUTES,
    TRAIN_DELAYED_TARGET_COLUMN,
    VALID_TARGET_FEATURES,
    VALID_TRAIN_PREDICTION_FEATURES,
    WEATHER_COLS_TO_MERGE,
    WEIGHT_DELAY_COLUMN,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    DEFAULT_TARGET_FEATURE,
    MAX_SAMPLE_WEIGHT_CLASSIFICATION,
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

        if state_machine.get("process_causes_column", False):
            if result["data"] is not None:
                try:
                    print(f"    → process_causes_column")
                    causes_df = self.process_causes_column(dataframe=result["data"])
                    
                    if causes_df is not None:
                        result["data"] = causes_df
                        result["steps_executed"].append("process_causes_column")
                        result["file_info"]["rows"] = len(causes_df)
                        result["file_info"]["columns"] = len(causes_df.columns)
                        print(f"      ✓ Processed causes column for {len(causes_df)} rows")
                    else:
                        result["errors"].append("process_causes_column failed")
                        print(f"      ✗ Failed to process causes column")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"process_causes_column failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ process_causes_column (no data available)")
                result["errors"].append("process_causes_column skipped - no data available")
        else:
            print(f"    ⊝ process_causes_column (disabled)")
        
        if state_machine.get("add_train_delayed_feature", False):
            if result["data"] is not None:
                try:
                    print(f"    → add_train_delayed_feature")
                    delayed_df = self.add_train_delayed_feature(dataframe=result["data"])
                    
                    if delayed_df is not None:
                        result["data"] = delayed_df
                        result["steps_executed"].append("add_train_delayed_feature")
                        result["file_info"]["rows"] = len(delayed_df)
                        result["file_info"]["columns"] = len(delayed_df.columns)
                        print(f"      ✓ Added trainDelayed feature to {len(delayed_df)} rows")
                    else:
                        result["errors"].append("add_train_delayed_feature failed")
                        print(f"      ✗ Failed to add trainDelayed feature")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"add_train_delayed_feature failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ add_train_delayed_feature (no data available)")
                result["errors"].append("add_train_delayed_feature skipped - no data available")
        else:
            print(f"    ⊝ add_train_delayed_feature (disabled)")

        if state_machine.get("merge_weather_columns", False):
            if result["data"] is not None:
                try:
                    print(f"    → merge_weather_columns")
                    merged_df = self.merge_weather_columns(dataframe=result["data"], month_id=file_id)
                    
                    if merged_df is not None:
                        result["data"] = merged_df
                        result["steps_executed"].append("merge_weather_columns")
                        result["file_info"]["rows"] = len(merged_df)
                        result["file_info"]["columns"] = len(merged_df.columns)
                        print(f"      ✓ Merged weather columns for {len(merged_df)} rows")
                    else:
                        result["errors"].append("merge_weather_columns failed")
                        print(f"      ✗ Failed to merge weather columns")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"merge_weather_columns failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ merge_weather_columns (no data available)")
                result["errors"].append("merge_weather_columns skipped - no data available")
        else:
            print(f"    ⊝ merge_weather_columns (disabled)")


        if state_machine.get("process_actual_time_column", False):
            if result["data"] is not None:
                try:
                    print(f"    → process_actual_time_column")
                    time_df = self.process_actual_time_column(dataframe=result["data"], month_id=file_id)
                    
                    if time_df is not None:
                        result["data"] = time_df
                        result["steps_executed"].append("process_actual_time_column")
                        result["file_info"]["rows"] = len(time_df)
                        result["file_info"]["columns"] = len(time_df.columns)
                        print(f"      ✓ Extracted temporal features for {len(time_df)} rows")
                    else:
                        result["errors"].append("process_actual_time_column failed")
                        print(f"      ✗ Failed to process actualTime column")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"process_actual_time_column failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ process_actual_time_column (no data available)")
                result["errors"].append("process_actual_time_column skipped - no data available")
        else:
            print(f"    ⊝ process_actual_time_column (disabled)")

        if state_machine.get("filter_columns", False):
            if result["data"] is not None:
                try:
                    print(f"    → filter_columns")
                    filtered_df = self.filter_columns(dataframe=result["data"], month_id=file_id)
                    
                    if filtered_df is not None:
                        result["data"] = filtered_df
                        result["steps_executed"].append("filter_columns")
                        result["file_info"]["rows"] = len(filtered_df)
                        result["file_info"]["columns"] = len(filtered_df.columns)
                        print(f"      ✓ Filtered columns for {len(filtered_df)} rows, {len(filtered_df.columns)} columns")
                    else:
                        result["errors"].append("filter_columns failed")
                        print(f"      ✗ Failed to filter columns")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"filter_columns failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ filter_columns (no data available)")
                result["errors"].append("filter_columns skipped - no data available")
        else:
            print(f"    ⊝ filter_columns (disabled)")

        if state_machine.get("convert_boolean_to_numeric", False):
            if result["data"] is not None:
                try:
                    print(f"    → convert_boolean_to_numeric")
                    numeric_df = self.convert_boolean_to_numeric(dataframe=result["data"], month_id=file_id)
                    
                    if numeric_df is not None:
                        result["data"] = numeric_df
                        result["steps_executed"].append("convert_boolean_to_numeric")
                        result["file_info"]["rows"] = len(numeric_df)
                        result["file_info"]["columns"] = len(numeric_df.columns)
                        print(f"      ✓ Converted boolean columns to numeric for {len(numeric_df)} rows")
                    else:
                        result["errors"].append("convert_boolean_to_numeric failed")
                        print(f"      ✗ Failed to convert boolean columns to numeric")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"convert_boolean_to_numeric failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ convert_boolean_to_numeric (no data available)")
                result["errors"].append("convert_boolean_to_numeric skipped - no data available")
        else:
            print(f"    ⊝ convert_boolean_to_numeric (disabled)")


from contextlib import contextmanager
import os
import pandas as pd
import re
import ast
import joblib
import logging
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.file_utils import generate_output_path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
            

from config.const import (
    ALL_PREPROCESSED_OUTPUT_FOLDER,
    ALL_WEATHER_FEATURES,
    DATA_FILE_PREFIX_FOR_TRAINING,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_WEATHER_CONDITIONS,
    BOOLEAN_FEATURES, 
    OUTPUT_FOLDER,
    PREPROCESSING_STATE_MACHINE,
    PREPROCESSED_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    REGRESSION_PROBLEM,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    SCORE_METRIC,
    TRAIN_DELAY_MINUTES,
    TRAIN_DELAYED_TARGET_COLUMN,
    VALID_TARGET_FEATURES,
    VALID_TRAIN_PREDICTION_FEATURES,
    WEATHER_COLS_TO_MERGE,
    WEATHER_MISSING_THRESHOLD,
    WEIGHT_DELAY_COLUMN,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    DEFAULT_TARGET_FEATURE,
    MAX_SAMPLE_WEIGHT_CLASSIFICATION,
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

        if state_machine.get("process_causes_column", False):
            if result["data"] is not None:
                try:
                    print(f"    → process_causes_column")
                    causes_df = self.process_causes_column(dataframe=result["data"])
                    
                    if causes_df is not None:
                        result["data"] = causes_df
                        result["steps_executed"].append("process_causes_column")
                        result["file_info"]["rows"] = len(causes_df)
                        result["file_info"]["columns"] = len(causes_df.columns)
                        print(f"      ✓ Processed causes column for {len(causes_df)} rows")
                    else:
                        result["errors"].append("process_causes_column failed")
                        print(f"      ✗ Failed to process causes column")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"process_causes_column failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ process_causes_column (no data available)")
                result["errors"].append("process_causes_column skipped - no data available")
        else:
            print(f"    ⊝ process_causes_column (disabled)")
        
        if state_machine.get("add_train_delayed_feature", False):
            if result["data"] is not None:
                try:
                    print(f"    → add_train_delayed_feature")
                    delayed_df = self.add_train_delayed_feature(dataframe=result["data"])
                    
                    if delayed_df is not None:
                        result["data"] = delayed_df
                        result["steps_executed"].append("add_train_delayed_feature")
                        result["file_info"]["rows"] = len(delayed_df)
                        result["file_info"]["columns"] = len(delayed_df.columns)
                        print(f"      ✓ Added trainDelayed feature to {len(delayed_df)} rows")
                    else:
                        result["errors"].append("add_train_delayed_feature failed")
                        print(f"      ✗ Failed to add trainDelayed feature")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"add_train_delayed_feature failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ add_train_delayed_feature (no data available)")
                result["errors"].append("add_train_delayed_feature skipped - no data available")
        else:
            print(f"    ⊝ add_train_delayed_feature (disabled)")

        if state_machine.get("merge_weather_columns", False):
            if result["data"] is not None:
                try:
                    print(f"    → merge_weather_columns")
                    merged_df = self.merge_weather_columns(dataframe=result["data"], month_id=file_id)
                    
                    if merged_df is not None:
                        result["data"] = merged_df
                        result["steps_executed"].append("merge_weather_columns")
                        result["file_info"]["rows"] = len(merged_df)
                        result["file_info"]["columns"] = len(merged_df.columns)
                        print(f"      ✓ Merged weather columns for {len(merged_df)} rows")
                    else:
                        result["errors"].append("merge_weather_columns failed")
                        print(f"      ✗ Failed to merge weather columns")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"merge_weather_columns failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ merge_weather_columns (no data available)")
                result["errors"].append("merge_weather_columns skipped - no data available")
        else:
            print(f"    ⊝ merge_weather_columns (disabled)")


        if state_machine.get("process_actual_time_column", False):
            if result["data"] is not None:
                try:
                    print(f"    → process_actual_time_column")
                    time_df = self.process_actual_time_column(dataframe=result["data"], month_id=file_id)
                    
                    if time_df is not None:
                        result["data"] = time_df
                        result["steps_executed"].append("process_actual_time_column")
                        result["file_info"]["rows"] = len(time_df)
                        result["file_info"]["columns"] = len(time_df.columns)
                        print(f"      ✓ Extracted temporal features for {len(time_df)} rows")
                    else:
                        result["errors"].append("process_actual_time_column failed")
                        print(f"      ✗ Failed to process actualTime column")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"process_actual_time_column failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ process_actual_time_column (no data available)")
                result["errors"].append("process_actual_time_column skipped - no data available")
        else:
            print(f"    ⊝ process_actual_time_column (disabled)")

        if state_machine.get("filter_columns", False):
            if result["data"] is not None:
                try:
                    print(f"    → filter_columns")
                    filtered_df = self.filter_columns(dataframe=result["data"], month_id=file_id)
                    
                    if filtered_df is not None:
                        result["data"] = filtered_df
                        result["steps_executed"].append("filter_columns")
                        result["file_info"]["rows"] = len(filtered_df)
                        result["file_info"]["columns"] = len(filtered_df.columns)
                        print(f"      ✓ Filtered columns for {len(filtered_df)} rows, {len(filtered_df.columns)} columns")
                    else:
                        result["errors"].append("filter_columns failed")
                        print(f"      ✗ Failed to filter columns")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"filter_columns failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ filter_columns (no data available)")
                result["errors"].append("filter_columns skipped - no data available")
        else:
            print(f"    ⊝ filter_columns (disabled)")

        if state_machine.get("convert_boolean_to_numeric", False):
            if result["data"] is not None:
                try:
                    print(f"    → convert_boolean_to_numeric")
                    numeric_df = self.convert_boolean_to_numeric(dataframe=result["data"], month_id=file_id)
                    
                    if numeric_df is not None:
                        result["data"] = numeric_df
                        result["steps_executed"].append("convert_boolean_to_numeric")
                        result["file_info"]["rows"] = len(numeric_df)
                        result["file_info"]["columns"] = len(numeric_df.columns)
                        print(f"      ✓ Converted boolean columns to numeric for {len(numeric_df)} rows")
                    else:
                        result["errors"].append("convert_boolean_to_numeric failed")
                        print(f"      ✗ Failed to convert boolean columns to numeric")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"convert_boolean_to_numeric failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ convert_boolean_to_numeric (no data available)")
                result["errors"].append("convert_boolean_to_numeric skipped - no data available")
        else:
            print(f"    ⊝ convert_boolean_to_numeric (disabled)")

        if state_machine.get("handle_missing_values", False):
            if result["data"] is not None:
                try:
                    print(f"    → handle_missing_values")
                    cleaned_df = self.handle_missing_values(dataframe=result["data"], month_id=file_id)
                    
                    if cleaned_df is not None:
                        result["data"] = cleaned_df
                        result["steps_executed"].append("handle_missing_values")
                        result["file_info"]["rows"] = len(cleaned_df)
                        result["file_info"]["columns"] = len(cleaned_df.columns)
                        print(f"      ✓ Handled missing values for {len(cleaned_df)} rows")
                    else:
                        result["errors"].append("handle_missing_values failed")
                        print(f"      ✗ Failed to handle missing values")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"handle_missing_values failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ handle_missing_values (no data available)")
                result["errors"].append("handle_missing_values skipped - no data available")
        else:
            print(f"    ⊝ handle_missing_values (disabled)")

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
            
            # Return the processed DataFrame
            return cross_df
            
        except Exception as e:
            print(f"Error processing file {input_file_path}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return None

    def process_causes_column(self, dataframe=None):
        """
        Process the 'causes' column to extract detailedCategoryCode and create weather-related indicator.
        
        This method handles the 'causes' column which may contain:
        - Empty values (NaN, None, empty string, empty list)
        - Complex nested data like: "[{'categoryCode': 'L', 'detailedCategoryCode': 'L2', ...}]"
        
        From the complex data, only the 'detailedCategoryCode' is extracted and kept.
        Additionally, creates a 'causes_related_to_weather' column based on detailedCategoryCode values:
        - 3: Strong weather indicator ('I1', 'I2')
        - 2: Possible weather indicator ('A1', 'K1', 'O1', 'P1', 'S1', 'S2', 'T2', 'T3', 'V3')
        - 1: Weak weather indicator (other non-empty categories)
        - 0: No indicator (empty/None values)
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with processed 'causes' column and new 'causes_related_to_weather' column.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Processing 'causes' column in dataframe with {len(df)} rows")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Check if 'causes' column exists
        if 'causes' not in df.columns:
            print("Warning: 'causes' column not found in dataframe. Skipping causes processing.")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        try:
            print(f"Found 'causes' column with {len(df)} rows to process")
            
            # Debug: Check sample values and analyze data structure
            print("\n--- ANALYZING CAUSES COLUMN STRUCTURE ---")
            sample_size = min(10, len(df))
            sample_indices = df.sample(n=sample_size, random_state=42).index if len(df) > sample_size else df.index
            sample_values = df.loc[sample_indices, 'causes'].tolist()
            
            print(f"Sample values (random {sample_size}):")
            for i, val in enumerate(sample_values[:5]):  # Show first 5 for readability
                print(f"  {i+1}: {repr(val)} (type: {type(val).__name__})")
            
            # Analyze value types and patterns
            value_analysis = {
                'total_rows': len(df),
                'null_values': 0,
                'empty_strings': 0,
                'empty_lists': 0,
                'string_representations': 0,
                'direct_objects': 0,
                'other_types': 0
            }
            
            for val in df['causes']:
                if pd.isna(val) or val is None:
                    value_analysis['null_values'] += 1
                elif val == "" or val == "[]":
                    value_analysis['empty_strings'] += 1
                elif isinstance(val, list) and len(val) == 0:
                    value_analysis['empty_lists'] += 1
                elif isinstance(val, str) and val.strip():
                    value_analysis['string_representations'] += 1
                elif isinstance(val, (list, dict)):
                    value_analysis['direct_objects'] += 1
                else:
                    value_analysis['other_types'] += 1
            
            print(f"\nData structure analysis:")
            for key, count in value_analysis.items():
                percentage = (count / value_analysis['total_rows']) * 100
                print(f"  {key}: {count:,} ({percentage:.1f}%)")
            print("--- END ANALYSIS ---\n")
            
            # Define weather-related category mappings
            WEATHER_INDICATORS = {
                # Strong weather indicators (score: 3)
                'strong': {'I1', 'I2'},
                # Possible weather indicators (score: 2) 
                'possible': {'A1', 'K1', 'O1', 'P1', 'S1', 'S2', 'T2', 'T3', 'V3'}
            }
            
            print(f"Weather indicator categories:")
            print(f"  Strong indicators (score 3): {sorted(WEATHER_INDICATORS['strong'])}")
            print(f"  Possible indicators (score 2): {sorted(WEATHER_INDICATORS['possible'])}")
            print(f"  Other categories (score 1): Any other non-empty detailedCategoryCode")
            print(f"  No indicator (score 0): Empty/None values")
            
            # Process each row more efficiently using vectorized operations where possible
            processed_values = []
            weather_scores = []
            
            # Counters for detailed reporting
            stats = {
                'empty_values': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'parsing_errors': 0,
                'weather_strong': 0,
                'weather_possible': 0,
                'weather_weak': 0,
                'weather_none': 0
            }
            
            print("\nProcessing causes data...")
            
            for index, cause_value in df['causes'].items():
                category_code, weather_score = self._process_single_cause(cause_value, index, stats)
                processed_values.append(category_code)
                weather_scores.append(weather_score)
            
            # Update the dataframe with processed values
            df['causes'] = processed_values
            df['causes_related_to_weather'] = weather_scores
            
            # Generate comprehensive statistics
            unique_categories = set(code for code in processed_values if code is not None)
            
            print(f"\n=== CAUSES PROCESSING RESULTS ===")
            print(f"Processing Statistics:")
            print(f"  Total rows processed: {len(df):,}")
            print(f"  Empty values: {stats['empty_values']:,}")
            print(f"  Successful extractions: {stats['successful_extractions']:,}")
            print(f"  Failed extractions: {stats['failed_extractions']:,}")
            print(f"  Parsing errors: {stats['parsing_errors']:,}")
            
            print(f"\nWeather Indicator Distribution:")
            print(f"  Strong indicators (3): {stats['weather_strong']:,} ({stats['weather_strong']/len(df)*100:.1f}%)")
            print(f"  Possible indicators (2): {stats['weather_possible']:,} ({stats['weather_possible']/len(df)*100:.1f}%)")
            print(f"  Weak indicators (1): {stats['weather_weak']:,} ({stats['weather_weak']/len(df)*100:.1f}%)")
            print(f"  No indicators (0): {stats['weather_none']:,} ({stats['weather_none']/len(df)*100:.1f}%)")
            
            if unique_categories:
                print(f"\nUnique category codes found ({len(unique_categories)}): {sorted(unique_categories)}")
                
                # Show category frequency distribution
                category_counts = pd.Series(processed_values).value_counts().sort_index()
                print(f"\nCategory Code Frequency:")
                for category, count in category_counts.items():
                    if category is not None:
                        percentage = (count / len(df)) * 100
                        # Determine weather indicator level
                        if category in STRONG_INDICATORS:
                            level = "Strong"
                        elif category in POSSIBLE_INDICATORS:
                            level = "Possible"
                        else:
                            level = "Weak"
                        print(f"  {category}: {count:,} ({percentage:.1f}%) - {level} weather indicator")
            else:
                print("\nNo category codes were successfully extracted")
            
            # Validate weather score distribution
            weather_score_counts = pd.Series(weather_scores).value_counts().sort_index()
            print(f"\nWeather Score Validation:")
            for score, count in weather_score_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  Score {score}: {count:,} ({percentage:.1f}%)")
            
            print("=== END PROCESSING RESULTS ===\n")
            
            return df
            
        except Exception as e:
            print(f"Error processing causes column: {e}")
            import traceback
            traceback.print_exc()
            return dataframe  # Return original dataframe on error

    def _process_single_cause(self, cause_value, index, stats):
        """
        Helper method to process a single cause value and extract detailedCategoryCode.
        
        Parameters:
        -----------
        cause_value : various
            The cause value to process (can be string, list, dict, None, etc.)
        index : int
            Row index for error reporting
        stats : dict
            Statistics dictionary to update
            
        Returns:
        --------
        tuple
            (category_code, weather_score) where:
            - category_code: Extracted detailedCategoryCode or None
            - weather_score: Weather indicator score (0-3)
        """
        # Import weather indicator constants
        from config.const import STRONG_INDICATORS, POSSIBLE_INDICATORS
        
        # Check if value is empty (comprehensive check)
        is_empty = (
            pd.isna(cause_value) or 
            cause_value is None or 
            cause_value == "" or 
            cause_value == "[]" or
            (isinstance(cause_value, list) and len(cause_value) == 0) or
            (isinstance(cause_value, str) and cause_value.strip() == "")
        )
        
        if is_empty:
            stats['empty_values'] += 1
            return None, 0  # No category code, no weather indicator
        
        try:
            # Process non-empty values
            if isinstance(cause_value, str):
                # Parse string representation
                cause_fixed = cause_value.replace("nan", "None")
                parsed_causes = ast.literal_eval(cause_fixed)
            else:
                # Assume it's already a Python object (list, dict, etc.)
                parsed_causes = cause_value
            
            # Extract categoryCode from the parsed data
            category_code = None
            
            if isinstance(parsed_causes, list) and len(parsed_causes) > 0:
                # List format: get first item
                first_cause = parsed_causes[0]
                if isinstance(first_cause, dict) and 'detailedCategoryCode' in first_cause:
                    category_code = first_cause['detailedCategoryCode']
            elif isinstance(parsed_causes, dict) and 'detailedCategoryCode' in parsed_causes:
                # Direct dictionary format
                category_code = parsed_causes['detailedCategoryCode']
            
            if category_code is not None:
                stats['successful_extractions'] += 1
                
                # Determine weather score based on detailedCategoryCode
                if category_code in STRONG_INDICATORS:
                    weather_score = 3
                    stats['weather_strong'] += 1
                elif category_code in POSSIBLE_INDICATORS:
                    weather_score = 2
                    stats['weather_possible'] += 1
                else:
                    weather_score = 1  # Other non-empty category
                    stats['weather_weak'] += 1
                    
                return category_code, weather_score
            else:
                stats['failed_extractions'] += 1
                return None, 0
                
        except Exception as e:
            # Parsing failed
            stats['parsing_errors'] += 1
            if stats['parsing_errors'] <= 3:  # Only print first few errors
                print(f"    Warning: Failed to parse causes in row {index}: {e}")
            return None, 0

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

    def process_actual_time_column(self, dataframe=None, month_id=None):
        """
        Process the 'actualTime' column to extract temporal features.
        
        From the 'actualTime' column (format: "2023-11-01T03:15:46.000Z"), extract:
        - month: Month number (1-12)
        - hour: Hour and minute in HH:MM format (e.g., "03:15", "14:30")
        - day_of_week: Day of week (1-7, where 1=Sunday, 7=Saturday)
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
        month_id : str, optional
            Month identifier for logging purposes.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with added temporal feature columns.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Processing 'actualTime' column in dataframe with {len(df)} rows")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Check if 'actualTime' column exists
        if 'actualTime' not in df.columns:
            print("Warning: 'actualTime' column not found in dataframe. Skipping actualTime processing.")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        # Use the logging method for detailed logging
        with self.get_logger("process_actual_time.log", "process_actual_time", month_id) as logger:
            try:
                logger.info(f"Starting actualTime processing for {len(df)} rows")
                
                print(f"Found 'actualTime' column with {len(df)} rows to process")
                
                # Debug: Check sample values
                print("\n--- DEBUGGING ACTUALTIME COLUMN ---")
                sample_values = df['actualTime'].dropna().head(5).tolist()
                print(f"Sample values (first 5 non-null): {sample_values}")
                
                # Check for missing values
                missing_count = df['actualTime'].isna().sum()
                valid_count = len(df) - missing_count
                print(f"Valid actualTime values: {valid_count}")
                print(f"Missing actualTime values: {missing_count}")
                
                if valid_count == 0:
                    print("Warning: All actualTime values are missing. Cannot extract temporal features.")
                    logger.warning("All actualTime values are missing")
                    return df
                
                print("--- END DEBUGGING ---\n")
                
                # Convert actualTime to datetime
                print("Converting actualTime to datetime...")
                df['actualTime_parsed'] = pd.to_datetime(df['actualTime'], errors='coerce')
                
                # Check how many successfully parsed
                parsed_count = df['actualTime_parsed'].notna().sum()
                failed_count = len(df) - missing_count - parsed_count
                
                print(f"Successfully parsed datetime: {parsed_count}")
                if failed_count > 0:
                    print(f"Failed to parse datetime: {failed_count}")
                    logger.warning(f"Failed to parse {failed_count} datetime values")
                
                # Extract temporal features only for successfully parsed dates
                mask = df['actualTime_parsed'].notna()
                
                # Extract month (1-12)
                df.loc[mask, 'month'] = df.loc[mask, 'actualTime_parsed'].dt.month
                
                # Extract hour and minute in HH:MM format
                hour_values = df.loc[mask, 'actualTime_parsed'].dt.hour
                minute_values = df.loc[mask, 'actualTime_parsed'].dt.minute
                df.loc[mask, 'hour'] = hour_values.astype(str).str.zfill(2) + ':' + minute_values.astype(str).str.zfill(2)
                
                # Extract day of week (1-7, where 1=Sunday, 7=Saturday)
                pandas_dayofweek = df.loc[mask, 'actualTime_parsed'].dt.dayofweek
                df.loc[mask, 'day_of_week'] = np.where(pandas_dayofweek == 6, 1, pandas_dayofweek + 2)
                
                # Drop the temporary parsed column
                df = df.drop('actualTime_parsed', axis=1)
                
                # Convert new columns to appropriate data types
                for col in ['month', 'day_of_week']:
                    if col in df.columns:
                        # Fill NaN values with 0 for rows that couldn't be parsed, then convert to int
                        df[col] = df[col].fillna(0).astype(int)
                
                # For hour column, fill NaN values with "00:00" for rows that couldn't be parsed
                if 'hour' in df.columns:
                    df['hour'] = df['hour'].fillna("00:00")
                
                # Print summary statistics
                print(f"\nTemporal feature extraction completed:")
                
                if parsed_count > 0:
                    print(f"Month values range: {df['month'].min()} - {df['month'].max()}")
                    print(f"Hour format: HH:MM (e.g., {df[df['hour'] != '00:00']['hour'].iloc[0] if (df['hour'] != '00:00').any() else '00:00'})")
                    print(f"Day of week values range: {df['day_of_week'].min()} - {df['day_of_week'].max()}")
                    
                    # Show distribution
                    print(f"\nMonth distribution:")
                    month_counts = df[df['month'] > 0]['month'].value_counts().sort_index()
                    for month, count in month_counts.items():
                        print(f"  Month {month:2d}: {count:,} records")
                    
                    print(f"\nHour distribution (showing top 5 most frequent times):")
                    hour_counts = df[df['hour'] != '00:00']['hour'].value_counts().head(5)
                    for hour_time, count in hour_counts.items():
                        print(f"  {hour_time}: {count:,} records")
                    
                    print(f"\nDay of week distribution:")
                    day_names = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 
                               5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
                    dow_counts = df[df['day_of_week'] > 0]['day_of_week'].value_counts().sort_index()
                    for day_num, count in dow_counts.items():
                        day_name = day_names.get(day_num, f'Unknown({day_num})')
                        print(f"  {day_num} ({day_name}): {count:,} records")
                
                # Log summary
                logger.info(f"Temporal feature extraction completed")
                logger.info(f"Successfully processed: {parsed_count} records")
                logger.info(f"Failed to parse: {failed_count} records") 
                logger.info(f"Missing actualTime: {missing_count} records")
                logger.info(f"Added columns: month, hour, day_of_week")
                
                return df
                
            except Exception as e:
                error_msg = f"Error processing actualTime column: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                import traceback
                traceback_str = traceback.format_exc()
                logger.error(f"Traceback: {traceback_str}")
                return dataframe  # Return original dataframe on error

    def filter_columns(self, dataframe=None, month_id=None):
        """
        Filter dataframe to keep only specified columns from the constants.
        
        This method keeps only the columns defined in:
        - VALID_TARGET_FEATURES: Target variables for prediction
        - VALID_TRAIN_PREDICTION_FEATURES: Train-specific features
        - ALL_WEATHER_FEATURES: Weather-related features
        - Additional utility columns: 'data_year', 'train_id', 'causes' (if they exist)
        
        All other columns are dropped to focus the dataset on relevant features only.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to filter.
        month_id : str, optional
            Month identifier for logging purposes.
            
        Returns:
        --------
        pandas.DataFrame
            The filtered dataframe with only specified columns.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Filtering columns in dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Use the logging method for detailed logging
        with self.get_logger("filter_columns.log", "filter_columns", month_id) as logger:
            try:
                logger.info(f"Starting column filtering for dataframe with {len(df.columns)} columns")
                logger.info(f"Original columns: {list(df.columns)}")
                
                # Define all columns we want to keep
                columns_to_keep = []
                
                # Add target features that exist in the dataframe
                target_cols_found = [col for col in VALID_TARGET_FEATURES if col in df.columns]
                columns_to_keep.extend(target_cols_found)
                print(f"Target columns found: {target_cols_found}")
                logger.info(f"Target columns found: {target_cols_found}")
                
                # Add train prediction features that exist in the dataframe
                train_cols_found = [col for col in VALID_TRAIN_PREDICTION_FEATURES if col in df.columns]
                columns_to_keep.extend(train_cols_found)
                print(f"Train prediction columns found: {train_cols_found}")
                logger.info(f"Train prediction columns found: {train_cols_found}")
                
                # Add weather features that exist in the dataframe
                weather_cols_found = [col for col in ALL_WEATHER_FEATURES if col in df.columns]
                columns_to_keep.extend(weather_cols_found)
                print(f"Weather columns found: {weather_cols_found}")
                logger.info(f"Weather columns found: {weather_cols_found}")
                
                # Add utility columns if they exist
                #utility_columns = ['data_year', 'train_id', 'causes']
                #utility_cols_found = [col for col in utility_columns if col in df.columns]
                #columns_to_keep.extend(utility_cols_found)
                #print(f"Utility columns found: {utility_cols_found}")
                #logger.info(f"Utility columns found: {utility_cols_found}")
                
                # Remove duplicates while preserving order
                columns_to_keep = list(dict.fromkeys(columns_to_keep))
                
                # Check if we have any columns to keep
                if not columns_to_keep:
                    error_msg = "No specified columns found in dataframe"
                    print(f"Error: {error_msg}")
                    logger.error(error_msg)
                    logger.error(f"Available columns: {list(df.columns)}")
                    logger.error(f"Expected target columns: {VALID_TARGET_FEATURES}")
                    logger.error(f"Expected train columns: {VALID_TRAIN_PREDICTION_FEATURES}")
                    logger.error(f"Expected weather columns: {ALL_WEATHER_FEATURES}")
                    return None
                
                # Identify columns that will be dropped
                original_columns = set(df.columns)
                columns_to_keep_set = set(columns_to_keep)
                columns_to_drop = original_columns - columns_to_keep_set
                
                print(f"\nColumn filtering summary:")
                print(f"- Original columns: {len(original_columns)}")
                print(f"- Columns to keep: {len(columns_to_keep)}")
                print(f"- Columns to drop: {len(columns_to_drop)}")
                
                logger.info(f"Column filtering summary:")
                logger.info(f"Original columns: {len(original_columns)}")
                logger.info(f"Columns to keep: {len(columns_to_keep)}")
                logger.info(f"Columns to drop: {len(columns_to_drop)}")
                
                if columns_to_drop:
                    print(f"\nColumns being dropped:")
                    dropped_list = sorted(list(columns_to_drop))
                    for col in dropped_list:
                        print(f"  - {col}")
                    logger.info(f"Dropped columns: {dropped_list}")
                
                print(f"\nColumns being kept:")
                for i, col in enumerate(columns_to_keep, 1):
                    print(f"  {i:2d}. {col}")
                
                logger.info(f"Kept columns: {columns_to_keep}")
                
                # Filter the dataframe to keep only specified columns
                filtered_df = df[columns_to_keep]
                
                # Verify the filtering worked
                print(f"\nFiltering completed:")
                print(f"- Final dataframe shape: {filtered_df.shape}")
                print(f"- Columns retained: {len(filtered_df.columns)}")
                
                logger.info(f"Column filtering completed successfully")
                logger.info(f"Final dataframe shape: {filtered_df.shape}")
                logger.info(f"Final columns: {list(filtered_df.columns)}")
                
                return filtered_df
                
            except Exception as e:
                error_msg = f"Error filtering columns: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                import traceback
                traceback_str = traceback.format_exc()
                logger.error(f"Traceback: {traceback_str}")
                return dataframe  # Return original dataframe on error

    def convert_boolean_to_numeric(self, dataframe=None, month_id=None):
            """
            Convert boolean columns to numeric values (0/1).
            
            This method processes columns specified in BOOLEAN_FEATURES:
            - Fills NaN values with 0 before conversion
            - Converts boolean values to integers (False -> 0, True -> 1)
            
            Parameters:
            -----------
            dataframe : pandas.DataFrame
                The dataframe to process.
            month_id : str, optional
                Month identifier for logging purposes.
                
            Returns:
            --------
            pandas.DataFrame
                The dataframe with boolean columns converted to numeric.
            """
            # Check if dataframe is provided
            if dataframe is None:
                print("Error: Dataframe must be provided")
                return None
                
            df = dataframe.copy()
            print(f"Converting boolean columns to numeric in dataframe with {len(df)} rows")
            
            if df.empty:
                print("Warning: Empty dataframe")
                return df
            
            # Use the logging method for detailed logging
            with self.get_logger("convert_boolean_to_numeric.log", "convert_boolean", month_id) as logger:
                try:
                    logger.info(f"Starting boolean to numeric conversion for {len(BOOLEAN_FEATURES)} columns")
                    logger.info(f"Columns to process: {BOOLEAN_FEATURES}")
                    
                    columns_processed = 0
                    columns_not_found = 0
                    total_nulls_filled = 0
                    
                    for col in BOOLEAN_FEATURES:
                        if col in df.columns:
                            # Count NaN values before processing
                            nulls = df[col].isna().sum()
                            if nulls > 0:
                                print(f"Filling {nulls} NaN values in '{col}' with 0 before conversion")
                                logger.info(f"Filling {nulls} NaN values in '{col}' with 0")
                                total_nulls_filled += nulls
                            
                            # Fill NaN values with 0 before converting to integer
                            df[col] = df[col].fillna(0)
                            
                            # Convert boolean values to integers (False -> 0, True -> 1)
                            df[col] = df[col].astype(int)
                            
                            print(f"Converted '{col}' to numeric (0/1)")
                            logger.info(f"Converted '{col}' to numeric (0/1)")
                            columns_processed += 1
                            
                            # Log value distribution after conversion
                            value_counts = df[col].value_counts().sort_index()
                            logger.info(f"'{col}' value distribution: {dict(value_counts)}")
                            
                        else:
                            print(f"Column '{col}' not found in dataframe. Skipping.")
                            logger.warning(f"Column '{col}' not found in dataframe")
                            columns_not_found += 1
                    
                    # Summary
                    print(f"\nBoolean to numeric conversion completed:")
                    print(f"- Columns processed: {columns_processed}")
                    print(f"- Columns not found: {columns_not_found}")
                    print(f"- Total NaN values filled: {total_nulls_filled}")
                    
                    logger.info(f"Boolean to numeric conversion completed")
                    logger.info(f"Columns processed: {columns_processed}")
                    logger.info(f"Columns not found: {columns_not_found}")
                    logger.info(f"Total NaN values filled: {total_nulls_filled}")
                    logger.info(f"Final dataframe shape: {df.shape}")
                    
                    return df
                    
                except Exception as e:
                    error_msg = f"Error converting boolean columns to numeric: {str(e)}"
                    print(error_msg)
                    logger.error(error_msg)
                    import traceback
                    traceback_str = traceback.format_exc()
                    logger.error(f"Traceback: {traceback_str}")
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
        - Use month-specific median imputation for all weather columns
        - Use global median for all other columns that still have missing values
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The preprocessed dataframe to handle missing values in.
        month_id : str, optional
            Month identifier for logging purposes.
                
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
            
            # Count rows before cleaning
            original_row_count = len(df)
            original_col_count = len(df.columns)
            
            # Fill missing values in non numeric features
            from config.const import CATEGORICAL_FEATURES, BOOLEAN_FEATURES
            for col in CATEGORICAL_FEATURES or BOOLEAN_FEATURES:
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

            # Check required columns
            from config.const import VALID_TARGET_FEATURES
            required_cols = [col for col in VALID_TARGET_FEATURES if col in df.columns]
            
            if required_cols:
                print(f"Checking for missing values in required columns: {required_cols}")
                logger.info(f"Checking for missing values in required columns: {required_cols}")
                # Store the count before dropping rows
                before_required_drop = len(df)
                # Drop rows where any of the required columns are None/NaN
                df = df.dropna(subset=required_cols)
                # Calculate dropped rows
                dropped_required = before_required_drop - len(df)
                dropped_percentage = (dropped_required / before_required_drop) * 100 if before_required_drop > 0 else 0
                print(f"- Dropped {dropped_required} rows with missing values in required columns ({dropped_percentage:.2f}%)")
                logger.info(f"Dropped {dropped_required} rows with missing values in required columns ({dropped_percentage:.2f}%)")
            else:
                print("Warning: Required columns not found in dataframe")
                logger.warning("Required columns not found in dataframe")
                dropped_required = 0
            
            # Step 3: Handle weather condition columns
            print(f"\n--- WEATHER COLUMN FILTERING ---")
            logger.info("=== Weather Column Filtering ===")
            
            # Drop weather columns that exceed the missing value threshold
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
            
            # ===== ENHANCED MISSING VALUE HANDLING - WEATHER FEATURES WITH MONTH-SPECIFIC MEDIAN =====
            
            # Identify all weather-related columns (after filtering)
            weather_cols = [col for col in df.columns if any(weather_condition in col for weather_condition in ALL_WEATHER_FEATURES)]
        
            
            if weather_cols:
                print(f"\n--- WEATHER FEATURES IMPUTATION WITH MONTH-SPECIFIC MEDIANS ---")
                logger.info("=== Weather Features Imputation with Month-Specific Medians ===")
                
                # Check if month column exists
                if 'month' not in df.columns:
                    print("Warning: 'month' column not found. Cannot perform month-specific imputation.")
                    logger.warning("'month' column not found. Cannot perform month-specific imputation.")
                    
                    # Fallback to global median for weather features
                    for col in weather_cols:
                        nulls = df[col].isna().sum()
                        if nulls > 0:
                            percentage = (nulls / len(df)) * 100
                            median_value = df[col].median()
                            df[col] = df[col].fillna(median_value)
                            print(f"- Filled {nulls} missing values in '{col}' with global median: {median_value:.2f} ({percentage:.2f}%)")
                            logger.info(f"Filled {nulls} missing values in '{col}' with global median: {median_value:.2f} ({percentage:.2f}%)")
                else:
                    print(f"Found {len(weather_cols)} weather-related columns for month-specific median imputation:")
                    for col in weather_cols:
                        print(f"  - {col}")
                    logger.info(f"Found {len(weather_cols)} weather-related columns: {weather_cols}")
                    
                    # Perform month-specific median imputation for each weather column
                    for col in weather_cols:
                        nulls_before = df[col].isna().sum()
                        if nulls_before > 0:
                            percentage = (nulls_before / len(df)) * 100
                            
                            # Calculate median for each month
                            monthly_medians = df.groupby('month')[col].median()
                            
                            # Show monthly medians for this column
                            print(f"\n  Processing '{col}' ({nulls_before} missing values, {percentage:.2f}%):")
                            print(f"    Monthly medians:")
                            for month, median_val in monthly_medians.items():
                                month_count = df[df['month'] == month][col].isna().sum()
                                if month_count > 0:
                                    print(f"      Month {month}: {median_val:.2f} (filling {month_count} values)")
                                else:
                                    print(f"      Month {month}: {median_val:.2f} (no missing values)")
                            
                            # Fill missing values with month-specific medians
                            def fill_with_month_median(row):
                                if pd.isna(row[col]):
                                    return monthly_medians[row['month']]
                                return row[col]
                            
                            df[col] = df.apply(fill_with_month_median, axis=1)
                            
                            # Verify the imputation worked
                            nulls_after = df[col].isna().sum()
                            filled_count = nulls_before - nulls_after
                            
                            print(f"    Result: Filled {filled_count} missing values using month-specific medians")
                            logger.info(f"Filled {filled_count} missing values in '{col}' using month-specific medians ({percentage:.2f}%)")
                            
                            if nulls_after > 0:
                                print(f"    Warning: {nulls_after} values still missing (possibly months with no data)")
                                logger.warning(f"{nulls_after} values still missing in '{col}' after month-specific imputation")
                        else:
                            print(f"  '{col}': No missing values")

            # Handle any remaining non-weather columns with missing values using global median
            remaining_cols_with_na = [col for col in df.columns 
                                    if df[col].isna().sum() > 0 
                                    and not any(weather_condition in col for weather_condition in self.important_conditions)]

            if remaining_cols_with_na:
                print(f"\n--- NON-WEATHER FEATURES IMPUTATION WITH GLOBAL MEDIAN ---")
                logger.info("=== Non-Weather Features Imputation with Global Median ===")
                
                for col in remaining_cols_with_na:
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        percentage = (nulls / len(df)) * 100
                        median_value = df[col].median()
                        df[col] = df[col].fillna(median_value)
                        print(f"- Filled {nulls} missing values in '{col}' with global median: {median_value:.2f} ({percentage:.2f}%)")
                        logger.info(f"Filled {nulls} missing values in '{col}' with global median: {median_value:.2f} ({percentage:.2f}%)")
            
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
            
            # Additional validation: check for any remaining missing values
            final_missing = df.isna().sum().sum()
            if final_missing > 0:
                print(f"\nWarning: {final_missing} missing values still remain after imputation")
                logger.warning(f"{final_missing} missing values still remain after imputation")
                # Show which columns still have missing values
                remaining_missing_cols = df.columns[df.isna().any()].tolist()
                print(f"Columns with remaining missing values: {remaining_missing_cols}")
                logger.warning(f"Columns with remaining missing values: {remaining_missing_cols}")
            else:
                print(f"\n✓ All missing values successfully handled")
                logger.info("All missing values successfully handled")
            
            return df

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