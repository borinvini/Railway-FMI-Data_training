from contextlib import contextmanager
from datetime import datetime
import glob
import os
import joblib
import json
import pandas as pd
import re
import ast
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from src.file_utils import format_param_distributions_for_json, generate_output_path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, 
    average_precision_score, cohen_kappa_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
            

from config.const import (
    ALL_WEATHER_FEATURES,
    BORDERLINE_SMOTE_CONFIG,
    BORDERLINE_SMOTE_OUTPUT_FOLDER,
    CLASSIFICATION_PROBLEM,
    DATA_FILE_PREFIX_FOR_TRAINING,
    DECISION_TREE_PARAM_DISTRIBUTIONS,
    DECISION_TREE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_WEATHER_CONDITIONS,
    BOOLEAN_FEATURES,
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_TRAINING_READY_OUTPUT_FOLDER, 
    OUTPUT_FOLDER,
    POSSIBLE_INDICATORS,
    PREPROCESSING_STATE_MACHINE,
    PREPROCESSED_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    RANDOM_SEARCH_CV_FOLDS,
    RANDOM_SEARCH_ITERATIONS,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    REGRESSION_PROBLEM,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    SCORE_METRIC,
    STRONG_INDICATORS,
    TEST_SIZE,
    THRESHOLD_OPTIMIZATION_CONFIG,
    TRAIN_DELAY_MINUTES,
    TRAIN_DELAYED_TARGET_COLUMN,
    TRAINING_READY_OUTPUT_FOLDER,
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
                pipeline_result = self.execute_preprocessing_pipeline_steps(
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

    def execute_preprocessing_pipeline_steps(self, input_file_path, file_id, year, state_machine):
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

        if state_machine.get("convert_hour_to_sincos", False):
            if result["data"] is not None:
                try:
                    print(f"    → convert_hour_to_sincos")
                    sincos_df = self.convert_hour_to_sincos(
                        dataframe=result["data"], 
                        month_id=file_id
                    )
                    
                    if sincos_df is not None and not sincos_df.empty:
                        # Clear previous dataframe from memory
                        del result["data"]
                        result["data"] = sincos_df
                        result["steps_executed"].append("convert_hour_to_sincos")
                        result["file_info"]["rows"] = len(sincos_df)
                        result["file_info"]["columns"] = len(sincos_df.columns)
                        print(f"      ✓ Converted hour to sin/cos features ({len(sincos_df)} rows, {len(sincos_df.columns)} columns)")
                    else:
                        result["errors"].append("convert_hour_to_sincos returned empty data")
                        print(f"      ✗ Failed - empty result")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"convert_hour_to_sincos failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ convert_hour_to_sincos (no data available)")
                result["errors"].append("convert_hour_to_sincos skipped - no data available")
        else:
            print(f"    ⊝ convert_hour_to_sincos (disabled)")

        if state_machine.get("convert_month_to_sincos", False):
            if result["data"] is not None:
                try:
                    print(f"    → convert_month_to_sincos")
                    month_sincos_df = self.convert_month_to_sincos(
                        dataframe=result["data"], 
                        month_id=file_id
                    )
                    
                    if month_sincos_df is not None and not month_sincos_df.empty:
                        # Clear previous dataframe from memory
                        del result["data"]
                        result["data"] = month_sincos_df
                        result["steps_executed"].append("convert_month_to_sincos")
                        result["file_info"]["rows"] = len(month_sincos_df)
                        result["file_info"]["columns"] = len(month_sincos_df.columns)
                        print(f"      ✓ Converted month to sin/cos features ({len(month_sincos_df)} rows, {len(month_sincos_df.columns)} columns)")
                    else:
                        result["errors"].append("convert_month_to_sincos returned empty data")
                        print(f"      ✗ Failed - empty result")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"convert_month_to_sincos failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ convert_month_to_sincos (no data available)")
                result["errors"].append("convert_month_to_sincos skipped - no data available")
        else:
            print(f"    ⊝ convert_month_to_sincos (disabled)")

        if state_machine.get("convert_dayofweek_to_sincos", False):
            if result["data"] is not None:
                try:
                    print(f"    → convert_dayofweek_to_sincos")
                    dayofweek_sincos_df = self.convert_dayofweek_to_sincos(
                        dataframe=result["data"], 
                        month_id=file_id
                    )
                    
                    if dayofweek_sincos_df is not None and not dayofweek_sincos_df.empty:
                        # Clear previous dataframe from memory
                        del result["data"]
                        result["data"] = dayofweek_sincos_df
                        result["steps_executed"].append("convert_dayofweek_to_sincos")
                        result["file_info"]["rows"] = len(dayofweek_sincos_df)
                        result["file_info"]["columns"] = len(dayofweek_sincos_df.columns)
                        print(f"      ✓ Converted day_of_week to sin/cos features ({len(dayofweek_sincos_df)} rows, {len(dayofweek_sincos_df.columns)} columns)")
                    else:
                        result["errors"].append("convert_dayofweek_to_sincos returned empty data")
                        print(f"      ✗ Failed - empty result")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"convert_dayofweek_to_sincos failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ convert_dayofweek_to_sincos (no data available)")
                result["errors"].append("convert_dayofweek_to_sincos skipped - no data available")
        else:
            print(f"    ⊝ convert_dayofweek_to_sincos (disabled)")

        if state_machine.get("drop_original_temporal_columns", False):
            if result["data"] is not None:
                try:
                    print(f"    → drop_original_temporal_columns")
                    temporal_dropped_df = self.drop_original_temporal_columns(
                        dataframe=result["data"], 
                        month_id=file_id
                    )
                    
                    if temporal_dropped_df is not None and not temporal_dropped_df.empty:
                        # Clear previous dataframe from memory
                        del result["data"]
                        result["data"] = temporal_dropped_df
                        result["steps_executed"].append("drop_original_temporal_columns")
                        result["file_info"]["rows"] = len(temporal_dropped_df)
                        result["file_info"]["columns"] = len(temporal_dropped_df.columns)
                        print(f"      ✓ Dropped original temporal columns ({len(temporal_dropped_df)} rows, {len(temporal_dropped_df.columns)} columns)")
                    else:
                        result["errors"].append("drop_original_temporal_columns returned empty data")
                        print(f"      ✗ Failed - empty result")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"drop_original_temporal_columns failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ drop_original_temporal_columns (no data available)")
                result["errors"].append("drop_original_temporal_columns skipped - no data available")
        else:
            print(f"    ⊝ drop_original_temporal_columns (disabled)")

        if state_machine.get("select_target", False):
            if result["data"] is not None:
                try:
                    print(f"    → select_target")
                    target_df = self.select_target_feature(
                        dataframe=result["data"], 
                        target_feature=DEFAULT_TARGET_FEATURE
                    )
                    
                    if target_df is not None and not target_df.empty:
                        # Clear previous dataframe from memory
                        del result["data"]
                        result["data"] = target_df
                        result["steps_executed"].append("select_target")
                        result["file_info"]["rows"] = len(target_df)
                        result["file_info"]["columns"] = len(target_df.columns)
                        print(f"      ✓ Selected target feature '{DEFAULT_TARGET_FEATURE}' ({len(target_df)} rows, {len(target_df.columns)} columns)")
                    else:
                        result["errors"].append("select_target returned empty data")
                        print(f"      ✗ Failed - empty result")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"select_target failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ select_target (no data available)")
                result["errors"].append("select_target skipped - no data available")
        else:
            print(f"    ⊝ select_target (disabled)")

        if state_machine.get("filter_strong_weather_causes", False):
            if result["data"] is not None:
                try:
                    print(f"    → filter_strong_weather_causes")
                    filtered_df = self.filter_strong_weather_causes(
                        dataframe=result["data"], 
                        month_id=file_id, 
                        filename=os.path.basename(input_file_path)
                    )
                    
                    if filtered_df is not None:
                        result["data"] = filtered_df
                        result["steps_executed"].append("filter_strong_weather_causes")
                        result["file_info"]["rows"] = len(filtered_df)
                        print(f"      ✓ Filtered to {len(filtered_df)} rows with strong weather causes")
                    else:
                        result["errors"].append("filter_strong_weather_causes returned None")
                        print(f"      ✗ Failed - returned None")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"filter_strong_weather_causes failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                result["errors"].append("filter_strong_weather_causes: No data available")
                print(f"    ✗ filter_strong_weather_causes (no data)")
                return result
        else:
            print(f"    ⊝ filter_strong_weather_causes (disabled)")

        if state_machine.get("remove_duplicates", False):
            if result["data"] is not None:
                try:
                    print(f"    → remove_duplicates")
                    deduplicated_df = self.remove_duplicates(
                        dataframe=result["data"], 
                        month_id=file_id
                    )
                    
                    if deduplicated_df is not None:
                        # Clear previous dataframe from memory
                        del result["data"]
                        result["data"] = deduplicated_df
                        result["steps_executed"].append("remove_duplicates")
                        result["file_info"]["rows"] = len(deduplicated_df)
                        print(f"      ✓ Removed duplicates ({len(deduplicated_df)} rows remaining)")
                    else:
                        result["errors"].append("remove_duplicates returned None")
                        print(f"      ✗ Failed - returned None")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"remove_duplicates failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                result["errors"].append("remove_duplicates: No data available")
                print(f"    ✗ remove_duplicates (no data)")
                return result
        else:
            print(f"    ⊝ remove_duplicates (disabled)")

        if state_machine.get("save_training_ready_csv", False):
            if result["data"] is not None:
                try:
                    print(f"    → save_training_ready_csv")
                    save_success = self.save_training_ready_csv(file_id, result["data"])
                    
                    if save_success:
                        result["steps_executed"].append("save_training_ready_csv")
                        print(f"      ✓ Saved training-ready data for {file_id}")
                    else:
                        result["errors"].append("save_training_ready_csv failed")
                        print(f"      ✗ Failed to save training-ready data for {file_id}")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"save_training_ready_csv failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ save_training_ready_csv (no data available)")
                result["errors"].append("save_training_ready_csv skipped - no data available")
        else:
            print(f"    ⊝ save_training_ready_csv (disabled)")

        
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
            print(f"Processing {len(df)} rows to extract train stops from timeTableRows column. Please wait...")
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
            print("Processing weather_conditions column - expanding into separate columns. Please wait...")
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
            return df
        
        try:
            # Initialize tracking variables
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
            
            processed_values = []
            weather_scores = []
            
            # Process each value directly in the loop
            for index, cause_value in df['causes'].items():
                # FIXED: Safe empty value check that handles arrays/series
                try:
                    # Handle numpy arrays and pandas Series
                    if hasattr(cause_value, '__len__') and hasattr(cause_value, '__iter__'):
                        # For array-like objects, check if it's a single-element array first
                        if hasattr(cause_value, 'shape') and cause_value.shape == ():
                            # Single-element numpy scalar
                            cause_value = cause_value.item()
                        elif hasattr(cause_value, '__len__') and len(cause_value) == 1:
                            # Single-element array/series
                            cause_value = cause_value[0] if not pd.isna(cause_value[0]) else None
                    
                    # Now perform safe empty checks
                    is_empty = False
                    
                    # Check for pandas NA/NaN
                    if pd.isna(cause_value):
                        is_empty = True
                    # Check for None
                    elif cause_value is None:
                        is_empty = True
                    # Check for empty string (safe comparison)
                    elif isinstance(cause_value, str):
                        is_empty = (cause_value == "" or cause_value == "[]" or cause_value.strip() == "")
                    # Check for empty list
                    elif isinstance(cause_value, list):
                        is_empty = (len(cause_value) == 0)
                    # Check for other empty-like values
                    elif str(cause_value) in ["", "[]", "nan", "None"]:
                        is_empty = True
                        
                except Exception as e:
                    print(f"Warning: Error checking if value is empty at index {index}: {e}")
                    print(f"Cause value type: {type(cause_value)}, value: {cause_value}")
                    # Treat problematic values as non-empty to avoid data loss
                    is_empty = False
                
                if is_empty:
                    stats['empty_values'] += 1
                    stats['weather_none'] += 1
                    processed_values.append(None)
                    weather_scores.append(0)
                    continue
                
                try:
                    # Process non-empty values
                    if isinstance(cause_value, str):
                        # Parse string representation
                        cause_fixed = cause_value.replace("nan", "None")
                        parsed_causes = ast.literal_eval(cause_fixed)
                    else:
                        # Assume it's already a Python object (list, dict, etc.)
                        parsed_causes = cause_value
                    
                    # Extract detailed category codes
                    detailed_codes = []
                    
                    if isinstance(parsed_causes, list):
                        for cause_dict in parsed_causes:
                            if isinstance(cause_dict, dict) and 'detailedCategoryCode' in cause_dict:
                                detailed_codes.append(cause_dict['detailedCategoryCode'])
                    elif isinstance(parsed_causes, dict) and 'detailedCategoryCode' in parsed_causes:
                        detailed_codes.append(parsed_causes['detailedCategoryCode'])
                    
                    # Store the extracted codes or None if empty
                    if detailed_codes:
                        processed_values.append(detailed_codes)
                        stats['successful_extractions'] += 1
                    else:
                        processed_values.append(None)
                        stats['failed_extractions'] += 1
                    
                    # Calculate weather indicator score using constants from const.py
                    weather_score = 0
                    for code in detailed_codes:
                        if code in STRONG_INDICATORS:
                            weather_score = max(weather_score, 3)  # Strong indicator
                        elif code in POSSIBLE_INDICATORS:
                            weather_score = max(weather_score, 2)  # Possible indicator
                        elif code:  # Any other non-empty code
                            weather_score = max(weather_score, 1)  # Weak indicator
                    
                    weather_scores.append(weather_score)
                    
                    # Update stats
                    if weather_score == 3:
                        stats['weather_strong'] += 1
                    elif weather_score == 2:
                        stats['weather_possible'] += 1
                    elif weather_score == 1:
                        stats['weather_weak'] += 1
                    else:
                        stats['weather_none'] += 1
                        
                except Exception as e:
                    print(f"Warning: Failed to parse causes in row {index}: {e}")
                    print(f"Problematic value: {cause_value}")
                    print(f"Value type: {type(cause_value)}")
                    stats['parsing_errors'] += 1
                    processed_values.append(None)
                    weather_scores.append(0)
            
            # Update the dataframe with processed values
            df['causes'] = processed_values
            df['causes_related_to_weather'] = weather_scores
            
            # Generate summary statistics
            print(f"Causes processing completed:")
            print(f"  Successful extractions: {stats['successful_extractions']:,}")
            print(f"  Weather indicators found: {stats['weather_strong'] + stats['weather_possible'] + stats['weather_weak']:,}")
            
            if stats['parsing_errors'] > 0:
                print(f"  Parsing errors: {stats['parsing_errors']:,}")
            
            return df
            
        except Exception as e:
            print(f"Error processing causes column: {e}")
            import traceback
            traceback.print_exc()  # This will help debug the exact location
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
        
        # Check if required columns exist
        missing_columns = []
        if 'actualTime' not in df.columns:
            missing_columns.append('actualTime')
        if 'scheduledTime' not in df.columns:
            missing_columns.append('scheduledTime')
        
        if missing_columns:
            print(f"Warning: Required columns not found in dataframe: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        # NEW: Check for NULL/NaN values in scheduledTime OR actualTime and drop those rows
        initial_row_count = len(df)
        
        # Create mask for rows where either scheduledTime OR actualTime is NULL/NaN
        scheduled_time_null = df['scheduledTime'].isna()
        actual_time_null = df['actualTime'].isna()
        rows_to_drop = scheduled_time_null | actual_time_null
        
        # Count rows to be dropped
        scheduled_nulls = scheduled_time_null.sum()
        actual_nulls = actual_time_null.sum()
        total_rows_to_drop = rows_to_drop.sum()
        
        print(f"Data quality check:")
        print(f"- Rows with NULL/NaN scheduledTime: {scheduled_nulls}")
        print(f"- Rows with NULL/NaN actualTime: {actual_nulls}")
        print(f"- Total rows to drop (either column NULL/NaN): {total_rows_to_drop}")
        
        # Drop rows where either column is NULL/NaN
        if total_rows_to_drop > 0:
            df = df[~rows_to_drop].copy()
            print(f"- Dropped {total_rows_to_drop} rows")
            print(f"- Remaining rows: {len(df)} (was {initial_row_count})")
        else:
            print("- No rows to drop (all time values are valid)")
        
        # If no rows remain after dropping, return empty dataframe
        if df.empty:
            print("Warning: All rows were dropped due to NULL/NaN time values")
            return df
        
        # Use the logging method for detailed logging
        with self.get_logger("process_actual_time.log", "process_actual_time", month_id) as logger:
            try:
                logger.info(f"Starting actualTime processing for {len(df)} rows (after dropping {total_rows_to_drop} rows with NULL/NaN time values)")
                
                print(f"Found 'actualTime' column with {len(df)} rows to process")
                
                # Debug: Check sample values
                print("\n--- DEBUGGING ACTUALTIME COLUMN ---")
                sample_values = df['actualTime'].dropna().head(5).tolist()
                print(f"Sample values (first 5 non-null): {sample_values}")
                
                # Check for missing values (should be 0 after our filtering)
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
                
                # Log statistics if we have valid data
                if len(df) > 0 and 'month' in df.columns:
                    valid_temporal_data = df[df['month'] > 0]
                    print(f"\nTemporal feature extraction summary:")
                    print(f"Total rows processed: {len(df):,}")
                    print(f"Rows with valid temporal data: {len(valid_temporal_data):,}")
                    
                    # Show distribution
                    print(f"\nMonth distribution:")
                    month_counts = df[df['month'] > 0]['month'].value_counts().sort_index()
                    for month, count in month_counts.items():
                        print(f"  Month {month:2d}: {count:,} records")
                    
                    print(f"\nHour distribution (showing top 5 most frequent times):")
                    hour_counts = df[df['hour'] != '00:00']['hour'].value_counts().head(5)
                    for hour, count in hour_counts.items():
                        print(f"  {hour}: {count:,} records")
                    
                    print(f"\nDay of week distribution:")
                    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                    day_counts = df[df['day_of_week'] > 0]['day_of_week'].value_counts().sort_index()
                    for day_num, count in day_counts.items():
                        day_name = day_names[day_num - 1] if 1 <= day_num <= 7 else f"Unknown({day_num})"
                        print(f"  {day_num} ({day_name}): {count:,} records")
                
                # Final summary
                print(f"\nFinal temporal columns added:")
                for col in ['month', 'hour', 'day_of_week']:
                    if col in df.columns:
                        print(f"  - {col}: {df[col].dtype}")
                
                logger.info(f"Temporal feature extraction completed successfully for {len(df)} rows")
                logger.info(f"Added columns: month, hour, day_of_week")
                
                return df
                
            except Exception as e:
                error_msg = f"Error processing actualTime column: {e}"
                print(error_msg)
                logger.error(error_msg)
                logger.error(f"Exception details: {str(e)}")
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

        # Ensure all column names are strings to avoid type comparison issues
        df.columns = df.columns.astype(str)
        
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
                            
                            # Calculate single median for all data in the file
                            file_median = df[col].median()

                            # Show median for this column
                            print(f"\n  Processing '{col}' ({nulls_before} missing values, {percentage:.2f}%):")
                            print(f"    File median: {file_median:.2f}")

                            # Fill missing values with the single file median
                            df[col] = df[col].fillna(file_median)
                            
                            # Verify the imputation worked
                            nulls_after = df[col].isna().sum()
                            filled_count = nulls_before - nulls_after
                            
                            print(f"    Result: Filled {filled_count} missing values using file median")
                            logger.info(f"Filled {filled_count} missing values in '{col}' using file median ({percentage:.2f}%)")
                            
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

    def convert_hour_to_sincos(self, dataframe=None, month_id=None):
        """
        Convert the hour column from HH:MM format to cyclical sin/cos features.
        
        This method transforms the hour column (e.g., "03:15", "14:30") into two 
        continuous cyclical features that better represent the temporal nature
        of time for machine learning models:
        - hour_sin: sine component of the hour
        - hour_cos: cosine component of the hour
        
        The original hour column is removed after transformation.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
        month_id : str, optional
            Month identifier for logging purposes.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with hour_sin and hour_cos columns replacing the original hour column.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Converting hour column to sin/cos format in dataframe with {len(df)} rows")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Check if 'hour' column exists
        if 'hour' not in df.columns:
            print("Warning: 'hour' column not found in dataframe. Skipping hour to sin/cos conversion.")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        # Use the logging method for detailed logging
        with self.get_logger("convert_hour_to_sincos.log", "convert_hour_to_sincos", month_id) as logger:
            try:
                logger.info(f"Starting hour to sin/cos conversion for {len(df)} rows")
                
                # Debug: Check sample values
                print("\n--- DEBUGGING HOUR COLUMN ---")
                sample_values = df['hour'].dropna().head(5).tolist()
                print(f"Sample hour values (first 5 non-null): {sample_values}")
                
                # Check for missing values
                missing_count = df['hour'].isna().sum()
                valid_count = len(df) - missing_count
                print(f"Valid hour values: {valid_count}")
                print(f"Missing hour values: {missing_count}")
                
                if valid_count == 0:
                    print("Warning: All hour values are missing. Cannot convert to sin/cos.")
                    logger.warning("All hour values are missing")
                    return df
                
                print("--- END DEBUGGING ---\n")
                
                # Convert hour from HH:MM to decimal hours
                print("Converting hour from HH:MM format to decimal hours...")
                
                # Create a mask for valid (non-null) hour values
                valid_mask = df['hour'].notna()
                
                # Convert HH:MM to decimal hours for valid entries
                hour_decimal = pd.Series(index=df.index, dtype=float)
                
                for idx in df[valid_mask].index:
                    try:
                        hour_str = str(df.loc[idx, 'hour'])
                        if ':' in hour_str:
                            hour_part, minute_part = hour_str.split(':')
                            decimal_hour = int(hour_part) + int(minute_part) / 60.0
                            hour_decimal.loc[idx] = decimal_hour
                        else:
                            # Handle case where hour might not be in HH:MM format
                            print(f"Warning: Unexpected hour format '{hour_str}' at index {idx}")
                            logger.warning(f"Unexpected hour format '{hour_str}' at index {idx}")
                            hour_decimal.loc[idx] = np.nan
                    except (ValueError, AttributeError) as e:
                        print(f"Warning: Could not parse hour '{df.loc[idx, 'hour']}' at index {idx}: {e}")
                        logger.warning(f"Could not parse hour '{df.loc[idx, 'hour']}' at index {idx}: {e}")
                        hour_decimal.loc[idx] = np.nan
                
                # Count successfully converted values
                converted_count = hour_decimal.notna().sum()
                failed_conversion_count = valid_count - converted_count
                
                print(f"Successfully converted to decimal: {converted_count}")
                if failed_conversion_count > 0:
                    print(f"Failed to convert: {failed_conversion_count}")
                    logger.warning(f"Failed to convert {failed_conversion_count} hour values to decimal")
                
                # Create sin/cos features using the cyclical transformation
                print("Creating sin/cos cyclical features...")
                
                # Calculate sin and cos for the 24-hour cycle
                # Formula: sin(2π × hour / 24) and cos(2π × hour / 24)
                df['hour_sin'] = np.sin(2 * np.pi * hour_decimal / 24.0)
                df['hour_cos'] = np.cos(2 * np.pi * hour_decimal / 24.0)
                
                # Fill NaN values in sin/cos with 0 (neutral position on the unit circle)
                df['hour_sin'] = df['hour_sin'].fillna(0.0)
                df['hour_cos'] = df['hour_cos'].fillna(0.0)
                
                # Reorder columns to put hour_sin and hour_cos right after month
                print("Reordering columns to place hour_sin and hour_cos after month...")
                
                # Get current column order
                current_cols = list(df.columns)
                
                # Find the position of month column
                if 'month' in current_cols:
                    month_index = current_cols.index('month')
                    
                    # Remove hour_sin and hour_cos from their current positions
                    cols_without_sincos = [col for col in current_cols if col not in ['hour_sin', 'hour_cos']]
                    
                    # Insert hour_sin and hour_cos right after month
                    reordered_cols = (
                        cols_without_sincos[:month_index + 1] +  # Everything up to and including month
                        ['hour_sin', 'hour_cos'] +               # Our new temporal features
                        cols_without_sincos[month_index + 1:]    # Everything after month
                    )
                    
                    # Reorder the dataframe
                    df = df[reordered_cols]
                    
                    print(f"✓ Columns reordered: hour_sin and hour_cos placed after month")
                    logger.info("Columns reordered: hour_sin and hour_cos placed after month column")
                else:
                    print("Warning: 'month' column not found. hour_sin and hour_cos will remain in their current positions.")
                    logger.warning("month column not found during reordering")
                
                print(f"Hour to sin/cos conversion completed:")
                print(f"- Original hour column removed")
                print(f"- Added hour_sin column (range: {df['hour_sin'].min():.3f} to {df['hour_sin'].max():.3f})")
                print(f"- Added hour_cos column (range: {df['hour_cos'].min():.3f} to {df['hour_cos'].max():.3f})")
                print(f"- Successfully converted {converted_count} out of {len(df)} rows")
                print(f"- Temporal features grouped: month → hour_sin → hour_cos")
                
                # Show some examples of the conversion
                if converted_count > 0:
                    print(f"\nExample conversions:")
                    sample_indices = df[df['hour_sin'].notna()].head(3).index
                    for idx in sample_indices:
                        sin_val = df.loc[idx, 'hour_sin']
                        cos_val = df.loc[idx, 'hour_cos']
                        print(f"  Index {idx}: sin={sin_val:.3f}, cos={cos_val:.3f}")
                
                logger.info(f"Hour to sin/cos conversion completed successfully for {converted_count} rows")
                logger.info(f"Added columns: hour_sin, hour_cos")
                logger.info(f"Removed column: hour")
                
                return df
                
            except Exception as e:
                error_msg = f"Error converting hour to sin/cos: {e}"
                print(error_msg)
                logger.error(error_msg)
                logger.error(f"Exception details: {str(e)}")
                return dataframe  # Return original dataframe on error

    def convert_month_to_sincos(self, dataframe=None, month_id=None):
        """
        Convert the month column to cyclical sin/cos features while keeping the original.
        
        This method transforms the month column (1-12) into two continuous cyclical 
        features that better represent the seasonal nature of months for machine 
        learning models:
        - month_sin: sine component of the month
        - month_cos: cosine component of the month
        
        The original month column is preserved for flexibility (tree-based models
        can still use the categorical representation).
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
        month_id : str, optional
            Month identifier for logging purposes.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with month_sin and month_cos columns added after the original month column.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Converting month column to sin/cos format in dataframe with {len(df)} rows")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Check if 'month' column exists
        if 'month' not in df.columns:
            print("Warning: 'month' column not found in dataframe. Skipping month to sin/cos conversion.")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        # Use the logging method for detailed logging
        with self.get_logger("convert_month_to_sincos.log", "convert_month_to_sincos", month_id) as logger:
            try:
                logger.info(f"Starting month to sin/cos conversion for {len(df)} rows")
                
                # Debug: Check sample values
                print("\n--- DEBUGGING MONTH COLUMN ---")
                sample_values = df['month'].dropna().head(5).tolist()
                print(f"Sample month values (first 5 non-null): {sample_values}")
                
                # Check for missing values
                missing_count = df['month'].isna().sum()
                valid_count = len(df) - missing_count
                print(f"Valid month values: {valid_count}")
                print(f"Missing month values: {missing_count}")
                
                # Check for invalid month values (not 1-12)
                invalid_months = df[(df['month'].notna()) & ((df['month'] < 1) | (df['month'] > 12))]
                invalid_count = len(invalid_months)
                
                if invalid_count > 0:
                    print(f"Warning: Found {invalid_count} invalid month values (not 1-12)")
                    print(f"Invalid values: {sorted(invalid_months['month'].unique())}")
                    logger.warning(f"Found {invalid_count} invalid month values: {sorted(invalid_months['month'].unique())}")
                
                if valid_count == 0:
                    print("Warning: All month values are missing. Cannot convert to sin/cos.")
                    logger.warning("All month values are missing")
                    return df
                
                print("--- END DEBUGGING ---\n")
                
                # Create sin/cos features using the cyclical transformation for 12-month cycle
                print("Creating month sin/cos cyclical features...")
                
                # Calculate sin and cos for the 12-month cycle
                # Formula: sin(2π × month / 12) and cos(2π × month / 12)
                # Note: We use the month values directly since they're already in 1-12 format
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
                
                # For missing month values, set sin/cos to 0 (neutral position on the unit circle)
                month_mask = df['month'].isna()
                df.loc[month_mask, 'month_sin'] = 0.0
                df.loc[month_mask, 'month_cos'] = 0.0
                
                # Count successfully converted values (non-missing original values)
                converted_count = df['month'].notna().sum()
                
                # Reorder columns to group temporal features: month → month_sin → month_cos → (rest)
                print("Reordering columns to place month_sin and month_cos right after month...")
                
                # Get current column order
                current_cols = list(df.columns)
                
                # Find the position of month column
                if 'month' in current_cols:
                    month_index = current_cols.index('month')
                    
                    # Remove month_sin and month_cos from their current positions
                    cols_without_sincos = [col for col in current_cols if col not in ['month_sin', 'month_cos']]
                    
                    # Insert month_sin and month_cos right after month
                    reordered_cols = (
                        cols_without_sincos[:month_index + 1] +  # Everything up to and including month
                        ['month_sin', 'month_cos'] +             # Our new seasonal features
                        cols_without_sincos[month_index + 1:]    # Everything after month
                    )
                    
                    # Reorder the dataframe
                    df = df[reordered_cols]
                    
                    print(f"✓ Columns reordered: month_sin and month_cos placed after month")
                    logger.info("Columns reordered: month_sin and month_cos placed after month column")
                else:
                    print("Warning: 'month' column not found during reordering (this shouldn't happen).")
                    logger.warning("month column not found during reordering")
                
                print(f"Month to sin/cos conversion completed:")
                print(f"- Original month column preserved")
                print(f"- Added month_sin column (range: {df['month_sin'].min():.3f} to {df['month_sin'].max():.3f})")
                print(f"- Added month_cos column (range: {df['month_cos'].min():.3f} to {df['month_cos'].max():.3f})")
                print(f"- Successfully converted {converted_count} out of {len(df)} rows")
                print(f"- Temporal features order: month → month_sin → month_cos → hour_sin → hour_cos")
                
                # Show some examples of the conversion
                if converted_count > 0:
                    print(f"\nExample seasonal conversions:")
                    # Try to show examples from different seasons if available
                    example_months = []
                    for target_month in [1, 4, 7, 10]:  # Winter, Spring, Summer, Fall
                        month_examples = df[df['month'] == target_month].head(1)
                        if not month_examples.empty:
                            example_months.extend(month_examples.index.tolist())
                    
                    # If we don't have examples from all seasons, just show first few
                    if not example_months:
                        example_months = df[df['month'].notna()].head(3).index.tolist()
                    
                    for idx in example_months[:3]:  # Show max 3 examples
                        month_val = df.loc[idx, 'month']
                        sin_val = df.loc[idx, 'month_sin']
                        cos_val = df.loc[idx, 'month_cos']
                        
                        # Add season name for context
                        season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
                                    7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
                        season = season_map.get(int(month_val) if pd.notna(month_val) else 0, 'Unknown')
                        
                        print(f"  Month {int(month_val)} ({season}): sin={sin_val:.3f}, cos={cos_val:.3f}")
                
                logger.info(f"Month to sin/cos conversion completed successfully for {converted_count} rows")
                logger.info(f"Added columns: month_sin, month_cos")
                logger.info("Original month column preserved")
                
                return df
                
            except Exception as e:
                error_msg = f"Error converting month to sin/cos: {e}"
                print(error_msg)
                logger.error(error_msg)
                logger.error(f"Exception details: {str(e)}")
                return dataframe  # Return original dataframe on error

    def convert_dayofweek_to_sincos(self, dataframe=None, month_id=None):
            """
            Convert the day_of_week column to cyclical sin/cos features while keeping the original.
            
            This method transforms the day_of_week column (1-7, where 1=Sunday, 7=Saturday) 
            into two continuous cyclical features that better represent the weekly cycle for 
            machine learning models:
            - day_week_sin: sine component of the day of week
            - day_week_cos: cosine component of the day of week
            
            The original day_of_week column is preserved for flexibility (tree-based models
            can still use the categorical representation).
            
            Parameters:
            -----------
            dataframe : pandas.DataFrame
                The dataframe to process.
            month_id : str, optional
                Month identifier for logging purposes.
                
            Returns:
            --------
            pandas.DataFrame
                The dataframe with day_week_sin and day_week_cos columns added after the original day_of_week column.
            """
            # Check if dataframe is provided
            if dataframe is None:
                print("Error: Dataframe must be provided")
                return None
                
            df = dataframe.copy()
            print(f"Converting day_of_week column to sin/cos format in dataframe with {len(df)} rows")
            
            if df.empty:
                print("Warning: Empty dataframe")
                return df
            
            # Check if 'day_of_week' column exists
            if 'day_of_week' not in df.columns:
                print("Warning: 'day_of_week' column not found in dataframe. Skipping day_of_week to sin/cos conversion.")
                print(f"Available columns: {list(df.columns)}")
                return df
            
            # Use the logging method for detailed logging
            with self.get_logger("convert_dayofweek_to_sincos.log", "convert_dayofweek_to_sincos", month_id) as logger:
                try:
                    logger.info(f"Starting day_of_week to sin/cos conversion for {len(df)} rows")
                    
                    # Debug: Check sample values
                    print("\n--- DEBUGGING DAY_OF_WEEK COLUMN ---")
                    sample_values = df['day_of_week'].dropna().head(5).tolist()
                    print(f"Sample day_of_week values (first 5 non-null): {sample_values}")
                    
                    # Check for missing values
                    missing_count = df['day_of_week'].isna().sum()
                    valid_count = len(df) - missing_count
                    print(f"Valid day_of_week values: {valid_count}")
                    print(f"Missing day_of_week values: {missing_count}")
                    
                    # Check for invalid day_of_week values (not 1-7)
                    invalid_days = df[(df['day_of_week'].notna()) & ((df['day_of_week'] < 1) | (df['day_of_week'] > 7))]
                    invalid_count = len(invalid_days)
                    
                    if invalid_count > 0:
                        print(f"Warning: Found {invalid_count} invalid day_of_week values (not 1-7)")
                        print(f"Invalid values: {sorted(invalid_days['day_of_week'].unique())}")
                        logger.warning(f"Found {invalid_count} invalid day_of_week values: {sorted(invalid_days['day_of_week'].unique())}")
                    
                    if valid_count == 0:
                        print("Warning: All day_of_week values are missing. Cannot convert to sin/cos.")
                        logger.warning("All day_of_week values are missing")
                        return df
                    
                    print("--- END DEBUGGING ---\n")
                    
                    # Create sin/cos features using the cyclical transformation for 7-day cycle
                    print("Creating day_of_week sin/cos cyclical features...")
                    
                    # Calculate sin and cos for the 7-day cycle
                    # Formula: sin(2π × day_of_week / 7) and cos(2π × day_of_week / 7)
                    # Note: We use the day_of_week values directly since they're already in 1-7 format
                    df['day_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
                    df['day_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
                    
                    # For missing day_of_week values, set sin/cos to 0 (neutral position on the unit circle)
                    day_mask = df['day_of_week'].isna()
                    df.loc[day_mask, 'day_week_sin'] = 0.0
                    df.loc[day_mask, 'day_week_cos'] = 0.0
                    
                    # Count successfully converted values (non-missing original values)
                    converted_count = df['day_of_week'].notna().sum()
                    
                    # Reorder columns to group temporal features: find day_of_week and place sin/cos after it
                    print("Reordering columns to place day_week_sin and day_week_cos right after day_of_week...")
                    
                    # Get current column order
                    current_cols = list(df.columns)
                    
                    # Find the position of day_of_week column
                    if 'day_of_week' in current_cols:
                        day_index = current_cols.index('day_of_week')
                        
                        # Remove day_week_sin and day_week_cos from their current positions
                        cols_without_sincos = [col for col in current_cols if col not in ['day_week_sin', 'day_week_cos']]
                        
                        # Insert day_week_sin and day_week_cos right after day_of_week
                        reordered_cols = (
                            cols_without_sincos[:day_index + 1] +    # Everything up to and including day_of_week
                            ['day_week_sin', 'day_week_cos'] +       # Our new weekly cyclical features
                            cols_without_sincos[day_index + 1:]      # Everything after day_of_week
                        )
                        
                        # Reorder the dataframe
                        df = df[reordered_cols]
                        
                        print(f"✓ Columns reordered: day_week_sin and day_week_cos placed after day_of_week")
                        logger.info("Columns reordered: day_week_sin and day_week_cos placed after day_of_week column")
                    else:
                        print("Warning: 'day_of_week' column not found during reordering (this shouldn't happen).")
                        logger.warning("day_of_week column not found during reordering")
                    
                    print(f"Day of week to sin/cos conversion completed:")
                    print(f"- Original day_of_week column preserved")
                    print(f"- Added day_week_sin column (range: {df['day_week_sin'].min():.3f} to {df['day_week_sin'].max():.3f})")
                    print(f"- Added day_week_cos column (range: {df['day_week_cos'].min():.3f} to {df['day_week_cos'].max():.3f})")
                    print(f"- Successfully converted {converted_count} out of {len(df)} rows")
                    print(f"- Temporal features order: month → month_sin → month_cos → day_of_week → day_week_sin → day_week_cos")
                    
                    # Show some examples of the conversion
                    if converted_count > 0:
                        print(f"\nExample weekly conversions:")
                        # Try to show examples from different days if available
                        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                        example_days = []
                        for target_day in [1, 2, 6, 7]:  # Sunday, Monday, Friday, Saturday
                            day_examples = df[df['day_of_week'] == target_day].head(1)
                            if not day_examples.empty:
                                example_days.extend(day_examples.index.tolist())
                        
                        # If we don't have examples from all target days, just show first few
                        if not example_days:
                            example_days = df[df['day_of_week'].notna()].head(3).index.tolist()
                        
                        for idx in example_days[:3]:  # Show max 3 examples
                            day_val = df.loc[idx, 'day_of_week']
                            sin_val = df.loc[idx, 'day_week_sin']
                            cos_val = df.loc[idx, 'day_week_cos']
                            
                            # Get day name for context
                            day_name = day_names[int(day_val) - 1] if 1 <= day_val <= 7 else f"Unknown({day_val})"
                            print(f"  Index {idx}: Day {day_val} ({day_name}) -> sin={sin_val:.3f}, cos={cos_val:.3f}")
                    
                    logger.info(f"Day of week to sin/cos conversion completed successfully for {converted_count} rows")
                    logger.info(f"Added columns: day_week_sin, day_week_cos")
                    logger.info(f"Preserved column: day_of_week")
                    
                    return df
                    
                except Exception as e:
                    error_msg = f"Error converting day_of_week to sin/cos: {e}"
                    print(error_msg)
                    logger.error(error_msg)
                    logger.error(f"Exception details: {str(e)}")
                    return dataframe  # Return original dataframe on error

    def drop_original_temporal_columns(self, dataframe=None, month_id=None):
        """
        Drop the original temporal columns that have been converted to cyclical features.
        
        This method removes the original categorical temporal columns after they have been
        converted to sin/cos cyclical features, keeping only the engineered features:
        - Drops 'month' (already converted to month_sin, month_cos)
        - Drops 'hour' (already converted to hour_sin, hour_cos)  
        - Drops 'day_of_week' (categorical temporal feature)
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to process.
        month_id : str, optional
            Month identifier for logging purposes.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with original temporal columns removed.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        print(f"Dropping original temporal columns in dataframe with {len(df)} rows")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
        
        # Define columns to drop
        columns_to_drop = ['month', 'hour', 'day_of_week']
        
        # Use the logging method for detailed logging
        with self.get_logger("drop_original_temporal_columns.log", "drop_original_temporal_columns", month_id) as logger:
            try:
                logger.info(f"Starting original temporal columns drop for {len(df)} rows")
                
                # Check which columns exist and which are missing
                existing_columns = []
                missing_columns = []
                
                for col in columns_to_drop:
                    if col in df.columns:
                        existing_columns.append(col)
                    else:
                        missing_columns.append(col)
                
                print(f"\n--- TEMPORAL COLUMNS ANALYSIS ---")
                print(f"Columns to drop: {columns_to_drop}")
                print(f"Existing columns to drop: {existing_columns}")
                if missing_columns:
                    print(f"Missing columns (already dropped?): {missing_columns}")
                print(f"--- END ANALYSIS ---\n")
                
                # Log the analysis
                logger.info(f"Columns to drop: {columns_to_drop}")
                logger.info(f"Existing columns: {existing_columns}")
                if missing_columns:
                    logger.warning(f"Missing columns: {missing_columns}")
                
                if not existing_columns:
                    print("Info: None of the target temporal columns exist in the dataframe")
                    logger.info("No temporal columns to drop - they may have been dropped already")
                    return df
                
                # Drop the existing columns
                print(f"Dropping {len(existing_columns)} temporal columns: {existing_columns}")
                df = df.drop(columns=existing_columns)
                
                # Verify the columns were dropped
                still_existing = [col for col in existing_columns if col in df.columns]
                if still_existing:
                    error_msg = f"Failed to drop columns: {still_existing}"
                    print(f"Error: {error_msg}")
                    logger.error(error_msg)
                    return dataframe  # Return original on error
                
                print(f"Original temporal columns drop completed:")
                print(f"- Dropped columns: {existing_columns}")
                print(f"- Remaining columns: {len(df.columns)}")
                print(f"- Final dataset shape: {df.shape}")
                
                # Log successful completion
                logger.info(f"Successfully dropped {len(existing_columns)} temporal columns")
                logger.info(f"Dropped columns: {existing_columns}")
                logger.info(f"Final dataset shape: {df.shape}")
                
                return df
                
            except Exception as e:
                error_msg = f"Error dropping original temporal columns: {e}"
                print(error_msg)
                logger.error(error_msg)
                logger.error(f"Exception details: {str(e)}")
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
            The dataframe with only the selected target feature retained and optional 
            train features dropped based on configuration.
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
        # Note: You may need to define DROP_TRAIN_FEATURES and NON_NUMERIC_FEATURES in const.py
        # For now, I'll comment this out since these constants weren't found in the search
        """
        if hasattr(self, 'DROP_TRAIN_FEATURES') and self.DROP_TRAIN_FEATURES:
            features_to_drop = [col for col in NON_NUMERIC_FEATURES if col in df.columns]
            if features_to_drop:
                df = df.drop(columns=features_to_drop)
                print(f"Dropped train features as configured: {features_to_drop}")
            else:
                print("No train features found to drop")
        """
        
        print(f"Final dataframe shape: {df.shape}")
        return df

    def filter_strong_weather_causes(self, dataframe=None, month_id=None, filename=None):
        """
        Filter the dataframe to keep only rows where causes_related_to_weather equals 3 (strong weather indicators).
        
        This method filters the data to focus on train delays that are strongly related to weather conditions.
        Based on the causes_related_to_weather column:
        - 3: Strong weather indicator ('I1', 'I2' category codes)
        - 2: Possible weather indicator 
        - 1: Weak weather indicator
        - 0: No weather indicator
        
        Only rows with value 3 (strong weather indicators) are retained.
        
        If the 'causes_related_to_weather' column is not found, the entire dataframe is dropped
        and a CSV file with only column names (no data) is saved.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to filter.
        month_id : str, optional
            Month identifier for logging purposes (format: YYYY_MM).
        filename : str, optional
            Original filename to extract date information if month_id is not available.
            
        Returns:
        --------
        pandas.DataFrame
            The filtered dataframe containing only strong weather-related delays.
            Returns empty dataframe with column names if causes_related_to_weather column is missing.
        """
        import re
        import os
        from datetime import datetime
        
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        
        # Extract month and year information
        month_info = "Unknown"
        year_info = "Unknown"
        
        # Priority 1: Try to get month from dataframe's 'month' column
        if 'month' in df.columns and not df.empty:
            try:
                # Get the first non-null month value
                month_values = df['month'].dropna()
                if not month_values.empty:
                    first_month = month_values.iloc[0]
                    if isinstance(first_month, (int, float)) and not pd.isna(first_month):
                        month_info = f"{int(first_month):02d}"
                    elif isinstance(first_month, str):
                        month_info = first_month.strip()
                    print(f"Extracted month from dataframe 'month' column: {month_info}")
            except Exception as e:
                print(f"Could not extract month from dataframe 'month' column: {e}")
        
        # Priority 2: Try to get from month_id parameter
        if month_info == "Unknown" and month_id:
            try:
                # month_id should be in format YYYY_MM
                if '_' in month_id:
                    year_part, month_part = month_id.split('_')
                    month_info = month_part
                    year_info = year_part
                    print(f"Extracted from month_id parameter: Year={year_info}, Month={month_info}")
            except Exception as e:
                print(f"Could not extract month from month_id parameter: {e}")
        
        # Priority 3: Try to extract from filename
        if month_info == "Unknown" and filename:
            try:
                # Extract from filename pattern: matched_data_YYYY_MM.csv
                match = re.search(r'(\d{4}).*?(\d{2})', filename)
                if match:
                    year_info, month_info = match.groups()
                    print(f"Extracted from filename: Year={year_info}, Month={month_info}")
            except Exception as e:
                print(f"Could not extract month from filename: {e}")
        
        # Set current year if not found
        if year_info == "Unknown":
            year_info = str(datetime.now().year)
            print(f"Using current year: {year_info}")
        
        # Use the logging context manager for comprehensive logging
        with self.get_logger("filter_strong_weather_causes.log", "strong_weather_filter", month_id) as logger:
            print(f"Filtering for strong weather causes in dataframe with {len(df)} rows")
            logger.info(f"=== STRONG WEATHER CAUSES FILTERING START ===")
            logger.info(f"Input dataframe shape: {df.shape}")
            logger.info(f"Processing month: {month_info}, Year: {year_info}")
            
            if df.empty:
                print("Warning: Empty dataframe")
                logger.warning("Empty dataframe provided - no filtering performed")
                return df
            
            # Check if 'causes_related_to_weather' column exists
            if 'causes_related_to_weather' not in df.columns:
                warning_msg = f"'causes_related_to_weather' column not found in dataframe for {year_info}-{month_info}. Dropping entire dataframe and saving empty CSV with column names only."
                print(f"Warning: {warning_msg}")
                logger.warning(warning_msg)
                logger.info(f"Available columns: {list(df.columns)}")
                
                # Create empty dataframe with same columns but no data
                empty_df = pd.DataFrame(columns=df.columns)
                logger.info(f"Returning empty dataframe with {len(empty_df.columns)} columns and 0 rows")
                
                return empty_df
            
            # Log initial data distribution
            logger.info(f"Initial data analysis:")
            logger.info(f"  - Total rows: {len(df):,}")
            logger.info(f"  - Total columns: {len(df.columns)}")
            logger.info(f"  - Month/Year: {month_info}/{year_info}")
            
            # Store original row count
            original_rows = len(df)
            
            # Show distribution of causes_related_to_weather values in original data
            value_counts = df['causes_related_to_weather'].value_counts().sort_index()
            
            print(f"\nOriginal causes_related_to_weather distribution for {year_info}-{month_info}:")
            logger.info(f"Original causes_related_to_weather distribution for {year_info}-{month_info}:")
            
            for value, count in value_counts.items():
                percentage = (count / original_rows) * 100
                if value == 0:
                    label = "No weather indicator"
                elif value == 1:
                    label = "Weak weather indicator"
                elif value == 2:
                    label = "Possible weather indicator"
                elif value == 3:
                    label = "Strong weather indicator"
                else:
                    label = f"Unknown value ({value})"
                
                log_msg = f"  {value}: {count:,} rows ({percentage:.1f}%) - {label}"
                print(log_msg)
                logger.info(log_msg)
            
            # Filter to keep only rows where causes_related_to_weather equals 3 (strong weather indicators)
            logger.info("Applying filter: causes_related_to_weather == 3")
            filtered_df = df[df['causes_related_to_weather'] == 3].copy()
            
            # Calculate filtering statistics
            filtered_rows = len(filtered_df)
            filtered_percentage = (filtered_rows / original_rows) * 100 if original_rows > 0 else 0
            removed_rows = original_rows - filtered_rows
            removed_percentage = (removed_rows / original_rows) * 100 if original_rows > 0 else 0
            
            # Display and log filtering results
            print(f"\n=== STRONG WEATHER CAUSES FILTERING RESULTS ({year_info}-{month_info}) ===")
            logger.info(f"=== FILTERING RESULTS SUMMARY ({year_info}-{month_info}) ===")
            
            results_summary = [
                f"Original rows: {original_rows:,}",
                f"Rows with strong weather causes (value = 3): {filtered_rows:,} ({filtered_percentage:.1f}%)",
                f"Rows removed: {removed_rows:,} ({removed_percentage:.1f}%)",
                f"Data retention rate: {filtered_percentage:.1f}%"
            ]
            
            for result in results_summary:
                print(result)
                logger.info(result)
            
            # Log detailed statistics
            logger.info("=== DETAILED FILTERING STATISTICS ===")
            logger.info(f"Filter criteria: Keep rows where causes_related_to_weather == 3")
            logger.info(f"Rows meeting criteria: {filtered_rows:,}")
            logger.info(f"Rows not meeting criteria: {removed_rows:,}")
            logger.info(f"Percentage of data retained: {filtered_percentage:.2f}%")
            logger.info(f"Percentage of data removed: {removed_percentage:.2f}%")
            
            # Warn if no data remains after filtering
            if filtered_rows == 0:
                warning_msg = f"WARNING: No rows with strong weather causes found for {year_info}-{month_info}! The filtered dataset is empty."
                recommendation = "Consider using a different filter criteria or checking the data."
                print(warning_msg)
                print(recommendation)
                logger.warning(warning_msg)
                logger.warning(recommendation)

            # Drop the causes_related_to_weather column since it's no longer needed for training
            # (all remaining rows have the same value: 3)
            if 'causes_related_to_weather' in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=['causes_related_to_weather'])
                print(f"Dropped 'causes_related_to_weather' column (no longer needed for training)")
                logger.info("Dropped 'causes_related_to_weather' column - all remaining rows had value 3")

            logger.info(f"=== STRONG WEATHER CAUSES FILTERING COMPLETE ({year_info}-{month_info}) ===")
            return filtered_df

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
        month_id : str, optional
            Month identifier for logging purposes.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with duplicates removed.
        """
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
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
            print(f"\n=== DUPLICATE REMOVAL RESULTS ===")
            print(f"Original row count: {original_row_count:,}")
            print(f"Duplicate rows removed: {removed_duplicates:,}")
            print(f"Remaining rows: {len(df_deduplicated):,}")
            print(f"Data retention: {retention_percentage:.2f}%")
            print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
            
            # Log the essential summary information
            logger.info(f"Original row count: {original_row_count}")
            logger.info(f"Duplicate rows removed: {removed_duplicates}")
            logger.info(f"Remaining rows: {len(df_deduplicated)}")
            logger.info(f"Data retention: {retention_percentage:.2f}%")
            logger.info(f"Duplicate percentage: {duplicate_percentage:.2f}%")
            
            # Warn if high percentage of duplicates found
            if duplicate_percentage > 10:
                warning_msg = f"WARNING: High duplicate percentage ({duplicate_percentage:.1f}%) detected!"
                print(f"⚠️  {warning_msg}")  # Use emoji only for console output
                logger.warning(warning_msg)  # Use plain text for logging
            elif removed_duplicates > 0:
                success_msg = f"Successfully removed {removed_duplicates:,} duplicate rows"
                print(f"✓ {success_msg}")  # Use checkmark only for console output
                logger.info(success_msg)  # Log success message too
            else:
                success_msg = "No duplicate rows found - data is already clean"
                print(f"✓ {success_msg}")  # Use checkmark only for console output
                logger.info(success_msg)  # Log success message too
        
        return df_deduplicated

    def save_training_ready_csv(self, month_id, dataframe):
        """
        Save a training-ready dataframe to the training-ready directory.
        
        Parameters:
        -----------
        month_id : str
            Month identifier in format "YYYY_MM" for the filename.
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
                
            # Create the training-ready directory path
            training_ready_dir = TRAINING_READY_OUTPUT_FOLDER
            
            # Create the output filename with training_ready prefix
            filename = f"training_ready_{month_id}.csv"
            file_path = os.path.join(training_ready_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(training_ready_dir, exist_ok=True)
            
            # Save the dataframe
            dataframe.to_csv(file_path, index=False)
            print(f"Successfully saved training-ready dataframe to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving training-ready dataframe for {month_id}: {e}")
            return False
        
    def execute_training_pipeline_steps(self, csv_files, state_machine):
        """
        Execute training pipeline steps based on the training state machine configuration.
        
        This method processes CSV files through the configured training pipeline steps,
        maintaining data flow between steps and handling errors gracefully.
        
        Parameters:
        -----------
        csv_files : list
            List of CSV file paths to process
        state_machine : dict
            Training state machine configuration defining which steps to execute
            
        Returns:
        --------
        dict
            Results of training pipeline execution including success status and metadata
        """
        result = {
            "success": False,
            "data": None,
            "steps_executed": [],
            "errors": [],
            "file_info": {
                "total_files": len(csv_files),
                "processed_files": 0
            }
        }
        
        print(f"  Executing training pipeline steps based on state machine configuration...")
        print(f"  Training state machine: {state_machine}")
        
        # Execute merge_data_files if enabled
        if state_machine.get("merge_data_files", False):
            try:
                print(f"    → merge_data_files")
                merged_result = self.merge_data_files(csv_files)
                
                if merged_result and merged_result.get("success", False):
                    result["data"] = merged_result.get("data")
                    result["steps_executed"].append("merge_data_files")
                    result["file_info"]["processed_files"] = merged_result.get("processed_files", 0)
                    print(f"      ✓ Successfully merged data files")
                    result["success"] = True
                else:
                    error_msg = merged_result.get("error", "merge_data_files returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"merge_data_files failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ merge_data_files (disabled)")

        if state_machine.get("split_dataset", False):
            try:
                print(f"    → split_dataset")
                split_result = self.split_dataset(csv_files)
                
                if split_result and split_result.get("success", False):
                    result["steps_executed"].append("split_dataset")
                    result["file_info"]["processed_files"] = split_result.get("processed_files", 0)
                    print(f"      ✓ Successfully split datasets into train/test sets")
                    print(f"      ✓ Total train rows: {split_result.get('total_train_rows', 0):,}")
                    print(f"      ✓ Total test rows: {split_result.get('total_test_rows', 0):,}")
                    result["success"] = True
                else:
                    error_msg = split_result.get("error", "split_dataset returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"split_dataset failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ split_dataset (disabled)")

        if state_machine.get("scale_weather_features", False):
            try:
                print(f"    → scale_weather_features")
                scaling_result = self.scale_weather_features(csv_files)
                
                if scaling_result and scaling_result.get("success", False):
                    result["steps_executed"].append("scale_weather_features")
                    result["file_info"]["processed_files"] = scaling_result.get("processed_files", 0)
                    print(f"      ✓ Successfully scaled weather features")
                    print(f"      ✓ Total train rows: {scaling_result.get('total_train_rows', 0):,}")
                    print(f"      ✓ Total test rows: {scaling_result.get('total_test_rows', 0):,}")
                    print(f"      ✓ Weather features scaled: {scaling_result.get('weather_features_available', [])}")
                    result["success"] = True
                else:
                    error_msg = scaling_result.get("error", "scale_weather_features returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"scale_weather_features failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ scale_weather_features (disabled)")

        if state_machine.get("correlation_analysis", False):
            try:
                print(f"    → correlation_analysis (Point-Biserial)")
                correlation_result = self.correlation_analysis(csv_files)
                
                if correlation_result and correlation_result.get("success", False):
                    result["steps_executed"].append("correlation_analysis")
                    result["file_info"]["processed_files"] = correlation_result.get("processed_files", 0)
                    print(f"      ✓ Successfully completed Point-Biserial correlation analysis")
                    print(f"      ✓ Files analyzed: {correlation_result.get('processed_files', 0)}")
                    print(f"      ✓ Weather features: {correlation_result.get('total_weather_features', 0)}")
                    print(f"      ✓ Analysis type: {correlation_result.get('analysis_type', 'Point-Biserial Correlation')}")
                    print(f"      ✓ Results saved to: {correlation_result.get('output_path', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = correlation_result.get("error", "Point-Biserial correlation_analysis returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"Point-Biserial correlation_analysis failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ correlation_analysis (Point-Biserial) (disabled)")

        if state_machine.get("non_weather_correlation_analysis", False):
                    try:
                        print(f"    → non_weather_correlation_analysis (Non-Weather Features)")
                        non_weather_correlation_result = self.non_weather_correlation_analysis(csv_files)
                        
                        if non_weather_correlation_result and non_weather_correlation_result.get("success", False):
                            result["steps_executed"].append("non_weather_correlation_analysis")
                            result["file_info"]["non_weather_processed_files"] = non_weather_correlation_result.get("processed_files", 0)
                            print(f"      ✓ Successfully completed non-weather correlation analysis")
                            print(f"      ✓ Files analyzed: {non_weather_correlation_result.get('processed_files', 0)}")
                            print(f"      ✓ Boolean features: {non_weather_correlation_result.get('total_boolean_features', 0)}")
                            print(f"      ✓ Temporal features: {non_weather_correlation_result.get('total_temporal_features', 0)}")
                            print(f"      ✓ Analysis type: {non_weather_correlation_result.get('analysis_type', 'Non-Weather Features Correlation')}")
                            print(f"      ✓ Results saved to: {non_weather_correlation_result.get('output_path', 'N/A')}")
                            result["success"] = True
                        else:
                            error_msg = non_weather_correlation_result.get("error", "Non-weather correlation_analysis returned unsuccessful result")
                            result["errors"].append(error_msg)
                            print(f"      ✗ Failed - {error_msg}")
                            return result
                            
                    except Exception as e:
                        result["errors"].append(f"Non-weather correlation_analysis failed: {str(e)}")
                        print(f"      ✗ Failed - {str(e)}")
                        return result
        else:
            print(f"    ⊝ non_weather_correlation_analysis (Non-Weather Features) (disabled)")


        if state_machine.get("train_decision_tree", False):
            try:
                print(f"    → train_decision_tree")
                dt_result = self.train_decision_tree()
                
                if dt_result and dt_result.get("success", False):
                    result["steps_executed"].append("train_decision_tree")
                    result["file_info"]["decision_tree_files"] = dt_result.get("files_processed", 0)
                    print(f"      ✓ Successfully trained Decision Tree classifier")
                    print(f"      ✓ Target feature: {dt_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Files processed: {dt_result.get('files_processed', 0)}")
                    print(f"      ✓ Total train samples: {dt_result.get('total_train_samples', 0):,}")
                    print(f"      ✓ Total test samples: {dt_result.get('total_test_samples', 0):,}")
                    print(f"      ✓ Results saved to: {dt_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = dt_result.get("error", "train_decision_tree returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"train_decision_tree failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_decision_tree (disabled)")

        if state_machine.get("threshold_optimization_decision_tree", False):
            try:
                print(f"    → threshold_optimization_decision_tree")
                threshold_opt_result = self.threshold_optimization_decision_tree()
                
                if threshold_opt_result and threshold_opt_result.get("success", False):
                    result["steps_executed"].append("threshold_optimization_decision_tree")
                    result["file_info"]["threshold_optimized_models"] = threshold_opt_result.get("models_optimized", 0)
                    print(f"      ✓ Successfully optimized decision tree thresholds")
                    print(f"      ✓ Target feature: {threshold_opt_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Models optimized: {threshold_opt_result.get('models_optimized', 0)}")
                    print(f"      ✓ Average optimal threshold: {threshold_opt_result.get('average_optimal_threshold', 0):.3f}")
                    print(f"      ✓ Average optimized F1: {threshold_opt_result.get('average_optimized_f1', 0):.3f}")
                    print(f"      ✓ Results saved to: {threshold_opt_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = threshold_opt_result.get("error", "threshold_optimization_decision_tree returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"threshold_optimization_decision_tree failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ threshold_optimization_decision_tree (disabled)")

        if state_machine.get("generate_borderline_smote_data", False):
            try:
                print(f"    → generate_borderline_smote_data")
                smote_result = self.generate_borderline_smote_data()
                
                if smote_result and smote_result.get("success", False):
                    result["steps_executed"].append("generate_borderline_smote_data")
                    result["file_info"]["synthetic_samples_generated"] = smote_result.get("total_synthetic_samples", 0)
                    result["file_info"]["smote_files_processed"] = smote_result.get("files_processed", 0)
                    print(f"      ✓ Successfully generated synthetic data using BorderlineSMOTE")
                    print(f"      ✓ Target feature: {smote_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Files processed: {smote_result.get('files_processed', 0)}")
                    print(f"      ✓ Synthetic samples generated: {smote_result.get('total_synthetic_samples', 0):,}")
                    print(f"      ✓ Average augmentation ratio: {smote_result.get('average_augmentation_ratio', 0):.2f}x")
                    print(f"      ✓ Results saved to: {smote_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = smote_result.get("error", "generate_borderline_smote_data returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"generate_borderline_smote_data failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ generate_borderline_smote_data (disabled)")


        
        return result
    
    def correlation_analysis(self, csv_files=None):
        """
        Analyze Point-Biserial correlations between binary target feature and continuous weather features.
        
        This method finds all merged scaled training data files in data/output/4-merged_scaled_training_ready,
        calculates Point-Biserial correlations between the target feature (trainDelayed - binary) 
        and all weather features (continuous and scaled), and creates Point-Biserial correlation visualizations.
        
        Point-Biserial correlation is the appropriate correlation measure when one variable 
        is binary (trainDelayed: True/False) and the other is continuous (weather measurements).
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (currently not used - method discovers files automatically)
            
        Returns:
        --------
        dict
            Results of the Point-Biserial correlation analysis including success status and analysis info
        """
        try:
            print(f"    correlation_analysis: Starting Point-Biserial correlation analysis on scaled training data...")
            
            # Create output directory for correlation analysis results
            correlation_output_dir = os.path.join(self.project_root, "data/output/correlation_analysis")
            os.makedirs(correlation_output_dir, exist_ok=True)
            
            # Find all merged scaled training data files using glob pattern
            merged_data_pattern = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*_train_scaled.csv")
            merged_data_files = glob.glob(merged_data_pattern)
            
            if not merged_data_files:
                error_msg = "No merged scaled training data files found for Point-Biserial correlation analysis"
                print(f"    correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    correlation_analysis: Found {len(merged_data_files)} merged scaled training data files")
            for file_path in merged_data_files:
                print(f"      - {os.path.basename(file_path)}")
            
            # Initialize storage for results
            correlation_results = []
            all_correlations = []
            total_files_processed = 0
            
            # Process each merged data file
            for file_path in merged_data_files:
                try:
                    filename = os.path.basename(file_path)
                    print(f"    correlation_analysis: Processing {filename}...")
                    
                    # Read the merged dataset
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"    correlation_analysis: Warning - File {filename} is empty. Skipping.")
                        continue
                    
                    # Check if target feature exists and is binary
                    if DEFAULT_TARGET_FEATURE not in df.columns:
                        print(f"    correlation_analysis: Warning - Target feature '{DEFAULT_TARGET_FEATURE}' not found in {filename}. Skipping.")
                        continue
                    
                    # Verify target is binary for Point-Biserial correlation
                    target_unique_values = df[DEFAULT_TARGET_FEATURE].dropna().unique()
                    if len(target_unique_values) > 2:
                        print(f"    correlation_analysis: Warning - Target feature has more than 2 unique values. Point-Biserial correlation assumes binary target.")
                    
                    # Filter for available weather features in the dataset
                    available_weather_features = [col for col in ALL_WEATHER_FEATURES if col in df.columns]
                    
                    if not available_weather_features:
                        print(f"    correlation_analysis: Warning - No weather features found in {filename}. Skipping.")
                        continue
                    
                    print(f"      Found {len(available_weather_features)} weather features for Point-Biserial analysis")
                    
                    # Calculate Point-Biserial correlations between binary target and continuous weather features
                    target_series = df[DEFAULT_TARGET_FEATURE]
                    correlations = {}
                    
                    for weather_feature in available_weather_features:
                        try:
                            # Calculate Point-Biserial correlation (using Pearson's formula which is equivalent)
                            weather_series = df[weather_feature]
                            
                            # Only calculate correlation if both series have valid data
                            mask = pd.notna(target_series) & pd.notna(weather_series)
                            if mask.sum() < 10:  # Need at least 10 valid pairs
                                print(f"        Warning: Insufficient valid data for {weather_feature} Point-Biserial correlation")
                                correlations[weather_feature] = np.nan
                                continue
                            
                            # Point-Biserial correlation (mathematically equivalent to Pearson when one variable is binary)
                            correlation = target_series[mask].corr(weather_series[mask])
                            correlations[weather_feature] = correlation
                            
                            print(f"        {weather_feature}: r_pb = {correlation:.4f}")
                            
                        except Exception as e:
                            print(f"        Warning: Failed to calculate Point-Biserial correlation for {weather_feature}: {str(e)}")
                            correlations[weather_feature] = np.nan
                    
                    # Store results for this file
                    target_proportion = target_series.mean() if target_series.dtype in [bool, int, float] else None
                    file_result = {
                        'filename': filename,
                        'correlations': correlations,
                        'available_features': available_weather_features,
                        'total_samples': len(df),
                        'target_proportion': target_proportion,  # Proportion of positive cases
                        'target_name': DEFAULT_TARGET_FEATURE
                    }
                    correlation_results.append(file_result)
                    all_correlations.append(correlations)
                    
                    # Create individual Point-Biserial correlation plot for this file
                    self._create_correlation_plot(
                        correlations, 
                        filename,
                        correlation_output_dir,
                        f"Point-Biserial Correlation: {DEFAULT_TARGET_FEATURE} vs Weather Features\nFile: {filename}"
                    )
                    
                    # Create distribution plots for this file
                    self._create_distribution_plots(
                        df,
                        available_weather_features,
                        filename,
                        correlation_output_dir
                    )
                    
                    total_files_processed += 1
                    print(f"      Successfully processed Point-Biserial correlation analysis for {filename}")
                    
                except Exception as e:
                    print(f"    correlation_analysis: Error processing {filename}: {str(e)}")
                    continue
            
            # Check if we processed any files successfully
            if not correlation_results:
                error_msg = "No files were successfully processed for Point-Biserial correlation analysis"
                print(f"    correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            # Create combined Point-Biserial correlation analysis across all files
            if len(correlation_results) > 1:
                self._create_combined_correlation_analysis(correlation_results, correlation_output_dir)
            
            # Save detailed Point-Biserial correlation summary
            summary_filename = "point_biserial_analysis_summary.txt"
            summary_path = os.path.join(correlation_output_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write("Point-Biserial Correlation Analysis Summary\n")
                f.write("=" * 55 + "\n\n")
                
                f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Target feature: {DEFAULT_TARGET_FEATURE} (Binary variable)\n")
                f.write(f"Weather features analyzed: {len(ALL_WEATHER_FEATURES)} (Continuous variables)\n")
                f.write(f"Files processed: {total_files_processed}\n\n")
                
                f.write("Analysis Components Generated:\n")
                f.write("• Point-Biserial correlation plots for each scaled training file\n")
                f.write("• Feature distribution plots for each scaled training file\n")
                f.write("• Comparative distribution plots (delayed vs not delayed) using scaled features\n")
                f.write("• Combined correlation analysis across all scaled training files\n\n")
                
                f.write("Point-Biserial Correlation Overview:\n")
                f.write("This analysis measures the correlation between a binary variable\n")
                f.write(f"({DEFAULT_TARGET_FEATURE}: True/False) and continuous scaled variables (weather measurements).\n")
                f.write("The correlation coefficient ranges from -1 to +1, where:\n")
                f.write("• Positive values: Higher scaled weather values associate with more train delays\n")
                f.write("• Negative values: Higher scaled weather values associate with fewer train delays\n")
                f.write("• Values near 0: Little to no linear relationship\n\n")
                
                # File-by-file results
                f.write("File-by-file Analysis:\n")
                f.write("-" * 30 + "\n")
                for result in correlation_results:
                    f.write(f"\nFile: {result['filename']}\n")
                    f.write(f"  Total samples: {result['total_samples']:,}\n")
                    if result['target_proportion'] is not None:
                        f.write(f"  Delay proportion: {result['target_proportion']:.4f} ({result['target_proportion']*100:.2f}%)\n")
                    f.write(f"  Weather features available: {len(result['available_features'])}\n")
                    
                    f.write(f"  Point-Biserial Correlations:\n")
                    sorted_correlations = sorted(
                        [(k, v) for k, v in result['correlations'].items() if not pd.isna(v)],
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    
                    for feature, corr in sorted_correlations:
                        f.write(f"    {feature}: r_pb = {corr:+.4f}\n")
                    
                    # Identify strongest correlations
                    if sorted_correlations:
                        strongest = sorted_correlations[0]
                        f.write(f"  Strongest correlation: {strongest[0]} (r_pb = {strongest[1]:+.4f})\n")
            
            print(f"    correlation_analysis: Point-Biserial analysis on scaled training data completed for {total_files_processed} files")
            print(f"    correlation_analysis: Results saved to {correlation_output_dir}")
            
            # Return success result
            result = {
                "success": True,
                "processed_files": total_files_processed,
                "output_path": correlation_output_dir,
                "summary_path": summary_path,
                "correlation_results": correlation_results,
                "total_weather_features": len(ALL_WEATHER_FEATURES),
                "analysis_type": "Point-Biserial Correlation",
                "message": f"Successfully analyzed Point-Biserial correlations for {total_files_processed} scaled training files"
            }
            
            print(f"    correlation_analysis: Completed successfully - {total_files_processed} scaled training files analyzed")
            
            return result
            
        except Exception as e:
            error_msg = f"Point-Biserial correlation_analysis failed: {str(e)}"
            print(f"    correlation_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }

    def non_weather_correlation_analysis(self, csv_files=None):
        """
        Analyze correlations between trainDelayed (binary target) and non-weather features.
        
        This method finds all merged scaled training data files in data/output/4-merged_scaled_training_ready,
        calculates correlations between the target feature (trainDelayed - binary) and:
        - Boolean features: trainStopping, commercialStop (using Phi coefficient)
        - Temporal sin-cos features: month_sin, month_cos, hour_sin, hour_cos, day_week_sin, day_week_cos (using Point-Biserial)
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (currently not used - method discovers files automatically)
            
        Returns:
        --------
        dict
            Results of the non-weather correlation analysis including success status and analysis info
        """
        try:
            print(f"    non_weather_correlation_analysis: Starting correlation analysis for non-weather features...")
            
            # Create output directory for non-weather correlation analysis results
            output_dir = os.path.join(self.project_root, "data/output/correlation_analysis")
            os.makedirs(output_dir, exist_ok=True)
            
            # Define feature categories to analyze
            BOOLEAN_FEATURES = ['trainStopping', 'commercialStop']
            TEMPORAL_SINCOS_FEATURES = [
                'month_sin', 'month_cos', 
                'hour_sin', 'hour_cos', 
                'day_week_sin', 'day_week_cos'
            ]
            ALL_NON_WEATHER_FEATURES = BOOLEAN_FEATURES + TEMPORAL_SINCOS_FEATURES
            
            # Find all merged scaled training data files using glob pattern
            merged_data_pattern = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*_train_scaled.csv")
            merged_data_files = glob.glob(merged_data_pattern)
            
            if not merged_data_files:
                error_msg = "No merged scaled training data files found for non-weather correlation analysis"
                print(f"    non_weather_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    non_weather_correlation_analysis: Found {len(merged_data_files)} merged scaled training data files")
            for file_path in merged_data_files:
                print(f"      - {os.path.basename(file_path)}")
            
            # Initialize storage for results
            correlation_results = []
            total_files_processed = 0
            
            # Process each merged data file
            for file_path in merged_data_files:
                try:
                    filename = os.path.basename(file_path)
                    print(f"    non_weather_correlation_analysis: Processing {filename}...")
                    
                    # Read the merged dataset
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"    non_weather_correlation_analysis: Warning - File {filename} is empty. Skipping.")
                        continue
                    
                    # Check if target feature exists and is binary
                    if DEFAULT_TARGET_FEATURE not in df.columns:
                        print(f"    non_weather_correlation_analysis: Warning - Target feature '{DEFAULT_TARGET_FEATURE}' not found in {filename}. Skipping.")
                        continue
                    
                    # Verify target is binary
                    target_unique_values = df[DEFAULT_TARGET_FEATURE].dropna().unique()
                    if len(target_unique_values) > 2:
                        print(f"    non_weather_correlation_analysis: Warning - Target feature has more than 2 unique values.")
                    
                    # Filter for available non-weather features in the dataset
                    available_boolean_features = [col for col in BOOLEAN_FEATURES if col in df.columns]
                    available_temporal_features = [col for col in TEMPORAL_SINCOS_FEATURES if col in df.columns]
                    
                    if not available_boolean_features and not available_temporal_features:
                        print(f"    non_weather_correlation_analysis: Warning - No non-weather features found in {filename}. Skipping.")
                        continue
                    
                    print(f"      Found {len(available_boolean_features)} boolean features and {len(available_temporal_features)} temporal features")
                    
                    # Calculate correlations for different feature types
                    target_series = df[DEFAULT_TARGET_FEATURE]
                    correlations = {}
                    correlation_types = {}
                    
                    # Boolean features - using Point-Biserial (equivalent to Phi coefficient for binary variables)
                    for feature in available_boolean_features:
                        try:
                            feature_series = df[feature]
                            
                            # Check if feature is actually boolean/binary
                            unique_vals = feature_series.dropna().unique()
                            if len(unique_vals) > 2:
                                print(f"        Warning: {feature} has more than 2 unique values, treating as categorical")
                            
                            # Only calculate correlation if both series have valid data
                            mask = pd.notna(target_series) & pd.notna(feature_series)
                            if mask.sum() < 10:  # Need at least 10 valid pairs
                                print(f"        Warning: Insufficient valid data for {feature}")
                                correlations[feature] = np.nan
                                correlation_types[feature] = "Boolean (insufficient data)"
                                continue
                            
                            # Point-Biserial correlation (mathematically equivalent to Phi when both variables are binary)
                            correlation = target_series[mask].corr(feature_series[mask])
                            correlations[feature] = correlation
                            correlation_types[feature] = "Boolean (Phi coefficient)"
                            
                            print(f"        {feature}: φ = {correlation:.4f}")
                            
                        except Exception as e:
                            print(f"        Warning: Failed to calculate correlation for {feature}: {str(e)}")
                            correlations[feature] = np.nan
                            correlation_types[feature] = "Boolean (error)"
                    
                    # Temporal sin-cos features - using Point-Biserial correlation
                    for feature in available_temporal_features:
                        try:
                            feature_series = df[feature]
                            
                            # Only calculate correlation if both series have valid data
                            mask = pd.notna(target_series) & pd.notna(feature_series)
                            if mask.sum() < 10:  # Need at least 10 valid pairs
                                print(f"        Warning: Insufficient valid data for {feature}")
                                correlations[feature] = np.nan
                                correlation_types[feature] = "Temporal (insufficient data)"
                                continue
                            
                            # Point-Biserial correlation
                            correlation = target_series[mask].corr(feature_series[mask])
                            correlations[feature] = correlation
                            correlation_types[feature] = "Temporal (Point-Biserial)"
                            
                            print(f"        {feature}: r_pb = {correlation:.4f}")
                            
                        except Exception as e:
                            print(f"        Warning: Failed to calculate correlation for {feature}: {str(e)}")
                            correlations[feature] = np.nan
                            correlation_types[feature] = "Temporal (error)"
                    
                    # Calculate target proportion for context
                    target_proportion = None
                    if DEFAULT_TARGET_FEATURE in df.columns:
                        target_count = df[DEFAULT_TARGET_FEATURE].sum()
                        total_count = len(df)
                        target_proportion = target_count / total_count if total_count > 0 else 0
                    
                    # Store results for this file
                    file_result = {
                        'filename': filename,
                        'total_samples': len(df),
                        'target_proportion': target_proportion,
                        'available_boolean_features': available_boolean_features,
                        'available_temporal_features': available_temporal_features,
                        'correlations': correlations,
                        'correlation_types': correlation_types
                    }
                    correlation_results.append(file_result)
                    
                    # Create visualization for this file
                    self._create_non_weather_correlation_plot(
                        correlations, correlation_types, filename, output_dir, 
                        f'Non-Weather Features Correlation Analysis - {filename}'
                    )
                    
                    total_files_processed += 1
                    
                except Exception as e:
                    print(f"    non_weather_correlation_analysis: Error processing file {filename}: {str(e)}")
                    continue
            
            # Check if we processed any files successfully
            if not correlation_results:
                error_msg = "No files were successfully processed for non-weather correlation analysis"
                print(f"    non_weather_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            # Create combined analysis across all files
            if len(correlation_results) > 1:
                self._create_combined_non_weather_correlation_analysis(correlation_results, output_dir)
            
            # Save detailed summary
            summary_filename = "non_weather_correlation_summary.txt"
            summary_path = os.path.join(output_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write("Non-Weather Features Correlation Analysis Summary\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Target feature: {DEFAULT_TARGET_FEATURE} (Binary variable)\n")
                f.write(f"Boolean features analyzed: {len(BOOLEAN_FEATURES)}\n")
                f.write(f"Temporal features analyzed: {len(TEMPORAL_SINCOS_FEATURES)}\n")
                f.write(f"Files processed: {total_files_processed}\n\n")
                
                f.write("Analysis Components Generated:\n")
                f.write("• Correlation plots for each scaled training file\n")
                f.write("• Feature distribution comparisons (delayed vs not delayed)\n")
                f.write("• Combined correlation analysis across all files\n\n")
                
                f.write("Correlation Types Used:\n")
                f.write("• Boolean features (trainStopping, commercialStop): Phi coefficient\n")
                f.write("• Temporal features (sin-cos encoded): Point-Biserial correlation\n\n")
                
                f.write("Interpretation Guide:\n")
                f.write("• Positive values: Higher feature values associate with more train delays\n")
                f.write("• Negative values: Higher feature values associate with fewer train delays\n")
                f.write("• Values near 0: Little to no linear relationship\n\n")
                
                # File-by-file results
                f.write("File-by-file Analysis:\n")
                f.write("-" * 30 + "\n")
                for result in correlation_results:
                    f.write(f"\nFile: {result['filename']}\n")
                    f.write(f"  Total samples: {result['total_samples']:,}\n")
                    if result['target_proportion'] is not None:
                        f.write(f"  Delay proportion: {result['target_proportion']:.4f} ({result['target_proportion']*100:.2f}%)\n")
                    f.write(f"  Boolean features available: {len(result['available_boolean_features'])}\n")
                    f.write(f"  Temporal features available: {len(result['available_temporal_features'])}\n")
                    
                    f.write(f"  Correlations:\n")
                    sorted_correlations = sorted(
                        [(k, v, result['correlation_types'][k]) for k, v in result['correlations'].items() if not pd.isna(v)],
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    
                    for feature, corr, corr_type in sorted_correlations:
                        f.write(f"    {feature}: {corr:+.4f} ({corr_type})\n")
                    
                    # Identify strongest correlations
                    if sorted_correlations:
                        strongest = sorted_correlations[0]
                        f.write(f"  Strongest correlation: {strongest[0]} ({strongest[1]:+.4f})\n")
            
            print(f"    non_weather_correlation_analysis: Analysis completed for {total_files_processed} files")
            print(f"    non_weather_correlation_analysis: Results saved to {output_dir}")
            
            # Return success result
            result = {
                "success": True,
                "processed_files": total_files_processed,
                "output_path": output_dir,
                "summary_path": summary_path,
                "correlation_results": correlation_results,
                "total_boolean_features": len(BOOLEAN_FEATURES),
                "total_temporal_features": len(TEMPORAL_SINCOS_FEATURES),
                "analysis_type": "Non-Weather Features Correlation",
                "message": f"Successfully analyzed non-weather correlations for {total_files_processed} scaled training files"
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Non-weather correlation analysis failed: {str(e)}"
            print(f"    non_weather_correlation_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }

    def merge_data_files(self, csv_files):
        """
        Merge multiple training-ready data files into a single dataset for training.
        
        This method loads all files from data/output/preprocessed_training_ready,
        combines them into a unified dataset, and saves the result to 
        data/output/merged_training_ready. It adds source tracking columns and 
        creates detailed summary statistics.
        
        Parameters:
        -----------
        csv_files : list
            List of CSV file paths to merge (currently not used - method discovers files automatically)
            
        Returns:
        --------
        dict
            Results of the merge operation including success status and merged data info
        """
        try:
            print(f"    merge_data_files: Starting merge operation...")
            
            # Create output directory using the constant from const.py
            merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(merged_training_ready_dir, exist_ok=True)
            
            # Find all training-ready CSV files using glob pattern
            training_ready_pattern = os.path.join(self.project_root, TRAINING_READY_OUTPUT_FOLDER, "training_ready_*.csv")
            training_ready_files = glob.glob(training_ready_pattern)
            
            if not training_ready_files:
                error_msg = "No training-ready files found to merge"
                print(f"    merge_data_files: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    merge_data_files: Found {len(training_ready_files)} training-ready files")
            for file_path in training_ready_files:
                print(f"      - {os.path.basename(file_path)}")
            
            # Initialize storage for dataframes and file information
            all_dataframes = []
            file_info = []
            
            # Process each training-ready file
            for file_path in training_ready_files:
                try:
                    filename = os.path.basename(file_path)
                    print(f"    merge_data_files: Processing {filename}...")
                    
                    # Extract month information from filename using regex
                    # Expected format: training_ready_YYYY_MM.csv
                    month_match = re.search(r'training_ready_(\d{4})_(\d{2})\.csv$', filename)
                    
                    if not month_match:
                        print(f"    merge_data_files: Warning - Could not extract date from {filename}. Skipping.")
                        continue
                    
                    year = int(month_match.group(1))
                    month_number = int(month_match.group(2))
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"    merge_data_files: Warning - File {filename} is empty. Skipping.")
                        continue
                    
                    print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")
                    
                    # Add source tracking columns
                    df = df.copy()  # Avoid modifying the original dataframe
                    df['source_month'] = month_number
                    df['source_year'] = year
                    df['source_file'] = filename
                    
                    # Store the dataframe and file info
                    all_dataframes.append(df)
                    file_info.append({
                        'filename': filename,
                        'year': year,
                        'month': month_number,
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
                    
                    print(f"      Successfully processed {filename}")
                    
                except Exception as e:
                    print(f"    merge_data_files: Error processing {filename}: {str(e)}")
                    continue
            
            # Check if we have any dataframes to merge
            if not all_dataframes:
                error_msg = "No valid dataframes found to merge"
                print(f"    merge_data_files: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            # Merge all dataframes
            print(f"    merge_data_files: Merging {len(all_dataframes)} dataframes...")
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            print(f"    merge_data_files: Merged dataset shape: {merged_df.shape}")

            # Generate summary statistics
            month_distribution = merged_df['source_month'].value_counts().sort_index()
            year_distribution = merged_df['source_year'].value_counts().sort_index()

            # Remove source tracking columns before saving
            columns_to_remove = ['source_month', 'source_year', 'source_file']
            print(f"    merge_data_files: Dropping source tracking columns: {', '.join(columns_to_remove)}")
            merged_df = merged_df.drop(columns=columns_to_remove, errors='ignore')
            
            print(f"    merge_data_files: Removed source tracking columns. Final shape: {merged_df.shape}")

            # Generate output filename
            sorted_files = sorted(file_info, key=lambda x: (x['year'], x['month']))
            first_file = sorted_files[0]
            last_file = sorted_files[-1]

            # Format: merged_data_YYYY-MM_to_YYYY-MM.csv
            output_filename = f"merged_data_{first_file['year']}-{first_file['month']:02d}_to_{last_file['year']}-{last_file['month']:02d}.csv"
            output_path = os.path.join(merged_training_ready_dir, output_filename)
            
            # Save merged dataset
            merged_df.to_csv(output_path, index=False)
            print(f"    merge_data_files: Saved merged dataset to {output_path}")

            # Save summary information
            summary_filename = "merge_summary.txt"
            summary_path = os.path.join(merged_training_ready_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write("Merged Training Dataset Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Merge timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output file: {output_filename}\n")
                f.write(f"Total rows: {len(merged_df):,}\n")
                f.write(f"Total columns: {len(merged_df.columns)}\n")
                f.write(f"Files merged: {len(all_dataframes)}\n\n")
                
                # File details
                f.write("Files processed:\n")
                f.write("-" * 20 + "\n")
                for info in file_info:
                    f.write(f"{info['filename']}: {info['rows']:,} rows, {info['columns']} columns\n")
                
                # Month distribution
                f.write("\nMonth distribution:\n")
                f.write("-" * 20 + "\n")
                for month, count in month_distribution.items():
                    f.write(f"Month {month:02d}: {count:,} rows\n")
                
                # Year distribution
                f.write("\nYear distribution:\n")
                f.write("-" * 20 + "\n")
                for year, count in year_distribution.items():
                    f.write(f"Year {year}: {count:,} rows\n")
                
                # Column information
                f.write("\nColumns in merged dataset:\n")
                f.write("-" * 20 + "\n")
                for col in merged_df.columns:
                    f.write(f"{col}\n")
                
                # Data quality summary
                f.write("\nData Quality Summary:\n")
                f.write("-" * 20 + "\n")
                missing_values = merged_df.isnull().sum()
                if missing_values.sum() > 0:
                    f.write(f"Missing values per column:\n")
                    for col, missing_count in missing_values.items():
                        if missing_count > 0:
                            missing_pct = (missing_count / len(merged_df)) * 100
                            f.write(f"  {col}: {missing_count:,} ({missing_pct:.2f}%)\n")
                else:
                    f.write("No missing values found\n")
            
            print(f"    merge_data_files: Summary saved to {summary_filename}")
            
            # Return success result following the pattern of other methods
            result = {
                "success": True,
                "data": merged_df,  # Include the merged dataframe for potential chaining
                "output_path": output_path,
                "summary_path": summary_path,
                "processed_files": len(all_dataframes),
                "total_rows": len(merged_df),
                "total_columns": len(merged_df.columns),
                "files_merged": len(all_dataframes),
                "month_distribution": month_distribution.to_dict(),
                "file_details": file_info,
                "message": f"Successfully merged {len(all_dataframes)} files into {len(merged_df):,} rows"
            }
            
            print(f"    merge_data_files: Completed successfully - {len(all_dataframes)} files merged into {len(merged_df):,} rows")
            
            return result
            
        except Exception as e:
            error_msg = f"merge_data_files failed: {str(e)}"
            print(f"    merge_data_files: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }

    def split_dataset(self, csv_files=None, test_size=TEST_SIZE, random_state=42, stratify_column=None):
        """
        Split merged training datasets into train and test sets.
        
        This method finds all merged data files in data/output/merged_training_ready,
        splits each dataset into training and testing sets, and saves them as separate files.
        Automatically uses stratified splitting for classification problems based on the target feature.
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (currently not used - method discovers files automatically)
        test_size : float, optional
            Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
        stratify_column : str, optional
            Column name to use for stratified splitting. If None, uses automatic detection based on target feature.
            
        Returns:
        --------
        dict
            Results of the split operation including success status and split info
        """
        try:
            print(f"    split_dataset: Starting dataset splitting operation...")
            
            # Create/ensure output directory exists
            merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(merged_training_ready_dir, exist_ok=True)
            
            # Find all merged data files using glob pattern
            merged_data_pattern = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*.csv")
            merged_data_files = glob.glob(merged_data_pattern)
            
            # Filter out existing train/test files to avoid re-splitting them
            merged_data_files = [f for f in merged_data_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
            
            if not merged_data_files:
                error_msg = "No merged data files found to split"
                print(f"    split_dataset: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    split_dataset: Found {len(merged_data_files)} merged data files")
            for file_path in merged_data_files:
                print(f"      - {os.path.basename(file_path)}")
            
            # Initialize storage for processing results
            split_results = []
            total_train_rows = 0
            total_test_rows = 0
            
            # Process each merged data file
            for file_path in merged_data_files:
                try:
                    filename = os.path.basename(file_path)
                    print(f"    split_dataset: Processing {filename}...")
                    
                    # Read the merged dataset
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"    split_dataset: Warning - File {filename} is empty. Skipping.")
                        continue
                    
                    print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")
                    
                    # Determine target column and stratification logic
                    target_column = DEFAULT_TARGET_FEATURE
                    
                    # Check if the default target feature exists in the dataset
                    if target_column not in df.columns:
                        # Try to find an alternative target from valid options
                        target_options = VALID_TARGET_FEATURES
                        target_column = None
                        
                        for option in target_options:
                            if option in df.columns:
                                target_column = option
                                break
                        
                        if not target_column:
                            print(f"      Warning: No target column found in {filename}. Skipping.")
                            continue
                    
                    print(f"      Using target column: {target_column}")
                    
                    # Determine if this is a classification or regression problem
                    is_classification = True
                    if target_column in REGRESSION_PROBLEM:
                        is_classification = False
                        print(f"      Target '{target_column}' indicates a regression problem")
                    else:
                        print(f"      Target '{target_column}' indicates a classification problem")
                    
                    # Calculate class distribution for classification problems (BEFORE split)
                    class_distribution_before = None
                    class_distribution_train = None
                    class_distribution_test = None
                    
                    if is_classification:
                        # Calculate class distribution before split
                        value_counts = df[target_column].value_counts()
                        total_samples = len(df)
                        class_distribution_before = {}
                        
                        for class_value, count in value_counts.items():
                            percentage = (count / total_samples) * 100
                            class_distribution_before[class_value] = {
                                'count': count,
                                'percentage': percentage
                            }
                        
                        print(f"      Class distribution before split:")
                        for class_value, stats in class_distribution_before.items():
                            print(f"        Class {class_value}: {stats['count']} samples ({stats['percentage']:.2f}%)")
                    
                    # Prepare stratification
                    stratify = None
                    if stratify_column:
                        # Use explicitly provided stratify column
                        if stratify_column in df.columns:
                            stratify = df[stratify_column]
                            print(f"      Using explicit stratified split on column: {stratify_column}")
                        else:
                            print(f"      Warning: Specified stratify column '{stratify_column}' not found. Using automatic detection.")
                    
                    # Auto-detect stratification for classification problems
                    if stratify is None and is_classification:
                        # Use the target column for stratification in classification
                        stratify = df[target_column]
                        print(f"      Using stratified split on target column: {target_column}")
                    
                    # Perform the train-test split
                    train_df, test_df = train_test_split(
                        df,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=stratify
                    )
                    
                    # Calculate class distribution for classification problems (AFTER split)
                    if is_classification:
                        # Calculate class distribution for train set
                        train_value_counts = train_df[target_column].value_counts()
                        train_total = len(train_df)
                        class_distribution_train = {}
                        
                        for class_value, count in train_value_counts.items():
                            percentage = (count / train_total) * 100
                            class_distribution_train[class_value] = {
                                'count': count,
                                'percentage': percentage
                            }
                        
                        # Calculate class distribution for test set
                        test_value_counts = test_df[target_column].value_counts()
                        test_total = len(test_df)
                        class_distribution_test = {}
                        
                        for class_value, count in test_value_counts.items():
                            percentage = (count / test_total) * 100
                            class_distribution_test[class_value] = {
                                'count': count,
                                'percentage': percentage
                            }
                        
                        print(f"      Class distribution after split:")
                        print(f"        Train set:")
                        for class_value, stats in class_distribution_train.items():
                            print(f"          Class {class_value}: {stats['count']} samples ({stats['percentage']:.2f}%)")
                        print(f"        Test set:")
                        for class_value, stats in class_distribution_test.items():
                            print(f"          Class {class_value}: {stats['count']} samples ({stats['percentage']:.2f}%)")
                    
                    # Generate output filenames
                    base_filename = filename.replace('.csv', '')
                    train_filename = f"{base_filename}_train.csv"
                    test_filename = f"{base_filename}_test.csv"
                    
                    train_path = os.path.join(merged_training_ready_dir, train_filename)
                    test_path = os.path.join(merged_training_ready_dir, test_filename)
                    
                    # Save the train and test sets
                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)
                    
                    print(f"      Saved train set to: {train_filename}")
                    print(f"      Saved test set to: {test_filename}")
                    
                    # Store results for this file
                    result_data = {
                        'original_file': filename,
                        'train_file': train_filename,
                        'test_file': test_filename,
                        'target_column': target_column,
                        'is_classification': is_classification,
                        'stratified': stratify is not None,
                        'original_rows': len(df),
                        'train_rows': len(train_df),
                        'test_rows': len(test_df),
                        'test_size_actual': len(test_df) / len(df)
                    }
                    
                    # Add class distribution data for classification problems
                    if is_classification:
                        result_data['class_distribution_before'] = class_distribution_before
                        result_data['class_distribution_train'] = class_distribution_train
                        result_data['class_distribution_test'] = class_distribution_test
                    
                    split_results.append(result_data)
                    
                    total_train_rows += len(train_df)
                    total_test_rows += len(test_df)
                    
                    print(f"      Successfully processed {filename}")
                    
                except Exception as e:
                    print(f"    split_dataset: Error processing {filename}: {str(e)}")
                    continue
            
            # Check if we processed any files successfully
            if not split_results:
                error_msg = "No files were successfully split"
                print(f"    split_dataset: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    split_dataset: Split operation completed for {len(split_results)} files")
            print(f"    split_dataset: Total train rows: {total_train_rows:,}")
            print(f"    split_dataset: Total test rows: {total_test_rows:,}")
            
            # Save enhanced summary information
            summary_filename = "split_summary.txt"
            summary_path = os.path.join(merged_training_ready_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write("Dataset Split Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Split timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Test size: {test_size}\n")
                f.write(f"Random state: {random_state}\n")
                f.write(f"Default target feature: {DEFAULT_TARGET_FEATURE}\n")
                f.write(f"Files processed: {len(split_results)}\n")
                f.write(f"Total train rows: {total_train_rows:,}\n")
                f.write(f"Total test rows: {total_test_rows:,}\n\n")
                
                # File details with enhanced class distribution information
                f.write("Split details:\n")
                f.write("-" * 30 + "\n")
                for result in split_results:
                    f.write(f"Original: {result['original_file']}\n")
                    f.write(f"  Target: {result['target_column']}\n")
                    f.write(f"  Problem type: {'Classification' if result['is_classification'] else 'Regression'}\n")
                    f.write(f"  Stratified: {'Yes' if result['stratified'] else 'No'}\n")
                    f.write(f"  Train: {result['train_file']} ({result['train_rows']:,} rows)\n")
                    f.write(f"  Test: {result['test_file']} ({result['test_rows']:,} rows)\n")
                    f.write(f"  Actual test ratio: {result['test_size_actual']:.3f}\n")
                    
                    # Add class distribution information for classification problems
                    if result['is_classification'] and 'class_distribution_before' in result:
                        f.write(f"\n  Class Distribution Analysis:\n")
                        f.write(f"  {'-' * 25}\n")
                        
                        # Before split
                        f.write(f"  Before Split (Total: {result['original_rows']:,} samples):\n")
                        for class_value, stats in result['class_distribution_before'].items():
                            f.write(f"    Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)\n")
                        
                        # After split - Train set
                        f.write(f"\n  After Split - Train Set ({result['train_rows']:,} samples):\n")
                        for class_value, stats in result['class_distribution_train'].items():
                            f.write(f"    Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)\n")
                        
                        # After split - Test set
                        f.write(f"\n  After Split - Test Set ({result['test_rows']:,} samples):\n")
                        for class_value, stats in result['class_distribution_test'].items():
                            f.write(f"    Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)\n")
                    
                    f.write(f"\n")
            
            print(f"    split_dataset: Summary saved to {summary_filename}")
            
            # Return success result following the pattern of other methods
            result = {
                "success": True,
                "processed_files": len(split_results),
                "total_train_rows": total_train_rows,
                "total_test_rows": total_test_rows,
                "test_size": test_size,
                "split_details": split_results,
                "summary_path": summary_path,
                "message": f"Successfully split {len(split_results)} datasets into train/test sets"
            }
            
            print(f"    split_dataset: Completed successfully - {len(split_results)} datasets split")
            
            return result
            
        except Exception as e:
            error_msg = f"split_dataset failed: {str(e)}"
            print(f"    split_dataset: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }
        
    def scale_weather_features(self, csv_files=None):
        """
        Scale and normalize weather features using RobustScaler.
        
        This method finds all train/test split files, fits a RobustScaler on the weather 
        features from training data only, then transforms both train and test sets using 
        the training parameters. Saves scaled datasets to the scaled output folder.
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (currently not used - method discovers files automatically)
            
        Returns:
        --------
        dict
            Results of the scaling operation including success status and scaling info
        """
        try:
            print(f"    scale_weather_features: Starting weather feature scaling operation...")
            
            # Create output directory
            scaled_training_ready_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(scaled_training_ready_dir, exist_ok=True)
            
            # Find all train/test split files
            merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
            train_pattern = os.path.join(merged_training_ready_dir, "merged_data_*_train.csv")
            test_pattern = os.path.join(merged_training_ready_dir, "merged_data_*_test.csv")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if not train_files:
                error_msg = "No training files found to scale"
                print(f"    scale_weather_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            if not test_files:
                error_msg = "No test files found to scale"
                print(f"    scale_weather_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    scale_weather_features: Found {len(train_files)} train files and {len(test_files)} test files")
            
            # Initialize storage for processing results
            scaling_results = []
            total_train_rows = 0
            total_test_rows = 0
            
            # Process each pair of train/test files
            for train_file in train_files:
                try:
                    # Find corresponding test file
                    train_filename = os.path.basename(train_file)
                    test_filename = train_filename.replace('_train.csv', '_test.csv')
                    test_file = os.path.join(merged_training_ready_dir, test_filename)
                    
                    if not os.path.exists(test_file):
                        print(f"    scale_weather_features: Warning - No corresponding test file for {train_filename}. Skipping.")
                        continue
                    
                    print(f"    scale_weather_features: Processing {train_filename} and {test_filename}...")
                    
                    # Read the datasets
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                    
                    if train_df.empty or test_df.empty:
                        print(f"    scale_weather_features: Warning - Empty datasets found. Skipping.")
                        continue
                    
                    # Identify weather features that exist in the dataset
                    available_weather_features = [col for col in ALL_WEATHER_FEATURES if col in train_df.columns]
                    
                    if not available_weather_features:
                        print(f"    scale_weather_features: Warning - No weather features found in {train_filename}. Skipping.")
                        continue
                    
                    print(f"    scale_weather_features: Found {len(available_weather_features)} weather features to scale")
                    print(f"      Weather features: {available_weather_features}")
                    
                    # Create and fit scaler on training data only
                    scaler = RobustScaler()
                    
                    # Extract weather features for scaling
                    train_weather_features = train_df[available_weather_features]
                    test_weather_features = test_df[available_weather_features]
                    
                    # Fit scaler on training data only
                    scaler.fit(train_weather_features)
                    
                    # Transform both train and test sets using training parameters
                    train_weather_scaled = scaler.transform(train_weather_features)
                    test_weather_scaled = scaler.transform(test_weather_features)
                    
                    # Create scaled DataFrames
                    train_scaled_df = train_df.copy()
                    test_scaled_df = test_df.copy()
                    
                    # Replace weather feature columns with scaled versions
                    train_scaled_df[available_weather_features] = train_weather_scaled
                    test_scaled_df[available_weather_features] = test_weather_scaled
                    
                    # Generate output filenames
                    scaled_train_filename = train_filename.replace('.csv', '_scaled.csv')
                    scaled_test_filename = test_filename.replace('.csv', '_scaled.csv')
                    
                    scaled_train_path = os.path.join(scaled_training_ready_dir, scaled_train_filename)
                    scaled_test_path = os.path.join(scaled_training_ready_dir, scaled_test_filename)
                    
                    # Save scaled datasets
                    train_scaled_df.to_csv(scaled_train_path, index=False)
                    test_scaled_df.to_csv(scaled_test_path, index=False)
                    
                    # Save scaler for future use
                    scaler_filename = train_filename.replace('_train.csv', '_weather_scaler.pkl')
                    scaler_path = os.path.join(scaled_training_ready_dir, scaler_filename)
                    joblib.dump(scaler, scaler_path)
                    
                    print(f"    scale_weather_features: ✓ Saved scaled train data to {scaled_train_filename}")
                    print(f"    scale_weather_features: ✓ Saved scaled test data to {scaled_test_filename}")
                    print(f"    scale_weather_features: ✓ Saved scaler to {scaler_filename}")
                    
                    # Store scaling statistics
                    scaling_info = {
                        "original_train_file": train_filename,
                        "original_test_file": test_filename,
                        "scaled_train_file": scaled_train_filename,
                        "scaled_test_file": scaled_test_filename,
                        "scaler_file": scaler_filename,
                        "weather_features_scaled": available_weather_features,
                        "train_rows": len(train_scaled_df),
                        "test_rows": len(test_scaled_df),
                        "features_count": len(available_weather_features),
                        "scaler_stats": {
                            "center_": scaler.center_.tolist() if hasattr(scaler, 'center_') else [],
                            "scale_": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else []
                        }
                    }
                    
                    scaling_results.append(scaling_info)
                    total_train_rows += len(train_scaled_df)
                    total_test_rows += len(test_scaled_df)
                    
                except Exception as e:
                    error_msg = f"Failed to process {train_filename}: {str(e)}"
                    print(f"    scale_weather_features: ✗ {error_msg}")
                    scaling_results.append({
                        "error": error_msg,
                        "file": train_filename
                    })
                    continue
            
            # Save scaling summary
            summary_filename = "scaling_summary.txt"
            summary_path = os.path.join(scaled_training_ready_dir, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Weather Feature Scaling Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Scaling timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Scaler type: RobustScaler\n")
                f.write(f"Total file pairs processed: {len([r for r in scaling_results if 'error' not in r])}\n")
                f.write(f"Total train rows: {total_train_rows:,}\n")
                f.write(f"Total test rows: {total_test_rows:,}\n\n")
                
                if scaling_results:
                    f.write("Weather features scaled:\n")
                    f.write("-" * 25 + "\n")
                    for feature in ALL_WEATHER_FEATURES:
                        feature_found = any(feature in result.get('weather_features_scaled', []) 
                                        for result in scaling_results if 'error' not in result)
                        status = "✓ Scaled" if feature_found else "✗ Not found"
                        f.write(f"  {feature}: {status}\n")
                    
                    f.write("\nFile processing details:\n")
                    f.write("-" * 25 + "\n")
                    for result in scaling_results:
                        if 'error' in result:
                            f.write(f"  ✗ {result['file']}: {result['error']}\n")
                        else:
                            f.write(f"  ✓ {result['original_train_file']} -> {result['scaled_train_file']}\n")
                            f.write(f"    Features: {len(result['weather_features_scaled'])}\n")
                            f.write(f"    Train rows: {result['train_rows']:,}\n")
                            f.write(f"    Test rows: {result['test_rows']:,}\n\n")
            
            print(f"    scale_weather_features: Summary saved to {summary_filename}")
            
            # Return success result
            successful_files = len([r for r in scaling_results if 'error' not in r])
            result = {
                "success": True,
                "processed_files": successful_files,
                "total_train_rows": total_train_rows,
                "total_test_rows": total_test_rows,
                "scaling_results": scaling_results,
                "weather_features_available": list(set([
                    feature for result in scaling_results 
                    for feature in result.get('weather_features_scaled', [])
                    if 'error' not in result
                ])),
                "output_directory": scaled_training_ready_dir,
                "summary_path": summary_path,
                "message": f"Successfully scaled weather features for {successful_files} file pairs"
            }
            
            print(f"    scale_weather_features: Completed successfully - {successful_files} file pairs processed")
            print(f"    scale_weather_features: Total rows - Train: {total_train_rows:,}, Test: {total_test_rows:,}")
            
            return result
            
        except Exception as e:
            error_msg = f"scale_weather_features failed: {str(e)}"
            print(f"    scale_weather_features: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }
        
    def train_decision_tree(self):
        """
        Train Decision Tree classifiers on scaled training data using RandomizedSearchCV.
        
        This method trains on the scaled data from the previous pipeline stages.
        Uses the DEFAULT_TARGET_FEATURE and checks if it's a classification problem.
        Saves results and feature importance to data/output/decision_tree folder.
        Now generates a single consolidated JSON file instead of separate files.
        
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics.
        """
        try:
            print(f"    train_decision_tree: Starting Decision Tree training...")
            
            # Check if target feature is a classification problem
            if DEFAULT_TARGET_FEATURE not in CLASSIFICATION_PROBLEM:
                error_msg = f"Target feature '{DEFAULT_TARGET_FEATURE}' is not a classification problem. Expected one of: {CLASSIFICATION_PROBLEM}"
                print(f"    train_decision_tree: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"    train_decision_tree: Using target feature '{DEFAULT_TARGET_FEATURE}' for classification")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, "data/output/decision_tree")
            os.makedirs(output_dir, exist_ok=True)
            
            # Find train and test files
            scaled_data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train_scaled.csv")
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test_scaled.csv")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if not train_files:
                error_msg = f"No training files found matching pattern: {train_pattern}"
                print(f"    train_decision_tree: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            if not test_files:
                error_msg = f"No test files found matching pattern: {test_pattern}"
                print(f"    train_decision_tree: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"    train_decision_tree: Found {len(train_files)} training files and {len(test_files)} test files")
            
            # Initialize consolidated results structure
            consolidated_results = {
                "training_overview": {
                    "training_completed": datetime.now().isoformat(),
                    "target_feature": DEFAULT_TARGET_FEATURE,
                    "score_metric": SCORE_METRIC,
                    "total_files_processed": 0,
                    "total_train_samples": 0,
                    "total_test_samples": 0,
                    "successful_files": 0,
                    "failed_files": 0
                },
                "hyperparameter_search": {
                    "method": "RandomizedSearchCV",
                    "iterations": RANDOM_SEARCH_ITERATIONS,
                    "cv_folds": RANDOM_SEARCH_CV_FOLDS,
                    "param_distributions": format_param_distributions_for_json(DECISION_TREE_PARAM_DISTRIBUTIONS)
                },
                "file_results": [],
                "aggregate_metrics": {
                    "average_accuracy": 0.0,
                    "average_f1": 0.0,
                    "average_balanced_accuracy": 0.0,
                    "average_roc_auc": 0.0,
                    "best_performing_file": None,
                    "worst_performing_file": None
                }
            }
            
            # Process all train/test file pairs
            total_train_samples = 0
            total_test_samples = 0
            successful_files = 0
            failed_files = 0
            
            # Metrics for aggregation
            all_accuracies = []
            all_f1_scores = []
            all_balanced_accuracies = []
            all_roc_aucs = []
            best_f1 = -1
            worst_f1 = 2
            best_file = None
            worst_file = None
            
            for train_file in train_files:
                try:
                    # Find corresponding test file
                    train_filename = os.path.basename(train_file)
                    test_filename = train_filename.replace('_train_scaled.csv', '_test_scaled.csv')
                    test_file = os.path.join(scaled_data_dir, test_filename)
                    
                    if not os.path.exists(test_file):
                        print(f"    train_decision_tree: Warning - No corresponding test file for {train_filename}. Skipping.")
                        failed_files += 1
                        continue
                    
                    print(f"    train_decision_tree: Processing {train_filename} and {test_filename}")
                    
                    # Load data
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                    
                    # Check if target column exists
                    if DEFAULT_TARGET_FEATURE not in train_df.columns:
                        print(f"    train_decision_tree: Warning - Target feature '{DEFAULT_TARGET_FEATURE}' not found in {train_filename}. Skipping.")
                        failed_files += 1
                        continue
                    
                    # Prepare features and target
                    X_train = train_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    y_train = train_df[DEFAULT_TARGET_FEATURE]
                    X_test = test_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    y_test = test_df[DEFAULT_TARGET_FEATURE]
                    
                    # Remove any non-numeric columns
                    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
                    X_train = X_train[numeric_cols]
                    X_test = X_test[numeric_cols]
                    
                    print(f"    train_decision_tree: Using {len(numeric_cols)} numeric features")
                    print(f"    train_decision_tree: Train samples: {len(X_train)}, Test samples: {len(X_test)}")
                    
                    # Create sample weights based on delay magnitude if weight column exists
                    sample_weights = None
                    if WEIGHT_DELAY_COLUMN and WEIGHT_DELAY_COLUMN in train_df.columns:
                        print(f"    train_decision_tree: Creating sample weights based on '{WEIGHT_DELAY_COLUMN}'")
                        sample_weights = np.ones(len(y_train))
                        
                        # Get delay values for each training sample
                        delays = train_df[WEIGHT_DELAY_COLUMN].values
                        
                        # Apply weights - higher delays get higher weights
                        delayed_idx = (delays > TRAIN_DELAY_MINUTES)
                        if np.any(delayed_idx):
                            # Normalize delay values by mean positive delay
                            mean_delay = delays[delayed_idx].mean()
                            # Use configured maximum weight
                            sample_weights[delayed_idx] = np.minimum(
                                MAX_SAMPLE_WEIGHT_CLASSIFICATION, 
                                1 + delays[delayed_idx]/mean_delay
                            )
                        
                        print(f"    train_decision_tree: Sample weights range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                    
                    # Set up cross-validation strategy
                    cv_strategy = StratifiedKFold(
                        n_splits=RANDOM_SEARCH_CV_FOLDS,
                        shuffle=True,
                        random_state=42
                    )
                    
                    # Initialize Decision Tree and RandomizedSearchCV
                    dt = DecisionTreeClassifier(random_state=42)
                    
                    print(f"    train_decision_tree: Starting RandomizedSearchCV with {RANDOM_SEARCH_ITERATIONS} iterations and {RANDOM_SEARCH_CV_FOLDS}-fold CV...")
                    
                    random_search = RandomizedSearchCV(
                        dt, 
                        param_distributions=DECISION_TREE_PARAM_DISTRIBUTIONS,
                        n_iter=RANDOM_SEARCH_ITERATIONS,
                        cv=cv_strategy,
                        scoring=SCORE_METRIC,
                        random_state=42,
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    # Fit RandomizedSearchCV with sample weights if available
                    if sample_weights is not None:
                        random_search.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        random_search.fit(X_train, y_train)
                    
                    best_params = random_search.best_params_
                    best_cv_score = random_search.best_score_
                    
                    print(f"    train_decision_tree: Best CV Score ({SCORE_METRIC}): {best_cv_score:.4f}")
                    print(f"    train_decision_tree: Best Parameters: {best_params}")
                    
                    # Train final model with best parameters
                    best_dt = DecisionTreeClassifier(**best_params, random_state=42)
                    
                    if sample_weights is not None:
                        best_dt.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        best_dt.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = best_dt.predict(X_test)
                    y_pred_proba = best_dt.predict_proba(X_test)
                    
                    # Define file_identifier
                    file_identifier = train_filename.replace('merged_data_', '').replace('_train_scaled.csv', '')
                    
                    # Calculate comprehensive metrics
                    metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Add CV score to metrics
                    metrics['best_cv_score'] = best_cv_score
                    metrics['optimized_metric_name'] = SCORE_METRIC
                    
                    # Generate and save confusion matrix
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    conf_matrix_result = self._save_confusion_matrix(
                        conf_matrix, y_test, y_pred, file_identifier, output_dir
                    )
                    
                    # Get feature importance
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': best_dt.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    print(f"    train_decision_tree: Test {SCORE_METRIC}: {metrics.get(SCORE_METRIC, 'N/A'):.4f}")
                    print(f"    train_decision_tree: Test Accuracy: {metrics['accuracy']:.4f}")
                    print(f"    train_decision_tree: Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                    
                    # Save model
                    model_filename = f"decision_tree_model_{file_identifier}.joblib"
                    model_path = os.path.join(output_dir, model_filename)
                    joblib.dump(best_dt, model_path)
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_{file_identifier}.csv"
                    importance_path = os.path.join(output_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    
                    print(f"    train_decision_tree: Saved model, feature importance, and confusion matrix for {file_identifier}")
                    
                    # Prepare individual file result for consolidated JSON
                    file_result = {
                        "file_identifier": file_identifier,
                        "files": {
                            "train_file": train_filename,
                            "test_file": test_filename,
                            "model_file": model_filename,
                            "feature_importance_file": importance_filename,
                            "confusion_matrix_plot": conf_matrix_result.get('plot_file', '')
                        },
                        "data_info": {
                            "train_samples": len(X_train),
                            "test_samples": len(X_test),
                            "features_used": len(X_train.columns),
                            "feature_names": X_train.columns.tolist(),
                            "sample_weights_used": sample_weights is not None
                        },
                        "hyperparameters": {
                            "best_parameters": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                            for k, v in best_params.items()},
                            "best_cv_score": float(best_cv_score)
                        },
                        "notation": {
                            "TP": "True Positive",
                            "TN": "True Negative",
                            "FP": "False Positive",
                            "FN": "False Negative",
                            "Accuracy": "(TP + TN) / (TP + FP + FN + TN) -> Overall proportion of correct predictions",
                            "Balanced Accuracy": "(TP/(TP+FN) + TN/(TN+FP)) / 2 -> Average of per-class accuracies",
                            "Recall": "TP / (TP + FN) -> Known as True Positive Rate",
                            "Specificity": "TN / (FP + TN) -> Known as True Negative Rate",
                            "Precision": "TP / (TP + FP)",
                            "F1_Score": "2 * (Precision * Recall) / (Precision + Recall)"
                        },
                        "performance_metrics": {
                            k: float(v) if isinstance(v, (int, float, np.number)) else v 
                            for k, v in metrics.items()
                        },
                        "confusion_matrix": {
                            "matrix": conf_matrix.tolist(),
                            "labels": ["Not Delayed", "Delayed"],
                            "true_negatives": int(conf_matrix[0, 0]),
                            "false_positives": int(conf_matrix[0, 1]),
                            "false_negatives": int(conf_matrix[1, 0]),
                            "true_positives": int(conf_matrix[1, 1])
                        },
                        "feature_importance_top_10": [
                            {
                                "feature": row['feature'],
                                "importance": float(row['importance'])
                            }
                            for _, row in feature_importance.head(10).iterrows()
                        ]
                    }
                    
                    # Add to consolidated results
                    consolidated_results["file_results"].append(file_result)
                    
                    # Update aggregates
                    total_train_samples += len(X_train)
                    total_test_samples += len(X_test)
                    successful_files += 1
                    
                    # Track metrics for averaging
                    current_accuracy = metrics['accuracy']
                    current_f1 = metrics.get('f1', metrics.get(SCORE_METRIC, 0))
                    current_balanced_accuracy = metrics['balanced_accuracy']
                    current_roc_auc = metrics.get('roc_auc', 0)
                    
                    all_accuracies.append(current_accuracy)
                    all_f1_scores.append(current_f1)
                    all_balanced_accuracies.append(current_balanced_accuracy)
                    all_roc_aucs.append(current_roc_auc)
                    
                    # Track best/worst performing files
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_file = file_identifier
                    if current_f1 < worst_f1:
                        worst_f1 = current_f1
                        worst_file = file_identifier
                    
                    print(f"    train_decision_tree: Completed processing {file_identifier}")
                    
                except Exception as e:
                    print(f"    train_decision_tree: Error processing {train_file}: {str(e)}")
                    failed_files += 1
                    continue
            
            # Finalize consolidated results
            if successful_files > 0:
                consolidated_results["training_overview"].update({
                    "total_files_processed": successful_files + failed_files,
                    "total_train_samples": total_train_samples,
                    "total_test_samples": total_test_samples,
                    "successful_files": successful_files,
                    "failed_files": failed_files
                })
                
                consolidated_results["aggregate_metrics"].update({
                    "average_accuracy": float(np.mean(all_accuracies)),
                    "average_f1": float(np.mean(all_f1_scores)),
                    "average_balanced_accuracy": float(np.mean(all_balanced_accuracies)),
                    "average_roc_auc": float(np.mean(all_roc_aucs)),
                    "best_performing_file": {
                        "file_identifier": best_file,
                        "f1_score": float(best_f1)
                    } if best_file else None,
                    "worst_performing_file": {
                        "file_identifier": worst_file,
                        "f1_score": float(worst_f1)
                    } if worst_file else None
                })
                
                # Save consolidated results to single JSON file
                consolidated_filename = "decision_tree_training_results.json"
                consolidated_path = os.path.join(output_dir, consolidated_filename)
                
                with open(consolidated_path, 'w') as f:
                    json.dump(consolidated_results, f, indent=2)
                
                print(f"    train_decision_tree: Processed {successful_files} file pairs")
                print(f"    train_decision_tree: Generated models, feature importance, and confusion matrices")
                print(f"    train_decision_tree: Consolidated results saved to: {consolidated_filename}")
                print(f"    train_decision_tree: Results saved to: {output_dir}")
                print(f"    train_decision_tree: Training completed successfully!")
                
                return {
                    "success": True,
                    "files_processed": successful_files,
                    "total_train_samples": total_train_samples,
                    "total_test_samples": total_test_samples,
                    "output_directory": output_dir,
                    "target_feature": DEFAULT_TARGET_FEATURE,
                    "score_metric": SCORE_METRIC,
                    "consolidated_results_file": consolidated_filename
                }
            else:
                error_msg = "No files were successfully processed"
                print(f"    train_decision_tree: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"Error in Decision Tree training: {str(e)}"
            print(f"    train_decision_tree: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def threshold_optimization_decision_tree(self):
        """
        Optimize decision tree classification thresholds using ROC analysis.
        
        This method loads trained decision tree models from the previous stage,
        analyzes ROC curves to find optimal thresholds, and retrains models
        with optimized thresholds. Results are saved to decision_tree_threshold_optimized folder.
        
        Returns:
        --------
        dict
            A summary of the threshold optimization results including optimal thresholds and performance metrics.
        """
        try:
            print(f"    threshold_optimization_decision_tree: Starting threshold optimization...")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, DECISION_TREE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if decision tree results exist from previous stage
            dt_output_dir = os.path.join(self.project_root, "data/output/decision_tree")
            dt_results_file = os.path.join(dt_output_dir, "decision_tree_training_results.json")
            
            if not os.path.exists(dt_results_file):
                error_msg = "Decision tree training results not found. Run train_decision_tree stage first."
                print(f"    threshold_optimization_decision_tree: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Load decision tree training results
            with open(dt_results_file, 'r') as f:
                dt_results = json.load(f)
            
            print(f"    threshold_optimization_decision_tree: Found {len(dt_results.get('file_results', []))} trained models")
            
            # Find train and test files
            scaled_data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train_scaled.csv")
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test_scaled.csv")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if not train_files or not test_files:
                error_msg = f"Training/test files not found in {scaled_data_dir}"
                print(f"    threshold_optimization_decision_tree: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Initialize results structure
            optimization_results = {
                "optimization_overview": {
                    "optimization_completed": datetime.now().isoformat(),
                    "target_feature": DEFAULT_TARGET_FEATURE,
                    "optimization_metric": THRESHOLD_OPTIMIZATION_CONFIG["optimization_metric"],
                    "total_models_optimized": 0,
                    "successful_optimizations": 0,
                    "failed_optimizations": 0
                },
                "threshold_optimization_config": THRESHOLD_OPTIMIZATION_CONFIG,
                "file_results": [],
                "aggregate_metrics": {
                    "average_optimal_threshold": 0.0,
                    "average_optimized_f1": 0.0,
                    "average_optimized_precision": 0.0,
                    "average_optimized_recall": 0.0,
                    "best_performing_file": None
                }
            }
            
            successful_optimizations = 0
            failed_optimizations = 0
            all_optimal_thresholds = []
            all_optimized_f1_scores = []
            all_optimized_precisions = []
            all_optimized_recalls = []
            best_f1 = -1
            best_file = None
            
            # Process each train/test file pair
            for train_file in train_files:
                try:
                    # Find corresponding test file and model
                    train_filename = os.path.basename(train_file)
                    test_filename = train_filename.replace('_train_scaled.csv', '_test_scaled.csv')
                    test_file = os.path.join(scaled_data_dir, test_filename)
                    
                    if not os.path.exists(test_file):
                        print(f"    threshold_optimization_decision_tree: Warning - No test file for {train_filename}. Skipping.")
                        continue
                    
                    # Find corresponding trained model
                    file_identifier = train_filename.replace('merged_data_', '').replace('_train_scaled.csv', '')
                    model_file = os.path.join(dt_output_dir, f"decision_tree_model_{file_identifier}.joblib")
                    
                    if not os.path.exists(model_file):
                        print(f"    threshold_optimization_decision_tree: Warning - No model file for {file_identifier}. Skipping.")
                        continue
                    
                    print(f"    threshold_optimization_decision_tree: Optimizing threshold for {file_identifier}")
                    
                    # Load the trained model
                    model = joblib.load(model_file)
                    
                    # Load test data
                    test_df = pd.read_csv(test_file)
                    
                    if DEFAULT_TARGET_FEATURE not in test_df.columns:
                        print(f"    threshold_optimization_decision_tree: Target feature '{DEFAULT_TARGET_FEATURE}' not found in {test_filename}")
                        failed_optimizations += 1
                        continue
                    
                    # Prepare test features and target
                    y_test = test_df[DEFAULT_TARGET_FEATURE]
                    X_test = test_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    
                    # Extract feature importance from the trained model
                    feature_importance = pd.DataFrame({
                        'feature': X_test.columns,  # Use X_test.columns to get the feature names
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    # Save feature importance for threshold optimized model
                    importance_filename = f"threshold_optimized_feature_importance_{file_identifier}.csv"
                    importance_path = os.path.join(output_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)

                    print(f"    threshold_optimization_decision_tree: Saved feature importance for {file_identifier}")

                    # Get probability predictions for positive class
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Calculate Precision-Recall curve
                    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
                    pr_auc = average_precision_score(y_test, y_proba)
                    
                    # Find optimal threshold based on different metrics
                    optimal_thresholds = self._find_optimal_thresholds(
                        y_test, y_proba, fpr, tpr, thresholds, precision, recall, pr_thresholds
                    )
                    
                    # Use the metric specified in config
                    optimization_metric = THRESHOLD_OPTIMIZATION_CONFIG["optimization_metric"]
                    optimal_threshold = optimal_thresholds[optimization_metric]

                    # Make predictions with optimal threshold
                    y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
                    
                    # Calculate comprehensive metrics using the private method
                    metrics = self._calculate_classification_metrics(y_test, y_pred_optimized, y_proba)

                    # Generate and save confusion matrix
                    conf_matrix = confusion_matrix(y_test, y_pred_optimized)
                    conf_matrix_result = self._save_confusion_matrix(
                        conf_matrix, y_test, y_pred_optimized, file_identifier, output_dir
                    )
                    
                    # Convert numpy types to JSON-serializable types
                    def convert_numpy_types(obj):
                        """Recursively convert numpy types to native Python types."""
                        if isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(v) for v in obj]
                        elif isinstance(obj, (np.integer, np.int32, np.int64)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif obj is None:
                            return None
                        else:
                            return obj
                    
                    # Apply conversion to metrics
                    metrics = convert_numpy_types(metrics)
                    
                    # Calculate individual metrics for backward compatibility
                    optimized_precision = precision_score(y_test, y_pred_optimized)
                    optimized_recall = recall_score(y_test, y_pred_optimized)
                    optimized_f1 = f1_score(y_test, y_pred_optimized)
                    optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
                    
                    # Create and save plots
                    if THRESHOLD_OPTIMIZATION_CONFIG["plot_roc_curve"]:
                        self._plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold, 
                                        file_identifier, output_dir)
                    
                    if THRESHOLD_OPTIMIZATION_CONFIG["plot_precision_recall"]:
                        self._plot_precision_recall_curve(precision, recall, pr_auc, optimal_threshold,
                                                        file_identifier, output_dir)
                    
                    # Plot threshold analysis
                    self._plot_threshold_analysis(thresholds, fpr, tpr, precision[:-1], recall[:-1],
                                                file_identifier, output_dir)
                    
                    # Save optimized model if configured
                    if THRESHOLD_OPTIMIZATION_CONFIG["save_optimized_models"]:
                        optimized_model_data = {
                            "original_model": model,
                            "optimal_threshold": optimal_threshold,
                            "optimization_metric": optimization_metric,
                            "roc_auc": roc_auc,
                            "pr_auc": pr_auc
                        }
                        optimized_model_file = os.path.join(output_dir, f"optimized_decision_tree_{file_identifier}.joblib")
                        joblib.dump(optimized_model_data, optimized_model_file)
                    
                    # Store results with comprehensive metrics
                    file_result = {
                        "file_identifier": file_identifier,
                        "optimization_successful": True,
                        "optimal_thresholds": optimal_thresholds,
                        "used_threshold": optimal_threshold,
                        "used_optimization_metric": optimization_metric,
                        "performance_metrics": {
                            "roc_auc": float(roc_auc),
                            "pr_auc": float(pr_auc),
                            "optimized_precision": float(optimized_precision),
                            "optimized_recall": float(optimized_recall),
                            "optimized_f1": float(optimized_f1),
                            "optimized_accuracy": float(optimized_accuracy)
                        },
                        "comprehensive_metrics": metrics,  # Add the comprehensive metrics here
                        "test_samples": len(y_test)
                    }
                    
                    optimization_results["file_results"].append(file_result)
                    
                    # Update tracking variables
                    all_optimal_thresholds.append(optimal_threshold)
                    all_optimized_f1_scores.append(optimized_f1)
                    all_optimized_precisions.append(optimized_precision)
                    all_optimized_recalls.append(optimized_recall)
                    
                    if optimized_f1 > best_f1:
                        best_f1 = optimized_f1
                        best_file = file_identifier
                    
                    successful_optimizations += 1
                    print(f"    threshold_optimization_decision_tree: ✓ {file_identifier} - Optimal threshold: {optimal_threshold:.3f}, F1: {optimized_f1:.3f}")
                    
                except Exception as e:
                    print(f"    threshold_optimization_decision_tree: Error processing {train_file}: {str(e)}")
                    failed_optimizations += 1
                    
                    # Add failed result
                    file_identifier = os.path.basename(train_file).replace('_train_scaled.csv', '')
                    optimization_results["file_results"].append({
                        "file_identifier": file_identifier,
                        "optimization_successful": False,
                        "error": str(e)
                    })
                    continue
            
            # Finalize results
            if successful_optimizations > 0:
                optimization_results["optimization_overview"].update({
                    "total_models_optimized": successful_optimizations + failed_optimizations,
                    "successful_optimizations": successful_optimizations,
                    "failed_optimizations": failed_optimizations
                })
                
                optimization_results["aggregate_metrics"].update({
                    "average_optimal_threshold": float(np.mean(all_optimal_thresholds)),
                    "average_optimized_f1": float(np.mean(all_optimized_f1_scores)),
                    "average_optimized_precision": float(np.mean(all_optimized_precisions)),
                    "average_optimized_recall": float(np.mean(all_optimized_recalls)),
                    "best_performing_file": {
                        "file_identifier": best_file,
                        "f1_score": float(best_f1)
                    } if best_file else None
                })
                
                # Save consolidated results
                results_file = os.path.join(output_dir, "threshold_optimization_results.json")
                with open(results_file, 'w') as f:
                    json.dump(optimization_results, f, indent=2)
                
                # Create summary plot showing all optimal thresholds
                self._plot_threshold_summary(all_optimal_thresholds, all_optimized_f1_scores, output_dir)
                
                print(f"    threshold_optimization_decision_tree: Completed successfully!")
                print(f"    threshold_optimization_decision_tree: Optimized {successful_optimizations} models")
                print(f"    threshold_optimization_decision_tree: Average optimal threshold: {np.mean(all_optimal_thresholds):.3f}")
                print(f"    threshold_optimization_decision_tree: Average optimized F1: {np.mean(all_optimized_f1_scores):.3f}")
                print(f"    threshold_optimization_decision_tree: Results saved to: {output_dir}")
                
                return {
                    "success": True,
                    "models_optimized": successful_optimizations,
                    "average_optimal_threshold": float(np.mean(all_optimal_thresholds)),
                    "average_optimized_f1": float(np.mean(all_optimized_f1_scores)),
                    "output_directory": output_dir,
                    "results_file": results_file,
                    "target_feature": DEFAULT_TARGET_FEATURE
                }
            else:
                error_msg = "No models were successfully optimized"
                print(f"    threshold_optimization_decision_tree: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"threshold_optimization_decision_tree failed: {str(e)}"
            print(f"    threshold_optimization_decision_tree: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def generate_borderline_smote_data(self):
        """
        Generate synthetic data using BorderlineSMOTE on training datasets.
        
        This method applies BorderlineSMOTE to generate synthetic samples for the minority class
        in the training datasets. It processes all merged_data_*_train_scaled.csv files and
        creates augmented versions with synthetic samples.
        
        BorderlineSMOTE focuses on borderline samples (samples that are close to the decision boundary)
        to generate more realistic synthetic samples compared to regular SMOTE.
        
        Returns:
        --------
        dict
            A summary of the BorderlineSMOTE data generation results including files processed and sample counts.
        """
        try:
            print(f"    generate_borderline_smote_data: Starting BorderlineSMOTE synthetic data generation...")
            
            # Import required libraries
            from imblearn.over_sampling import BorderlineSMOTE
            from collections import Counter
            
            # Create output directory
            output_dir = os.path.join(self.project_root, BORDERLINE_SMOTE_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            
            # Find training files
            scaled_data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train_scaled.csv")
            train_files = glob.glob(train_pattern)
            
            if not train_files:
                error_msg = f"No training files found matching pattern: {train_pattern}"
                print(f"    generate_borderline_smote_data: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Initialize results structure
            smote_results = {
                "generation_overview": {
                    "generation_completed": datetime.now().isoformat(),
                    "target_feature": DEFAULT_TARGET_FEATURE,
                    "total_files_processed": 0,
                    "successful_generations": 0,
                    "failed_generations": 0,
                    "borderline_smote_config": BORDERLINE_SMOTE_CONFIG
                },
                "file_results": [],
                "aggregate_metrics": {
                    "total_original_samples": 0,
                    "total_synthetic_samples": 0,
                    "total_final_samples": 0,
                    "average_augmentation_ratio": 0.0,
                    "overall_class_distributions": {
                        "before_smote": {
                            "total_samples": 0,
                            "distribution": {}
                        },
                        "after_smote": {
                            "total_samples": 0,
                            "distribution": {}
                        }
                    },
                    "best_performing_file": None
                }
            }
            
            successful_generations = 0
            failed_generations = 0
            total_original_samples = 0
            total_synthetic_samples = 0
            total_final_samples = 0
            augmentation_ratios = []
            overall_original_distribution = Counter()
            overall_final_distribution = Counter()
            
            # Process each training file
            for train_file in train_files:
                try:
                    train_filename = os.path.basename(train_file)
                    file_identifier = train_filename.replace('merged_data_', '').replace('_train_scaled.csv', '')
                    
                    print(f"    generate_borderline_smote_data: Processing {file_identifier}")
                    
                    # Load training data
                    train_df = pd.read_csv(train_file)
                    
                    if DEFAULT_TARGET_FEATURE not in train_df.columns:
                        print(f"    generate_borderline_smote_data: Target feature '{DEFAULT_TARGET_FEATURE}' not found in {train_filename}")
                        failed_generations += 1
                        continue
                    
                    # Prepare features and target
                    y_train = train_df[DEFAULT_TARGET_FEATURE]
                    X_train = train_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    
                    # Check class distribution before SMOTE
                    original_distribution = Counter(y_train)
                    original_sample_count = len(y_train)
                    
                    print(f"    generate_borderline_smote_data: Original distribution for {file_identifier}: {dict(original_distribution)}")
                    
                    # Check if minority class has enough samples for BorderlineSMOTE
                    minority_class_count = min(original_distribution.values())
                    if minority_class_count < BORDERLINE_SMOTE_CONFIG["k_neighbors"] + 1:
                        print(f"    generate_borderline_smote_data: Warning - Not enough minority samples ({minority_class_count}) for BorderlineSMOTE (requires >{BORDERLINE_SMOTE_CONFIG['k_neighbors']}). Skipping {file_identifier}")
                        failed_generations += 1
                        continue
                    
                    # Initialize BorderlineSMOTE
                    borderline_smote = BorderlineSMOTE(
                        k_neighbors=BORDERLINE_SMOTE_CONFIG["k_neighbors"],
                        m_neighbors=BORDERLINE_SMOTE_CONFIG["m_neighbors"],
                        kind=BORDERLINE_SMOTE_CONFIG["kind"],
                        random_state=BORDERLINE_SMOTE_CONFIG["random_state"],
                        sampling_strategy=BORDERLINE_SMOTE_CONFIG["sampling_strategy"]
                    )
                    
                    # Apply BorderlineSMOTE
                    X_resampled, y_resampled = borderline_smote.fit_resample(X_train, y_train)
                    
                    # Calculate metrics
                    final_distribution = Counter(y_resampled)
                    final_sample_count = len(y_resampled)
                    synthetic_samples_added = final_sample_count - original_sample_count
                    augmentation_ratio = final_sample_count / original_sample_count
                    
                    print(f"    generate_borderline_smote_data: Final distribution for {file_identifier}: {dict(final_distribution)}")
                    print(f"    generate_borderline_smote_data: Added {synthetic_samples_added:,} synthetic samples (augmentation ratio: {augmentation_ratio:.2f}x)")
                    
                    # Create augmented dataframe
                    augmented_df = pd.DataFrame(X_resampled, columns=X_train.columns)
                    augmented_df[DEFAULT_TARGET_FEATURE] = y_resampled
                    
                    # Save augmented training data
                    output_filename = f"borderline_smote_data_{file_identifier}_train_augmented.csv"
                    output_path = os.path.join(output_dir, output_filename)
                    augmented_df.to_csv(output_path, index=False)
                    
                    # Calculate detailed class distributions with percentages
                    original_distribution_detailed = {}
                    final_distribution_detailed = {}
                    
                    # Original distribution with counts and percentages
                    for class_label, count in original_distribution.items():
                        percentage = (count / original_sample_count) * 100
                        original_distribution_detailed[str(class_label)] = {
                            "count": count,
                            "percentage": round(percentage, 2)
                        }
                    
                    # Final distribution with counts and percentages
                    for class_label, count in final_distribution.items():
                        percentage = (count / final_sample_count) * 100
                        final_distribution_detailed[str(class_label)] = {
                            "count": count,
                            "percentage": round(percentage, 2)
                        }
                    
                    # Record results
                    file_result = {
                        "file_identifier": file_identifier,
                        "original_file": train_filename,
                        "augmented_file": output_filename,
                        "original_samples": original_sample_count,
                        "synthetic_samples_added": synthetic_samples_added,
                        "final_samples": final_sample_count,
                        "augmentation_ratio": float(augmentation_ratio),
                        "class_distributions": {
                            "before_smote": {
                                "total_samples": original_sample_count,
                                "distribution": original_distribution_detailed
                            },
                            "after_smote": {
                                "total_samples": final_sample_count,
                                "distribution": final_distribution_detailed
                            }
                        },
                        "borderline_smote_successful": True
                    }
                    
                    smote_results["file_results"].append(file_result)
                    
                    # Update tracking variables
                    total_original_samples += original_sample_count
                    total_synthetic_samples += synthetic_samples_added
                    total_final_samples += final_sample_count
                    augmentation_ratios.append(augmentation_ratio)
                    
                    # Update overall distributions
                    overall_original_distribution.update(original_distribution)
                    overall_final_distribution.update(final_distribution)
                    
                    successful_generations += 1
                    print(f"    generate_borderline_smote_data: ✓ {file_identifier} - Added {synthetic_samples_added:,} synthetic samples")
                    
                except Exception as e:
                    print(f"    generate_borderline_smote_data: Error processing {train_file}: {str(e)}")
                    failed_generations += 1
                    
                    # Add failed result
                    file_identifier = os.path.basename(train_file).replace('_train_scaled.csv', '')
                    smote_results["file_results"].append({
                        "file_identifier": file_identifier,
                        "borderline_smote_successful": False,
                        "error": str(e)
                    })
            
            # Finalize results
            if successful_generations > 0:
                # Calculate overall class distributions with percentages
                overall_original_detailed = {}
                overall_final_detailed = {}
                
                # Overall original distribution
                for class_label, count in overall_original_distribution.items():
                    percentage = (count / total_original_samples) * 100
                    overall_original_detailed[str(class_label)] = {
                        "count": count,
                        "percentage": round(percentage, 2)
                    }
                
                # Overall final distribution
                for class_label, count in overall_final_distribution.items():
                    percentage = (count / total_final_samples) * 100
                    overall_final_detailed[str(class_label)] = {
                        "count": count,
                        "percentage": round(percentage, 2)
                    }
                
                smote_results["generation_overview"].update({
                    "total_files_processed": successful_generations + failed_generations,
                    "successful_generations": successful_generations,
                    "failed_generations": failed_generations
                })
                
                smote_results["aggregate_metrics"].update({
                    "total_original_samples": total_original_samples,
                    "total_synthetic_samples": total_synthetic_samples,
                    "total_final_samples": total_final_samples,
                    "average_augmentation_ratio": float(np.mean(augmentation_ratios)) if augmentation_ratios else 0.0,
                    "overall_class_distributions": {
                        "before_smote": {
                            "total_samples": total_original_samples,
                            "distribution": overall_original_detailed
                        },
                        "after_smote": {
                            "total_samples": total_final_samples,
                            "distribution": overall_final_detailed
                        }
                    },
                    "best_performing_file": {
                        "file_identifier": smote_results["file_results"][np.argmax(augmentation_ratios)]["file_identifier"],
                        "augmentation_ratio": float(max(augmentation_ratios))
                    } if augmentation_ratios else None
                })
                
                # Save consolidated results
                results_file = os.path.join(output_dir, "borderline_smote_generation_results.json")
                with open(results_file, 'w') as f:
                    json.dump(smote_results, f, indent=2)
                
                # Create summary visualization
                self._plot_smote_summary(smote_results, output_dir)
                
                print(f"    generate_borderline_smote_data: Completed successfully!")
                print(f"    generate_borderline_smote_data: Generated {total_synthetic_samples:,} synthetic samples across {successful_generations} files")
                print(f"    generate_borderline_smote_data: Average augmentation ratio: {np.mean(augmentation_ratios):.2f}x")
                print(f"    generate_borderline_smote_data: Results saved to: {output_dir}")
                
                return {
                    "success": True,
                    "files_processed": successful_generations,
                    "total_synthetic_samples": total_synthetic_samples,
                    "average_augmentation_ratio": float(np.mean(augmentation_ratios)),
                    "output_directory": output_dir,
                    "results_file": results_file,
                    "target_feature": DEFAULT_TARGET_FEATURE
                }
            else:
                error_msg = "No files were successfully processed for BorderlineSMOTE generation"
                print(f"    generate_borderline_smote_data: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"generate_borderline_smote_data failed: {str(e)}"
            print(f"    generate_borderline_smote_data: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def _plot_smote_summary(self, smote_results, output_dir):
        """
        Create summary visualizations for BorderlineSMOTE data generation.
        
        Parameters:
        -----------
        smote_results : dict
            Results from BorderlineSMOTE generation
        output_dir : str
            Directory to save the plots
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract data for plotting
            file_identifiers = []
            original_samples = []
            synthetic_samples = []
            augmentation_ratios = []
            
            for result in smote_results["file_results"]:
                if result.get("borderline_smote_successful", False):
                    file_identifiers.append(result["file_identifier"])
                    original_samples.append(result["original_samples"])
                    synthetic_samples.append(result["synthetic_samples_added"])
                    augmentation_ratios.append(result["augmentation_ratio"])
            
            # Plot 1: Original vs Synthetic Samples
            x_pos = np.arange(len(file_identifiers))
            width = 0.35
            
            ax1.bar(x_pos - width/2, original_samples, width, label='Original Samples', alpha=0.8, color='skyblue')
            ax1.bar(x_pos + width/2, synthetic_samples, width, label='Synthetic Samples Added', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('File Identifier')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Original vs Synthetic Samples by File')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(file_identifiers, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Augmentation Ratios
            bars = ax2.bar(file_identifiers, augmentation_ratios, alpha=0.8, color='lightgreen')
            ax2.set_xlabel('File Identifier')
            ax2.set_ylabel('Augmentation Ratio')
            ax2.set_title('Data Augmentation Ratio by File')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, ratio in zip(bars, augmentation_ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{ratio:.2f}x', ha='center', va='bottom')
            
            # Plot 3: Total Sample Distribution
            total_original = sum(original_samples)
            total_synthetic = sum(synthetic_samples)
            
            ax3.pie([total_original, total_synthetic], 
                    labels=['Original Samples', 'Synthetic Samples'], 
                    autopct='%1.1f%%', 
                    startangle=90,
                    colors=['skyblue', 'lightcoral'])
            ax3.set_title('Overall Distribution: Original vs Synthetic Samples')
            
            # Plot 4: Summary Statistics
            ax4.axis('off')
            
            # Get overall class distribution info for display
            overall_before = smote_results["aggregate_metrics"]["overall_class_distributions"]["before_smote"]["distribution"]
            overall_after = smote_results["aggregate_metrics"]["overall_class_distributions"]["after_smote"]["distribution"]
            
            # Format class distribution strings
            before_dist_str = ", ".join([f"Class {k}: {v['count']:,} ({v['percentage']}%)" for k, v in overall_before.items()])
            after_dist_str = ", ".join([f"Class {k}: {v['count']:,} ({v['percentage']}%)" for k, v in overall_after.items()])
            
            summary_text = f"""
    BorderlineSMOTE Generation Summary

    Total Files Processed: {len(file_identifiers)}

    Sample Counts:
    Original Samples: {total_original:,}
    Synthetic Samples: {total_synthetic:,}
    Final Samples: {total_original + total_synthetic:,}

    Class Distribution Before SMOTE:
    {before_dist_str}

    Class Distribution After SMOTE:
    {after_dist_str}

    Augmentation Ratios:
    Average: {np.mean(augmentation_ratios):.2f}x
    Min: {min(augmentation_ratios):.2f}x
    Max: {max(augmentation_ratios):.2f}x

    BorderlineSMOTE Configuration:
    k_neighbors: {BORDERLINE_SMOTE_CONFIG['k_neighbors']}
    m_neighbors: {BORDERLINE_SMOTE_CONFIG['m_neighbors']}
    kind: {BORDERLINE_SMOTE_CONFIG['kind']}
    sampling_strategy: {BORDERLINE_SMOTE_CONFIG['sampling_strategy']}
            """
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = os.path.join(output_dir, "borderline_smote_generation_summary.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    generate_borderline_smote_data: Summary plot saved to {plot_filename}")
            
        except Exception as e:
            print(f"    generate_borderline_smote_data: Error creating summary plot: {str(e)}")

    def _create_correlation_plot(self, correlations, filename, output_dir, title):
            """
            Create a Point-Biserial Correlation Analysis plot for a single file.
            
            Parameters:
            -----------
            correlations : dict
                Dictionary of feature names and their correlation values
            filename : str
                Name of the source file
            output_dir : str
                Directory to save the plot
            title : str
                Title for the plot
            """
            try:
                # Filter out NaN correlations
                valid_correlations = {k: v for k, v in correlations.items() if not pd.isna(v)}
                
                if not valid_correlations:
                    print(f"        No valid correlations to plot for {filename}")
                    return
                
                # Sort correlations by absolute value for better visualization
                sorted_correlations = sorted(valid_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                features = [item[0] for item in sorted_correlations]
                corr_values = [item[1] for item in sorted_correlations]
                
                # Create figure with single plot
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Horizontal bar chart of Point-Biserial correlations
                colors = ['red' if val < 0 else 'steelblue' for val in corr_values]
                bars = ax.barh(range(len(features)), corr_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Customize plot
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, fontsize=11)
                ax.set_xlabel('Point-Biserial Correlation Coefficient (r_pb)', fontsize=12)
                ax.set_title(f'Point-Biserial Correlation Analysis\n{DEFAULT_TARGET_FEATURE} vs Weather Features', fontsize=14, pad=20)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                ax.grid(axis='x', alpha=0.3)
                ax.set_xlim(-1, 1)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, corr_values)):
                    label_x = value + (0.02 if value >= 0 else -0.02)
                    ha = 'left' if value >= 0 else 'right'
                    ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                            ha=ha, va='center', fontsize=10, fontweight='bold')
                
                # Add interpretation guide
                interpretation_text = (
                    "Interpretation Guide:\n"
                    "• |r_pb| ≥ 0.7: Very Strong relationship\n"
                    "• 0.5 ≤ |r_pb| < 0.7: Strong relationship\n"
                    "• 0.3 ≤ |r_pb| < 0.5: Moderate relationship\n"
                    "• 0.1 ≤ |r_pb| < 0.3: Weak relationship\n"
                    "• |r_pb| < 0.1: Very weak/no relationship\n\n"
                    "Positive: Higher weather values → More delays\n"
                    "Negative: Higher weather values → Fewer delays"
                )
                
                fig.text(0.02, 0.02, interpretation_text, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                        verticalalignment='bottom')
                
                plt.suptitle(f'Point-Biserial Correlation Analysis: {filename}', fontsize=15, y=0.95)
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.25)
                
                # Save plot
                safe_filename = filename.replace('.csv', '').replace(' ', '_')
                plot_filename = f"point_biserial_correlation_{safe_filename}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"        Saved point-biserial correlation plot: {plot_filename}")
                
            except Exception as e:
                print(f"        Warning: Failed to create plot for {filename}: {str(e)}")

    def _create_distribution_plots(self, df, weather_features, filename, output_dir):
        """
        Create distribution plots for target feature and weather features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset containing all features
        weather_features : list
            List of weather feature column names
        filename : str
            Name of the source file
        output_dir : str
            Directory to save the plots
        """
        try:
            print(f"        Creating distribution plots for {filename}...")
            
            # Check if target feature exists
            if DEFAULT_TARGET_FEATURE not in df.columns:
                print(f"        Warning: Target feature '{DEFAULT_TARGET_FEATURE}' not found for distribution plots")
                return
            
            # Filter weather features that exist in the dataframe
            valid_weather_features = [col for col in weather_features if col in df.columns]
            
            if not valid_weather_features:
                print(f"        Warning: No valid weather features found for distribution plots")
                return
            
            # Create comprehensive distribution analysis
            n_features = len(valid_weather_features)
            n_cols = 3
            n_rows = max(2, (n_features + n_cols - 1) // n_cols + 1)  # +1 for target distribution
            
            fig = plt.figure(figsize=(18, 6 * n_rows))
            gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
            
            # 1. Target feature distribution (first row, span all columns)
            ax_target = fig.add_subplot(gs[0, :])
            target_counts = df[DEFAULT_TARGET_FEATURE].value_counts()
            target_proportions = df[DEFAULT_TARGET_FEATURE].value_counts(normalize=True)
            
            bars = ax_target.bar(
                ['Not Delayed (False)', 'Delayed (True)'], 
                [target_counts.get(False, 0), target_counts.get(True, 0)],
                color=['lightgreen', 'lightcoral'], 
                alpha=0.7,
                edgecolor='black',
                linewidth=1
            )
            
            ax_target.set_title(f'{DEFAULT_TARGET_FEATURE} Distribution\nTotal samples: {len(df):,}', 
                            fontsize=14, fontweight='bold')
            ax_target.set_ylabel('Count', fontsize=12)
            
            # Add percentage labels on bars
            for i, (bar, count) in enumerate(zip(bars, [target_counts.get(False, 0), target_counts.get(True, 0)])):
                percentage = (count / len(df)) * 100
                ax_target.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(df)*0.01,
                            f'{count:,}\n({percentage:.1f}%)', 
                            ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax_target.grid(axis='y', alpha=0.3)
            
            # 2. Weather feature distributions and comparative box plots
            for i, feature in enumerate(valid_weather_features):
                row = (i // n_cols) + 1
                col = i % n_cols
                
                # Create subplot for this feature
                ax = fig.add_subplot(gs[row, col])
                
                # Get feature data, handling missing values
                feature_data = df[feature].dropna()
                if len(feature_data) == 0:
                    ax.text(0.5, 0.5, f'No valid data\nfor {feature}', 
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_title(feature, fontsize=12, fontweight='bold')
                    continue
                
                # Create combined plot: histogram + box plot comparison
                # Top: Histogram of the feature
                ax_hist = ax
                ax_hist.hist(feature_data, bins=30, alpha=0.7, color='steelblue', 
                            edgecolor='black', linewidth=0.5)
                ax_hist.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
                ax_hist.set_xlabel(feature, fontsize=10)
                ax_hist.set_ylabel('Frequency', fontsize=10)
                ax_hist.grid(axis='y', alpha=0.3)
                
                # Add statistics text
                stats_text = (f'Mean: {feature_data.mean():.2f}\n'
                            f'Std: {feature_data.std():.2f}\n'
                            f'Missing: {df[feature].isna().sum():,}')
                ax_hist.text(0.02, 0.98, stats_text, transform=ax_hist.transAxes,
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.suptitle(f'Feature Distributions Analysis: {filename}', fontsize=16, y=0.98)
            plt.tight_layout()
            
            # Save distribution plot
            safe_filename = filename.replace('.csv', '').replace(' ', '_')
            dist_plot_filename = f"distributions_{safe_filename}.png"
            dist_plot_path = os.path.join(output_dir, dist_plot_filename)
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create separate comparative box plots (delayed vs non-delayed)
            self._create_comparative_distributions(df, valid_weather_features, filename, output_dir)
            
            print(f"        Saved distribution plots: {dist_plot_filename}")
            
        except Exception as e:
            print(f"        Warning: Failed to create distribution plots for {filename}: {str(e)}")

    def _create_comparative_distributions(self, df, weather_features, filename, output_dir):
        """
        Create comparative distribution plots showing weather features for delayed vs non-delayed trains.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset containing all features
        weather_features : list
            List of weather feature column names
        filename : str
            Name of the source file
        output_dir : str
            Directory to save the plots
        """
        try:
            print(f"        Creating comparative distribution plots for {filename}...")
            
            # Separate data by target feature
            delayed_data = df[df[DEFAULT_TARGET_FEATURE] == True]
            not_delayed_data = df[df[DEFAULT_TARGET_FEATURE] == False]
            
            if len(delayed_data) == 0 or len(not_delayed_data) == 0:
                print(f"        Warning: Insufficient data for comparative analysis")
                return
            
            n_features = len(weather_features)
            n_cols = 2
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, feature in enumerate(weather_features):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # Get feature data for both groups
                delayed_feature = delayed_data[feature].dropna()
                not_delayed_feature = not_delayed_data[feature].dropna()
                
                if len(delayed_feature) == 0 or len(not_delayed_feature) == 0:
                    ax.text(0.5, 0.5, f'Insufficient data\nfor {feature}', 
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_title(feature, fontsize=12, fontweight='bold')
                    continue
                
                # Create box plot comparison
                box_data = [not_delayed_feature, delayed_feature]
                box_labels = ['Not Delayed', 'Delayed']
                
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightgreen')
                bp['boxes'][0].set_alpha(0.7)
                bp['boxes'][1].set_facecolor('lightcoral')
                bp['boxes'][1].set_alpha(0.7)
                
                ax.set_title(f'{feature}\nDelayed vs Not Delayed', fontsize=11, fontweight='bold')
                ax.set_ylabel(feature, fontsize=10)
                ax.grid(axis='y', alpha=0.3)
                
                # Add statistical comparison
                mean_not_delayed = not_delayed_feature.mean()
                mean_delayed = delayed_feature.mean()
                difference = mean_delayed - mean_not_delayed
                
                stats_text = (f'Not Delayed: μ={mean_not_delayed:.2f}\n'
                            f'Delayed: μ={mean_delayed:.2f}\n'
                            f'Difference: {difference:+.2f}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            # Hide empty subplots
            for i in range(n_features, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.suptitle(f'Comparative Distributions: Delayed vs Not Delayed Trains\n{filename}', 
                        fontsize=14, y=0.98)
            plt.tight_layout()
            
            # Save comparative plot
            safe_filename = filename.replace('.csv', '').replace(' ', '_')
            comp_plot_filename = f"comparative_distributions_{safe_filename}.png"
            comp_plot_path = os.path.join(output_dir, comp_plot_filename)
            plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"        Saved comparative distribution plot: {comp_plot_filename}")
            
        except Exception as e:
            print(f"        Warning: Failed to create comparative distribution plots for {filename}: {str(e)}")

    def _create_combined_correlation_analysis(self, correlation_results, output_dir):
        """
        Create combined Point-Biserial correlation analysis across all files.
        
        Parameters:
        -----------
        correlation_results : list
            List of correlation result dictionaries from each file
        output_dir : str
            Directory to save the combined analysis
        """
        try:
            print("    correlation_analysis: Creating combined point-biserial correlation analysis...")
            
            # Collect all unique weather features
            all_features = set()
            for result in correlation_results:
                all_features.update(result['correlations'].keys())
            all_features = sorted(list(all_features))
            
            # Calculate average correlations across files
            avg_correlations = {}
            correlation_ranges = {}
            
            for feature in all_features:
                feature_values = []
                for result in correlation_results:
                    corr_value = result['correlations'].get(feature, np.nan)
                    if not pd.isna(corr_value):
                        feature_values.append(corr_value)
                
                if feature_values:
                    avg_correlations[feature] = {
                        'mean': np.mean(feature_values),
                        'std': np.std(feature_values),
                        'min': np.min(feature_values),
                        'max': np.max(feature_values),
                        'count': len(feature_values)
                    }
                    correlation_ranges[feature] = (np.min(feature_values), np.max(feature_values))
            
            # Sort by absolute mean correlation
            sorted_features = sorted(avg_correlations.keys(), 
                                key=lambda x: abs(avg_correlations[x]['mean']), 
                                reverse=True)
            
            # Create comprehensive combined plot
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 2, 1])
            
            # Main plot: Average correlations with error bars
            ax1 = fig.add_subplot(gs[0, 0])
            
            y_positions = range(len(sorted_features))
            means = [avg_correlations[f]['mean'] for f in sorted_features]
            stds = [avg_correlations[f]['std'] for f in sorted_features]
            colors = ['red' if val < 0 else 'steelblue' for val in means]
            
            bars = ax1.barh(y_positions, means, xerr=stds, color=colors, alpha=0.7, 
                        capsize=5, edgecolor='black', linewidth=0.5)
            
            ax1.set_yticks(y_positions)
            ax1.set_yticklabels(sorted_features, fontsize=10)
            ax1.set_xlabel('Average Point-Biserial Correlation', fontsize=12)
            ax1.set_title('Average Point-Biserial Correlations\n(with Standard Deviation)', fontsize=12)
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax1.grid(axis='x', alpha=0.3)
            ax1.set_xlim(-1, 1)
            
            # Add value labels
            for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                label_x = mean_val + (0.05 if mean_val >= 0 else -0.05)
                ha = 'left' if mean_val >= 0 else 'right'
                ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                        f'{mean_val:.3f}±{std_val:.3f}', 
                        ha=ha, va='center', fontsize=8, fontweight='bold')
            
            # Range analysis plot
            ax2 = fig.add_subplot(gs[0, 1])
            
            for i, feature in enumerate(sorted_features):
                min_val, max_val = correlation_ranges[feature]
                mean_val = avg_correlations[feature]['mean']
                
                # Plot range as line
                ax2.plot([min_val, max_val], [i, i], 'k-', linewidth=2, alpha=0.6)
                # Plot min and max as points
                ax2.plot(min_val, i, 'ro', markersize=6, alpha=0.8)
                ax2.plot(max_val, i, 'bo', markersize=6, alpha=0.8)
                # Plot mean as diamond
                ax2.plot(mean_val, i, 'gD', markersize=8, alpha=0.9)
            
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels(sorted_features, fontsize=10)
            ax2.set_xlabel('Correlation Range Across Files', fontsize=12)
            ax2.set_title('Correlation Variability\n(Min-Max Range)', fontsize=12)
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.grid(axis='x', alpha=0.3)
            ax2.set_xlim(-1, 1)
            ax2.legend(['Range', 'Minimum', 'Maximum', 'Mean'], loc='lower right', fontsize=9)
            
            # Strength classification plot
            ax3 = fig.add_subplot(gs[0, 2])
            
            strength_categories = {'Very Strong (≥0.7)': [], 'Strong (0.5-0.7)': [], 
                                'Moderate (0.3-0.5)': [], 'Weak (0.1-0.3)': [], 
                                'Very Weak (<0.1)': []}
            
            for feature in sorted_features:
                abs_mean = abs(avg_correlations[feature]['mean'])
                if abs_mean >= 0.7:
                    strength_categories['Very Strong (≥0.7)'].append(feature)
                elif abs_mean >= 0.5:
                    strength_categories['Strong (0.5-0.7)'].append(feature)
                elif abs_mean >= 0.3:
                    strength_categories['Moderate (0.3-0.5)'].append(feature)
                elif abs_mean >= 0.1:
                    strength_categories['Weak (0.1-0.3)'].append(feature)
                else:
                    strength_categories['Very Weak (<0.1)'].append(feature)
            
            category_counts = [len(features) for features in strength_categories.values()]
            category_labels = list(strength_categories.keys())
            colors_pie = ['darkred', 'red', 'orange', 'yellow', 'lightgray']
            
            wedges, texts, autotexts = ax3.pie(category_counts, labels=category_labels, autopct='%1.0f%%', 
                                            colors=colors_pie, startangle=90)
            ax3.set_title('Correlation Strength\nDistribution', fontsize=12)
            
            # File-by-file heatmap
            ax4 = fig.add_subplot(gs[1, :2])
            
            # Create matrix for heatmap
            heatmap_data = []
            file_labels = []
            
            for result in correlation_results:
                row_data = []
                for feature in sorted_features:
                    corr_value = result['correlations'].get(feature, np.nan)
                    row_data.append(corr_value)
                heatmap_data.append(row_data)
                file_labels.append(result['filename'].replace('.csv', ''))
            
            heatmap_matrix = np.array(heatmap_data)
            mask = np.isnan(heatmap_matrix)
            
            im = ax4.imshow(heatmap_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1, shrink=0.8)
            cbar.set_label('Point-Biserial Correlation Coefficient', fontsize=10)
            
            ax4.set_xticks(range(len(sorted_features)))
            ax4.set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=9)
            ax4.set_yticks(range(len(file_labels)))
            ax4.set_yticklabels(file_labels, fontsize=9)
            ax4.set_title('Point-Biserial Correlations Across All Files', fontsize=12)
            
            # Add text annotations for non-NaN values
            for i in range(len(file_labels)):
                for j in range(len(sorted_features)):
                    if not mask[i, j]:
                        text_color = 'white' if abs(heatmap_matrix[i, j]) > 0.5 else 'black'
                        ax4.text(j, i, f'{heatmap_matrix[i, j]:.2f}', 
                                ha='center', va='center', color=text_color, fontsize=7)
            
            # Summary statistics text
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.axis('off')
            
            summary_text = "Summary Statistics:\n\n"
            summary_text += f"Total Features: {len(sorted_features)}\n"
            summary_text += f"Files Analyzed: {len(correlation_results)}\n\n"
            
            if sorted_features:
                strongest_feature = sorted_features[0]
                strongest_corr = avg_correlations[strongest_feature]['mean']
                summary_text += f"Strongest Correlation:\n{strongest_feature}\n(r = {strongest_corr:.3f})\n\n"
            
            # Count by strength
            for category, features in strength_categories.items():
                if features:
                    summary_text += f"{category}: {len(features)}\n"
            
            ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor="lightblue", alpha=0.8))
            
            plt.suptitle(f'Comprehensive Point-Biserial Correlation Analysis\n{DEFAULT_TARGET_FEATURE} vs Weather Features', 
                        fontsize=16, y=0.98)
            plt.tight_layout()
            
            # Save combined plot
            combined_plot_path = os.path.join(output_dir, "combined_point_biserial_analysis.png")
            plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save detailed average correlations summary
            avg_summary_path = os.path.join(output_dir, "point_biserial_summary.txt")
            with open(avg_summary_path, 'w') as f:
                f.write("Point-Biserial Correlation Analysis Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Target Variable: {DEFAULT_TARGET_FEATURE} (Binary)\n")
                f.write(f"Weather Features: {len(all_features)} (Continuous)\n")
                f.write(f"Files Analyzed: {len(correlation_results)}\n\n")
                
                f.write("Correlation Strength Interpretation:\n")
                f.write("• |r_pb| ≥ 0.7: Very Strong relationship\n")
                f.write("• 0.5 ≤ |r_pb| < 0.7: Strong relationship\n")
                f.write("• 0.3 ≤ |r_pb| < 0.5: Moderate relationship\n")
                f.write("• 0.1 ≤ |r_pb| < 0.3: Weak relationship\n")
                f.write("• |r_pb| < 0.1: Very weak/no relationship\n\n")
                
                f.write("Weather Features (sorted by absolute correlation strength):\n")
                f.write("-" * 70 + "\n")
                for i, feature in enumerate(sorted_features, 1):
                    stats = avg_correlations[feature]
                    strength = ""
                    abs_mean = abs(stats['mean'])
                    if abs_mean >= 0.7:
                        strength = "Very Strong"
                    elif abs_mean >= 0.5:
                        strength = "Strong"
                    elif abs_mean >= 0.3:
                        strength = "Moderate"
                    elif abs_mean >= 0.1:
                        strength = "Weak"
                    else:
                        strength = "Very Weak"
                    
                    f.write(f"{i:2d}. {feature}:\n")
                    f.write(f"    Mean correlation: {stats['mean']:+.4f} ({strength})\n")
                    f.write(f"    Std deviation: {stats['std']:.4f}\n")
                    f.write(f"    Range: [{stats['min']:+.4f}, {stats['max']:+.4f}]\n")
                    f.write(f"    Files available: {stats['count']}/{len(correlation_results)}\n")
                    
                    # Interpretation
                    if stats['mean'] > 0:
                        direction = "Higher values → More train delays"
                    elif stats['mean'] < 0:
                        direction = "Higher values → Fewer train delays"
                    else:
                        direction = "No clear relationship"
                    f.write(f"    Interpretation: {direction}\n\n")
            
            print(f"        Saved comprehensive point-biserial correlation analysis")
            
        except Exception as e:
            print(f"        Warning: Failed to create combined analysis: {str(e)}")

    def _create_non_weather_correlation_plot(self, correlations, correlation_types, filename, output_dir, title):
        """
        Create a Non-Weather Features Correlation Analysis plot for a single file.
        
        This creates separate subplots for boolean and temporal features with their correlations.
        """
        try:
            # Separate correlations by type
            boolean_correlations = {}
            temporal_correlations = {}
            
            for feature, corr in correlations.items():
                if pd.isna(corr):
                    continue
                corr_type = correlation_types.get(feature, "")
                if "Boolean" in corr_type:
                    boolean_correlations[feature] = corr
                elif "Temporal" in corr_type:
                    temporal_correlations[feature] = corr
            
            # Calculate subplot layout
            n_plots = 0
            if boolean_correlations:
                n_plots += 1
            if temporal_correlations:
                n_plots += 1
            
            if n_plots == 0:
                print(f"      No valid correlations to plot for {filename}")
                return
            
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6 * n_plots))
            if n_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # Plot boolean features
            if boolean_correlations:
                ax = axes[plot_idx]
                features = list(boolean_correlations.keys())
                values = list(boolean_correlations.values())
                colors = ['darkblue' if v >= 0 else 'darkred' for v in values]
                
                bars = ax.bar(features, values, color=colors, alpha=0.7, edgecolor='black')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title(f'Boolean Features - Phi Coefficient\n({len(features)} features)')
                ax.set_ylabel('Phi Coefficient (φ)')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
                
                # Rotate x-labels if needed
                if len(features) > 3:
                    ax.tick_params(axis='x', rotation=45)
                
                plot_idx += 1
            
            # Plot temporal features
            if temporal_correlations:
                ax = axes[plot_idx]
                features = list(temporal_correlations.keys())
                values = list(temporal_correlations.values())
                colors = ['steelblue' if v >= 0 else 'crimson' for v in values]
                
                bars = ax.bar(features, values, color=colors, alpha=0.7, edgecolor='black')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title(f'Temporal Features (Sin-Cos) - Point-Biserial Correlation\n({len(features)} features)')
                ax.set_ylabel('Point-Biserial Correlation (r_pb)')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
                
                # Rotate x-labels if needed
                if len(features) > 3:
                    ax.tick_params(axis='x', rotation=45)
            
            plt.suptitle(title, fontsize=14, y=0.98)
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"non_weather_correlation_{filename.replace('.csv', '.png')}"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      Saved non-weather correlation plot: {plot_filename}")
            
        except Exception as e:
            print(f"      Error creating non-weather correlation plot: {str(e)}")

    def _create_combined_non_weather_correlation_analysis(self, correlation_results, output_dir):
        """
        Create combined analysis across all non-weather correlation results.
        """
        try:
            print(f"    Creating combined non-weather correlation analysis...")
            
            # Aggregate correlations across all files
            all_features = set()
            for result in correlation_results:
                all_features.update(result['correlations'].keys())
            
            all_features = sorted(list(all_features))
            
            # Calculate average correlations
            avg_correlations = {}
            for feature in all_features:
                values = []
                for result in correlation_results:
                    if feature in result['correlations'] and not pd.isna(result['correlations'][feature]):
                        values.append(result['correlations'][feature])
                
                if values:
                    avg_correlations[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            # Create combined plot
            if avg_correlations:
                # Sort features by absolute correlation strength
                sorted_features = sorted(avg_correlations.keys(), 
                                    key=lambda x: abs(avg_correlations[x]['mean']), 
                                    reverse=True)
                
                # Separate by feature type for plotting
                boolean_features = [f for f in sorted_features if f in ['trainStopping', 'commercialStop']]
                temporal_features = [f for f in sorted_features if f not in boolean_features]
                
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                
                # Plot boolean features
                if boolean_features:
                    ax = axes[0]
                    values = [avg_correlations[f]['mean'] for f in boolean_features]
                    errors = [avg_correlations[f]['std'] for f in boolean_features]
                    colors = ['darkblue' if v >= 0 else 'darkred' for v in values]
                    
                    bars = ax.bar(boolean_features, values, yerr=errors, color=colors, 
                                alpha=0.7, edgecolor='black', capsize=5)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax.set_title(f'Boolean Features - Average Phi Coefficient Across All Files\n({len(boolean_features)} features)')
                    ax.set_ylabel('Average Phi Coefficient (φ)')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, value, error in zip(bars, values, errors):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + (error + 0.01 if height >= 0 else -error - 0.01),
                            f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=10, fontweight='bold')
                else:
                    axes[0].text(0.5, 0.5, 'No Boolean Features Available', 
                            ha='center', va='center', transform=axes[0].transAxes, fontsize=14)
                    axes[0].set_title('Boolean Features - No Data')
                
                # Plot temporal features
                if temporal_features:
                    ax = axes[1]
                    values = [avg_correlations[f]['mean'] for f in temporal_features]
                    errors = [avg_correlations[f]['std'] for f in temporal_features]
                    colors = ['steelblue' if v >= 0 else 'crimson' for v in values]
                    
                    bars = ax.bar(temporal_features, values, yerr=errors, color=colors, 
                                alpha=0.7, edgecolor='black', capsize=5)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax.set_title(f'Temporal Features - Average Point-Biserial Correlation Across All Files\n({len(temporal_features)} features)')
                    ax.set_ylabel('Average Point-Biserial Correlation (r_pb)')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, value, error in zip(bars, values, errors):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + (error + 0.01 if height >= 0 else -error - 0.01),
                            f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=10, fontweight='bold')
                    
                    # Rotate x-labels for temporal features
                    ax.tick_params(axis='x', rotation=45)
                else:
                    axes[1].text(0.5, 0.5, 'No Temporal Features Available', 
                            ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
                    axes[1].set_title('Temporal Features - No Data')
                
                plt.suptitle(f'Combined Non-Weather Features Correlation Analysis\n{DEFAULT_TARGET_FEATURE} vs Non-Weather Features', 
                            fontsize=16, y=0.98)
                plt.tight_layout()
                
                # Save combined plot
                combined_plot_path = os.path.join(output_dir, "combined_non_weather_correlation_analysis.png")
                plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save detailed average correlations summary
                avg_summary_path = os.path.join(output_dir, "non_weather_correlation_summary.txt")
                with open(avg_summary_path, 'w') as f:
                    f.write("Combined Non-Weather Features Correlation Analysis\n")
                    f.write("=" * 65 + "\n\n")
                    f.write(f"Target Variable: {DEFAULT_TARGET_FEATURE} (Binary)\n")
                    f.write(f"Non-Weather Features: {len(all_features)}\n")
                    f.write(f"Files Analyzed: {len(correlation_results)}\n\n")
                    
                    f.write("Correlation Strength Interpretation:\n")
                    f.write("• |correlation| ≥ 0.7: Very Strong relationship\n")
                    f.write("• 0.5 ≤ |correlation| < 0.7: Strong relationship\n")
                    f.write("• 0.3 ≤ |correlation| < 0.5: Moderate relationship\n")
                    f.write("• 0.1 ≤ |correlation| < 0.3: Weak relationship\n")
                    f.write("• |correlation| < 0.1: Very weak/no relationship\n\n")
                    
                    f.write("Non-Weather Features (sorted by absolute correlation strength):\n")
                    f.write("-" * 80 + "\n")
                    for i, feature in enumerate(sorted_features, 1):
                        stats = avg_correlations[feature]
                        strength = ""
                        abs_mean = abs(stats['mean'])
                        if abs_mean >= 0.7:
                            strength = "Very Strong"
                        elif abs_mean >= 0.5:
                            strength = "Strong"
                        elif abs_mean >= 0.3:
                            strength = "Moderate"
                        elif abs_mean >= 0.1:
                            strength = "Weak"
                        else:
                            strength = "Very Weak"
                        
                        feature_type = "Boolean" if feature in ['trainStopping', 'commercialStop'] else "Temporal"
                        
                        f.write(f"{i:2d}. {feature:20s} | {stats['mean']:+.4f} ± {stats['std']:.4f} | {strength:12s} | {feature_type}\n")
                        f.write(f"    Files: {stats['count']}/{len(correlation_results)} | Range: [{stats['min']:+.4f}, {stats['max']:+.4f}]\n\n")
                
                print(f"    Combined non-weather correlation analysis saved")
            
        except Exception as e:
            print(f"    Error creating combined non-weather correlation analysis: {str(e)}")

    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive classification metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like  
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities for positive class
            
        Returns:
        --------
        dict
            Dictionary containing various classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Classification report metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        #metrics['weighted_avg_f1'] = report['weighted avg']['f1-score']
        #metrics['weighted_avg_precision'] = report['weighted avg']['precision']
        #metrics['weighted_avg_recall'] = report['weighted avg']['recall']
        
        # Probability-based metrics (if available)
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                # For binary classification, use positive class probabilities
                if y_pred_proba.ndim > 1:
                    pos_proba = y_pred_proba[:, 1]
                else:
                    pos_proba = y_pred_proba
                    
                metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
                metrics['pr_auc'] = average_precision_score(y_true, pos_proba)
            except:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        
        # Class-specific metrics
        for class_label in report:
            if class_label not in ['weighted avg', 'macro avg', 'accuracy']:
                metrics[f'class_{class_label}_f1'] = report[class_label]['f1-score']
                metrics[f'class_{class_label}_precision'] = report[class_label]['precision']
                metrics[f'class_{class_label}_recall'] = report[class_label]['recall']
                metrics[f'class_{class_label}_support'] = report[class_label]['support']
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        for class_val, count in zip(unique, counts):
            metrics[f'class_{class_val}_test_count'] = count
            metrics[f'class_{class_val}_test_percentage'] = (count / len(y_true)) * 100
        
        return metrics

    def _save_confusion_matrix(self, conf_matrix, y_test, y_pred, file_identifier, output_dir):
        """
        Save confusion matrix as both CSV data and visualization plot.
        
        Parameters:
        -----------
        conf_matrix : array-like
            Confusion matrix from sklearn.metrics.confusion_matrix
        y_test : array-like
            True labels
        y_pred : array-like
            Predicted labels  
        file_identifier : str
            Identifier for the file being processed
        output_dir : str
            Directory to save the confusion matrix files
            
        Returns:
        --------
        dict
            Dictionary with paths to saved confusion matrix files
        """
        try:
            # Get unique class labels
            unique_labels = np.unique(np.concatenate([y_test, y_pred]))
            
            # Create confusion matrix DataFrame
            conf_df = pd.DataFrame(
                conf_matrix,
                index=[f'True_{label}' for label in unique_labels],
                columns=[f'Pred_{label}' for label in unique_labels]
            )
            
            # Save confusion matrix as CSV
            csv_filename = f"confusion_matrix_{file_identifier}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            conf_df.to_csv(csv_path)
            
            # Create and save confusion matrix visualization
            plt.figure(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(
                conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=[f'Predicted {label}' for label in unique_labels],
                yticklabels=[f'Actual {label}' for label in unique_labels],
                cbar_kws={'label': 'Count'}
            )
            
            plt.title(f'Confusion Matrix - {file_identifier}', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('Actual Label', fontsize=12)
            
            # Add classification metrics as text
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle binary vs multiclass metrics
            if len(unique_labels) == 2:
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
            else:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Add metrics text box
            metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"confusion_matrix_{file_identifier}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"    train_decision_tree: Saved confusion matrix to {csv_filename} and {plot_filename}")
            
            return {
                'csv_file': csv_filename,
                'plot_file': plot_filename,
                'csv_path': csv_path,
                'plot_path': plot_path
            }
            
        except Exception as e:
            print(f"    train_decision_tree: Warning - Failed to save confusion matrix: {str(e)}")
            return {
                'csv_file': None,
                'plot_file': None,
                'csv_path': None,
                'plot_path': None,
                'error': str(e)
            }
        
    def _find_optimal_thresholds(self, y_true, y_proba, fpr, tpr, thresholds, precision, recall, pr_thresholds):
        """
        Find optimal thresholds based on different metrics.
        
        Returns:
        --------
        dict
            Dictionary with optimal thresholds for different metrics
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        
        # Youden's J statistic (TPR - FPR)
        j_scores = tpr - fpr
        youden_optimal_idx = np.argmax(j_scores)
        youden_threshold = thresholds[youden_optimal_idx]
        
        # F1 score optimization
        f1_scores = []
        test_thresholds = np.arange(
            THRESHOLD_OPTIMIZATION_CONFIG["min_threshold"],
            THRESHOLD_OPTIMIZATION_CONFIG["max_threshold"],
            THRESHOLD_OPTIMIZATION_CONFIG["threshold_step"]
        )
        
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        
        f1_optimal_idx = np.argmax(f1_scores)
        f1_threshold = test_thresholds[f1_optimal_idx]
        
        # Precision optimization (highest precision with reasonable recall)
        precision_scores = []
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            if np.sum(y_pred) > 0:  # Avoid division by zero
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                # Only consider thresholds with recall > 0.1
                if rec > 0.1:
                    precision_scores.append(prec)
                else:
                    precision_scores.append(0)
            else:
                precision_scores.append(0)
        
        precision_optimal_idx = np.argmax(precision_scores)
        precision_threshold = test_thresholds[precision_optimal_idx]
        
        # Recall optimization (highest recall with reasonable precision)
        recall_scores = []
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            if np.sum(y_pred) > 0:
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                # Only consider thresholds with precision > 0.1
                if prec > 0.1:
                    recall_scores.append(rec)
                else:
                    recall_scores.append(0)
            else:
                recall_scores.append(0)
        
        recall_optimal_idx = np.argmax(recall_scores)
        recall_threshold = test_thresholds[recall_optimal_idx]
        
        # Accuracy optimization
        accuracy_scores = []
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            acc = accuracy_score(y_true, y_pred)
            accuracy_scores.append(acc)
        
        accuracy_optimal_idx = np.argmax(accuracy_scores)
        accuracy_threshold = test_thresholds[accuracy_optimal_idx]
        
        return {
            "youden": float(youden_threshold),
            "f1": float(f1_threshold),
            "precision": float(precision_threshold),
            "recall": float(recall_threshold),
            "accuracy": float(accuracy_threshold)
        }
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc, optimal_threshold, file_identifier, output_dir):
        """Plot and save ROC curve with optimal threshold marked."""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        
        # Mark optimal threshold
        from sklearn.metrics import roc_curve
        # Find closest threshold point
        optimal_idx = np.argmin(np.abs(np.array([0.5]) - optimal_threshold))  # Simplified for visualization
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
                label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {file_identifier}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{file_identifier}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, precision, recall, pr_auc, optimal_threshold, file_identifier, output_dir):
        """Plot and save Precision-Recall curve."""
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {file_identifier}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'precision_recall_curve_{file_identifier}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_analysis(self, thresholds, fpr, tpr, precision, recall, file_identifier, output_dir):
        """Plot threshold analysis showing how metrics change with threshold."""
        plt.figure(figsize=(15, 10))
        
        # Ensure all arrays have the same length
        min_len = min(len(thresholds), len(fpr), len(tpr), len(precision), len(recall))
        thresholds = thresholds[:min_len]
        fpr = fpr[:min_len]
        tpr = tpr[:min_len]
        precision = precision[:min_len]
        recall = recall[:min_len]
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, tpr, 'b-', label='True Positive Rate')
        plt.plot(thresholds, fpr, 'r-', label='False Positive Rate')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('TPR and FPR vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, precision, 'g-', label='Precision')
        plt.plot(thresholds, recall, 'b-', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        plt.plot(thresholds, f1_scores, 'purple', label='F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Calculate Youden's J statistic
        j_scores = tpr - fpr
        plt.plot(thresholds, j_scores, 'orange', label="Youden's J")
        plt.xlabel('Threshold')
        plt.ylabel("Youden's J")
        plt.title("Youden's J Statistic vs Threshold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Threshold Analysis - {file_identifier}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'threshold_analysis_{file_identifier}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_summary(self, optimal_thresholds, f1_scores, output_dir):
        """Plot summary of optimal thresholds across all models."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.hist(optimal_thresholds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Optimal Threshold')
        plt.ylabel('Frequency')
        plt.title('Distribution of Optimal Thresholds Across All Models')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.scatter(optimal_thresholds, f1_scores, alpha=0.7, color='green')
        plt.xlabel('Optimal Threshold')
        plt.ylabel('Optimized F1 Score')
        plt.title('Optimal Threshold vs F1 Score Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_optimization_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()