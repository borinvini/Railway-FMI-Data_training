import ast
from contextlib import contextmanager
import logging
import os
import re

import numpy as np
import pandas as pd

from config.const import ALL_WEATHER_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES, DATA_FILE_PREFIX_FOR_TRAINING, DEFAULT_TARGET_FEATURE, IMPORTANT_WEATHER_FEATURES, POSSIBLE_INDICATORS, PREPROCESSED_OUTPUT_FOLDER, PREPROCESSING_STATE_MACHINE, STRONG_INDICATORS, TARGET_STATION_CODE, TRAIN_DELAY_MINUTES, TRAIN_DELAYED_TARGET_COLUMN, TRAINING_READY_OUTPUT_FOLDER, VALID_TARGET_FEATURES, VALID_TRAIN_PREDICTION_FEATURES, WEATHER_COLS_TO_MERGE, WEATHER_MISSING_THRESHOLD
from src.file_utils import generate_output_path



class PreprocessingPipeline:
    def __init__(self):
        """Initialize the PreprocessingPipeline class with default values."""
        # Move relevant initialization from TrainingPipeline
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.preprocessed_dir = os.path.join(self.project_root, PREPROCESSED_OUTPUT_FOLDER)
        self.training_ready_dir = os.path.join(self.project_root, TRAINING_READY_OUTPUT_FOLDER)
        
        # Create log directory
        self.log_dir = os.path.join(self.project_root, "data", "output", "log")
        os.makedirs(self.log_dir, exist_ok=True)

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

        if state_machine.get("filter_by_target_station", False):
            if result["data"] is not None:
                try:
                    print(f"    → filter_by_target_station")
                    station_filtered_df = self.filter_by_target_station(
                        dataframe=result["data"], 
                        month_id=file_id
                    )
                    
                    if station_filtered_df is not None:
                        # Clear previous dataframe from memory
                        del result["data"]
                        result["data"] = station_filtered_df
                        result["steps_executed"].append("filter_by_target_station")
                        result["file_info"]["rows"] = len(station_filtered_df)
                        result["file_info"]["columns"] = len(station_filtered_df.columns)
                        print(f"      ✓ Filtered to {len(station_filtered_df)} rows for target station")
                        
                        # Check if filtering resulted in empty data
                        if station_filtered_df.empty:
                            print(f"      ⚠️  Warning: No data remains after station filtering")
                            # You might want to continue or stop here depending on requirements
                            # For now, we'll continue with empty dataframe
                            
                    else:
                        result["errors"].append("filter_by_target_station returned None")
                        print(f"      ✗ Failed - returned None")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"filter_by_target_station failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                result["errors"].append("filter_by_target_station: No data available")
                print(f"    ✗ filter_by_target_station (no data)")
                return result
        else:
            print(f"    ⊝ filter_by_target_station (disabled)")

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

    def filter_by_target_station(self, dataframe=None, month_id=None):
        """
        Filter the dataframe to keep only rows for a specific target station.
        
        This method filters the data to focus on train stops at a specific train station
        based on the stationShortCode column. Only exact matches are kept - for example,
        if TARGET_STATION_CODE is 'OL', stations like 'OLL' or 'OLK' will be dropped.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to filter.
        month_id : str, optional
            Month identifier for logging purposes (format: YYYY_MM).
            
        Returns:
        --------
        pandas.DataFrame
            The filtered dataframe containing only data for the target station.
            Returns None if stationShortCode column is missing or if no data matches.
        """
       
        # Check if dataframe is provided
        if dataframe is None:
            print("Error: Dataframe must be provided")
            return None
            
        df = dataframe.copy()
        
        # Log initial state
        initial_rows = len(df)
        initial_stations = df['stationShortCode'].nunique() if 'stationShortCode' in df.columns else 0
        
        print(f"Filtering data for target station '{TARGET_STATION_CODE}' from dataframe with {initial_rows} rows")
        
        if df.empty:
            print("Warning: Empty dataframe")
            return df
            
        # Check if stationShortCode column exists
        if 'stationShortCode' not in df.columns:
            print("Error: 'stationShortCode' column not found in dataframe")
            print(f"Available columns: {list(df.columns)}")
            return None
            
        # Log station distribution before filtering
        if initial_stations > 0:
            station_counts = df['stationShortCode'].value_counts().head(10)
            print(f"Found {initial_stations} unique stations. Top 10 stations by frequency:")
            for station, count in station_counts.items():
                print(f"  {station}: {count:,} rows")
                
        # Filter for exact match with target station code
        filtered_df = df[df['stationShortCode'] == TARGET_STATION_CODE].copy()
        
        # Log filtering results
        final_rows = len(filtered_df)
        rows_removed = initial_rows - final_rows
        
        if final_rows == 0:
            print(f"⚠️  WARNING: No data found for station '{TARGET_STATION_CODE}'!")
            print(f"Available stations in dataset: {sorted(df['stationShortCode'].unique().tolist())}")
            return filtered_df  # Return empty dataframe with same structure
            
        print(f"✓ Station filtering completed:")
        print(f"  • Target station: {TARGET_STATION_CODE}")
        print(f"  • Rows kept: {final_rows:,} ({final_rows/initial_rows*100:.1f}%)")
        print(f"  • Rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.1f}%)")
        
        # Optional: Log unique train numbers for this station
        if 'train_id' in filtered_df.columns:
            unique_trains = filtered_df['train_id'].nunique()
            print(f"  • Unique trains at {TARGET_STATION_CODE}: {unique_trains}")
        elif 'trainNumber' in filtered_df.columns:
            unique_trains = filtered_df['trainNumber'].nunique()
            print(f"  • Unique trains at {TARGET_STATION_CODE}: {unique_trains}")
            
        return filtered_df

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
                # FIXED: Enhanced empty value check that handles all edge cases
                try:
                    is_empty = False
                    
                    # Handle pandas NA/NaN (safe for all types)
                    try:
                        if pd.isna(cause_value):
                            is_empty = True
                    except (TypeError, ValueError):
                        # pd.isna() might fail on some complex objects, continue checking
                        pass
                    
                    if not is_empty:
                        # Handle None
                        if cause_value is None:
                            is_empty = True
                        # Handle strings
                        elif isinstance(cause_value, str):
                            is_empty = cause_value.strip() in ["", "[]", "nan", "None", "null"]
                        # Handle lists (this is the key fix for your error)
                        elif isinstance(cause_value, list):
                            is_empty = len(cause_value) == 0
                        # Handle dictionaries
                        elif isinstance(cause_value, dict):
                            is_empty = len(cause_value) == 0
                        # Handle numpy arrays and pandas Series
                        elif hasattr(cause_value, '__len__') and hasattr(cause_value, '__iter__'):
                            try:
                                # For array-like objects, check length safely
                                if hasattr(cause_value, 'shape'):
                                    # numpy array
                                    if cause_value.shape == ():
                                        # Single-element numpy scalar
                                        scalar_value = cause_value.item()
                                        is_empty = pd.isna(scalar_value) or scalar_value is None or str(scalar_value).strip() in ["", "[]", "nan", "None", "null"]
                                    elif cause_value.size == 0:
                                        # Empty array
                                        is_empty = True
                                    else:
                                        # Multi-element array - not empty
                                        is_empty = False
                                else:
                                    # Regular iterable (but not string, dict, or list which we handled above)
                                    try:
                                        is_empty = len(cause_value) == 0
                                    except:
                                        # If len() fails, assume non-empty
                                        is_empty = False
                            except:
                                # If anything fails with array-like objects, assume non-empty
                                is_empty = False
                        # Handle other cases by string conversion
                        else:
                            str_value = str(cause_value).lower().strip()
                            is_empty = str_value in ["", "[]", "nan", "none", "null"]
                            
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
                        for cause_item in parsed_causes:
                            if isinstance(cause_item, dict) and 'detailedCategoryCode' in cause_item:
                                detailed_codes.append(cause_item['detailedCategoryCode'])
                    elif isinstance(parsed_causes, dict) and 'detailedCategoryCode' in parsed_causes:
                        detailed_codes.append(parsed_causes['detailedCategoryCode'])
                    
                    # Store the extracted detailed codes (or None if no codes found)
                    if detailed_codes:
                        processed_values.append(detailed_codes)
                        
                        # Calculate weather-related score based on detailed codes (inline)
                        # Define weather-related category mappings
                        STRONG_WEATHER_CODES = {'I1', 'I2'}  # Strong weather indicators
                        POSSIBLE_WEATHER_CODES = {'A1', 'K1', 'O1', 'P1', 'S1', 'S2', 'T2', 'T3', 'V3'}  # Possible weather indicators
                        
                        weather_score = 0
                        for code in detailed_codes:
                            if code in STRONG_WEATHER_CODES:
                                weather_score = max(weather_score, 3)
                            elif code in POSSIBLE_WEATHER_CODES:
                                weather_score = max(weather_score, 2)
                            elif code:  # Any other non-empty code
                                weather_score = max(weather_score, 1)
                        
                        weather_scores.append(weather_score)
                        
                        # Update statistics
                        if weather_score == 3:
                            stats['weather_strong'] += 1
                        elif weather_score == 2:
                            stats['weather_possible'] += 1
                        elif weather_score == 1:
                            stats['weather_weak'] += 1
                        else:
                            stats['weather_none'] += 1
                        
                        stats['successful_extractions'] += 1
                    else:
                        processed_values.append(None)
                        weather_scores.append(0)
                        stats['weather_none'] += 1
                        stats['failed_extractions'] += 1
                        
                except Exception as e:
                    print(f"Warning: Error processing causes at index {index}: {e}")
                    print(f"Cause value: {cause_value}")
                    processed_values.append(None)
                    weather_scores.append(0)
                    stats['parsing_errors'] += 1
                    stats['weather_none'] += 1
            
            # Update the dataframe with processed values
            df['causes'] = processed_values
            df['causes_related_to_weather'] = weather_scores
            
            # Print processing statistics
            total_processed = len(df)
            print(f"Causes processing completed:")
            print(f"  - Total rows processed: {total_processed:,}")
            print(f"  - Empty values: {stats['empty_values']:,} ({stats['empty_values']/total_processed*100:.1f}%)")
            print(f"  - Successful extractions: {stats['successful_extractions']:,} ({stats['successful_extractions']/total_processed*100:.1f}%)")
            print(f"  - Failed extractions: {stats['failed_extractions']:,} ({stats['failed_extractions']/total_processed*100:.1f}%)")
            print(f"  - Parsing errors: {stats['parsing_errors']:,} ({stats['parsing_errors']/total_processed*100:.1f}%)")
            print(f"Weather-related distribution:")
            print(f"  - Strong weather (3): {stats['weather_strong']:,} ({stats['weather_strong']/total_processed*100:.1f}%)")
            print(f"  - Possible weather (2): {stats['weather_possible']:,} ({stats['weather_possible']/total_processed*100:.1f}%)")
            print(f"  - Weak weather (1): {stats['weather_weak']:,} ({stats['weather_weak']/total_processed*100:.1f}%)")
            print(f"  - No weather indicator (0): {stats['weather_none']:,} ({stats['weather_none']/total_processed*100:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"Critical error in process_causes_column: {e}")
            return None

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
            important_weather_features = [col for col in IMPORTANT_WEATHER_FEATURES if col in df.columns]
            if important_weather_features:
                # Count rows where ALL important weather conditions are filled
                important_weather_complete = df[important_weather_features].notna().all(axis=1).sum()
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
            if important_weather_features:
                print(f"\nMissing data breakdown for important weather conditions:")
                logger.info("Missing data breakdown for important weather conditions:")
                for col in important_weather_features:
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
            
            # Identify all weather-related columns (not just important ones)
            all_weather_cols = [col for col in df.columns if any(weather_condition in col for weather_condition in IMPORTANT_WEATHER_FEATURES)]
            
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
            important_weather_features = [col for col in ALL_WEATHER_FEATURES if col in df.columns]
            
            if not important_weather_features:
                print("Warning: None of the specified important weather conditions found in the dataframe after filtering")
                logger.warning("None of the specified important weather conditions found in the dataframe after filtering")
                return df
            
            print(f"Found {len(important_weather_features)} important weather condition columns after filtering: {important_weather_features}")
            logger.info(f"Found {len(important_weather_features)} important weather condition columns after filtering: {important_weather_features}")
            
            # Store count before dropping weather condition rows
            before_weather_drop = len(df)
            
            # Drop rows where ALL of the remaining important weather conditions are missing
            # (Keep rows with at least one of the specified conditions)
            df = df.dropna(subset=important_weather_features, how='all')
            
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
            important_weather_features = [col for col in df.columns if any(weather_condition in col for weather_condition in ALL_WEATHER_FEATURES)]
        
            
            if important_weather_features:
                print(f"\n--- WEATHER FEATURES IMPUTATION WITH MONTH-SPECIFIC MEDIANS ---")
                logger.info("=== Weather Features Imputation with Month-Specific Medians ===")
                
                # Check if month column exists
                if 'month' not in df.columns:
                    print("Warning: 'month' column not found. Cannot perform month-specific imputation.")
                    logger.warning("'month' column not found. Cannot perform month-specific imputation.")
                    
                    # Fallback to global median for weather features
                    for col in important_weather_features:
                        nulls = df[col].isna().sum()
                        if nulls > 0:
                            percentage = (nulls / len(df)) * 100
                            median_value = df[col].median()
                            df[col] = df[col].fillna(median_value)
                            print(f"- Filled {nulls} missing values in '{col}' with global median: {median_value:.2f} ({percentage:.2f}%)")
                            logger.info(f"Filled {nulls} missing values in '{col}' with global median: {median_value:.2f} ({percentage:.2f}%)")
                else:
                    print(f"Found {len(important_weather_features)} weather-related columns for month-specific median imputation:")
                    for col in important_weather_features:
                        print(f"  - {col}")
                    logger.info(f"Found {len(important_weather_features)} weather-related columns: {important_weather_features}")
                    
                    # Perform month-specific median imputation for each weather column
                    for col in important_weather_features:
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
                                    and not any(weather_condition in col for weather_condition in IMPORTANT_WEATHER_FEATURES)]

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
        