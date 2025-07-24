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
    DATA_FILE_PREFIX_FOR_TRAINING,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_WEATHER_CONDITIONS,
    NON_NUMERIC_FEATURES, 
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
        
        # Step 2: Add train delayed feature if enabled and we have data
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
        
        # Step 3: Save to CSV if enabled and we have data
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