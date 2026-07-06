from contextlib import contextmanager
from datetime import datetime
import glob
import os
import shutil
from typing import Counter
import joblib
import json
import pandas as pd
import re
import ast
import logging
import numpy as np
import psutil
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler

import xgboost as xgb
import lightgbm as lgb

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score,
    f1_score, r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.file_utils import (
    save_dataframe_to_parquet,
)

from config.const import (
    OUTPUT_FOLDER,
)

from config.const_preprocessing import (
    PREPROCESSED_OUTPUT_FOLDER,
    DATA_FILE_PREFIX_FOR_TRAINING,
    DEFAULT_TARGET_FEATURE,
    ALL_WEATHER_FEATURES,
    SKEWED_WEATHER_FEATURES,
    TRAINING_READY_OUTPUT_FOLDER,
    VALID_TARGET_FEATURES,
    REGRESSION_PROBLEM,
    CLASSIFICATION_PROBLEM,
    TRAIN_DELAY_MINUTES,
)

from config.const_training import (
    MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER,
    FILTER_LOWER_QUANTILE,
    FILTER_UPPER_QUANTILE,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_TRAINING_READY_OUTPUT_FOLDER,
    TEST_SIZE,
    RANDOM_STATE,
    SCORE_METRIC,
    RANDOM_SEARCH_ITERATIONS,
    RANDOM_SEARCH_CV_FOLDS,
    WEIGHT_DELAY_COLUMN,
    MAX_SAMPLE_WEIGHT_CLASSIFICATION,
    XGBOOST_PARAM_DISTRIBUTIONS,
    LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    LIGHTGBM_PARAM_DISTRIBUTIONS,
    MAX_SAMPLE_WEIGHT_REGRESSION,
    MERGED_BALANCED_OUTPUT_FOLDER,
    RESAMPLING_METHOD,
    IMBALANCE_THRESHOLD,
    SMOTE_RANDOM_STATE,
    SPLIT_DATASET_OUTPUT_FOLDER,
    SELECTED_COLUMNS,
    SCHEMA_MISMATCH_STRATEGY,
    DELAY_THRESHOLD_MINUTES,
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

        if state_machine.get("filter_delay_outliers", False):
            if result["data"] is not None:
                try:
                    print(f"    → filter_delay_outliers")
                    filter_result = self.filter_delay_outliers(data=result["data"])

                    if filter_result and filter_result.get("success", False):
                        result["data"] = filter_result.get("data")
                        result["steps_executed"].append("filter_delay_outliers")
                        result["success"] = True
                        removed = filter_result.get("rows_removed_lower", 0) + filter_result.get("rows_removed_upper", 0)
                        print(f"      ✓ Removed {removed:,} outlier rows")
                        print(f"      ✓ Rows remaining: {len(result['data']):,}")
                    else:
                        error_msg = filter_result.get("error", "filter_delay_outliers returned unsuccessful result")
                        result["errors"].append(error_msg)
                        print(f"      ✗ Failed - {error_msg}")
                        return result

                except Exception as e:
                    result["errors"].append(f"filter_delay_outliers failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ filter_delay_outliers (no data available)")
        else:
            print(f"    ⊝ filter_delay_outliers (disabled)")


        if state_machine.get("select_training_cols", False):
            try:
                print(f"    → select_training_cols")
                cols_selection_result = self.select_training_cols(
                    data=result["data"] if result["data"] is not None else None
                )
                
                if cols_selection_result and cols_selection_result.get("success", False):
                    result["data"] = cols_selection_result.get("data")
                    result["steps_executed"].append("select_training_cols")
                    result["column_analysis"] = {
                        "total_columns": cols_selection_result.get("total_columns", 0),
                        "file_analyzed": cols_selection_result.get("file_path", ""),
                        "dataset_shape": cols_selection_result.get("dataset_shape", (0, 0)),
                        "column_types": cols_selection_result.get("column_types", {})
                    }
                    print(f"      ✓ Successfully selected training columns")
                    print(f"      ✓ Total columns found: {cols_selection_result.get('total_columns', 0)}")
                    print(f"      ✓ File analyzed: {os.path.basename(cols_selection_result.get('file_path', '') or '')}")
                    result["success"] = True
                else:
                    error_msg = cols_selection_result.get("error", "select_training_cols returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"select_training_cols failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ select_training_cols (disabled)")

        if state_machine.get("split_dataset", False):
            try:
                print(f"    → split_dataset")
                _split_dir = (
                    os.path.join(self.project_root, MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER)
                    if state_machine.get("filter_delay_outliers", False) and not state_machine.get("select_training_cols", False)
                    else None
                )
                split_result = self.split_dataset(
                    data_dir=_split_dir,
                    output_dir=os.path.join(self.project_root, SPLIT_DATASET_OUTPUT_FOLDER),
                )
                
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

        if state_machine.get("balance_classes", False):
            try:
                print(f"    → balance_classes")
                _balance_input_dir = os.path.join(self.project_root, SPLIT_DATASET_OUTPUT_FOLDER)
                balance_result = self.balance_classes(data_dir=_balance_input_dir)

                if balance_result and balance_result.get("success", False):
                    result["steps_executed"].append("balance_classes")
                    result["success"] = True
                    if balance_result.get("skipped", False):
                        print(f"      ✓ Skipped — minority share already meets threshold")
                    else:
                        print(f"      ✓ Rows before: {balance_result.get('rows_before', 0):,}")
                        print(f"      ✓ Rows after:  {balance_result.get('rows_after', 0):,}")
                        print(f"      ✓ Method: {balance_result.get('resampling_method', 'N/A')}")
                else:
                    error_msg = balance_result.get("error", "balance_classes returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"balance_classes failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ balance_classes (disabled)")

        if state_machine.get("scale_weather_features", False):
            try:
                print(f"    → scale_weather_features")
                _use_filtered_for_scale = (
                    state_machine.get("filter_delay_outliers", False)
                    and not state_machine.get("select_training_cols", False)
                )
                _scale_data_folder = (
                    MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered_for_scale
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                scaling_result = self.scale_weather_features(
                    csv_files,
                    data_dir=os.path.join(self.project_root, _scale_data_folder),
                )

                if scaling_result and scaling_result.get("success", False):
                    result["steps_executed"].append("scale_weather_features")
                    result["file_info"]["processed_files"] = scaling_result.get("processed_files", 0)
                    print(f"      ✓ Successfully scaled weather features")
                    print(f"      ✓ Total train rows: {scaling_result.get('total_train_rows', 0):,}")
                    print(f"      ✓ Total test rows: {scaling_result.get('total_test_rows', 0):,}")
                    print(f"      ✓ Weather features scaled: {scaling_result.get('weather_features_scaled', [])}")
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

        if state_machine.get("numeric_correlation_analysis", False):
            try:
                print(f"    → numeric_correlation_analysis (Comprehensive Numeric Correlation Analysis)")
                numeric_correlation_result = self.numeric_correlation_analysis(csv_files)
                
                if numeric_correlation_result and numeric_correlation_result.get("success", False):
                    result["steps_executed"].append("numeric_correlation_analysis")
                    result["file_info"]["numeric_correlation_processed_files"] = numeric_correlation_result.get("processed_files", 0)
                    print(f"      ✓ Successfully completed numeric correlation analysis")
                    print(f"      ✓ Files analyzed: {numeric_correlation_result.get('processed_files', 0)}")
                    print(f"      ✓ Target variable: {numeric_correlation_result.get('target_variable', 'N/A')}")
                    print(f"      ✓ Analysis types: {', '.join(numeric_correlation_result.get('analysis_types', []))}")
                    print(f"      ✓ Results saved to: {numeric_correlation_result.get('output_path', 'N/A')}")
                    
                    # Display summary statistics if available
                    if numeric_correlation_result.get('total_combined_records'):
                        total_records = numeric_correlation_result['total_combined_records']
                        print(f"      ✓ Total records analyzed: {total_records:,}")
                    
                    if numeric_correlation_result.get('combined_analysis'):
                        print(f"      ✓ Combined cross-file analysis completed")
                    
                    result["success"] = True
                else:
                    error_msg = numeric_correlation_result.get("error", "numeric_correlation_analysis returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"numeric_correlation_analysis failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ numeric_correlation_analysis (Comprehensive Numeric Correlation Analysis) (disabled)")
    
        if state_machine.get("data_distribution_analysis", False):
            try:
                print(f"    → data_distribution_analysis (Comprehensive Data Distribution Analysis)")
                distribution_result = self.data_distribution_analysis(csv_files)
                
                if distribution_result and distribution_result.get("success", False):
                    result["steps_executed"].append("data_distribution_analysis")
                    result["file_info"]["distribution_processed_files"] = distribution_result.get("processed_files", 0)
                    print(f"      ✓ Successfully completed data distribution analysis")
                    print(f"      ✓ Files analyzed: {distribution_result.get('processed_files', 0)}")
                    print(f"      ✓ Columns analyzed: {distribution_result.get('total_columns', 0)}")
                    print(f"      ✓ Histograms generated: {distribution_result.get('plots_generated', 0)}")
                    print(f"      ✓ Results saved to: {distribution_result.get('output_path', 'N/A')}")
                    
                    # Display summary statistics if available
                    if distribution_result.get('total_rows'):
                        total_records = distribution_result['total_rows']
                        print(f"      ✓ Total records analyzed: {total_records:,}")
                    
                    result["success"] = True
                else:
                    error_msg = distribution_result.get("error", "data_distribution_analysis returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"data_distribution_analysis failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ data_distribution_analysis (Comprehensive Data Distribution Analysis) (disabled)")

        if state_machine.get("target_feature_analysis", False):
            try:
                print(f"    → target_feature_analysis (Comprehensive Target Feature Analysis)")
                target_analysis_result = self.target_feature_analysis()
                
                if target_analysis_result and target_analysis_result.get("success", False):
                    result["steps_executed"].append("target_feature_analysis")
                    result["file_info"]["target_analysis_processed_files"] = target_analysis_result.get("processed_files", 0)
                    print(f"      ✓ Successfully completed target feature analysis")
                    print(f"      ✓ Files analyzed: {target_analysis_result.get('processed_files', 0)}")
                    print(f"      ✓ Target feature: {target_analysis_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Delay threshold: {target_analysis_result.get('delay_threshold', 'N/A')} minutes")
                    print(f"      ✓ Total samples: {target_analysis_result.get('valid_target_rows', 0):,}")
                    print(f"      ✓ Plots created: {target_analysis_result.get('plots_created', 0)}")
                    print(f"      ✓ Results saved to: {target_analysis_result.get('output_directory', 'N/A')}")
                    
                    # Display key statistics if available
                    if target_analysis_result.get('statistics'):
                        stats = target_analysis_result['statistics']
                        if 'zero_threshold_analysis' in stats:
                            zero_stats = stats['zero_threshold_analysis']
                            print(f"      ✓ Zero threshold (≥0): {zero_stats['delayed_count']:,} delayed ({zero_stats['delayed_percentage']:.1f}%)")
                        
                        if 'train_delay_threshold_analysis' in stats:
                            threshold_stats = stats['train_delay_threshold_analysis']
                            print(f"      ✓ {threshold_stats['threshold']}-min threshold: {threshold_stats['delayed_count']:,} delayed ({threshold_stats['delayed_percentage']:.1f}%)")
                    
                    result["success"] = True
                else:
                    error_msg = target_analysis_result.get("error", "target_feature_analysis returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"target_feature_analysis failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ target_feature_analysis (Comprehensive Target Feature Analysis) (disabled)")

        if state_machine.get("train_xgboost_with_randomized_search_cv", False):
            try:
                print(f"    → train_xgboost_with_randomized_search_cv")
                _use_scaled = state_machine.get("scale_weather_features", False)
                _use_filtered = (
                    state_machine.get("filter_delay_outliers", False)
                    and not state_machine.get("select_training_cols", False)
                )
                _data_folder = (
                    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                xgboost_result = self.train_xgboost_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )
                
                if xgboost_result and xgboost_result.get("success", False):
                    result["steps_executed"].append("train_xgboost_with_randomized_search_cv")
                    result["file_info"]["xgboost_models_trained"] = xgboost_result.get("models_trained", 0)
                    result["file_info"]["xgboost_problem_type"] = xgboost_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained XGBoost models")
                    print(f"      ✓ Problem type: {xgboost_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {xgboost_result.get('models_trained', 0)}")
                    print(f"      ✓ Target feature: {xgboost_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Average CV Score: {xgboost_result.get('cv_score', 0):.4f}")
                    
                    # Print problem-specific metrics
                    if xgboost_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {xgboost_result.get('test_f1', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {xgboost_result.get('test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {xgboost_result.get('test_r2', 0):.4f}")
                        
                    print(f"      ✓ Results saved to: {xgboost_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = xgboost_result.get("error", "train_xgboost_with_randomized_search_cv returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"train_xgboost_with_randomized_search_cv failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_xgboost_with_randomized_search_cv (disabled)")

        if state_machine.get("train_lightgbm_with_randomized_search_cv", False):
            try:
                print(f"    → train_lightgbm_with_randomized_search_cv")
                _use_scaled = state_machine.get("scale_weather_features", False)
                _use_filtered = (
                    state_machine.get("filter_delay_outliers", False)
                    and not state_machine.get("select_training_cols", False)
                )
                _data_folder = (
                    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                lightgbm_result = self.train_lightgbm_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )

                if lightgbm_result and lightgbm_result.get("success", False):
                    result["steps_executed"].append("train_lightgbm_with_randomized_search_cv")
                    result["file_info"]["lightgbm_models_trained"] = lightgbm_result.get("models_trained", 0)
                    result["file_info"]["lightgbm_problem_type"] = lightgbm_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained LightGBM models")
                    print(f"      ✓ Problem type: {lightgbm_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {lightgbm_result.get('models_trained', 0)}")
                    print(f"      ✓ Target feature: {lightgbm_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Average CV Score: {lightgbm_result.get('cv_score', 0):.4f}")

                    if lightgbm_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {lightgbm_result.get('test_f1', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {lightgbm_result.get('test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {lightgbm_result.get('test_r2', 0):.4f}")

                    print(f"      ✓ Results saved to: {lightgbm_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = lightgbm_result.get("error", "train_lightgbm_with_randomized_search_cv returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"train_lightgbm_with_randomized_search_cv failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_lightgbm_with_randomized_search_cv (disabled)")

        return result

    def filter_delay_outliers(self, data=None):
        """
        Remove rows where differenceInMinutes falls outside asymmetric quantile bounds.

        Uses FILTER_LOWER_QUANTILE and FILTER_UPPER_QUANTILE from const_training to
        compute bounds on the full merged dataset, then drops rows outside those bounds.
        Saves the filtered dataframe to MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER for inspection.

        Parameters
        ----------
        data : pd.DataFrame or None
            Merged dataframe from merge_data_files. Required.

        Returns
        -------
        dict
            {
              "success": bool,
              "data": pd.DataFrame | None,
              "rows_before": int,
              "rows_removed_lower": int,
              "rows_removed_upper": int,
              "lower_bound": float,
              "upper_bound": float,
            }
            On missing column: success=True, data unchanged, removal counts are 0.
            On data=None: success=False, error key present.
        """
        if data is None:
            print("    filter_delay_outliers: data is None — skipping")
            return {"success": False, "error": "data is None", "data": None}

        target_col = "differenceInMinutes"

        if target_col not in data.columns:
            print(f"    filter_delay_outliers: '{target_col}' not found — returning data unchanged")
            return {
                "success": True,
                "data": data,
                "rows_before": len(data),
                "rows_removed_lower": 0,
                "rows_removed_upper": 0,
                "lower_bound": None,
                "upper_bound": None,
            }

        df = data.copy()
        rows_before = len(df)

        lower_bound = float(df[target_col].quantile(FILTER_LOWER_QUANTILE))
        upper_bound = float(df[target_col].quantile(FILTER_UPPER_QUANTILE))

        lower_mask = df[target_col] < lower_bound
        upper_mask = df[target_col] > upper_bound

        rows_removed_lower = int(lower_mask.sum())
        rows_removed_upper = int(upper_mask.sum())

        df = df[~lower_mask & ~upper_mask].copy()
        rows_after = len(df)

        print(f"\n    filter_delay_outliers: Filtering {rows_before:,} rows...")
        print(f"      Lower bound (q={FILTER_LOWER_QUANTILE}): {lower_bound:.2f} min")
        print(f"      Upper bound (q={FILTER_UPPER_QUANTILE}): {upper_bound:.2f} min")
        if rows_before > 0:
            print(f"      Removed (lower tail): {rows_removed_lower:,} ({rows_removed_lower / rows_before * 100:.2f}%)")
            print(f"      Removed (upper tail): {rows_removed_upper:,} ({rows_removed_upper / rows_before * 100:.2f}%)")
            print(f"      Rows after: {rows_after:,} ({rows_after / rows_before * 100:.2f}% kept)")
        else:
            print(f"      Input was empty — nothing to filter.")

        if df.empty:
            print("⚠️  Warning: All rows removed after outlier filtering — check quantile thresholds.")

        output_folder = os.path.join(self.project_root, MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER)

        print(f"    filter_delay_outliers: Saving filtered data...")
        try:
            saved_path = save_dataframe_to_parquet(
                folder_path=output_folder,
                month_id="outlier_filtered",
                df=df,
                file_prefix="merged_data",
            )
            print(f"      ✓ Saved filtered data to: {saved_path}")
        except Exception as save_error:
            print(f"      ⚠️  Warning: Failed to save filtered data: {save_error}")
            print("      Continuing with in-memory filtered data.")

        print(f"    filter_delay_outliers: Saving dropped rows log...")
        try:
            dropped_lower = data[lower_mask].copy()
            dropped_lower["dropped_tail"] = "lower"
            dropped_upper = data[upper_mask].copy()
            dropped_upper["dropped_tail"] = "upper"
            dropped_df = pd.concat([dropped_lower, dropped_upper], ignore_index=True)
            os.makedirs(output_folder, exist_ok=True)
            csv_path = os.path.join(output_folder, "dropped_rows.csv")
            dropped_df.to_csv(csv_path, index=False)
            print(f"      ✓ Saved {len(dropped_df):,} dropped rows to: {csv_path}")
        except Exception as csv_error:
            print(f"      ⚠️  Warning: Failed to save dropped rows log: {csv_error}")
            print("      Continuing without dropped rows log.")

        return {
            "success": True,
            "data": df,
            "rows_before": rows_before,
            "rows_removed_lower": rows_removed_lower,
            "rows_removed_upper": rows_removed_upper,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def balance_classes(self, data_dir=None):
        if data_dir is None:
            print("    balance_classes: data_dir is None — skipping")
            return {"success": False, "error": "data_dir is None"}

        train_files = glob.glob(os.path.join(data_dir, "*_train.parquet"))
        test_files = glob.glob(os.path.join(data_dir, "*_test.parquet"))

        if not train_files:
            msg = f"No *_train.parquet found in {data_dir}"
            print(f"    balance_classes: {msg}")
            return {"success": False, "error": msg}

        train_path = train_files[0]
        df = pd.read_parquet(train_path)
        rows_before = len(df)

        target_col = DEFAULT_TARGET_FEATURE
        is_classification = target_col in CLASSIFICATION_PROBLEM
        is_regression = target_col in REGRESSION_PROBLEM

        if not (is_classification or is_regression):
            msg = f"Target feature '{target_col}' not recognized as classification or regression problem"
            print(f"    balance_classes: {msg}")
            return {"success": False, "error": msg}

        if target_col == "trainDelayed":
            counterpart_col = "differenceInMinutes"
        elif is_regression:
            counterpart_col = "trainDelayed"
        else:
            counterpart_col = None

        output_folder = os.path.join(self.project_root, MERGED_BALANCED_OUTPUT_FOLDER)
        os.makedirs(output_folder, exist_ok=True)

        # Copy test file(s) unchanged regardless of whether we resample
        test_output_path = None
        if test_files:
            test_src = test_files[0]
            test_output_path = os.path.join(output_folder, os.path.basename(test_src))
            shutil.copy2(test_src, test_output_path)
            print(f"    balance_classes: Copied test file to: {test_output_path}")

        dropped_counterpart_col = None
        if counterpart_col and counterpart_col in df.columns:
            print(f"    balance_classes: Dropping counterpart column '{counterpart_col}' to prevent leakage")
            df = df.drop(columns=[counterpart_col])
            dropped_counterpart_col = counterpart_col

        if target_col not in df.columns:
            print(f"    balance_classes: '{target_col}' not found — saving train unchanged")
            train_out = os.path.join(output_folder, os.path.basename(train_path))
            df.to_parquet(train_out, index=False)
            return {
                "success": True,
                "rows_before": rows_before,
                "rows_after": rows_before,
                "minority_share_before": None,
                "minority_share_after": None,
                "resampling_method": "NONE",
                "skipped": True,
                "dropped_non_numeric_cols": [],
                "dropped_counterpart_col": dropped_counterpart_col,
                "train_output_path": train_out,
                "test_output_path": test_output_path,
            }

        if is_classification:
            y = df[target_col].astype(int)
        else:
            y = (df[target_col] > TRAIN_DELAY_MINUTES).astype(int)

        class_counts = y.value_counts()
        total = len(y)
        minority_share = int(class_counts.min()) / total * 100

        print(f"\n    balance_classes: Class balance before resampling:")
        print(f"      Punctual (≤ {TRAIN_DELAY_MINUTES} min): {int(class_counts.get(0, 0)):,} ({int(class_counts.get(0, 0)) / total * 100:.1f}%)")
        print(f"      Delayed  (> {TRAIN_DELAY_MINUTES} min): {int(class_counts.get(1, 0)):,} ({int(class_counts.get(1, 0)) / total * 100:.1f}%)")
        print(f"      Minority share: {minority_share:.1f}% (threshold: {IMBALANCE_THRESHOLD}%)")

        train_out = os.path.join(output_folder, os.path.basename(train_path))

        if minority_share >= IMBALANCE_THRESHOLD:
            print(f"    balance_classes: Balance acceptable — saving train unchanged")
            df.to_parquet(train_out, index=False)
            return {
                "success": True,
                "rows_before": rows_before,
                "rows_after": rows_before,
                "minority_share_before": minority_share,
                "minority_share_after": minority_share,
                "resampling_method": "NONE",
                "skipped": True,
                "dropped_non_numeric_cols": [],
                "dropped_counterpart_col": dropped_counterpart_col,
                "train_output_path": train_out,
                "test_output_path": test_output_path,
            }

        feature_source_cols = [c for c in df.columns if c != target_col] if is_classification else list(df.columns)
        bool_cols = [c for c in feature_source_cols if df[c].dtype == bool]
        for c in bool_cols:
            df[c] = df[c].astype(int)

        numeric_cols = [c for c in feature_source_cols if c in df.select_dtypes(include=[np.number]).columns]
        non_numeric_cols = [c for c in feature_source_cols if c not in numeric_cols]
        if non_numeric_cols:
            print(f"    balance_classes: Dropping {len(non_numeric_cols)} non-numeric column(s): {non_numeric_cols}")

        X = df[numeric_cols].copy()
        nan_mask = X.notna().all(axis=1)
        rows_with_nan = int((~nan_mask).sum())
        if rows_with_nan > 0:
            print(f"    balance_classes: Dropping {rows_with_nan:,} rows with NaN values before resampling")
            X = X[nan_mask]
            y = y[nan_mask]

        if RESAMPLING_METHOD == "SMOTE_TOMEK":
            categorical_cols = [
                c for c in numeric_cols
                if c in bool_cols
                or c.startswith("weather_scenario_")
                or c.endswith("_sin")
                or c.endswith("_cos")
            ]
            categorical_indices = [numeric_cols.index(c) for c in categorical_cols]

            if categorical_indices:
                print(f"    balance_classes: Applying SMOTENC (random_state={SMOTE_RANDOM_STATE}, categorical_features={len(categorical_indices)})...")
                smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=SMOTE_RANDOM_STATE)
                X_over, y_over = smote_nc.fit_resample(X, y)
            else:
                print(f"    balance_classes: No categorical/boolean columns detected — applying plain SMOTE (random_state={SMOTE_RANDOM_STATE})...")
                smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
                X_over, y_over = smote.fit_resample(X, y)

            print(f"    balance_classes: Applying TomekLinks cleaning...")
            tomek = TomekLinks()
            X_res, y_res = tomek.fit_resample(X_over, y_over)
            used_method = "SMOTE_TOMEK"
        else:
            print(f"    balance_classes: RESAMPLING_METHOD='{RESAMPLING_METHOD}' not handled — saving unchanged")
            X_res, y_res = X.values, y.values
            used_method = "NONE"

        df_balanced = pd.DataFrame(X_res, columns=numeric_cols)
        if is_classification:
            df_balanced[target_col] = np.asarray(y_res)
        rows_after = len(df_balanced)

        y_after = pd.Series(y_res)
        counts_after = y_after.value_counts()
        minority_share_after = int(counts_after.min()) / len(y_after) * 100

        print(f"\n    balance_classes: Class balance after resampling:")
        print(f"      Punctual (≤ {TRAIN_DELAY_MINUTES} min): {int(counts_after.get(0, 0)):,} ({int(counts_after.get(0, 0)) / len(y_after) * 100:.1f}%)")
        print(f"      Delayed  (> {TRAIN_DELAY_MINUTES} min): {int(counts_after.get(1, 0)):,} ({int(counts_after.get(1, 0)) / len(y_after) * 100:.1f}%)")
        print(f"      Rows: {rows_before:,} → {rows_after:,}")

        df_balanced.to_parquet(train_out, index=False)
        print(f"      Saved balanced train to: {train_out}")

        return {
            "success": True,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "minority_share_before": minority_share,
            "minority_share_after": minority_share_after,
            "resampling_method": used_method,
            "skipped": False,
            "dropped_non_numeric_cols": non_numeric_cols,
            "dropped_counterpart_col": dropped_counterpart_col,
            "train_output_path": train_out,
            "test_output_path": test_output_path,
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
            
            # Find all training-ready parquet files using glob pattern
            training_ready_pattern = os.path.join(self.project_root, TRAINING_READY_OUTPUT_FOLDER, "training_ready_*.parquet")
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
                    # Expected format: training_ready_YYYY_MM.parquet
                    month_match = re.search(r'training_ready_(\d{4})_(\d{2})\.parquet$', filename)
                    
                    if not month_match:
                        print(f"    merge_data_files: Warning - Could not extract date from {filename}. Skipping.")
                        continue
                    
                    year = int(month_match.group(1))
                    month_number = int(month_match.group(2))

                    # Read the parquet file
                    df = pd.read_parquet(file_path)
                    
                    if df.empty:
                        print(f"    merge_data_files: Warning - File {filename} is empty. Skipping.")
                        continue
                    
                    print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")

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

            # Schema consistency check — find columns absent from at least one file
            all_col_sets = [set(df.columns) for df in all_dataframes]
            common_cols_set = set.intersection(*all_col_sets)
            # Preserve original column order from the source files
            common_cols = [col for col in all_dataframes[0].columns if col in common_cols_set]
            inconsistent_cols = sorted(set.union(*all_col_sets) - common_cols_set)

            if inconsistent_cols:
                print(f"    merge_data_files: The following {len(inconsistent_cols)} columns are not present in all files:")
                for col in inconsistent_cols:
                    print(f"      - {col}")
                if SCHEMA_MISMATCH_STRATEGY == 'intersect':
                    proceed = True
                elif SCHEMA_MISMATCH_STRATEGY == 'fail':
                    proceed = False
                else:
                    response = input(f"    Drop these columns and merge with {len(common_cols)} common columns? (y/n): ").strip().lower()
                    proceed = response == 'y'
                if proceed:
                    all_dataframes = [df[common_cols] for df in all_dataframes]
                    for info in file_info:
                        info['columns'] = len(common_cols)
                    print(f"    merge_data_files: Proceeding with {len(common_cols)} common columns.")
                else:
                    error_msg = (
                        f"Schema mismatch: {len(inconsistent_cols)} columns not present in all files: "
                        f"{inconsistent_cols}\n"
                        f"Fix source files so all have identical columns before merging."
                    )
                    return {
                        "success": False,
                        "error": error_msg,
                        "processed_files": 0
                    }

            # Merge all dataframes
            print(f"    merge_data_files: Merging {len(all_dataframes)} dataframes...")
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            print(f"    merge_data_files: Merged dataset shape: {merged_df.shape}")

            # Compute distribution stats from file_info without touching the dataframe
            month_distribution = {}
            year_distribution = {}
            for info in file_info:
                month_distribution[info['month']] = month_distribution.get(info['month'], 0) + info['rows']
                year_distribution[info['year']] = year_distribution.get(info['year'], 0) + info['rows']

            # Generate output filename
            sorted_files = sorted(file_info, key=lambda x: (x['year'], x['month']))
            first_file = sorted_files[0]
            last_file = sorted_files[-1]

            # Format: merged_data_YYYY-MM_to_YYYY-MM.parquet
            output_filename = f"merged_data_{first_file['year']}-{first_file['month']:02d}_to_{last_file['year']}-{last_file['month']:02d}.parquet"
            output_path = os.path.join(merged_training_ready_dir, output_filename)

            # Save merged dataset
            merged_df.to_parquet(output_path, index=False)
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
                for month, count in sorted(month_distribution.items()):
                    f.write(f"Month {month:02d}: {count:,} rows\n")

                # Year distribution
                f.write("\nYear distribution:\n")
                f.write("-" * 20 + "\n")
                for year_key, count in sorted(year_distribution.items()):
                    f.write(f"Year {year_key}: {count:,} rows\n")
                
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
                "month_distribution": month_distribution,
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

    def select_training_cols(self, data=None, original_file_path=None):
        """
        Select training columns interactively by displaying all columns and allowing 
        user to choose which ones to keep for training by entering comma-separated numbers.
        
        This method finds the most recent merged data file, displays all column names 
        with numbers, and allows the user to select which columns to use for training
        by entering their numbers separated by commas. The unselected columns are dropped.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            Pre-loaded dataframe (not used - method discovers files automatically)
        original_file_path : str, optional
            Path to specific file to analyze (optional override)
            
        Returns:
        --------
        dict
            Results including success status, filtered data, and column selection info
        """
        try:
            print(f"    select_training_cols: Starting column selection...")

            # Input source priority:
            # 1. in-memory DataFrame (data=)
            # 2. explicit file path (original_file_path=)
            # 3. disk discovery: 501/ then 500/
            if data is not None:
                df = data.copy()
                file_path = original_file_path  # may be None
                print(f"    select_training_cols: Using in-memory DataFrame ({len(df):,} rows)")
            elif original_file_path and os.path.exists(original_file_path):
                file_path = original_file_path
                print(f"    select_training_cols: Using specified file: {os.path.basename(file_path)}")
                try:
                    df = pd.read_parquet(file_path)
                except Exception as e:
                    return {"success": False, "error": f"Failed to load parquet file: {str(e)}", "file_path": file_path}
            else:
                file_path = None
                for folder_const in [MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER, MERGED_TRAINING_READY_OUTPUT_FOLDER]:
                    search_dir = os.path.join(self.project_root, folder_const)
                    if not os.path.isdir(search_dir):
                        continue
                    found = [
                        f for f in glob.glob(os.path.join(search_dir, "merged_data_*.parquet"))
                        if not (f.endswith("_train.parquet") or f.endswith("_test.parquet"))
                    ]
                    if found:
                        file_path = max(found, key=os.path.getmtime)
                        print(f"    select_training_cols: Found file: {os.path.basename(file_path)} (from {folder_const})")
                        break
                if file_path is None:
                    error_msg = "No merged data files found in 501/ or 500/"
                    print(f"    select_training_cols: {error_msg}")
                    return {"success": False, "error": error_msg}
                try:
                    df = pd.read_parquet(file_path)
                except Exception as e:
                    return {"success": False, "error": f"Failed to load parquet file: {str(e)}", "file_path": file_path}
            
            original_shape = df.shape
            column_names = list(df.columns)
            total_columns = len(column_names)
            
            print(f"    select_training_cols: Loaded dataset with shape: {original_shape}")
            print(f"")
            
            # Display all columns with numbers for selection
            print(f"    🎯 COLUMN SELECTION FOR TRAINING:")
            print(f"    " + "="*80)
            print(f"    Please select columns by entering their numbers separated by commas.")
            print(f"    Total columns available: {total_columns}")
            print(f"")
            
            # Analyze column types for summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            boolean_cols = [col for col in df.columns if df[col].dtype == 'bool']
            
            # Display columns in a numbered list format
            print(f"    📋 AVAILABLE COLUMNS:")
            print(f"    " + "-"*80)
            
            for i, col in enumerate(column_names):
                col_type = str(df[col].dtype)
                non_null_count = df[col].count()
                null_count = len(df) - non_null_count
                null_percentage = (null_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                print(f"    {i+1:3d}. {col:<30} | Type: {col_type:<10} | "
                    f"Non-null: {non_null_count:>6} | Nulls: {null_percentage:5.1f}% | "
                    f"Unique: {unique_count:>5}")
            
            print(f"    " + "-"*80)
            print(f"")
            
            # Display column type summary
            print(f"    🔍 COLUMN TYPE SUMMARY:")
            print(f"    📊 Numeric: {len(numeric_cols)} columns")
            print(f"    📝 Categorical: {len(categorical_cols)} columns")
            print(f"    📅 Datetime: {len(datetime_cols)} columns")
            print(f"    ✓ Boolean: {len(boolean_cols)} columns")
            print(f"")
            
            # Column selection — config bypass or interactive
            if SELECTED_COLUMNS:
                missing = [col for col in SELECTED_COLUMNS if col not in column_names]
                if missing:
                    return {
                        "success": False,
                        "error": f"SELECTED_COLUMNS references columns not in DataFrame: {missing}",
                    }
                selected_indices = sorted([column_names.index(col) for col in SELECTED_COLUMNS])
                print(f"    select_training_cols: Applying {len(SELECTED_COLUMNS)} columns from SELECTED_COLUMNS config")
            else:
                # Interactive selection
                while True:
                    try:
                        print(f"    💡 SELECTION INSTRUCTIONS:")
                        print(f"    • Enter column numbers separated by commas (e.g., 1,3,5,8,12)")
                        print(f"    • Example: To select columns 1, 5, 7, and 10, type: 1,5,7,10")
                        print(f"")

                        user_input = input("    Enter column numbers: ").strip()

                        if not user_input:
                            print(f"    ❌ Please enter at least one column number.")
                            continue

                        # Parse comma-separated numbers
                        try:
                            selected_indices = []
                            numbers = [num.strip() for num in user_input.split(',')]

                            for num_str in numbers:
                                if num_str:  # Skip empty strings
                                    col_num = int(num_str)
                                    if 1 <= col_num <= total_columns:
                                        selected_indices.append(col_num - 1)  # Convert to 0-based index
                                    else:
                                        print(f"    ❌ Invalid column number: {col_num}. Must be between 1 and {total_columns}")
                                        selected_indices = []
                                        break

                            if selected_indices:
                                # Remove duplicates and sort
                                selected_indices = sorted(list(set(selected_indices)))
                                break
                            else:
                                print(f"    ❌ No valid columns selected. Please try again.")
                                continue

                        except ValueError:
                            print(f"    ❌ Invalid input. Please enter only numbers separated by commas.")
                            continue

                    except KeyboardInterrupt:
                        print(f"\n    ⚠️  Operation cancelled by user.")
                        return {
                            "success": False,
                            "error": "Operation cancelled by user"
                        }
                    except Exception as e:
                        print(f"    ❌ Error processing input: {str(e)}")
                        continue
            
            # Apply column selection
            selected_columns = [column_names[i] for i in selected_indices]
            dropped_columns = [col for col in column_names if col not in selected_columns]
            
            print(f"")
            print(f"    ✅ COLUMN SELECTION SUMMARY:")
            print(f"    " + "="*60)
            print(f"    📊 Total columns: {total_columns}")
            print(f"    ✅ Selected: {len(selected_columns)} columns")
            print(f"    ❌ Dropped: {len(dropped_columns)} columns")
            print(f"")
            
            print(f"    📋 SELECTED COLUMNS ({len(selected_columns)}):")
            for i, col in enumerate(selected_columns, 1):
                print(f"    {i:3d}. {col}")
            
            if dropped_columns:
                print(f"")
                print(f"    🗑️  DROPPED COLUMNS ({len(dropped_columns)}):")
                for i, col in enumerate(dropped_columns, 1):
                    print(f"    {i:3d}. {col}")
            
            print(f"")
            
            # Create filtered dataframe
            df_filtered = df[selected_columns].copy()
            final_shape = df_filtered.shape
            
            print(f"    📏 DATASET TRANSFORMATION:")
            print(f"    Original shape: {original_shape}")
            print(f"    Final shape: {final_shape}")
            print(f"")
            
            # Save the filtered dataset to the selected training ready folder
            try:
                # Create the selected training ready output directory
                selected_training_ready_dir = os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
                os.makedirs(selected_training_ready_dir, exist_ok=True)
                
                # Generate output filename based on original file
                original_filename = os.path.basename(file_path) if file_path else "merged_data_selected.parquet"
                output_filename = original_filename  # already .parquet
                output_path = os.path.join(selected_training_ready_dir, output_filename)

                df_filtered.to_parquet(output_path, index=False)
                print(f"    💾 Saved filtered dataset to: {output_filename}")
                print(f"    📁 Location: {MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER}")
            except Exception as save_error:
                print(f"    ⚠️  Warning: Could not save filtered dataset: {str(save_error)}")
                output_path = None
            
            print(f"")
            
            # Prepare results
            result = {
                "success": True,
                "data": df_filtered,
                "file_path": file_path,
                "output_path": output_path,
                "original_shape": original_shape,
                "final_shape": final_shape,
                "total_columns": total_columns,
                "selected_columns": selected_columns,
                "dropped_columns": dropped_columns,
                "column_selection_summary": {
                    "total_available": total_columns,
                    "selected_count": len(selected_columns),
                    "dropped_count": len(dropped_columns)
                },
                "column_types": {
                    "numeric": len([col for col in selected_columns if col in numeric_cols]),
                    "categorical": len([col for col in selected_columns if col in categorical_cols]),
                    "datetime": len([col for col in selected_columns if col in datetime_cols]),
                    "boolean": len([col for col in selected_columns if col in boolean_cols])
                },
                "message": f"Successfully selected {len(selected_columns)} out of {total_columns} columns for training"
            }
            
            print(f"    ✅ Column selection completed successfully!")
            print(f"")
            
            return result
            
        except Exception as e:
            error_msg = f"select_training_cols failed: {str(e)}"
            print(f"    select_training_cols: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def split_dataset(self, data_dir=None, output_dir=None, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify_column=None):
        """
        Split merged training dataset into train and test sets.
        
        This method checks if exactly one CSV file exists in data/output/merged_training_ready,
        and if so, splits that dataset into training and testing sets, saving them as separate files.
        Automatically uses stratified splitting for classification problems based on the target feature.
        
        Parameters:
        -----------
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
            merged_training_ready_dir = data_dir if data_dir is not None else os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(merged_training_ready_dir, exist_ok=True)
            
            # Get all parquet files in the directory
            all_parquet_files = [f for f in os.listdir(merged_training_ready_dir)
                                 if f.endswith('.parquet') and os.path.isfile(os.path.join(merged_training_ready_dir, f))]

            # Filter out existing train/test files to get only source files
            source_parquet_files = [f for f in all_parquet_files if not (f.endswith('_train.parquet') or f.endswith('_test.parquet'))]

            # Check if exactly one parquet file exists
            if len(source_parquet_files) == 0:
                error_msg = "No parquet files found in the directory"
                print(f"    split_dataset: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            elif len(source_parquet_files) > 1:
                error_msg = f"Expected exactly one parquet file, but found {len(source_parquet_files)}: {', '.join(source_parquet_files)}"
                print(f"    split_dataset: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0,
                    "found_files": source_parquet_files
                }

            # Proceed with the single parquet file
            csv_filename = source_parquet_files[0]
            file_path = os.path.join(merged_training_ready_dir, csv_filename)

            print(f"    split_dataset: Found exactly one parquet file: {csv_filename}")
            print(f"    split_dataset: Processing {csv_filename}...")

            # Read the dataset
            df = pd.read_parquet(file_path)
            
            if df.empty:
                error_msg = f"File {csv_filename} is empty"
                print(f"    split_dataset: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"      Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Detect target column for stratification
            target_column = DEFAULT_TARGET_FEATURE if DEFAULT_TARGET_FEATURE in df.columns else None
            if stratify_column:
                target_column = stratify_column if stratify_column in df.columns else target_column
            
            # Determine if this is a classification problem
            is_classification = False
            stratify = None
            
            if target_column and target_column in df.columns:
                is_classification = target_column in CLASSIFICATION_PROBLEM

                if is_classification:
                    stratify = df[target_column]
                    print(f"      Detected classification problem with target '{target_column}'")
                    
                    # Calculate class distribution before split
                    class_distribution_before = {}
                    value_counts = df[target_column].value_counts()
                    total_samples = len(df)
                    
                    for class_value, count in value_counts.items():
                        class_distribution_before[class_value] = {
                            'count': int(count),
                            'percentage': (count / total_samples) * 100
                        }
                        
                    print(f"      Class distribution:")
                    for class_value, stats in class_distribution_before.items():
                        print(f"        Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)")
                else:
                    print(f"      Detected regression problem with target '{target_column}'")
            else:
                print(f"      No target column detected - using simple random split")
            
           
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            print(f"      Split results:")
            print(f"        Train set: {len(train_df):,} samples ({(len(train_df) / len(df)) * 100:.1f}%)")
            print(f"        Test set: {len(test_df):,} samples ({(len(test_df) / len(df)) * 100:.1f}%)")
            
            # Generate output filenames
            base_name = csv_filename.replace('.parquet', '')
            train_filename = f"{base_name}_train.parquet"
            test_filename = f"{base_name}_test.parquet"

            split_output_dir = output_dir if output_dir is not None else merged_training_ready_dir
            os.makedirs(split_output_dir, exist_ok=True)

            train_path = os.path.join(split_output_dir, train_filename)
            test_path = os.path.join(split_output_dir, test_filename)

            # Save train and test sets
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path, index=False)
            
            print(f"      Saved train set to: {train_filename}")
            print(f"      Saved test set to: {test_filename}")
            
            # Calculate class distribution after split for classification problems
            class_distribution_train = {}
            class_distribution_test = {}
            
            if is_classification and target_column:
                # Train set distribution
                train_value_counts = train_df[target_column].value_counts()
                for class_value, count in train_value_counts.items():
                    class_distribution_train[class_value] = {
                        'count': int(count),
                        'percentage': (count / len(train_df)) * 100
                    }
                
                # Test set distribution
                test_value_counts = test_df[target_column].value_counts()
                for class_value, count in test_value_counts.items():
                    class_distribution_test[class_value] = {
                        'count': int(count),
                        'percentage': (count / len(test_df)) * 100
                    }
            
            # Prepare result data
            result_data = {
                'filename': csv_filename,
                'original_rows': len(df),
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'train_file': train_filename,
                'test_file': test_filename,
                'test_size_requested': test_size,
                'test_size_actual': len(test_df) / len(df),
                'target_column': target_column,
                'is_classification': is_classification
            }
            
            # Add class distribution data for classification problems
            if is_classification:
                result_data['class_distribution_before'] = class_distribution_before
                result_data['class_distribution_train'] = class_distribution_train
                result_data['class_distribution_test'] = class_distribution_test
            
            print(f"      Successfully processed {csv_filename}")
            
            # Save summary information
            summary_filename = "split_summary.txt"
            summary_path = os.path.join(split_output_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write("Dataset Split Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Split timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input file: {csv_filename}\n")
                f.write(f"Test size: {test_size}\n")
                f.write(f"Random state: {random_state}\n")
                f.write(f"Stratify column: {target_column}\n\n")
                
                f.write(f"Dataset Information:\n")
                f.write(f"  Original rows: {result_data['original_rows']:,}\n")
                f.write(f"  Columns: {df.shape[1]}\n")
                f.write(f"  Problem type: {'Classification' if is_classification else 'Regression'}\n\n")
                
                f.write(f"Split Results:\n")
                f.write(f"  Train set: {result_data['train_rows']:,} samples ({(result_data['train_rows'] / result_data['original_rows']) * 100:.1f}%)\n")
                f.write(f"  Test set: {result_data['test_rows']:,} samples ({(result_data['test_rows'] / result_data['original_rows']) * 100:.1f}%)\n")
                f.write(f"  Actual test size: {result_data['test_size_actual']:.3f}\n\n")
                
                f.write(f"Output Files:\n")
                f.write(f"  Train: {result_data['train_file']}\n")
                f.write(f"  Test: {result_data['test_file']}\n\n")
                
                # Add class distribution info for classification
                if is_classification:
                    f.write(f"Class Distribution Analysis:\n")
                    f.write(f"  Target column: {target_column}\n\n")
                    
                    # Before split
                    f.write(f"  Before Split ({result_data['original_rows']:,} samples):\n")
                    for class_value, stats in class_distribution_before.items():
                        f.write(f"    Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)\n")
                    
                    # After split - Train set
                    f.write(f"\n  After Split - Train Set ({result_data['train_rows']:,} samples):\n")
                    for class_value, stats in class_distribution_train.items():
                        f.write(f"    Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)\n")
                    
                    # After split - Test set
                    f.write(f"\n  After Split - Test Set ({result_data['test_rows']:,} samples):\n")
                    for class_value, stats in class_distribution_test.items():
                        f.write(f"    Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)\n")
            
            print(f"    split_dataset: Summary saved to {summary_filename}")
            
            # Return success result
            result = {
                "success": True,
                "processed_files": 1,
                "total_train_rows": len(train_df),
                "total_test_rows": len(test_df),
                "test_size": test_size,
                "split_details": [result_data],
                "summary_path": summary_path,
                "message": f"Successfully split dataset {csv_filename} into train/test sets"
            }
            
            print(f"    split_dataset: Completed successfully - dataset split")
            
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

    def scale_weather_features(self, csv_files=None, data_dir=None):
        """
        Scale and normalize weather features using RobustScaler.
        
        This method finds the single train/test split file pair, fits a RobustScaler on the weather 
        features from training data only, then transforms both train and test sets using 
        the training parameters. Saves scaled datasets to the scaled output folder.
        
        If no weather features are found, it copies the train/test files as-is from the
        source directory to the destination directory.
        
        Modified to:
        1. Load data from MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
        2. Process only a single train/test file pair (no loop)
        3. Copy files if no weather features found (instead of returning error)
        
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
            
            merged_selected_training_ready_dir = (
                data_dir if data_dir is not None
                else os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
            )
            train_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_train.parquet")
            test_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_test.parquet")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if len(train_files) == 0:
                error_msg = "No training files found to scale"
                print(f"    scale_weather_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            if len(train_files) > 1:
                error_msg = f"Multiple training files found ({len(train_files)}). Expected exactly one train/test file pair."
                print(f"    scale_weather_features: {error_msg}")
                print(f"    scale_weather_features: Found files: {[os.path.basename(f) for f in train_files]}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            if len(test_files) == 0:
                error_msg = "No test files found to scale"
                print(f"    scale_weather_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            if len(test_files) > 1:
                error_msg = f"Multiple test files found ({len(test_files)}). Expected exactly one train/test file pair."
                print(f"    scale_weather_features: {error_msg}")
                print(f"    scale_weather_features: Found files: {[os.path.basename(f) for f in test_files]}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            # Get the single train/test file pair
            train_file_path = train_files[0]
            test_file_path = test_files[0]
            
            train_filename = os.path.basename(train_file_path)
            test_filename = os.path.basename(test_file_path)
            
            print(f"    scale_weather_features: Processing file pair:")
            print(f"      Train: {train_filename}")
            print(f"      Test: {test_filename}")
            
            # Load datasets
            try:
                train_df = pd.read_parquet(train_file_path)
                test_df = pd.read_parquet(test_file_path)
            except Exception as e:
                error_msg = f"Error loading datasets: {str(e)}"
                print(f"    scale_weather_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    scale_weather_features: Train dataset shape: {train_df.shape}")
            print(f"    scale_weather_features: Test dataset shape: {test_df.shape}")
            
            # Identify weather features that exist in the dataset
            available_weather_features = [col for col in ALL_WEATHER_FEATURES if col in train_df.columns]

            # Also include rolling window columns (12h, 24h, 72h) — same pattern as filter_columns
            window_patterns = ('(12h', '(24h', '(72h')
            available_window_features = [col for col in train_df.columns if any(p in col for p in window_patterns)]
            available_weather_features = available_weather_features + available_window_features

            # Zero-inflated/right-skewed features (and their rolling-window derivatives,
            # e.g. "Precipitation amount (24h cumulative)") get a log1p pre-transform
            # before RobustScaler so the heavy right tail doesn't dominate the scale.
            skewed_cols = [
                col for col in available_weather_features
                if any(col == base or col.startswith(base) for base in SKEWED_WEATHER_FEATURES)
            ]

            # NEW LOGIC: Handle case when no weather features are found
            if not available_weather_features:
                print(f"    scale_weather_features: No weather features found in {train_filename}")
                print(f"    scale_weather_features: Copying files as-is from source to scaled directory...")
                
                # Generate output filenames (keep same naming convention)
                scaled_train_filename = train_filename  # already .parquet
                scaled_test_filename = test_filename    # already .parquet
                
                scaled_train_path = os.path.join(scaled_training_ready_dir, scaled_train_filename)
                scaled_test_path = os.path.join(scaled_training_ready_dir, scaled_test_filename)
                
                try:
                    # Copy train file
                    shutil.copy2(train_file_path, scaled_train_path)
                    print(f"    scale_weather_features: ✓ Copied train file to: {scaled_train_filename}")
                    
                    # Copy test file
                    shutil.copy2(test_file_path, scaled_test_path)
                    print(f"    scale_weather_features: ✓ Copied test file to: {scaled_test_filename}")
                    
                except Exception as e:
                    error_msg = f"Error copying files: {str(e)}"
                    print(f"    scale_weather_features: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "processed_files": 0
                    }
                
                # Prepare result for no weather features case
                no_scaling_result = {
                    "original_train_file": train_filename,
                    "original_test_file": test_filename,
                    "scaled_train_file": scaled_train_filename,
                    "scaled_test_file": scaled_test_filename,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "weather_features_scaled": [],  # Empty list - no features scaled
                    "scaling_method": "None - Files copied as-is"
                }
                
                # Save summary for no scaling case
                summary_filename = "scaling_summary.txt"
                summary_path = os.path.join(scaled_training_ready_dir, summary_filename)
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("Weather Feature Scaling Summary\n")
                    f.write("=" * 40 + "\n\n")
                    
                    f.write(f"Scaling timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Scaler type: None (No weather features found)\n")
                    f.write(f"Source directory: {merged_selected_training_ready_dir}\n")
                    f.write(f"Output directory: {MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER}\n\n")
                    
                    f.write(f"Files processed:\n")
                    f.write(f"  Train file: {train_filename} ({len(train_df):,} rows)\n")
                    f.write(f"  Test file: {test_filename} ({len(test_df):,} rows)\n")
                    f.write(f"  Output train file: {scaled_train_filename}\n")
                    f.write(f"  Output test file: {scaled_test_filename}\n\n")
                    
                    f.write("Weather features status:\n")
                    f.write("-" * 25 + "\n")
                    f.write("No weather features were found in the dataset.\n")
                    f.write("Files were copied as-is without any scaling applied.\n\n")
                    
                    f.write("Available weather features to check:\n")
                    for feature in ALL_WEATHER_FEATURES:
                        f.write(f"  {feature}: ✗ Not found\n")
                
                print(f"    scale_weather_features: Scaling summary saved to: {summary_filename}")
                
                # Return successful result for no scaling case
                return {
                    "success": True,
                    "processed_files": 1,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "weather_features_scaled": [],  # Empty list
                    "scaling_summary": no_scaling_result,
                    "output_directory": scaled_training_ready_dir,
                    "message": f"No weather features found. Files copied as-is: {train_filename}, {test_filename}",
                    "scaling_applied": False  # Flag to indicate no scaling was applied
                }
            
            # EXISTING LOGIC: Continue with normal scaling when weather features are found
            print(f"    scale_weather_features: Found {len(available_weather_features)} weather features to scale")
            print(f"      Weather features: {available_weather_features}")
            
            # Extract weather features for scaling
            train_weather_features = train_df[available_weather_features].copy()
            test_weather_features = test_df[available_weather_features].copy()

            # Zero-inflated/right-skewed features get a log1p pre-transform (deterministic,
            # not fit on data) before RobustScaler. These features are physically
            # non-negative, but raw FMI data uses negative sentinels (e.g. -1) for
            # "not applicable" readings; log1p(-1) is -inf, so clip to 0 first.
            if skewed_cols:
                print(f"    scale_weather_features: Applying log1p to {len(skewed_cols)} skewed feature(s): {skewed_cols}")
                train_weather_features[skewed_cols] = np.log1p(train_weather_features[skewed_cols].clip(lower=0))
                test_weather_features[skewed_cols] = np.log1p(test_weather_features[skewed_cols].clip(lower=0))

            # Create and fit scaler on training data only
            scaler = RobustScaler()
            scaler.fit(train_weather_features)

            # Save the fitted scaler (plus which columns need log1p) so it can be
            # applied to new data at inference time
            scaler_filename = "weather_scaler.joblib"
            scaler_path = os.path.join(scaled_training_ready_dir, scaler_filename)
            joblib.dump(
                {
                    "scaler": scaler,
                    "weather_features": available_weather_features,
                    "skewed_features": skewed_cols,
                },
                scaler_path,
            )
            print(f"    scale_weather_features: ✓ Scaler saved to: {scaler_filename}")

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
            scaled_train_filename = train_filename  # already .parquet
            scaled_test_filename = test_filename    # already .parquet

            scaled_train_path = os.path.join(scaled_training_ready_dir, scaled_train_filename)
            scaled_test_path = os.path.join(scaled_training_ready_dir, scaled_test_filename)

            # Save scaled datasets
            train_scaled_df.to_parquet(scaled_train_path, index=False)
            test_scaled_df.to_parquet(scaled_test_path, index=False)
            
            print(f"    scale_weather_features: ✓ Scaled train dataset saved to: {scaled_train_filename}")
            print(f"    scale_weather_features: ✓ Scaled test dataset saved to: {scaled_test_filename}")
            
            # Prepare scaling result
            scaling_result = {
                "original_train_file": train_filename,
                "original_test_file": test_filename,
                "scaled_train_file": scaled_train_filename,
                "scaled_test_file": scaled_test_filename,
                "scaler_file": scaler_filename,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
                "skewed_features_log1p": skewed_cols,
                "scaling_method": "RobustScaler"
            }
            
            # Save scaling summary
            summary_filename = "scaling_summary.txt"
            summary_path = os.path.join(scaled_training_ready_dir, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Weather Feature Scaling Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Scaling timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Scaler type: RobustScaler\n")
                f.write(f"Source directory: {merged_selected_training_ready_dir}\n")
                f.write(f"Output directory: {MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER}\n\n")
                
                f.write(f"Files processed:\n")
                f.write(f"  Train file: {train_filename} ({len(train_df):,} rows)\n")
                f.write(f"  Test file: {test_filename} ({len(test_df):,} rows)\n")
                f.write(f"  Scaled train file: {scaled_train_filename}\n")
                f.write(f"  Scaled test file: {scaled_test_filename}\n\n")
                
                f.write("Weather features scaled:\n")
                f.write("-" * 25 + "\n")
                for feature in ALL_WEATHER_FEATURES:
                    status = "✓ Scaled" if feature in available_weather_features else "✗ Not found"
                    f.write(f"  {feature}: {status}\n")

                f.write("\nSkewed features (log1p applied before RobustScaler):\n")
                f.write("-" * 25 + "\n")
                if skewed_cols:
                    for feature in skewed_cols:
                        f.write(f"  {feature}\n")
                else:
                    f.write("  None\n")

            print(f"    scale_weather_features: Scaling summary saved to: {summary_filename}")
            
            # Return successful result
            return {
                "success": True,
                "processed_files": 1,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
                "skewed_features_log1p": skewed_cols,
                "scaler_path": scaler_path,
                "scaling_summary": scaling_result,
                "output_directory": scaled_training_ready_dir,
                "message": f"Successfully scaled weather features for single train/test file pair: {train_filename}, {test_filename}",
                "scaling_applied": True
            }
            
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

    def train_xgboost_with_randomized_search_cv(self, data_dir=None):
        """
        Modified XGBoost training method that tests different iteration counts and plots performance curve.
        
        This method:
        1. Tests RANDOM_SEARCH_ITERATIONS from 10 to 100 (step=10)
        2. Tracks test F1 binary scores for each iteration count
        3. Plots test_f1_binary vs n_iter curve
        4. Saves the best model and performance metrics
        """
        
        try:
            print(f"    train_xgboost_with_randomized_search_cv: Starting XGBoost training with iteration analysis...")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"    train_xgboost_with_randomized_search_cv: Output directory: {output_dir}")
            
            if data_dir is None:
                data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            scaled_data_dir = data_dir

            # Find training and test files in the specified location
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train.parquet")
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test.parquet")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if not train_files or not test_files:
                error_msg = f"Training or test files not found in {scaled_data_dir}. Train files: {len(train_files)}, Test files: {len(test_files)}"
                print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Check if we have exactly one train/test file pair
            if len(train_files) != 1 or len(test_files) != 1:
                error_msg = f"Expected exactly one train and one test file. Found {len(train_files)} train files and {len(test_files)} test files"
                print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            train_file = train_files[0]
            test_file = test_files[0]
            
            # Extract identifier from filename
            train_filename = os.path.basename(train_file)
            identifier_match = re.search(r'merged_data_(.+?)_train\.parquet', train_filename)
            if not identifier_match:
                error_msg = f"Could not extract identifier from {train_filename}"
                print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            file_identifier = identifier_match.group(1)
            print(f"    train_xgboost_with_randomized_search_cv: Processing dataset: {file_identifier}")
            
            # Determine problem type based on target feature
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM
            
            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            problem_type = "classification" if is_classification else "regression"
            print(f"    train_xgboost_with_randomized_search_cv: Detected {problem_type} problem for target '{target_feature}'")
            
            # Load training and test data
            print(f"      Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            print(f"      Loading test data from {test_file}")
            test_df = pd.read_parquet(test_file)
            
            # Check if target feature exists
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Prepare features and target
            feature_columns = [col for col in train_df.columns if col != target_feature]
            
            X_train = train_df[feature_columns]
            y_train = train_df[target_feature]
            X_test = test_df[feature_columns]
            y_test = test_df[target_feature]
            
            print(f"      Dataset info - Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"      Features: {len(feature_columns)}")
            
            # Calculate sample weights if applicable
            sample_weights = None
            if WEIGHT_DELAY_COLUMN in train_df.columns:
                print(f"      Calculating sample weights using '{WEIGHT_DELAY_COLUMN}'...")
                weights = train_df[WEIGHT_DELAY_COLUMN].values
                
                if is_classification:
                    weights = np.clip(weights, 1, MAX_SAMPLE_WEIGHT_CLASSIFICATION)
                else:
                    weights = np.clip(weights, 1, MAX_SAMPLE_WEIGHT_REGRESSION)
                
                sample_weights = weights
                print(f"      Sample weights - Min: {weights.min():.2f}, Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")
            
            # Set up cross-validation strategy
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = xgb.XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = xgb.XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                scoring_metric = 'neg_mean_absolute_error'
                # Shift target so all values are positive, then log-transform to compress
                # the extreme right tail (max 963 min) that deflates R² artificially.
                y_shift = max(0.0, -float(y_train.min())) + 1.0
                y_train_log = np.log1p(y_train + y_shift)
                print(f"      Log-transform target: shift={y_shift:.2f}, "
                      f"log range=[{y_train_log.min():.3f}, {y_train_log.max():.3f}] "
                      f"(original [{y_train.min():.1f}, {y_train.max():.1f}])")

            # Define iteration range: 10 to 100 (step=10)
            iteration_values = list(range(10, RANDOM_SEARCH_ITERATIONS + 1, 10))
            print(f"      Testing iteration values: {iteration_values}")

            # Initialize tracking lists
            iteration_results = []
            test_f1_scores = []
            cv_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []

            best_model = None
            best_cv_score = -np.inf  # CV score: higher is always better (F1 or neg_MAE)
            best_iteration = None
            
            # Train models with different iteration counts
            print(f"      Starting training with different iteration counts...")

            # Each run is an independent random draw — not cumulative. The curve shows
            # search-budget sensitivity, not convergence toward a global optimum.
            for i, n_iter in enumerate(iteration_values):
                print(f"      Progress: {i+1}/{len(iteration_values)} - Testing {n_iter} iterations...")
                
                # Create RandomizedSearchCV with current iteration count
                randomized_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train_log if not is_classification else y_train
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit)
                
                # Get best model for current iteration count
                current_best_model = randomized_search.best_estimator_
                current_cv_score = randomized_search.best_score_
                
                # Make predictions on test set; inverse-transform for regression
                y_pred = current_best_model.predict(X_test)
                if not is_classification:
                    y_pred = np.expm1(y_pred) - y_shift

                # Calculate test metrics
                if is_classification:
                    test_f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                    test_accuracy = accuracy_score(y_test, y_pred)

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}")

                else:
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred)
                    test_r2 = r2_score(y_test, y_pred)
                    denom = np.sum(np.abs(y_test))
                    test_wmape = (np.sum(np.abs(y_test - y_pred)) / denom * 100) if denom > 0 else 0.0

                    y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
                    y_pred_binary = (y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
                    test_bin_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_recall    = recall_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_f1        = f1_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_accuracy  = accuracy_score(y_test_binary, y_pred_binary)

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

                if is_classification:
                    test_f1_scores.append(test_f1)
                else:
                    test_f1_scores.append(test_rmse)
                    test_mae_scores.append(test_mae)
                    test_wmape_scores.append(test_wmape)
                    test_bin_precision_scores.append(test_bin_precision)
                    test_bin_recall_scores.append(test_bin_recall)
                    test_bin_f1_scores.append(test_bin_f1)
                    test_bin_accuracy_scores.append(test_bin_accuracy)
                
                cv_scores.append(current_cv_score)

                # Store detailed results
                iteration_results.append({
                    'n_iter': n_iter,
                    'cv_score': current_cv_score,
                    'test_metric': test_f1_scores[-1],
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'best_params': randomized_search.best_params_
                })

            # Calculate final metrics once on the best model found across all iterations
            print(f"      Calculating final metrics with best model (iteration {best_iteration})...")
            final_y_pred = best_model.predict(X_test)
            if not is_classification:
                final_y_pred = np.expm1(final_y_pred) - y_shift

            if is_classification:
                final_test_accuracy = accuracy_score(y_test, final_y_pred)
                final_test_f1 = f1_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                final_test_precision = precision_score(y_test, final_y_pred, average='weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='weighted')

                if hasattr(best_model, 'predict_proba'):
                    final_y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                    final_test_auc = roc_auc_score(y_test, final_y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
                else:
                    final_test_auc = 0.0

            else:  # regression
                final_test_rmse = np.sqrt(mean_squared_error(y_test, final_y_pred))
                final_test_mae = mean_absolute_error(y_test, final_y_pred)
                final_test_r2 = r2_score(y_test, final_y_pred)
                denom = np.sum(np.abs(y_test))
                final_test_wmape = (np.sum(np.abs(y_test - final_y_pred)) / denom * 100) if denom > 0 else 0.0

                y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
                final_y_pred_binary = (final_y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
                final_bin_precision = precision_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_recall    = recall_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_f1        = f1_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_accuracy  = accuracy_score(y_test_binary, final_y_pred_binary)

            # Create performance curve plot
            print(f"      Creating performance curve plot...")
            
            # Create single plot with dual y-axes
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Plot Test F1/RMSE on left y-axis
            color1 = 'tab:blue'
            ax1.set_xlabel('Number of RandomizedSearch Iterations', fontsize=13)
            ax1.set_ylabel(f'Test {"F1 Binary Score" if is_classification else "RMSE"}', color=color1, fontsize=13)
            line1 = ax1.plot(iteration_values, test_f1_scores, 'o-', color=color1, linewidth=3, markersize=8, 
                            label=f'Test {"F1 Binary" if is_classification else "RMSE"}')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for CV Score
            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Cross-Validation Score', color=color2, fontsize=13)
            line2 = ax2.plot(iteration_values, cv_scores, 's-', color=color2, linewidth=3, markersize=8, 
                            label='CV Score')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Highlight best iteration on both axes
            if best_iteration:
                best_idx = iteration_values.index(best_iteration)
                
                # Highlight on test score axis
                ax1.plot(best_iteration, test_f1_scores[best_idx], 'o', color='red', markersize=15, 
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
                
                # Highlight on CV score axis
                ax2.plot(best_iteration, cv_scores[best_idx], 's', color='red', markersize=15, 
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
                
                # Add annotation
                ax1.annotate(f'Best at {best_iteration} iterations\nTest Score: {test_f1_scores[best_idx]:.4f}\nCV Score: {cv_scores[best_idx]:.4f}', 
                            xy=(best_iteration, test_f1_scores[best_idx]), 
                            xytext=(20, 30), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='orange'),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red'),
                            fontsize=11, fontweight='bold')
            
            # Add title
            plt.title(f'XGBoost Performance vs RandomizedSearch Iterations\nDataset: {file_identifier}', 
                    fontsize=16, fontweight='bold', pad=20)
            
            # Create combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                    fontsize=12, framealpha=0.9)
            
            # Adjust layout
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'xgboost_iteration_analysis_{file_identifier}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Performance curve saved to: {plot_filename}")
            
            # CREATE AND SAVE FEATURE IMPORTANCE PLOT
            results = {}
            if best_model is not None:
                print(f"      Creating feature importance plot...")
                
                # Get feature importance from the best model
                feature_importance = best_model.feature_importances_
                feature_names = feature_columns
                
                # Create DataFrame for easier handling
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=True)
                
                # Create feature importance plot
                plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))
                
                # Create horizontal bar plot
                bars = plt.barh(range(len(importance_df)), importance_df['importance'])
                
                # Customize the plot
                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Feature Importance', fontsize=12)
                plt.title(f'XGBoost Feature Importance - Best Model ({problem_type.title()})\n'
                        f'Dataset: {file_identifier} | Best Iteration: {best_iteration}', fontsize=14)
                
                # Add value labels on bars
                for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
                    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{importance:.3f}', 
                            ha='left', va='center', fontsize=10)
                
                # Color bars based on importance (gradient from red to green)
                normalized_importance = (importance_df['importance'] - importance_df['importance'].min()) / \
                                    (importance_df['importance'].max() - importance_df['importance'].min() + 1e-8)
                
                colors = plt.cm.RdYlGn(normalized_importance)
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                # Save feature importance plot
                importance_plot_filename = os.path.join(output_dir, f'xgboost_feature_importance_{file_identifier}.png')
                plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
                print(f"      Feature importance plot saved to: {importance_plot_filename}")
                plt.close()
                
                # Save feature importance data as CSV
                importance_csv_filename = os.path.join(output_dir, f'xgboost_feature_importance_{file_identifier}.csv')
                importance_df.sort_values('importance', ascending=False).to_csv(importance_csv_filename, index=False)
                print(f"      Feature importance data saved to: {importance_csv_filename}")
        

                # Create comprehensive results dictionary
                results = {
                    "file_identifier": file_identifier,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "dataset_info": {
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "features_count": len(feature_columns),
                        "train_file": train_file,
                        "test_file": test_file
                    },
                    "best_iteration": best_iteration,
                    "iteration_analysis": iteration_results,
                    "best_model_params": best_model.get_params() if best_model else None,
                    "feature_importance": dict(zip(feature_columns, best_model.feature_importances_)) if best_model else None
                }

                # Add problem-specific metrics
                if is_classification:
                    results["final_metrics"] = {
                        "test_accuracy": float(final_test_accuracy),
                        "test_f1": float(final_test_f1),
                        "test_precision": float(final_test_precision),
                        "test_recall": float(final_test_recall),
                        "test_auc": float(final_test_auc)
                    }
                else:
                    results["final_metrics"] = {
                        "test_rmse": float(final_test_rmse),
                        "test_mae": float(final_test_mae),
                        "test_wmape": float(final_test_wmape),
                        "test_r2": float(final_test_r2),
                        "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES,
                        "test_bin_precision": float(final_bin_precision),
                        "test_bin_recall": float(final_bin_recall),
                        "test_bin_f1": float(final_bin_f1),
                        "test_bin_accuracy": float(final_bin_accuracy)
                    }

                # Add iteration-wise metrics summary
                if not is_classification:
                    results["iteration_metrics_summary"] = {
                        "rmse_values": [float(x) for x in test_f1_scores],
                        "mae_values": [float(x) for x in test_mae_scores],
                        "wmape_values": [float(x) for x in test_wmape_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_rmse": float(min(test_f1_scores)),
                        "best_mae": float(min(test_mae_scores)),
                        "best_wmape": float(min(test_wmape_scores)),
                        "average_rmse": float(np.mean(test_f1_scores)),
                        "average_mae": float(np.mean(test_mae_scores)),
                        "average_wmape": float(np.mean(test_wmape_scores)),
                        "bin_f1_values": [float(x) for x in test_bin_f1_scores],
                        "bin_accuracy_values": [float(x) for x in test_bin_accuracy_scores],
                        "bin_precision_values": [float(x) for x in test_bin_precision_scores],
                        "bin_recall_values": [float(x) for x in test_bin_recall_scores]
                    }
            
            results_file = os.path.join(output_dir, f"xgboost_iteration_analysis_{file_identifier}.json")
            with open(results_file, 'w') as f:
                results_str = json.loads(json.dumps(results, default=str))
                json.dump(results_str, f, indent=2)

            print(f"      Results saved to: {results_file}")
            
            # Save the best model
            if best_model:
                model_filename = os.path.join(output_dir, f'xgboost_best_model_{file_identifier}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"      Best model saved to: {model_filename}")
            
            # Find the index of the best iteration in the results list
            best_iteration_idx = iteration_values.index(best_iteration) if best_iteration else 0


            # Print summary
            print(f"      Training Summary:")
            print(f"        Problem Type: {problem_type}")
            print(f"        Best Iteration Count: {best_iteration}")
            print(f"        Best CV Score: {best_cv_score:.4f}")
            print(f"        RMSE Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
            if not is_classification:
                print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")

            if is_classification:
                return {
                    "success": True,
                    "models_trained": 1,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "cv_score": float(iteration_results[best_iteration_idx]["cv_score"]),
                    "test_f1": float(final_test_f1),
                    "output_directory": output_dir,
                    "results_file": results_file
                }
            else:
                return {
                    "success": True,
                    "models_trained": 1,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "cv_score": float(iteration_results[best_iteration_idx]["cv_score"]),
                    "test_rmse": float(final_test_rmse),
                    "test_mae": float(final_test_mae),
                    "test_wmape": float(final_test_wmape),
                    "test_r2": float(final_test_r2),
                    "test_bin_precision": float(final_bin_precision),
                    "test_bin_recall": float(final_bin_recall),
                    "test_bin_f1": float(final_bin_f1),
                    "test_bin_accuracy": float(final_bin_accuracy),
                    "output_directory": output_dir,
                    "results_file": results_file
                }
            
        except Exception as e:
            error_msg = f"Error in train_xgboost_with_randomized_search_cv: {str(e)}"
            print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def train_lightgbm_with_randomized_search_cv(self, data_dir=None):
        """
        LightGBM training method that tests different iteration counts and plots performance curve.

        This method:
        1. Tests RANDOM_SEARCH_ITERATIONS from 10 to 100 (step=10)
        2. Tracks test F1 binary scores for each iteration count
        3. Plots test_f1_binary vs n_iter curve
        4. Saves the best model and performance metrics
        """

        try:
            print(f"    train_lightgbm_with_randomized_search_cv: Starting LightGBM training with iteration analysis...")

            # Create output directory
            output_dir = os.path.join(self.project_root, LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)

            print(f"    train_lightgbm_with_randomized_search_cv: Output directory: {output_dir}")

            if data_dir is None:
                data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            scaled_data_dir = data_dir

            # Find training and test files in the specified location
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train.parquet")
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test.parquet")

            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)

            if not train_files or not test_files:
                error_msg = f"Training or test files not found in {scaled_data_dir}. Train files: {len(train_files)}, Test files: {len(test_files)}"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Check if we have exactly one train/test file pair
            if len(train_files) != 1 or len(test_files) != 1:
                error_msg = f"Expected exactly one train and one test file. Found {len(train_files)} train files and {len(test_files)} test files"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            train_file = train_files[0]
            test_file = test_files[0]

            # Extract identifier from filename
            train_filename = os.path.basename(train_file)
            identifier_match = re.search(r'merged_data_(.+?)_train\.parquet', train_filename)
            if not identifier_match:
                error_msg = f"Could not extract identifier from {train_filename}"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            file_identifier = identifier_match.group(1)
            print(f"    train_lightgbm_with_randomized_search_cv: Processing dataset: {file_identifier}")

            # Determine problem type based on target feature
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM

            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            problem_type = "classification" if is_classification else "regression"
            print(f"    train_lightgbm_with_randomized_search_cv: Detected {problem_type} problem for target '{target_feature}'")

            # Load training and test data
            print(f"      Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            print(f"      Loading test data from {test_file}")
            test_df = pd.read_parquet(test_file)

            # Check if target feature exists
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Prepare features and target
            feature_columns = [col for col in train_df.columns if col != target_feature]

            X_train = train_df[feature_columns]
            y_train = train_df[target_feature]
            X_test = test_df[feature_columns]
            y_test = test_df[target_feature]

            print(f"      Dataset info - Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"      Features: {len(feature_columns)}")

            # Calculate sample weights if applicable
            sample_weights = None
            if WEIGHT_DELAY_COLUMN in train_df.columns:
                print(f"      Calculating sample weights using '{WEIGHT_DELAY_COLUMN}'...")
                weights = train_df[WEIGHT_DELAY_COLUMN].values

                if is_classification:
                    weights = np.clip(weights, 1, MAX_SAMPLE_WEIGHT_CLASSIFICATION)
                else:
                    weights = np.clip(weights, 1, MAX_SAMPLE_WEIGHT_REGRESSION)

                sample_weights = weights
                print(f"      Sample weights - Min: {weights.min():.2f}, Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")

            # Set up cross-validation strategy
            y_train_log = None  # assigned in regression branch below
            y_shift = 0.0       # assigned in regression branch below
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = lgb.LGBMClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    subsample_freq=1,
                    verbose=-1,
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = lgb.LGBMRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    subsample_freq=1,
                    verbose=-1,
                )
                scoring_metric = 'neg_mean_absolute_error'
                # Shift target so all values are positive, then log-transform to compress
                # the extreme right tail (max 963 min) that deflates R² artificially.
                y_shift = max(0.0, -float(y_train.min())) + 1.0
                y_train_log = np.log1p(y_train + y_shift)
                print(f"      Log-transform target: shift={y_shift:.2f}, "
                      f"log range=[{y_train_log.min():.3f}, {y_train_log.max():.3f}] "
                      f"(original [{y_train.min():.1f}, {y_train.max():.1f}])")

            # Define iteration range: 10 to 100 (step=10)
            iteration_values = list(range(10, RANDOM_SEARCH_ITERATIONS + 1, 10))
            print(f"      Testing iteration values: {iteration_values}")

            # Initialize tracking lists
            iteration_results = []
            test_f1_scores = []
            cv_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []

            best_model = None
            best_cv_score = -np.inf
            best_iteration = None

            # Train models with different iteration counts
            print(f"      Starting training with different iteration counts...")

            # Each run is an independent random draw — not cumulative. The curve shows
            # search-budget sensitivity, not convergence toward a global optimum.
            for i, n_iter in enumerate(iteration_values):
                print(f"      Progress: {i+1}/{len(iteration_values)} - Testing {n_iter} iterations...")

                # Create RandomizedSearchCV with current iteration count
                randomized_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=LIGHTGBM_PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=0
                )

                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train_log if not is_classification else y_train
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit)

                # Get best model for current iteration count
                current_best_model = randomized_search.best_estimator_
                current_cv_score = randomized_search.best_score_

                # Make predictions on test set; inverse-transform for regression
                y_pred = current_best_model.predict(X_test)
                if not is_classification:
                    y_pred = np.expm1(y_pred) - y_shift

                # Calculate test metrics
                if is_classification:
                    test_f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                    test_accuracy = accuracy_score(y_test, y_pred)

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}")

                else:
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred)
                    test_r2 = r2_score(y_test, y_pred)
                    denom = np.sum(np.abs(y_test))
                    test_wmape = (np.sum(np.abs(y_test - y_pred)) / denom * 100) if denom > 0 else 0.0

                    y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
                    y_pred_binary = (y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
                    test_bin_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_recall    = recall_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_f1        = f1_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_accuracy  = accuracy_score(y_test_binary, y_pred_binary)

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

                if is_classification:
                    test_f1_scores.append(test_f1)
                else:
                    test_f1_scores.append(test_rmse)
                    test_mae_scores.append(test_mae)
                    test_wmape_scores.append(test_wmape)
                    test_bin_precision_scores.append(test_bin_precision)
                    test_bin_recall_scores.append(test_bin_recall)
                    test_bin_f1_scores.append(test_bin_f1)
                    test_bin_accuracy_scores.append(test_bin_accuracy)

                cv_scores.append(current_cv_score)

                # Store detailed results
                iteration_results.append({
                    'n_iter': n_iter,
                    'cv_score': current_cv_score,
                    'test_metric': test_f1_scores[-1],
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'best_params': randomized_search.best_params_
                })

            # Calculate final metrics once on the best model found across all iterations
            print(f"      Calculating final metrics with best model (iteration {best_iteration})...")
            final_y_pred = best_model.predict(X_test)
            if not is_classification:
                final_y_pred = np.expm1(final_y_pred) - y_shift

            if is_classification:
                final_test_accuracy = accuracy_score(y_test, final_y_pred)
                final_test_f1 = f1_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                final_test_precision = precision_score(y_test, final_y_pred, average='weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='weighted')

                if hasattr(best_model, 'predict_proba'):
                    final_y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                    final_test_auc = roc_auc_score(y_test, final_y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
                else:
                    final_test_auc = 0.0

            else:  # regression
                final_test_rmse = np.sqrt(mean_squared_error(y_test, final_y_pred))
                final_test_mae = mean_absolute_error(y_test, final_y_pred)
                final_test_r2 = r2_score(y_test, final_y_pred)
                denom = np.sum(np.abs(y_test))
                final_test_wmape = (np.sum(np.abs(y_test - final_y_pred)) / denom * 100) if denom > 0 else 0.0

                y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
                final_y_pred_binary = (final_y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
                final_bin_precision = precision_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_recall    = recall_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_f1        = f1_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_accuracy  = accuracy_score(y_test_binary, final_y_pred_binary)

            # Create performance curve plot
            print(f"      Creating performance curve plot...")

            fig, ax1 = plt.subplots(figsize=(14, 8))

            color1 = 'tab:blue'
            ax1.set_xlabel('Number of RandomizedSearch Iterations', fontsize=13)
            ax1.set_ylabel(f'Test {"F1 Binary Score" if is_classification else "RMSE"}', color=color1, fontsize=13)
            line1 = ax1.plot(iteration_values, test_f1_scores, 'o-', color=color1, linewidth=3, markersize=8,
                            label=f'Test {"F1 Binary" if is_classification else "RMSE"}')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Cross-Validation Score', color=color2, fontsize=13)
            line2 = ax2.plot(iteration_values, cv_scores, 's-', color=color2, linewidth=3, markersize=8,
                            label='CV Score')
            ax2.tick_params(axis='y', labelcolor=color2)

            if best_iteration:
                best_idx = iteration_values.index(best_iteration)

                ax1.plot(best_iteration, test_f1_scores[best_idx], 'o', color='red', markersize=15,
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)

                ax2.plot(best_iteration, cv_scores[best_idx], 's', color='red', markersize=15,
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)

                ax1.annotate(f'Best at {best_iteration} iterations\nTest Score: {test_f1_scores[best_idx]:.4f}\nCV Score: {cv_scores[best_idx]:.4f}',
                            xy=(best_iteration, test_f1_scores[best_idx]),
                            xytext=(20, 30), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='orange'),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red'),
                            fontsize=11, fontweight='bold')

            plt.title(f'LightGBM Performance vs RandomizedSearch Iterations\nDataset: {file_identifier}',
                    fontsize=16, fontweight='bold', pad=20)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                    fontsize=12, framealpha=0.9)

            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'lightgbm_iteration_analysis_{file_identifier}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Performance curve saved to: {plot_filename}")

            # CREATE AND SAVE FEATURE IMPORTANCE PLOT
            results = {}
            if best_model is not None:
                print(f"      Creating feature importance plot...")

                feature_importance = best_model.feature_importances_
                feature_names = feature_columns

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=True)

                plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))

                bars = plt.barh(range(len(importance_df)), importance_df['importance'])

                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Feature Importance', fontsize=12)
                plt.title(f'LightGBM Feature Importance - Best Model ({problem_type.title()})\n'
                        f'Dataset: {file_identifier} | Best Iteration: {best_iteration}', fontsize=14)

                for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
                    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                            f'{importance:.3f}',
                            ha='left', va='center', fontsize=10)

                normalized_importance = (importance_df['importance'] - importance_df['importance'].min()) / \
                                    (importance_df['importance'].max() - importance_df['importance'].min() + 1e-8)

                colors = plt.cm.RdYlGn(normalized_importance)
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()

                importance_plot_filename = os.path.join(output_dir, f'lightgbm_feature_importance_{file_identifier}.png')
                plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
                print(f"      Feature importance plot saved to: {importance_plot_filename}")
                plt.close()

                importance_csv_filename = os.path.join(output_dir, f'lightgbm_feature_importance_{file_identifier}.csv')
                importance_df.sort_values('importance', ascending=False).to_csv(importance_csv_filename, index=False)
                print(f"      Feature importance data saved to: {importance_csv_filename}")

                # Create comprehensive results dictionary
                results = {
                    "file_identifier": file_identifier,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "dataset_info": {
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "features_count": len(feature_columns),
                        "train_file": train_file,
                        "test_file": test_file
                    },
                    "best_iteration": best_iteration,
                    "iteration_analysis": iteration_results,
                    "best_model_params": best_model.get_params() if best_model else None,
                    "feature_importance": dict(zip(feature_columns, best_model.feature_importances_)) if best_model else None
                }

                # Add problem-specific metrics
                if is_classification:
                    results["final_metrics"] = {
                        "test_accuracy": float(final_test_accuracy),
                        "test_f1": float(final_test_f1),
                        "test_precision": float(final_test_precision),
                        "test_recall": float(final_test_recall),
                        "test_auc": float(final_test_auc)
                    }
                else:
                    results["final_metrics"] = {
                        "test_rmse": float(final_test_rmse),
                        "test_mae": float(final_test_mae),
                        "test_wmape": float(final_test_wmape),
                        "test_r2": float(final_test_r2),
                        "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES,
                        "test_bin_precision": float(final_bin_precision),
                        "test_bin_recall": float(final_bin_recall),
                        "test_bin_f1": float(final_bin_f1),
                        "test_bin_accuracy": float(final_bin_accuracy)
                    }

                # Add iteration-wise metrics summary
                if not is_classification:
                    results["iteration_metrics_summary"] = {
                        "rmse_values": [float(x) for x in test_f1_scores],
                        "mae_values": [float(x) for x in test_mae_scores],
                        "wmape_values": [float(x) for x in test_wmape_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_rmse": float(min(test_f1_scores)),
                        "best_mae": float(min(test_mae_scores)),
                        "best_wmape": float(min(test_wmape_scores)),
                        "average_rmse": float(np.mean(test_f1_scores)),
                        "average_mae": float(np.mean(test_mae_scores)),
                        "average_wmape": float(np.mean(test_wmape_scores)),
                        "bin_f1_values": [float(x) for x in test_bin_f1_scores],
                        "bin_accuracy_values": [float(x) for x in test_bin_accuracy_scores],
                        "bin_precision_values": [float(x) for x in test_bin_precision_scores],
                        "bin_recall_values": [float(x) for x in test_bin_recall_scores]
                    }

                results_file = os.path.join(output_dir, f"lightgbm_iteration_analysis_{file_identifier}.json")
                with open(results_file, 'w') as f:
                    results_str = json.loads(json.dumps(results, default=str))
                    json.dump(results_str, f, indent=2)

                print(f"      Results saved to: {results_file}")

                # Save the best model
                model_filename = os.path.join(output_dir, f'lightgbm_best_model_{file_identifier}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"      Best model saved to: {model_filename}")

                # Find the index of the best iteration in the results list
                best_iteration_idx = iteration_values.index(best_iteration) if best_iteration else 0

                # Print summary
                print(f"      Training Summary:")
                print(f"        Problem Type: {problem_type}")
                print(f"        Best Iteration Count: {best_iteration}")
                print(f"        Best CV Score: {best_cv_score:.4f}")
                print(f"        RMSE Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
                if not is_classification:
                    print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                    print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                    print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")

                if is_classification:
                    return {
                        "success": True,
                        "models_trained": 1,
                        "problem_type": problem_type,
                        "target_feature": target_feature,
                        "cv_score": float(iteration_results[best_iteration_idx]["cv_score"]),
                        "test_f1": float(final_test_f1),
                        "output_directory": output_dir,
                        "results_file": results_file
                    }
                else:
                    return {
                        "success": True,
                        "models_trained": 1,
                        "problem_type": problem_type,
                        "target_feature": target_feature,
                        "cv_score": float(iteration_results[best_iteration_idx]["cv_score"]),
                        "test_rmse": float(final_test_rmse),
                        "test_mae": float(final_test_mae),
                        "test_wmape": float(final_test_wmape),
                        "test_r2": float(final_test_r2),
                        "test_bin_precision": float(final_bin_precision),
                        "test_bin_recall": float(final_bin_recall),
                        "test_bin_f1": float(final_bin_f1),
                        "test_bin_accuracy": float(final_bin_accuracy),
                        "output_directory": output_dir,
                        "results_file": results_file
                    }
            else:
                return {
                    "success": False,
                    "error": "No model was trained — all iterations failed to produce a best estimator"
                }

        except Exception as e:
            error_msg = f"Error in train_lightgbm_with_randomized_search_cv: {str(e)}"
            print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def numeric_correlation_analysis(self, csv_files=None, target_file=None):
        """
        Performs comprehensive numeric correlation analysis on a single merged data file.
        
        This method provides comprehensive statistical analysis and visualizations including 
        correlation matrices, scatter plots, and feature importance ranking for a single file
        instead of processing multiple files in a loop.
        
        Key Analyses Performed:
        1. Correlation matrix heatmap for feature-to-feature correlations (excluding target and self-correlations)
        2. Individual scatter plots for top correlated features vs target
        3. Statistical correlation analysis (Pearson and Spearman)
        4. Feature correlation ranking and importance
        5. Distribution analysis of correlations
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (deprecated - use target_file instead)
        target_file : str, optional
            Specific file path to analyze. If None, will look for the first available merged file.
            
        Returns:
        --------
        dict
            Results of the numeric correlation analysis including plots and statistics
        """
        try:
            print(f"    numeric_correlation_analysis: Starting comprehensive numeric correlation analysis...")
            
            # ═══════════════════════════════════════════════════════════════════════
            # IEEE-COMPLIANT MATPLOTLIB CONFIGURATION
            # ═══════════════════════════════════════════════════════════════════════
            import matplotlib as mpl
            mpl.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                'font.size': 8,
                'axes.linewidth': 1.0,
                'xtick.major.width': 1.0,
                'ytick.major.width': 1.0,
                'pdf.fonttype': 42,
                'ps.fonttype': 42
            })
            # ═══════════════════════════════════════════════════════════════════════
            
            # Create output directories
            analysis_output_dir = os.path.join(self.project_root, "data/output/numeric_correlation_analysis")
            correlation_plots_dir = os.path.join(analysis_output_dir, "correlation_plots")
            scatter_plots_dir = os.path.join(analysis_output_dir, "scatter_plots")
            statistical_analysis_dir = os.path.join(analysis_output_dir, "statistical_analysis")
            feature_ranking_dir = os.path.join(analysis_output_dir, "feature_ranking")
            
            os.makedirs(analysis_output_dir, exist_ok=True)
            os.makedirs(correlation_plots_dir, exist_ok=True)
            os.makedirs(scatter_plots_dir, exist_ok=True)
            os.makedirs(statistical_analysis_dir, exist_ok=True)
            os.makedirs(feature_ranking_dir, exist_ok=True)
            
            # Determine target file to analyze
            target_file_path = None
            
            if target_file and os.path.exists(target_file):
                target_file_path = target_file
                print(f"    numeric_correlation_analysis: Using specified target file: {os.path.basename(target_file)}")
            else:
                # Look for scaled data first, then fallback to non-scaled
                merged_data_pattern = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*_train.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                if not merged_data_files:
                    merged_data_pattern = os.path.join(self.project_root, "data/output/3-merged_training_ready", "merged_data_*_train.csv")
                    merged_data_files = glob.glob(merged_data_pattern)
                    print(f"    numeric_correlation_analysis: Using non-scaled data files")
                
                if merged_data_files:
                    target_file_path = merged_data_files[0]  # Use first available file
                    print(f"    numeric_correlation_analysis: Using first available file: {os.path.basename(target_file_path)}")
                
            # Check if target file exists
            if not target_file_path or not os.path.exists(target_file_path):
                error_msg = f"No valid merged data file found for numeric correlation analysis."
                print(f"    numeric_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            try:
                # Load data
                filename = os.path.basename(target_file_path)
                print(f"      Loading data from {filename}...")
                df = pd.read_csv(target_file_path)
                
                # Identify target column
                target_column = DEFAULT_TARGET_FEATURE
                if target_column not in df.columns:
                    available_targets = [col for col in VALID_TARGET_FEATURES if col in df.columns]
                    if available_targets:
                        target_column = available_targets[0]
                        print(f"      Using target column: {target_column}")
                    else:
                        error_msg = f"No valid target column found in {filename}"
                        print(f"    numeric_correlation_analysis: {error_msg}")
                        return {
                            "success": False,
                            "error": error_msg,
                            "processed_files": 0,
                            "target_file": target_file_path,
                            "available_columns": list(df.columns)
                        }
                
                # Extract all numeric columns (excluding target)
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Remove target column and any index columns from features
                features_to_exclude = [target_column, 'Unnamed: 0', 'index']
                numeric_features = [col for col in numeric_columns if col not in features_to_exclude]
                
                if not numeric_features:
                    error_msg = f"No numeric features found for correlation analysis in {filename}"
                    print(f"    numeric_correlation_analysis: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "processed_files": 0,
                        "target_file": target_file_path,
                        "available_columns": list(df.columns)
                    }
                
                print(f"      Found {len(numeric_features)} numeric features for analysis")
                
                # Create analysis dataframe
                analysis_df = df[numeric_features + [target_column]].copy()
                
                # Remove rows with missing target values
                initial_rows = len(analysis_df)
                analysis_df = analysis_df.dropna(subset=[target_column])
                final_rows = len(analysis_df)
                
                if initial_rows != final_rows:
                    print(f"      Removed {initial_rows - final_rows} rows with missing target values")
                
                if analysis_df.empty:
                    error_msg = f"No valid data remaining after removing missing values in {filename}"
                    print(f"    numeric_correlation_analysis: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "processed_files": 0,
                        "target_file": target_file_path
                    }
                
                # Calculate correlations between features and target
                print(f"      Calculating correlations for {len(numeric_features)} features...")
                correlations = {}
                
                for feature in numeric_features:
                    if feature in analysis_df.columns:
                        # Remove missing values for this feature
                        feature_data = analysis_df[[feature, target_column]].dropna()
                        
                        if len(feature_data) > 1:  # Need at least 2 points for correlation
                            try:
                                # Pearson correlation
                                pearson_corr, pearson_p = pearsonr(feature_data[feature], feature_data[target_column])
                                
                                # Spearman correlation
                                spearman_corr, spearman_p = spearmanr(feature_data[feature], feature_data[target_column])
                                
                                correlations[feature] = {
                                    'pearson': pearson_corr,
                                    'spearman': spearman_corr,
                                    'pearson_p_value': pearson_p,
                                    'spearman_p_value': spearman_p,
                                    'sample_size': len(feature_data)
                                }
                            except Exception as e:
                                print(f"        Warning: Could not calculate correlation for {feature}: {e}")
                                continue
                
                if not correlations:
                    error_msg = f"Could not calculate any correlations in {filename}"
                    print(f"    numeric_correlation_analysis: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "processed_files": 0,
                        "target_file": target_file_path
                    }
                
                # Sort features by absolute Pearson correlation with target
                sorted_features = sorted(
                    correlations.items(),
                    key=lambda x: abs(x[1]['pearson']) if not np.isnan(x[1]['pearson']) else 0,
                    reverse=True
                )
                
                print(f"      Successfully calculated correlations for {len(correlations)} features")
                print(f"      Top 5 correlated features:")
                for i, (feature, corr_data) in enumerate(sorted_features[:5]):
                    print(f"        {i+1}. {feature}: r={corr_data['pearson']:.4f}, p={corr_data['pearson_p_value']:.4f}")
                
                # ═══════════════════════════════════════════════════════════════════════
                # CREATE IEEE-COMPLIANT CORRELATION HEATMAP
                # ═══════════════════════════════════════════════════════════════════════
                correlation_matrix = analysis_df[numeric_features].corr()
                
                # IEEE figure size: 7.16" for two-column
                fig_width = 7.16
                fig_height = fig_width * 0.85
                
                plt.figure(figsize=(fig_width, fig_height))
                
                # Create mask to hide both upper triangle AND diagonal (self-correlations)
                mask_upper = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
                mask_diagonal = np.eye(correlation_matrix.shape[0], dtype=bool)
                mask = mask_upper | mask_diagonal  # Combine both masks
                
                # IEEE-compliant heatmap
                sns.heatmap(correlation_matrix, 
                        annot=True, 
                        cmap='RdBu_r',
                        center=0,
                        mask=mask, 
                        square=True, 
                        fmt='.3f',
                        annot_kws={'fontsize': 6},
                        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                        vmin=-1.0,
                        vmax=1.0)
                
                # NO TITLE - IEEE uses captions below figures
                plt.tight_layout()
                
                # Save both PNG and PDF (PDF for IEEE submission)
                heatmap_path_png = os.path.join(correlation_plots_dir, f"correlation_heatmap_{filename.replace('.csv', '.png')}")
                heatmap_path_pdf = os.path.join(correlation_plots_dir, f"correlation_heatmap_{filename.replace('.csv', '.pdf')}")
                
                plt.savefig(heatmap_path_png, dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.savefig(heatmap_path_pdf, bbox_inches='tight', pad_inches=0.05)
                plt.close()
                
                print(f"        Saved IEEE-compliant correlation heatmap:")
                print(f"          PNG: {os.path.basename(heatmap_path_png)}")
                print(f"          PDF: {os.path.basename(heatmap_path_pdf)}")
                
                # ═══════════════════════════════════════════════════════════════════════
                # CREATE IEEE-COMPLIANT STRONG CORRELATIONS HEATMAP
                # ═══════════════════════════════════════════════════════════════════════
                correlation_threshold = 0.3  # Adjust as needed
                strong_corr_matrix = correlation_matrix.copy()
                strong_corr_matrix[abs(strong_corr_matrix) < correlation_threshold] = np.nan
                
                plt.figure(figsize=(fig_width, fig_height))
                
                sns.heatmap(strong_corr_matrix, 
                        annot=True, 
                        cmap='RdBu_r',
                        center=0,
                        mask=mask, 
                        square=True, 
                        fmt='.3f',
                        annot_kws={'fontsize': 6},
                        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                        linewidths=0.2,
                        linecolor='gray',
                        vmin=-1.0,
                        vmax=1.0)
                
                # NO TITLE for IEEE
                plt.tight_layout()
                
                strong_heatmap_png = os.path.join(correlation_plots_dir, f"strong_correlation_heatmap_{filename.replace('.csv', '.png')}")
                strong_heatmap_pdf = os.path.join(correlation_plots_dir, f"strong_correlation_heatmap_{filename.replace('.csv', '.pdf')}")
                
                plt.savefig(strong_heatmap_png, dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.savefig(strong_heatmap_pdf, bbox_inches='tight', pad_inches=0.05)
                plt.close()
                
                print(f"        Saved IEEE-compliant strong correlations heatmap:")
                print(f"          PNG: {os.path.basename(strong_heatmap_png)}")
                print(f"          PDF: {os.path.basename(strong_heatmap_pdf)}")
                # ═══════════════════════════════════════════════════════════════════════
                
                # Create scatter plots for ALL features vs target
                print(f"      Creating scatter plots for all {len(sorted_features)} features...")
                for i, (feature, corr_data) in enumerate(sorted_features):
                    try:
                        plt.figure(figsize=(10, 6))
                        
                        feature_data = analysis_df[[feature, target_column]].dropna()
                        
                        # Skip if insufficient data
                        if len(feature_data) < 2:
                            print(f"        Warning: Insufficient data for {feature}, skipping scatter plot")
                            plt.close()
                            continue
                        
                        plt.scatter(feature_data[feature], feature_data[target_column], alpha=0.6, s=15)
                        plt.xlabel(feature, fontsize=12)
                        plt.ylabel(target_column, fontsize=12)
                        
                        # Add correlation info and rank to title
                        plt.title(f'#{i+1}: {feature} vs {target_column}\n'
                                f'Pearson r = {corr_data["pearson"]:.4f} (p = {corr_data["pearson_p_value"]:.4f})\n'
                                f'Spearman ρ = {corr_data["spearman"]:.4f} (p = {corr_data["spearman_p_value"]:.4f})', 
                                fontsize=11)
                        
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        # Clean feature name for filename
                        clean_feature_name = re.sub(r'[^\w\s-]', '', feature).strip().replace(' ', '_')
                        scatter_path = os.path.join(scatter_plots_dir, f"scatter_{i+1:02d}_{clean_feature_name}_{filename.replace('.csv', '.png')}")
                        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        if i < 10:  # Only print first 10 to avoid spam
                            print(f"        Saved scatter plot: {os.path.basename(scatter_path)}")
                        elif i == 10:
                            print(f"        ... (continuing to save scatter plots for remaining {len(sorted_features) - 10} features)")
                            
                    except Exception as e:
                        print(f"        Warning: Could not create scatter plot for {feature}: {e}")
                        plt.close()
                
                # Create feature ranking analysis
                feature_ranking_stats = {
                    'filename': filename,
                    'total_features_analyzed': len(correlations),
                    'dataset_shape': list(df.shape),
                    'analysis_dataset_shape': list(analysis_df.shape),
                    'target_column': target_column,
                    'top_10_features': [
                        {
                            'rank': i + 1,
                            'feature': feature,
                            'pearson_correlation': float(corr_data['pearson']),
                            'spearman_correlation': float(corr_data['spearman']),
                            'pearson_p_value': float(corr_data['pearson_p_value']),
                            'spearman_p_value': float(corr_data['spearman_p_value']),
                            'sample_size': int(corr_data['sample_size'])
                        }
                        for i, (feature, corr_data) in enumerate(sorted_features[:10])
                    ]
                }
                
                # Save feature ranking
                ranking_path = os.path.join(feature_ranking_dir, f"feature_ranking_{filename.replace('.csv', '.json')}")
                with open(ranking_path, 'w') as f:
                    json.dump(feature_ranking_stats, f, indent=2)
                
                print(f"      Saved feature ranking: {os.path.basename(ranking_path)}")
                
                # Save comprehensive results summary
                results_summary_path = os.path.join(analysis_output_dir, f"analysis_summary_{filename.replace('.csv', '.txt')}")
                with open(results_summary_path, 'w') as f:
                    f.write(f"Comprehensive Numeric Correlation Analysis Summary\n")
                    f.write(f"=" * 80 + "\n\n")
                    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Target File: {filename}\n")
                    f.write(f"Target Column: {target_column}\n\n")
                    f.write(f"Dataset Statistics:\n")
                    f.write(f"  Original Shape: {df.shape}\n")
                    f.write(f"  Analysis Shape: {analysis_df.shape}\n")
                    f.write(f"  Features Analyzed: {len(correlations)}\n\n")
                    f.write(f"Top 10 Correlated Features (by absolute Pearson correlation):\n")
                    f.write(f"-" * 80 + "\n")
                    for i, (feature, corr_data) in enumerate(sorted_features[:10], 1):
                        f.write(f"{i:2d}. {feature:30s} | Pearson: {corr_data['pearson']:+.4f} | Spearman: {corr_data['spearman']:+.4f}\n")
                    f.write(f"\n")
                
                print(f"      Saved analysis summary: {os.path.basename(results_summary_path)}")
                
                # Final success message
                print(f"\n    numeric_correlation_analysis: Successfully completed analysis!")
                print(f"      Output directory: {analysis_output_dir}")
                
                return {
                    "success": True,
                    "processed_files": 1,
                    "target_file": target_file_path,
                    "target_variable": target_column,
                    "features_analyzed": len(correlations),
                    "output_path": analysis_output_dir,
                    "correlation_plots": correlation_plots_dir,
                    "scatter_plots": scatter_plots_dir,
                    "analysis_types": ["Feature-to-Feature Correlation Matrix", "Scatter Plots", "Statistical Analysis"]
                }
                
            except Exception as e:
                error_msg = f"Error during numeric correlation analysis: {str(e)}"
                print(f"    numeric_correlation_analysis: {error_msg}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
        
        except Exception as e:
            error_msg = f"numeric_correlation_analysis failed: {str(e)}"
            print(f"    numeric_correlation_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }


    def data_distribution_analysis(self, csv_files=None, target_file=None):
        """
        Data Distribution Analysis Pipeline Stage
        
        Analyzes the distribution of each feature in the training dataset by creating
        histogram plots for all columns. This stage helps understand data distributions
        and identify potential data quality issues or patterns.
        
        Key Analyses Performed:
        1. Histogram distribution plots for each column
        2. Basic distribution statistics (mean, median, std, etc.)
        3. Data quality assessment (missing values, outliers)
        4. Distribution summary report
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (deprecated - use target_file instead)
        target_file : str, optional
            Specific file path to analyze. If None, will look for the first available merged file.
            
        Returns:
        --------
        dict
            Results of the data distribution analysis including plots and statistics
        """
        try:
            print(f"    data_distribution_analysis: Starting comprehensive data distribution analysis...")
            
            # Create output directories
            analysis_output_dir = os.path.join(self.project_root, "data/output/data_distribution_analysis")
            histograms_dir = os.path.join(analysis_output_dir, "histograms")
            statistics_dir = os.path.join(analysis_output_dir, "statistics")
            
            os.makedirs(analysis_output_dir, exist_ok=True)
            os.makedirs(histograms_dir, exist_ok=True)
            os.makedirs(statistics_dir, exist_ok=True)
            
            # Determine target file to analyze
            target_file_path = None
            
            if target_file and os.path.exists(target_file):
                target_file_path = target_file
                print(f"    data_distribution_analysis: Using specified target file: {os.path.basename(target_file)}")
            else:
                # Look for merged data in the specified folder pattern
                merged_data_pattern = os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER, "merged*data**_train.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                if merged_data_files:
                    target_file_path = merged_data_files[0]  # Use first available file
                    print(f"    data_distribution_analysis: Using first available file: {os.path.basename(target_file_path)}")
                    print(f"    data_distribution_analysis: Found {len(merged_data_files)} matching files")
                
            # Check if target file exists
            if not target_file_path or not os.path.exists(target_file_path):
                error_msg = f"No valid merged data file found for distribution analysis. Pattern: {merged_data_pattern}"
                print(f"    data_distribution_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    data_distribution_analysis: Loading dataset from {os.path.basename(target_file_path)}...")
            
            # Read the dataset
            df = pd.read_csv(target_file_path)
            
            if df.empty:
                error_msg = f"Dataset is empty: {target_file_path}"
                print(f"    data_distribution_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            total_rows, total_columns = df.shape
            print(f"    data_distribution_analysis: Dataset shape: {total_rows:,} rows × {total_columns} columns")
            
            # Get all columns for distribution analysis
            all_columns = df.columns.tolist()
            
            # Exclude index columns
            columns_to_exclude = ['Unnamed: 0', 'index']
            analysis_columns = [col for col in all_columns if col not in columns_to_exclude]
            
            print(f"    data_distribution_analysis: Analyzing {len(analysis_columns)} columns")
            
            # Initialize results storage
            distribution_results = {
                "success": True,
                "processed_files": 1,
                "target_file": target_file_path,
                "total_rows": total_rows,
                "total_columns": len(analysis_columns),
                "analysis_columns": analysis_columns,
                "output_path": analysis_output_dir,
                "plots_generated": 0,
                "column_statistics": {}
            }
            
            # Create histogram for each column
            for i, column in enumerate(analysis_columns, 1):
                try:
                    print(f"    data_distribution_analysis: Processing column {i}/{len(analysis_columns)}: {column}")
                    
                    # Get column data
                    column_data = df[column].dropna()  # Remove NaN values for histogram
                    
                    if len(column_data) == 0:
                        print(f"      Warning: Column {column} contains only NaN values, skipping...")
                        continue
                    
                    # Calculate basic statistics
                    stats = {
                        "count": len(column_data),
                        "missing_values": df[column].isnull().sum(),
                        "missing_percentage": (df[column].isnull().sum() / len(df)) * 100,
                        "data_type": str(df[column].dtype),
                        "unique_values": df[column].nunique()
                    }
                    
                    # Add numeric statistics if column is numeric
                    if df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                        stats.update({
                            "mean": column_data.mean(),
                            "median": column_data.median(),
                            "std": column_data.std(),
                            "min": column_data.min(),
                            "max": column_data.max(),
                            "q25": column_data.quantile(0.25),
                            "q75": column_data.quantile(0.75)
                        })
                    
                    distribution_results["column_statistics"][column] = stats
                    
                    # Create histogram
                    plt.figure(figsize=(10, 6))
                    
                    # Handle different data types
                    if df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                        # Special handling for differenceInMinutes_eachStation_offset
                        if column == 'differenceInMinutes_eachStation_offset':
                            # Create bins with 1-minute intervals from -10 to 100
                            bins = np.arange(-10, 101, 1)  # Creates bins: -10, -9, -8, ..., 99, 100
                            plt.hist(column_data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
                            plt.xlim(-10, 50)
                            
                            # Set x-axis ticks to show every minute (step = 1)
                            plt.xticks(np.arange(-10, 51, 2))
                            
                            print(f"      Applied x-axis range [-10, 100] with 1-minute bins and 1-minute tick intervals for {column}")
                            
                            # Add info about data outside the range
                            data_below = (column_data < -10).sum()
                            data_above = (column_data > 100).sum()
                            data_in_range = len(column_data) - data_below - data_above
                            
                            if data_below > 0 or data_above > 0:
                                range_info = f"\nData outside range: {data_below + data_above:,} ({((data_below + data_above)/len(column_data)*100):.1f}%)"
                                range_info += f"\n  < -10: {data_below:,}, > 100: {data_above:,}"
                                print(f"      {column}: {data_in_range:,} values in range [-10,100], {data_below + data_above:,} outside range")
                            else:
                                range_info = ""
                        else:
                            # Standard numeric data - create histogram with automatic bins
                            n_bins = min(50, max(10, int(np.sqrt(len(column_data)))))  # Adaptive bins
                            plt.hist(column_data, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
                            range_info = ""
                        
                        plt.xlabel('Value')
                        
                        # Add statistics text box
                        stats_text = f'Count: {stats["count"]:,}\nMean: {stats["mean"]:.3f}\nStd: {stats["std"]:.3f}\nMissing: {stats["missing_values"]} ({stats["missing_percentage"]:.1f}%){range_info}'
                        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                    elif df[column].dtype == 'bool' or df[column].nunique() <= 10:
                        # Boolean or categorical data with few unique values
                        value_counts = df[column].value_counts()
                        plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
                        plt.xticks(range(len(value_counts)), [str(x) for x in value_counts.index], rotation=45)
                        plt.xlabel('Category')
                        
                        # Add statistics text box
                        stats_text = f'Count: {stats["count"]:,}\nUnique: {stats["unique_values"]}\nMissing: {stats["missing_values"]} ({stats["missing_percentage"]:.1f}%)'
                        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                    else:
                        # Categorical data with many unique values - show top 20
                        value_counts = df[column].value_counts().head(20)
                        plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
                        plt.xticks(range(len(value_counts)), [str(x)[:10] + '...' if len(str(x)) > 10 else str(x) for x in value_counts.index], rotation=45)
                        plt.xlabel('Category (Top 20)')
                        
                        # Add statistics text box
                        stats_text = f'Count: {stats["count"]:,}\nUnique: {stats["unique_values"]}\nMissing: {stats["missing_values"]} ({stats["missing_percentage"]:.1f}%)\nShowing: Top 20'
                        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of {column}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save histogram plot
                    safe_column_name = "".join(c for c in column if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    plot_filename = f"histogram_{safe_column_name.replace(' ', '_')}.png"
                    plot_path = os.path.join(histograms_dir, plot_filename)
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    distribution_results["plots_generated"] += 1
                    
                except Exception as e:
                    print(f"      Error processing column {column}: {str(e)}")
                    continue
            
            print(f"    data_distribution_analysis: Generated {distribution_results['plots_generated']} histogram plots")
            
            # Save distribution statistics summary
            stats_summary_path = os.path.join(statistics_dir, "distribution_statistics_summary.csv")
            
            # Create summary dataframe
            summary_data = []
            for column, stats in distribution_results["column_statistics"].items():
                row = {"column": column}
                row.update(stats)
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(stats_summary_path, index=False)
            print(f"    data_distribution_analysis: Saved statistics summary to {os.path.basename(stats_summary_path)}")
            
            # Create comprehensive analysis report
            report_path = os.path.join(analysis_output_dir, "distribution_analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write("DATA DISTRIBUTION ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {os.path.basename(target_file_path)}\n")
                f.write(f"Total Rows: {total_rows:,}\n")
                f.write(f"Total Columns Analyzed: {len(analysis_columns)}\n")
                f.write(f"Histograms Generated: {distribution_results['plots_generated']}\n\n")
                
                # Data Quality Summary
                f.write("DATA QUALITY SUMMARY\n")
                f.write("-" * 30 + "\n")
                
                columns_with_missing = sum(1 for stats in distribution_results["column_statistics"].values() if stats["missing_values"] > 0)
                f.write(f"Columns with missing values: {columns_with_missing}/{len(analysis_columns)}\n")
                
                # Find columns with high missing percentage
                high_missing_cols = [(col, stats) for col, stats in distribution_results["column_statistics"].items() 
                                if stats["missing_percentage"] > 50]
                
                if high_missing_cols:
                    f.write(f"\nColumns with >50% missing data:\n")
                    for col, stats in high_missing_cols:
                        f.write(f"  - {col}: {stats['missing_percentage']:.1f}% missing\n")
                
                # Data type distribution
                f.write(f"\nDATA TYPE DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                type_counts = {}
                for stats in distribution_results["column_statistics"].values():
                    dtype = stats["data_type"]
                    type_counts[dtype] = type_counts.get(dtype, 0) + 1
                
                for dtype, count in sorted(type_counts.items()):
                    f.write(f"{dtype}: {count} columns\n")
            
            print(f"    data_distribution_analysis: Saved comprehensive report to {os.path.basename(report_path)}")
            print(f"    data_distribution_analysis: Analysis completed successfully!")
            print(f"      ✓ Files processed: 1")
            print(f"      ✓ Columns analyzed: {len(analysis_columns)}")
            print(f"      ✓ Histograms generated: {distribution_results['plots_generated']}")
            print(f"      ✓ Output saved to: {analysis_output_dir}")
            
            return distribution_results
            
        except Exception as e:
            error_msg = f"data_distribution_analysis failed: {str(e)}"
            print(f"    data_distribution_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }
        
    def target_feature_analysis(self, target_file=None):
        """
        Analyzes target features and creates IEEE-standard publication-ready visualizations.
        
        Creates comprehensive distribution and statistical analysis for each target feature:
        - Regression features: Delay distribution with multiple thresholds
        - Classification features: Category distribution plots
        - Unknown features: Appropriate visualization based on data type
        
        IEEE-compliant plot specifications:
        - Font: Serif 12pt
        - Figure size: One-column width (3.5 inches)
        - Line width: 1.0pt for main lines
        - Marker size: 3pt
        - Grid lines: 0.5pt
        - Format: PDF (vector format)
        - No titles (use figure captions instead)
        - Y-axis: 30% headroom above maximum value
        - Labels: Normal weight (non-bold)
        - Side-by-side plots: Y-axis only on left plot, consistent scale
        
        Parameters:
        -----------
        target_file : str, optional
            Specific file path to analyze. If None, will look for the first available merged file.
            
        Returns:
        --------
        dict
            Results of the target feature analysis including plots and statistics for all features
        """
        try:
            # Set seaborn palette first
            sns.set_palette("husl")
            
            # IEEE-compliant plot settings - ALL FONTS SET TO 12, SERIF FAMILY
            # Applied AFTER style reset to ensure they take effect
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 12,
                'axes.labelsize': 12,
                'axes.titlesize': 12,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 12,
                'lines.linewidth': 1.0,
                'lines.markersize': 3,
                'grid.linewidth': 0.5,
                'axes.linewidth': 0.5,
                'xtick.major.width': 0.5,
                'ytick.major.width': 0.5,
                'pdf.fonttype': 42,  # TrueType fonts
                'ps.fonttype': 42
            })
            
            print(f"    target_feature_analysis: Starting comprehensive target feature analysis...")
            print(f"    target_feature_analysis: Analyzing {len(VALID_TARGET_FEATURES)} target features: {VALID_TARGET_FEATURES}")
            print(f"    target_feature_analysis: Delay threshold: {TRAIN_DELAY_MINUTES} minutes")
            
            # Create output directories
            analysis_output_dir = os.path.join(self.project_root, "data/output/target_feature_analysis")
            plots_dir = os.path.join(analysis_output_dir, "plots")
            
            os.makedirs(analysis_output_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Determine target file to analyze
            target_file_path = None
            
            if target_file and os.path.exists(target_file):
                target_file_path = target_file
                print(f"    target_feature_analysis: Using specified target file: {os.path.basename(target_file)}")
            else:
                # Look for merged data in the specified folder pattern
                merged_data_pattern = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                if merged_data_files:
                    target_file_path = merged_data_files[0]  # Use first available file
                    print(f"    target_feature_analysis: Using first available file: {os.path.basename(target_file_path)}")
                    print(f"    target_feature_analysis: Found {len(merged_data_files)} matching files")
            
            # Check if target file exists
            if not target_file_path or not os.path.exists(target_file_path):
                error_msg = f"No valid merged data file found for target feature analysis."
                error_msg += f"\nPattern searched: {os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER, 'merged_data_*.csv')}"
                print(f"    target_feature_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    target_feature_analysis: Loading dataset from {os.path.basename(target_file_path)}...")
            
            # Read the dataset
            df = pd.read_csv(target_file_path)
            
            if df.empty:
                error_msg = f"Dataset is empty: {target_file_path}"
                print(f"    target_feature_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            total_rows = len(df)
            print(f"    target_feature_analysis: Dataset shape: {total_rows:,} rows")
            
            # Initialize results tracking
            all_feature_results = {}
            total_plots_created = 0
            successful_analyses = 0
            
            # Process each target feature
            for feature_idx, target_feature in enumerate(VALID_TARGET_FEATURES, 1):
                print(f"\n    target_feature_analysis: [{feature_idx}/{len(VALID_TARGET_FEATURES)}] Analyzing target feature: {target_feature}")
                
                # Check if target feature exists
                if target_feature not in df.columns:
                    print(f"    target_feature_analysis: Warning - Target feature '{target_feature}' not found in dataset. Skipping.")
                    all_feature_results[target_feature] = {
                        "success": False,
                        "error": f"Feature '{target_feature}' not found in dataset",
                        "available_columns": list(df.columns)
                    }
                    continue
                
                # Extract target feature data
                target_data = df[target_feature].dropna()
                target_rows = len(target_data)
                
                if target_rows == 0:
                    print(f"    target_feature_analysis: Warning - No valid target data found for '{target_feature}' (all values are NaN). Skipping.")
                    all_feature_results[target_feature] = {
                        "success": False,
                        "error": f"No valid data found for '{target_feature}' (all values are NaN)",
                        "total_rows": total_rows,
                        "valid_rows": 0
                    }
                    continue
                
                print(f"    target_feature_analysis: Valid target data for '{target_feature}': {target_rows:,} rows")
                
                # Determine if this is a regression or classification problem
                is_regression = target_feature in REGRESSION_PROBLEM
                is_classification = target_feature in CLASSIFICATION_PROBLEM
                
                if is_regression:
                    # For continuous/regression target features - create delay analysis plots
                    print(f"    target_feature_analysis: Creating IEEE-compliant regression analysis plots for '{target_feature}'...")
                    
                    # Calculate statistics for different delay definitions
                    # Definition 1: Delayed if >= 0 (any positive delay)
                    delayed_zero_threshold = target_data > 0
                    delayed_zero_count = delayed_zero_threshold.sum()
                    non_delayed_zero_count = target_rows - delayed_zero_count
                    delayed_zero_percentage = (delayed_zero_count / target_rows) * 100
                    non_delayed_zero_percentage = (non_delayed_zero_count / target_rows) * 100
                    
                    # Definition 2: Delayed if >= TRAIN_DELAY_MINUTES
                    delayed_threshold = target_data >= TRAIN_DELAY_MINUTES
                    delayed_threshold_count = delayed_threshold.sum()
                    non_delayed_threshold_count = target_rows - delayed_threshold_count
                    delayed_threshold_percentage = (delayed_threshold_count / target_rows) * 100
                    non_delayed_threshold_percentage = (non_delayed_threshold_count / target_rows) * 100
                    
                    print(f"    target_feature_analysis: Statistics for '{target_feature}':")
                    print(f"      Zero threshold (>=0): {delayed_zero_count:,} delayed ({delayed_zero_percentage:.1f}%)")
                    print(f"      {TRAIN_DELAY_MINUTES}min threshold (>={TRAIN_DELAY_MINUTES}): {delayed_threshold_count:,} delayed ({delayed_threshold_percentage:.1f}%)")
                    
                    # Create IEEE-compliant figure with two subplots
                    # One-column width: 3.5 inches
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
                    
                    # Prepare data for both plots
                    categories_zero = ['Non-delayed\n(≤0 min)', 'Delayed\n(>0 min)']
                    counts_zero = [non_delayed_zero_count, delayed_zero_count]
                    percentages_zero = [non_delayed_zero_percentage, delayed_zero_percentage]
                    colors_zero = ['#2E8B57', '#DC143C']  # SeaGreen, Crimson
                    
                    categories_threshold = [f'Non-delayed\n(<{TRAIN_DELAY_MINUTES} min)', 
                                        f'Delayed\n(≥{TRAIN_DELAY_MINUTES} min)']
                    counts_threshold = [non_delayed_threshold_count, delayed_threshold_count]
                    percentages_threshold = [non_delayed_threshold_percentage, delayed_threshold_percentage]
                    colors_threshold = ['#4682B4', '#FF6347']  # SteelBlue, Tomato
                    
                    # Calculate overall maximum for consistent scaling across both plots
                    overall_max = max(max(counts_zero), max(counts_threshold))
                    
                    # Plot 1: Binary classification with zero threshold (>0)
                    bars1 = ax1.bar(categories_zero, counts_zero, color=colors_zero, alpha=0.7, 
                                edgecolor='black', linewidth=0.5)
                    ax1.set_ylabel('Number of Trains', fontsize=12)
                    ax1.set_xlabel('Train Status', fontsize=12)
                    ax1.tick_params(axis='both', labelsize=12)
                    
                    # Add percentage labels on bars (percentage only)
                    for bar, percentage in zip(bars1, percentages_zero):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{percentage:.1f}%',
                            ha='center', va='bottom', fontsize=12)
                    
                    # Add grid for better readability
                    ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
                    ax1.set_ylim(0, overall_max * 1.3)
                    
                    # Plot 2: Binary classification with threshold (>=TRAIN_DELAY_MINUTES)
                    bars2 = ax2.bar(categories_threshold, counts_threshold, color=colors_threshold, alpha=0.7, 
                                edgecolor='black', linewidth=0.5)
                    ax2.set_xlabel('Train Status', fontsize=12)
                    ax2.tick_params(axis='both', labelsize=12)
                    ax2.set_yticklabels([])  # Remove y-axis labels for right plot
                    
                    # Add percentage labels on bars (percentage only)
                    for bar, percentage in zip(bars2, percentages_threshold):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{percentage:.1f}%',
                            ha='center', va='bottom', fontsize=12)
                    
                    # Add grid for better readability
                    ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
                    ax2.set_ylim(0, overall_max * 1.3)
                    
                    plt.tight_layout()
                    
                    # Save the plot in PDF format
                    safe_feature_name = target_feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    plot_path = os.path.join(plots_dir, f"target_feature_{safe_feature_name}_delay_distribution.pdf")
                    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
                    plt.close()
                    
                    print(f"    target_feature_analysis: Saved IEEE-compliant delay distribution plot: {os.path.basename(plot_path)}")
                    
                    # Store results for this feature
                    all_feature_results[target_feature] = {
                        "success": True,
                        "feature_type": "regression",
                        "total_rows": total_rows,
                        "valid_rows": target_rows,
                        "statistics": {
                            "zero_threshold": {
                                "delayed_count": int(delayed_zero_count),
                                "non_delayed_count": int(non_delayed_zero_count),
                                "delayed_percentage": delayed_zero_percentage,
                                "non_delayed_percentage": non_delayed_zero_percentage
                            },
                            "train_delay_threshold": {
                                "delayed_count": int(delayed_threshold_count),
                                "non_delayed_count": int(non_delayed_threshold_count),
                                "delayed_percentage": delayed_threshold_percentage,
                                "non_delayed_percentage": non_delayed_threshold_percentage,
                                "threshold_minutes": TRAIN_DELAY_MINUTES
                            },
                            "basic_stats": {
                                "mean": float(target_data.mean()),
                                "median": float(target_data.median()),
                                "std": float(target_data.std()),
                                "min": float(target_data.min()),
                                "max": float(target_data.max())
                            }
                        },
                        "plots_created": 1,
                        "plot_path": plot_path
                    }
                    total_plots_created += 1
                        
                elif is_classification:
                    # For categorical/classification target features - create distribution plots
                    print(f"    target_feature_analysis: Creating IEEE-compliant classification analysis plots for '{target_feature}'...")
                    
                    # Get value counts and proportions
                    value_counts = target_data.value_counts()
                    value_proportions = target_data.value_counts(normalize=True) * 100
                    
                    print(f"    target_feature_analysis: Value distribution for '{target_feature}':")
                    for value, count in value_counts.items():
                        percentage = value_proportions[value]
                        print(f"      {value}: {count:,} ({percentage:.1f}%)")
                    
                    # Create IEEE-compliant single plot for categorical data
                    # One-column width: 3.5 inches
                    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
                    
                    # Create bar plot
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(value_counts)]
                    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                                color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # Set labels (IEEE standard: no title)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.set_xlabel('Categories', fontsize=12)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels([str(x) for x in value_counts.index], 
                                    rotation=45 if len(value_counts) > 3 else 0, fontsize=12)
                    ax.tick_params(axis='both', labelsize=12)
                    
                    # Add percentage labels on bars (percentage only)
                    for bar, percentage in zip(bars, value_proportions.values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{percentage:.1f}%',
                            ha='center', va='bottom', fontsize=12)
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)
                    ax.set_ylim(0, max(value_counts.values) * 1.3)
                    
                    plt.tight_layout()
                    
                    # Save the plot in PDF format
                    safe_feature_name = target_feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    plot_path = os.path.join(plots_dir, f"target_feature_{safe_feature_name}_distribution.pdf")
                    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
                    plt.close()
                    
                    print(f"    target_feature_analysis: Saved IEEE-compliant distribution plot: {os.path.basename(plot_path)}")
                    
                    # Store results for this feature
                    all_feature_results[target_feature] = {
                        "success": True,
                        "feature_type": "classification",
                        "total_rows": total_rows,
                        "valid_rows": target_rows,
                        "statistics": {
                            "value_counts": {str(k): int(v) for k, v in value_counts.items()},
                            "value_percentages": {str(k): float(v) for k, v in value_proportions.items()},
                            "unique_values": len(value_counts),
                            "most_frequent": str(value_counts.index[0]),
                            "most_frequent_count": int(value_counts.iloc[0]),
                            "most_frequent_percentage": float(value_proportions.iloc[0])
                        },
                        "plots_created": 1,
                        "plot_path": plot_path
                    }
                    total_plots_created += 1
                    
                else:
                    # Handle unknown feature type
                    print(f"    target_feature_analysis: Warning - '{target_feature}' not found in REGRESSION_PROBLEM or CLASSIFICATION_PROBLEM lists.")
                    print(f"    target_feature_analysis: Creating basic IEEE-compliant distribution analysis...")
                    
                    # Determine if numeric or categorical based on data type
                    is_numeric = pd.api.types.is_numeric_dtype(target_data)
                    
                    if is_numeric:
                        # Treat as regression-like
                        print(f"    target_feature_analysis: Treating '{target_feature}' as numeric/continuous...")
                        
                        # Basic statistics
                        mean_val = target_data.mean()
                        median_val = target_data.median()
                        std_val = target_data.std()
                        min_val = target_data.min()
                        max_val = target_data.max()
                        
                        print(f"    target_feature_analysis: Basic statistics for '{target_feature}':")
                        print(f"      Mean: {mean_val:.4f}, Median: {median_val:.4f}, Std: {std_val:.4f}")
                        print(f"      Range: [{min_val:.4f}, {max_val:.4f}]")
                        
                        # Create IEEE-compliant histogram
                        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
                        
                        ax.hist(target_data, bins=50, alpha=0.7, color='skyblue', 
                            edgecolor='black', linewidth=0.5)
                        ax.set_xlabel(target_feature, fontsize=12)
                        ax.set_ylabel('Frequency', fontsize=12)
                        ax.tick_params(axis='both', labelsize=12)
                        ax.grid(True, alpha=0.2, linewidth=0.5)
                        
                        # Add statistics text box
                        stats_text = (f'μ={mean_val:.2f}, σ={std_val:.2f}\n'
                                    f'Range: [{min_val:.2f}, {max_val:.2f}]')
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, 
                                    linewidth=0.5))
                        
                        plt.tight_layout()
                        
                        # Save the plot
                        safe_feature_name = target_feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        plot_path = os.path.join(plots_dir, f"target_feature_{safe_feature_name}_histogram.pdf")
                        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
                        plt.close()
                        
                        print(f"    target_feature_analysis: Saved IEEE-compliant histogram plot: {os.path.basename(plot_path)}")
                        
                        # Store results
                        all_feature_results[target_feature] = {
                            "success": True,
                            "feature_type": "numeric_unknown",
                            "total_rows": total_rows,
                            "valid_rows": target_rows,
                            "statistics": {
                                "basic_stats": {
                                    "mean": float(mean_val),
                                    "median": float(median_val),
                                    "std": float(std_val),
                                    "min": float(min_val),
                                    "max": float(max_val)
                                }
                            },
                            "plots_created": 1,
                            "plot_path": plot_path
                        }
                        total_plots_created += 1
                        
                    else:
                        # Treat as categorical
                        print(f"    target_feature_analysis: Treating '{target_feature}' as categorical...")
                        
                        # Get value counts
                        value_counts = target_data.value_counts()
                        value_proportions = target_data.value_counts(normalize=True) * 100
                        
                        print(f"    target_feature_analysis: Value distribution for '{target_feature}':")
                        for value, count in value_counts.head(10).items():  # Show top 10
                            percentage = value_proportions[value]
                            print(f"      {value}: {count:,} ({percentage:.1f}%)")
                        if len(value_counts) > 10:
                            print(f"      ... and {len(value_counts) - 10} more categories")
                        
                        # Create IEEE-compliant bar plot (show top 20 categories)
                        top_categories = min(20, len(value_counts))
                        plot_value_counts = value_counts.head(top_categories)
                        plot_value_proportions = value_proportions.head(top_categories)
                        
                        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))  # Slightly taller for many categories
                        
                        # Create horizontal bar plot for better label readability
                        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_value_counts)))
                        bars = ax.barh(range(len(plot_value_counts))[::-1], plot_value_counts.values, 
                                    color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                        
                        ax.set_xlabel('Count', fontsize=12)
                        ax.set_ylabel('Categories', fontsize=12)
                        ax.set_yticks(range(len(plot_value_counts))[::-1])
                        ax.set_yticklabels([str(x)[:20] + '...' if len(str(x)) > 20 else str(x) 
                                        for x in plot_value_counts.index], fontsize=12)
                        ax.tick_params(axis='both', labelsize=12)
                        
                        # Add percentage labels on bars (percentage only)
                        for i, (bar, percentage) in enumerate(zip(bars, plot_value_proportions.values)):
                            width = bar.get_width()
                            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                                f'{percentage:.1f}%',
                                ha='left', va='center', fontsize=12)
                        
                        # Add grid for better readability
                        ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
                        ax.set_xlim(0, max(plot_value_counts.values) * 1.3)
                        
                        plt.tight_layout()
                        
                        # Save the plot
                        safe_feature_name = target_feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        plot_path = os.path.join(plots_dir, f"target_feature_{safe_feature_name}_categories.pdf")
                        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
                        plt.close()
                        
                        print(f"    target_feature_analysis: Saved IEEE-compliant categories plot: {os.path.basename(plot_path)}")
                        
                        # Store results
                        all_feature_results[target_feature] = {
                            "success": True,
                            "feature_type": "categorical_unknown",
                            "total_rows": total_rows,
                            "valid_rows": target_rows,
                            "statistics": {
                                "value_counts": {str(k): int(v) for k, v in value_counts.head(50).items()},  # Top 50
                                "value_percentages": {str(k): float(v) for k, v in value_proportions.head(50).items()},
                                "unique_values": len(value_counts),
                                "most_frequent": str(value_counts.index[0]),
                                "most_frequent_count": int(value_counts.iloc[0]),
                                "most_frequent_percentage": float(value_proportions.iloc[0])
                            },
                            "plots_created": 1,
                            "plot_path": plot_path
                        }
                        total_plots_created += 1
                
                successful_analyses += 1
                print(f"    target_feature_analysis: ✓ Successfully analyzed '{target_feature}'")
            
            # Create summary report
            print(f"\n    target_feature_analysis: Creating summary analysis...")
            
            # Save detailed results to JSON
            results_json_path = os.path.join(analysis_output_dir, "target_features_analysis_results.json")
            with open(results_json_path, 'w') as f:
                json.dump(all_feature_results, f, indent=2, default=str)
            
            # Create summary text report
            summary_report_path = os.path.join(analysis_output_dir, "target_features_summary_report.txt")
            with open(summary_report_path, 'w') as f:
                f.write("TARGET FEATURES ANALYSIS SUMMARY REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {os.path.basename(target_file_path)}\n")
                f.write(f"Total Dataset Rows: {total_rows:,}\n\n")
                f.write(f"Target Features Analyzed: {len(VALID_TARGET_FEATURES)}\n")
                f.write(f"Successful Analyses: {successful_analyses}\n")
                f.write(f"Failed Analyses: {len(VALID_TARGET_FEATURES) - successful_analyses}\n")
                f.write(f"Total Plots Created: {total_plots_created}\n")
                f.write(f"Plot Format: IEEE-compliant PDF (vector)\n\n")
                
                f.write("="*60 + "\n")
                f.write("FEATURE ANALYSIS DETAILS\n")
                f.write("="*60 + "\n\n")
                
                for feature, result in all_feature_results.items():
                    f.write(f"Feature: {feature}\n")
                    f.write("-"*40 + "\n")
                    
                    if result.get("success"):
                        f.write(f"  Status: SUCCESS\n")
                        f.write(f"  Feature Type: {result.get('feature_type', 'unknown')}\n")
                        f.write(f"  Valid Rows: {result.get('valid_rows', 0):,}\n")
                        
                        if 'statistics' in result:
                            stats = result['statistics']
                            
                            if 'zero_threshold' in stats:
                                f.write(f"  Zero Threshold Analysis:\n")
                                f.write(f"    Delayed: {stats['zero_threshold']['delayed_count']:,} ({stats['zero_threshold']['delayed_percentage']:.1f}%)\n")
                                f.write(f"    Non-delayed: {stats['zero_threshold']['non_delayed_count']:,} ({stats['zero_threshold']['non_delayed_percentage']:.1f}%)\n")
                            
                            if 'train_delay_threshold' in stats:
                                f.write(f"  {stats['train_delay_threshold']['threshold_minutes']}-min Threshold Analysis:\n")
                                f.write(f"    Delayed: {stats['train_delay_threshold']['delayed_count']:,} ({stats['train_delay_threshold']['delayed_percentage']:.1f}%)\n")
                                f.write(f"    Non-delayed: {stats['train_delay_threshold']['non_delayed_count']:,} ({stats['train_delay_threshold']['non_delayed_percentage']:.1f}%)\n")
                            
                            if 'basic_stats' in stats:
                                f.write(f"  Basic Statistics:\n")
                                f.write(f"    Mean: {stats['basic_stats']['mean']:.4f}\n")
                                f.write(f"    Median: {stats['basic_stats']['median']:.4f}\n")
                                f.write(f"    Std Dev: {stats['basic_stats']['std']:.4f}\n")
                                f.write(f"    Range: [{stats['basic_stats']['min']:.4f}, {stats['basic_stats']['max']:.4f}]\n")
                            
                            if 'value_counts' in stats:
                                f.write(f"  Value Distribution (Top 10):\n")
                                for i, (value, count) in enumerate(list(stats['value_counts'].items())[:10]):
                                    percentage = stats['value_percentages'].get(value, 0)
                                    f.write(f"    {value}: {count:,} ({percentage:.1f}%)\n")
                    else:
                        f.write(f"  Status: FAILED\n")
                        f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
                    
                    f.write("\n")
            
            print(f"    target_feature_analysis: Analysis completed successfully!")
            print(f"    target_feature_analysis: Plots saved to: {plots_dir}")
            print(f"    target_feature_analysis: Results saved to: {analysis_output_dir}")
            
            return {
                "success": True,
                "processed_files": 1,
                "target_file": target_file_path,
                "total_rows": total_rows,
                "target_features_analyzed": len(VALID_TARGET_FEATURES),
                "successful_analyses": successful_analyses,
                "failed_analyses": len(VALID_TARGET_FEATURES) - successful_analyses,
                "plots_created": total_plots_created,
                "output_directory": analysis_output_dir,
                "plots_directory": plots_dir,
                "results_json": results_json_path,
                "summary_report": summary_report_path,
                "feature_results": all_feature_results
            }
            
        except Exception as e:
            error_msg = f"target_feature_analysis failed: {str(e)}"
            print(f"    target_feature_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }