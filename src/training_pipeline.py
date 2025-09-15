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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler

import xgboost as xgb

from imblearn.over_sampling import BorderlineSMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, recall_score, roc_auc_score, 
    average_precision_score, cohen_kappa_score, f1_score,
    classification_report, confusion_matrix, roc_curve, r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns
            
from src.file_utils import (
    format_param_distributions_for_json,
    generate_feature_importance_report, 
    generate_output_path,
    optimize_threshold_xgboost,
    plot_feature_importance,
    plot_precision_recall_xgboost,
    plot_roc_curve_smote,
    plot_precision_recall_curve_smote,
    plot_roc_curve_xgboost,
    plot_threshold_analysis_smote,
    plot_threshold_summary_smote,
    plot_smote_summary,
    create_correlation_plot,
    create_distribution_plots,
    create_combined_correlation_analysis,
    create_non_weather_correlation_plot,
    create_combined_non_weather_correlation_analysis,
    calculate_classification_metrics,
    save_confusion_matrix,
    find_optimal_thresholds,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_threshold_analysis,
    plot_threshold_summary,
    save_nan_cleanup_summary
)

from config.const import (
    OUTPUT_FOLDER,
)

from config.const_preprocessing import (
    PREPROCESSED_OUTPUT_FOLDER,
    DATA_FILE_PREFIX_FOR_TRAINING,
    DEFAULT_TARGET_FEATURE,
    ALL_WEATHER_FEATURES,
    TRAINING_READY_OUTPUT_FOLDER,
    USE_SIN_COS_APPROACH,
    VALID_TARGET_FEATURES,
    REGRESSION_PROBLEM,
    CLASSIFICATION_PROBLEM,
    TRAIN_DELAY_MINUTES,
    SELECTED_WEATHER_FEATURES,
)

from config.const_training import (
    MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER,
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    XGBOOST_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER,
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
    THRESHOLD_OPTIMIZATION_CONFIG,
    MERGED_TRAINING_READY_OUTPUT_FOLDER,
    TEST_SIZE,
    SCORE_METRIC,
    RANDOM_SEARCH_ITERATIONS,
    RANDOM_SEARCH_CV_FOLDS,
    DECISION_TREE_PARAM_DISTRIBUTIONS,    
    WEIGHT_DELAY_COLUMN,
    MAX_SAMPLE_WEIGHT_CLASSIFICATION,
    DECISION_TREE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER,
    BORDERLINE_SMOTE_OUTPUT_FOLDER,
    BORDERLINE_SMOTE_CONFIG,
    XGBOOST_PARAM_DISTRIBUTIONS,
    XGBOOST_METHODS_CONFIG,
    MAX_SAMPLE_WEIGHT_REGRESSION,
    XGBOOST_SELECTED_FEATURES_OUTPUT_FOLDER
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

        if state_machine.get("select_training_cols", False):
            try:
                print(f"    → select_training_cols")
                cols_selection_result = self.select_training_cols(
                    data=result["data"] if result["data"] is not None else None
                )
                
                if cols_selection_result and cols_selection_result.get("success", False):
                    # Note: This stage doesn't modify data, just displays column info
                    result["steps_executed"].append("select_training_cols")
                    result["column_analysis"] = {
                        "total_columns": cols_selection_result.get("total_columns", 0),
                        "file_analyzed": cols_selection_result.get("file_path", ""),
                        "dataset_shape": cols_selection_result.get("dataset_shape", (0, 0)),
                        "column_types": cols_selection_result.get("column_types", {})
                    }
                    print(f"      ✓ Successfully analyzed training columns")
                    print(f"      ✓ Total columns found: {cols_selection_result.get('total_columns', 0)}")
                    print(f"      ✓ File analyzed: {os.path.basename(cols_selection_result.get('file_path', ''))}")
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

        if state_machine.get("select_time_features", False):
            try:
                print(f"    → select_time_features")
                time_selection_result = self.select_time_features(
                    data=result["data"] if result["data"] is not None else None
                )
                
                if time_selection_result and time_selection_result.get("success", False):
                    result["data"] = time_selection_result.get("data")
                    result["steps_executed"].append("select_time_features")
                    result["time_feature_selection"] = time_selection_result.get("selection_summary")
                    
                    selection_summary = time_selection_result.get("selection_summary", {})
                    approach = selection_summary.get("approach_description", "unknown")
                    kept_count = selection_summary.get("columns_kept", 0)
                    dropped_count = selection_summary.get("columns_dropped", 0)
                    
                    print(f"      ✓ Time feature selection completed ({approach}: kept {kept_count}, dropped {dropped_count} columns)")
                else:
                    error_msg = time_selection_result.get("error", "select_time_features failed")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"select_time_features failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ select_time_features (disabled)")

        if state_machine.get("drop_nan_columns", False):
            try:
                print(f"    → drop_nan_columns")
                
                # Use the data from merge_data_files if available
                cleanup_result = self.drop_nan_columns(
                    data=result["data"] if result["data"] is not None else None
                )
                
                if cleanup_result and cleanup_result.get("success", False):
                    result["data"] = cleanup_result.get("data")  # Update with cleaned data
                    result["steps_executed"].append("drop_nan_columns")
                    print(f"      ✓ Successfully cleaned NaN columns - dropped {cleanup_result.get('columns_dropped', 0)} columns")
                    result["success"] = True
                else:
                    error_msg = cleanup_result.get("error", "drop_nan_columns returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"drop_nan_columns failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ drop_nan_columns (disabled)")

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

        if state_machine.get("correlation_analysis_by_station", False):
            try:
                print(f"    → correlation_analysis_by_station (Delayed Trains Only)")
                delayed_correlation_result = self.correlation_analysis_by_station(csv_files)
                
                if delayed_correlation_result and delayed_correlation_result.get("success", False):
                    result["steps_executed"].append("correlation_analysis_by_station")
                    result["file_info"]["delayed_correlation_processed_files"] = delayed_correlation_result.get("processed_files", 0)
                    print(f"      ✓ Successfully completed delayed trains correlation analysis")
                    print(f"      ✓ Files analyzed: {delayed_correlation_result.get('processed_files', 0)}")
                    print(f"      ✓ Total delayed trains: {delayed_correlation_result.get('total_delayed_trains', 0):,}")
                    print(f"      ✓ Analysis type: {delayed_correlation_result.get('analysis_type', 'Delayed Trains Correlation')}")
                    print(f"      ✓ Results saved to: {delayed_correlation_result.get('output_path', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = delayed_correlation_result.get("error", "correlation_analysis_by_station returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"correlation_analysis_by_station failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ correlation_analysis_by_station (Delayed Trains) (disabled)")

        if state_machine.get("snow_depth_delay_analysis", False):
            try:
                print(f"    → snow_depth_delay_analysis (Snow Depth vs Train Delays)")
                snow_depth_analysis_result = self.snow_depth_delay_analysis(csv_files)
                
                if snow_depth_analysis_result and snow_depth_analysis_result.get("success", False):
                    result["steps_executed"].append("snow_depth_delay_analysis")
                    result["file_info"]["snow_depth_analysis_processed_files"] = snow_depth_analysis_result.get("processed_files", 0)
                    print(f"      ✓ Successfully completed snow depth delay analysis")
                    print(f"      ✓ Files analyzed: {snow_depth_analysis_result.get('processed_files', 0)}")
                    print(f"      ✓ Total records analyzed: {snow_depth_analysis_result.get('total_records_analyzed', 0):,}")
                    print(f"      ✓ Analysis types: {', '.join(snow_depth_analysis_result.get('analysis_types', []))}")
                    print(f"      ✓ Results saved to: {snow_depth_analysis_result.get('output_path', 'N/A')}")
                    
                    # Display key findings if available
                    if snow_depth_analysis_result.get('combined_analysis'):
                        combined = snow_depth_analysis_result['combined_analysis']
                        if 'overall_correlations' in combined:
                            pearson_r = combined['overall_correlations']['pearson_correlation']
                            print(f"      ✓ Overall Pearson correlation: {pearson_r:.4f}")
                            
                            if combined.get('delay_summary'):
                                delay_diff = combined['delay_summary']['delay_difference']
                                if abs(delay_diff) > 0.1:  # Only show if meaningful difference
                                    if delay_diff > 0:
                                        print(f"      ✓ Snow increases delays by {delay_diff:.2f} minutes on average")
                                    else:
                                        print(f"      ✓ Snow decreases delays by {abs(delay_diff):.2f} minutes on average")
                                else:
                                    print(f"      ✓ No significant delay difference between snowy and clear conditions")
                    
                    result["success"] = True
                else:
                    error_msg = snow_depth_analysis_result.get("error", "snow_depth_delay_analysis returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"snow_depth_delay_analysis failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ snow_depth_delay_analysis (Snow Depth vs Train Delays) (disabled)")

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

        if state_machine.get("train_decision_tree_with_borderline_smote_data", False):
            try:
                print(f"    → train_decision_tree_with_borderline_smote_data")
                smote_dt_result = self.train_decision_tree_with_borderline_smote_data()
                
                if smote_dt_result and smote_dt_result.get("success", False):
                    result["steps_executed"].append("train_decision_tree_with_borderline_smote_data")
                    result["file_info"]["smote_dt_files_processed"] = smote_dt_result.get("files_processed", 0)
                    result["file_info"]["smote_dt_train_samples"] = smote_dt_result.get("total_train_samples", 0)
                    result["file_info"]["smote_dt_test_samples"] = smote_dt_result.get("total_test_samples", 0)
                    print(f"      ✓ Successfully trained Decision Trees with BorderlineSMOTE data")
                    print(f"      ✓ Target feature: {smote_dt_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Files processed: {smote_dt_result.get('files_processed', 0)}")
                    print(f"      ✓ Total train samples: {smote_dt_result.get('total_train_samples', 0):,}")
                    print(f"      ✓ Total test samples: {smote_dt_result.get('total_test_samples', 0):,}")
                    print(f"      ✓ Average test accuracy: {smote_dt_result.get('average_test_accuracy', 0):.4f}")
                    print(f"      ✓ Average test F1: {smote_dt_result.get('average_test_f1', 0):.4f}")
                    print(f"      ✓ Average test precision: {smote_dt_result.get('average_test_precision', 0):.4f}")
                    print(f"      ✓ Average test recall: {smote_dt_result.get('average_test_recall', 0):.4f}")
                    print(f"      ✓ Average CV score: {smote_dt_result.get('average_cv_score', 0):.4f}")
                    print(f"      ✓ Results saved to: {smote_dt_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = smote_dt_result.get("error", "train_decision_tree_with_borderline_smote_data returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"train_decision_tree_with_borderline_smote_data failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_decision_tree_with_borderline_smote_data (disabled)")

        if state_machine.get("threshold_optimization_decision_tree_borderline_smote", False):
            try:
                print(f"    → threshold_optimization_decision_tree_borderline_smote")
                smote_threshold_opt_result = self.threshold_optimization_decision_tree_borderline_smote()
                
                if smote_threshold_opt_result and smote_threshold_opt_result.get("success", False):
                    result["steps_executed"].append("threshold_optimization_decision_tree_borderline_smote")
                    result["file_info"]["smote_threshold_optimized_models"] = smote_threshold_opt_result.get("models_optimized", 0)
                    print(f"      ✓ Successfully optimized BorderlineSMOTE decision tree thresholds")
                    print(f"      ✓ Target feature: {smote_threshold_opt_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ SMOTE models optimized: {smote_threshold_opt_result.get('models_optimized', 0)}")
                    print(f"      ✓ Average optimal threshold: {smote_threshold_opt_result.get('average_optimal_threshold', 0):.3f}")
                    print(f"      ✓ Average optimized F1: {smote_threshold_opt_result.get('average_optimized_f1', 0):.3f}")
                    print(f"      ✓ Average optimized Precision: {smote_threshold_opt_result.get('average_optimized_precision', 0):.3f}")
                    print(f"      ✓ Average optimized Recall: {smote_threshold_opt_result.get('average_optimized_recall', 0):.3f}")
                    print(f"      ✓ Results saved to: {smote_threshold_opt_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = smote_threshold_opt_result.get("error", "threshold_optimization_decision_tree_borderline_smote returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"threshold_optimization_decision_tree_borderline_smote failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ threshold_optimization_decision_tree_borderline_smote (disabled)")

        if state_machine.get("train_xgboost_with_randomized_search_cv", False):
            try:
                print(f"    → train_xgboost_with_randomized_search_cv")
                xgboost_result = self.train_xgboost_with_randomized_search_cv()
                
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

        if state_machine.get("threshold_optimization_xgboost", False):
            try:
                print(f"    → threshold_optimization_xgboost")
                xgb_threshold_result = self.threshold_optimization_xgboost()
                
                if xgb_threshold_result and xgb_threshold_result.get("success", False):
                    result["steps_executed"].append("threshold_optimization_xgboost")
                    result["file_info"]["xgboost_models_optimized"] = xgb_threshold_result.get("models_optimized", 0)
                    print(f"      ✓ Successfully optimized XGBoost thresholds")
                    print(f"      ✓ Target feature: {xgb_threshold_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Models optimized: {xgb_threshold_result.get('models_optimized', 0)}")
                    print(f"      ✓ Average optimal threshold: {xgb_threshold_result.get('average_optimal_threshold', 0):.3f}")
                    print(f"      ✓ Average optimized F1: {xgb_threshold_result.get('average_optimized_f1', 0):.3f}")
                    print(f"      ✓ Average optimized Precision: {xgb_threshold_result.get('average_optimized_precision', 0):.3f}")
                    print(f"      ✓ Average optimized Recall: {xgb_threshold_result.get('average_optimized_recall', 0):.3f}")
                    print(f"      ✓ Results saved to: {xgb_threshold_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = xgb_threshold_result.get("error", "threshold_optimization_xgboost returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"threshold_optimization_xgboost failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ threshold_optimization_xgboost (disabled)")

        if state_machine.get("train_xgboost_selected_features", False):
            try:
                print(f"    → train_xgboost_selected_features")
                selected_xgboost_result = self.train_xgboost_selected_features()
                
                if selected_xgboost_result and selected_xgboost_result.get("success", False):
                    result["steps_executed"].append("train_xgboost_selected_features")
                    result["file_info"]["selected_xgboost_models_trained"] = selected_xgboost_result.get("models_trained", 0)
                    result["file_info"]["selected_xgboost_problem_type"] = selected_xgboost_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained XGBoost models with selected features")
                    print(f"      ✓ Problem type: {selected_xgboost_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {selected_xgboost_result.get('models_trained', 0)}")
                    print(f"      ✓ Selected weather features: {selected_xgboost_result.get('selected_weather_features', [])}")
                    print(f"      ✓ Average CV Score: {selected_xgboost_result.get('average_cv_score', 0):.4f}")
                    
                    # Print problem-specific metrics
                    if selected_xgboost_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {selected_xgboost_result.get('average_test_f1', 0):.4f}")
                        print(f"      ✓ Average Test Accuracy: {selected_xgboost_result.get('average_test_accuracy', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {selected_xgboost_result.get('average_test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {selected_xgboost_result.get('average_test_r2', 0):.4f}")
                        
                    print(f"      ✓ Results saved to: {selected_xgboost_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = selected_xgboost_result.get("error", "train_xgboost_selected_features returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"train_xgboost_selected_features failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_xgboost_selected_features (disabled)")

            
        return result

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
            print(f"    select_training_cols: Starting interactive column selection...")
            
            # Determine file path to analyze
            if original_file_path and os.path.exists(original_file_path):
                file_path = original_file_path
                print(f"    select_training_cols: Using specified file: {os.path.basename(file_path)}")
            else:
                # Auto-discover merged files using the MERGED_TRAINING_READY_OUTPUT_FOLDER
                merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
                
                # Create directory if it doesn't exist
                os.makedirs(merged_training_ready_dir, exist_ok=True)
                
                # Find merged data files using glob pattern
                merged_data_pattern = os.path.join(merged_training_ready_dir, "merged_data_*.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                # Filter out train/test files to get only the main merged files
                merged_data_files = [f for f in merged_data_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
                
                if merged_data_files:
                    # Use the most recent merged file
                    file_path = max(merged_data_files, key=os.path.getmtime)
                    print(f"    select_training_cols: Found merged file: {os.path.basename(file_path)}")
                else:
                    error_msg = f"No merged data files found in directory: {merged_training_ready_dir}"
                    print(f"    select_training_cols: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "searched_directory": merged_training_ready_dir,
                        "search_pattern": "merged_data_*.csv"
                    }
            
            # Load the CSV file
            print(f"    select_training_cols: Loading file: {file_path}")
            
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                error_msg = f"Failed to load CSV file: {str(e)}"
                print(f"    select_training_cols: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "file_path": file_path
                }
            
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
            
            # Interactive column selection (simple comma-separated numbers only)
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
                original_filename = os.path.basename(file_path)
                output_filename = original_filename.replace('.csv', '.csv')
                output_path = os.path.join(selected_training_ready_dir, output_filename)
                
                df_filtered.to_csv(output_path, index=False)
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

    def split_dataset(self, csv_files=None, test_size=TEST_SIZE, random_state=42, stratify_column=None):
        """
        Split merged training dataset into train and test sets.
        
        This method checks if exactly one CSV file exists in data/output/merged_training_ready,
        and if so, splits that dataset into training and testing sets, saving them as separate files.
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
            merged_training_ready_dir = os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(merged_training_ready_dir, exist_ok=True)
            
            # Get all CSV files in the directory
            all_csv_files = [f for f in os.listdir(merged_training_ready_dir) 
                            if f.endswith('.csv') and os.path.isfile(os.path.join(merged_training_ready_dir, f))]
            
            # Filter out existing train/test files to get only source files
            source_csv_files = [f for f in all_csv_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
            
            # Check if exactly one CSV file exists
            if len(source_csv_files) == 0:
                error_msg = "No CSV files found in the directory"
                print(f"    split_dataset: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            elif len(source_csv_files) > 1:
                error_msg = f"Expected exactly one CSV file, but found {len(source_csv_files)}: {', '.join(source_csv_files)}"
                print(f"    split_dataset: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0,
                    "found_files": source_csv_files
                }
            
            # Proceed with the single CSV file
            csv_filename = source_csv_files[0]
            file_path = os.path.join(merged_training_ready_dir, csv_filename)
            
            print(f"    split_dataset: Found exactly one CSV file: {csv_filename}")
            print(f"    split_dataset: Processing {csv_filename}...")
            
            # Read the dataset
            df = pd.read_csv(file_path)
            
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
                unique_values = df[target_column].nunique()
                is_classification = unique_values <= 10  # Assume classification if ≤ 10 unique values
                
                if is_classification:
                    stratify = df[target_column]
                    print(f"      Detected classification problem with target '{target_column}' ({unique_values} classes)")
                    
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
            base_name = csv_filename.replace('.csv', '')
            train_filename = f"{base_name}_train.csv"
            test_filename = f"{base_name}_test.csv"
            
            train_path = os.path.join(merged_training_ready_dir, train_filename)
            test_path = os.path.join(merged_training_ready_dir, test_filename)
            
            # Save train and test sets
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
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
            summary_path = os.path.join(merged_training_ready_dir, summary_filename)
            
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

    def scale_weather_features(self, csv_files=None):
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
            
            merged_selected_training_ready_dir = os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
            train_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_train.csv")
            test_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_test.csv")
            
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
                train_df = pd.read_csv(train_file_path)
                test_df = pd.read_csv(test_file_path)
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
            
            # NEW LOGIC: Handle case when no weather features are found
            if not available_weather_features:
                print(f"    scale_weather_features: No weather features found in {train_filename}")
                print(f"    scale_weather_features: Copying files as-is from source to scaled directory...")
                
                # Generate output filenames (keep same naming convention)
                scaled_train_filename = train_filename.replace('.csv', '.csv')  # Keep same name
                scaled_test_filename = test_filename.replace('.csv', '.csv')    # Keep same name
                
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
                    f.write(f"Source directory: {MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER}\n")
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
            scaled_train_filename = train_filename.replace('.csv', '.csv')
            scaled_test_filename = test_filename.replace('.csv', '.csv')
            
            scaled_train_path = os.path.join(scaled_training_ready_dir, scaled_train_filename)
            scaled_test_path = os.path.join(scaled_training_ready_dir, scaled_test_filename)
            
            # Save scaled datasets
            train_scaled_df.to_csv(scaled_train_path, index=False)
            test_scaled_df.to_csv(scaled_test_path, index=False)
            
            print(f"    scale_weather_features: ✓ Scaled train dataset saved to: {scaled_train_filename}")
            print(f"    scale_weather_features: ✓ Scaled test dataset saved to: {scaled_test_filename}")
            
            # Prepare scaling result
            scaling_result = {
                "original_train_file": train_filename,
                "original_test_file": test_filename,
                "scaled_train_file": scaled_train_filename,
                "scaled_test_file": scaled_test_filename,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
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
                f.write(f"Source directory: {MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER}\n")
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
            
            print(f"    scale_weather_features: Scaling summary saved to: {summary_filename}")
            
            # Return successful result
            return {
                "success": True,
                "processed_files": 1,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
                "scaling_summary": scaling_result,
                "output_directory": scaled_training_ready_dir,
                "message": f"Successfully scaled weather features for single train/test file pair: {train_filename}, {test_filename}",
                "scaling_applied": True  # Flag to indicate scaling was applied
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

    def train_xgboost_with_randomized_search_cv(self):
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
            
            scaled_data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            
            # Find training and test files in the specified location
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train.csv")
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test.csv")
            
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
            identifier_match = re.search(r'merged_data_(.+?)_train\.csv', train_filename)
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
            train_df = pd.read_csv(train_file)
            
            print(f"      Loading test data from {test_file}")
            test_df = pd.read_csv(test_file)
            
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
                cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                base_model = xgb.XGBClassifier(
                    random_state=42,
                    n_jobs=1,
                    eval_metric='logloss'
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                base_model = xgb.XGBRegressor(
                    random_state=42,
                    n_jobs=1,
                    eval_metric='mae'
                )
                scoring_metric = 'neg_mean_absolute_error'
            
            # Define iteration range: 10 to 100 (step=10)
            iteration_values = list(range(10, RANDOM_SEARCH_ITERATIONS + 1, 10))
            print(f"      Testing iteration values: {iteration_values}")
            
            # Initialize tracking lists
            iteration_results = []
            test_f1_scores = []
            cv_scores = []
            test_mae_scores = []  
            test_mape_scores = []  
            
            best_model = None
            best_f1_score = -np.inf if is_classification else np.inf
            best_iteration = None
            
            # Train models with different iteration counts
            print(f"      Starting training with different iteration counts...")
            
            for i, n_iter in enumerate(iteration_values):
                print(f"      Progress: {i+1}/{len(iteration_values)} - Testing {n_iter} iterations...")
                
                # Create RandomizedSearchCV with current iteration count
                randomized_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=42,
                    n_jobs=1,
                    verbose=0  # Reduced verbosity for cleaner output
                )
                
                # Fit with sample weights if available
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_train)
                
                # Get best model for current iteration count
                current_best_model = randomized_search.best_estimator_
                current_cv_score = randomized_search.best_score_
                
                # Make predictions on test set
                y_pred = current_best_model.predict(X_test)
                
                # Calculate test metrics
                if is_classification:
                    test_f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                    test_accuracy = accuracy_score(y_test, y_pred)
                    
                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}")
                    
                    # Track best model based on F1 score
                    if test_f1 > best_f1_score:
                        best_f1_score = test_f1
                        best_model = current_best_model
                        best_iteration = n_iter
                    
                    test_f1_scores.append(test_f1)
                    
                else:
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred) 
                    test_r2 = r2_score(y_test, y_pred)

                    epsilon = 1e-8
                    test_mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), epsilon))) * 100
    
                    
                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test RMSE = {test_rmse:.4f}, Test R² = {test_r2:.4f}, Test MAE = {test_mae:.4f}, Test MAPE = {test_mape:.2f}%")
                    
                    # Track best model based on RMSE (lower is better)
                    if test_rmse < best_f1_score:
                        best_f1_score = test_rmse
                        best_model = current_best_model
                        best_iteration = n_iter
                    
                    test_f1_scores.append(test_rmse) 
                    test_mae_scores.append(test_mae)      # ADD THIS
                    test_mape_scores.append(test_mape)    # ADD THIS
                
                cv_scores.append(current_cv_score)
                
                # Store detailed results
                iteration_results.append({
                    'n_iter': n_iter,
                    'cv_score': current_cv_score,
                    'test_metric': test_f1_scores[-1],
                    'test_mae': test_mae_scores[-1] if not is_classification else None,      # ADD THIS
                    'test_mape': test_mape_scores[-1] if not is_classification else None,    # ADD THIS
                    'best_params': randomized_search.best_params_
                })

                # Calculate final metrics on the best model
                print(f"      Calculating final metrics with best model (iteration {best_iteration})...")

                # Make final predictions with best model
                final_y_pred = best_model.predict(X_test)

                if is_classification:
                    # Your existing classification metrics
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
                    
                    # Calculate final MAPE
                    epsilon = 1e-8
                    final_test_mape = np.mean(np.abs((y_test - final_y_pred) / np.maximum(np.abs(y_test), epsilon))) * 100
            
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
            plt.show()
            print(f"      Performance curve saved to: {plot_filename}")
            
            # CREATE AND SAVE FEATURE IMPORTANCE PLOT
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
                        "test_mae": float(final_test_mae),          # ADD THIS
                        "test_mape": float(final_test_mape),        # ADD THIS
                        "test_r2": float(final_test_r2)
                    }

                # Add iteration-wise metrics summary
                if not is_classification:
                    results["iteration_metrics_summary"] = {
                        "rmse_values": [float(x) for x in test_f1_scores],
                        "mae_values": [float(x) for x in test_mae_scores],      # ADD THIS
                        "mape_values": [float(x) for x in test_mape_scores],    # ADD THIS
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_rmse": float(min(test_f1_scores)),
                        "best_mae": float(min(test_mae_scores)),                # ADD THIS
                        "best_mape": float(min(test_mape_scores)),              # ADD THIS
                        "average_rmse": float(np.mean(test_f1_scores)),
                        "average_mae": float(np.mean(test_mae_scores)),         # ADD THIS
                        "average_mape": float(np.mean(test_mape_scores))        # ADD THIS
                    }
            
            results_file = os.path.join(output_dir, f"xgboost_iteration_analysis_{file_identifier}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

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
            print(f"        Best Test Score: {best_f1_score:.4f}")
            print(f"        Score Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
            
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
                    "test_mape": float(final_test_mape),        
                    "test_r2": float(final_test_r2),
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

############################
## OLD METHODS
############################

    def threshold_optimization_xgboost(self):
        """
        Optimize XGBoost classification thresholds using ROC analysis.
        
        This method loads trained XGBoost models from the previous stage,
        analyzes ROC curves to find optimal thresholds, and saves optimized models
        with new thresholds. Results are saved to xgboost_threshold_optimized folder.
        
        Input Data Sources:
        - Models: data/output/xgboost_randomized_search/
        - Test Data: data/output/4-merged_scaled_training_ready/merged_data_*_test_scaled.csv
        
        Output:
        - Optimized models and thresholds saved to: data/output/xgboost_threshold_optimized/
        
        Returns:
        --------
        dict
            A summary of the threshold optimization results including optimal thresholds and performance metrics.
        """
        try:
            print(f"    threshold_optimization_xgboost: Starting XGBoost threshold optimization...")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, XGBOOST_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if XGBoost results exist from previous stage
            xgb_output_dir = os.path.join(self.project_root, "data/output/xgboost_randomized_search")
            xgb_results_file = os.path.join(xgb_output_dir, "xgboost_randomized_search_training_results.json")
            
            if not os.path.exists(xgb_results_file):
                error_msg = "XGBoost training results not found. Run train_xgboost_with_randomized_search_cv stage first."
                print(f"    threshold_optimization_xgboost: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Load XGBoost training results
            with open(xgb_results_file, 'r') as f:
                xgb_results = json.load(f)
            
            # Check if this is a classification problem
            problem_type = xgb_results.get("training_overview", {}).get("problem_type", "")
            if problem_type != "classification":
                error_msg = f"Threshold optimization is only supported for classification problems. Current problem type: {problem_type}"
                print(f"    threshold_optimization_xgboost: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"    threshold_optimization_xgboost: Found {len(xgb_results.get('file_results', []))} trained XGBoost models")
            
            # Find train and test files
            scaled_data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train_scaled.csv")
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test_scaled.csv")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if not train_files or not test_files:
                error_msg = f"Training/test files not found in {scaled_data_dir}"
                print(f"    threshold_optimization_xgboost: {error_msg}")
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
                    "failed_optimizations": 0,
                    "problem_type": "classification"
                },
                "threshold_optimization_config": THRESHOLD_OPTIMIZATION_CONFIG,
                "file_results": [],
                "aggregate_metrics": {
                    "average_optimal_threshold": 0.0,
                    "average_optimized_f1": 0.0,
                    "average_optimized_precision": 0.0,
                    "average_optimized_recall": 0.0,
                    "average_optimized_accuracy": 0.0,
                    "best_performing_file": None
                }
            }
            
            successful_optimizations = 0
            failed_optimizations = 0
            all_optimal_thresholds = []
            all_optimized_f1_scores = []
            all_optimized_precisions = []
            all_optimized_recalls = []
            all_optimized_accuracies = []
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
                        print(f"    threshold_optimization_xgboost: Warning - No test file for {train_filename}. Skipping.")
                        continue
                    
                    # Find corresponding trained XGBoost model
                    file_identifier = train_filename.replace('merged_data_', '').replace('_train_scaled.csv', '')
                    
                    # Look for XGBoost model files (there might be multiple methods/models per file)
                    model_pattern = os.path.join(xgb_output_dir, f"xgboost_model_{file_identifier}.joblib")
                    model_files = glob.glob(model_pattern)
                    
                    if not model_files:
                        print(f"    threshold_optimization_xgboost: Warning - No XGBoost model files for {file_identifier}. Skipping.")
                        failed_optimizations += 1
                        continue
                    
                    print(f"    threshold_optimization_xgboost: Optimizing threshold for {file_identifier}")
                    
                    # Load test data
                    test_df = pd.read_csv(test_file)
                    
                    if DEFAULT_TARGET_FEATURE not in test_df.columns:
                        print(f"    threshold_optimization_xgboost: Target feature '{DEFAULT_TARGET_FEATURE}' not found in {test_filename}")
                        failed_optimizations += 1
                        continue
                    
                    # Prepare test features and target
                    y_test = test_df[DEFAULT_TARGET_FEATURE]
                    X_test = test_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    
                    # Handle missing values (consistent with XGBoost training)
                    X_test = X_test.fillna(0)
                    
                    # Process each model file for this identifier
                    file_results = {
                        "file_identifier": file_identifier,
                        "test_samples": len(y_test),
                        "positive_samples": int(y_test.sum()),
                        "negative_samples": int(len(y_test) - y_test.sum()),
                        "models_optimized": [],
                        "optimization_successful": True,
                        "error_message": None
                    }
                    
                    file_successful = False
                    
                    for model_file in model_files:
                        try:
                            # Extract method name from model filename
                            model_filename = os.path.basename(model_file)
                            method_match = re.search(r'xgboost_model_.*?_(.+?)\.joblib', model_filename)
                            method_name = method_match.group(1) if method_match else "unknown"
                            
                            print(f"      Processing {method_name} model...")
                            
                            # Load the trained XGBoost model
                            model = joblib.load(model_file)
                            
                            # Get probability predictions
                            y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                            
                            # Perform threshold optimization
                            optimal_threshold, optimal_metrics = optimize_threshold_xgboost(
                                y_test, y_proba, file_identifier, method_name, output_dir
                            )
                            
                            # Create optimized model wrapper that uses the optimal threshold
                            optimized_model_data = {
                                "original_model": model,
                                "optimal_threshold": optimal_threshold,
                                "optimization_metrics": optimal_metrics,
                                "model_type": "xgboost_optimized"
                            }
                            
                            # Save optimized model
                            optimized_model_filename = f"xgboost_optimized_model_{file_identifier}_{method_name}.joblib"
                            optimized_model_path = os.path.join(output_dir, optimized_model_filename)
                            joblib.dump(optimized_model_data, optimized_model_path)

                            # Generate and save confusion matrix for optimized threshold
                            y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
                            conf_matrix = confusion_matrix(y_test, y_pred_optimized)
                            conf_matrix_result = save_confusion_matrix(
                                conf_matrix, y_test, y_pred_optimized, 
                                f"{file_identifier}_{method_name}_optimized", output_dir
                            )
                            
                            # Store results for this model
                            model_result = {
                                "method_name": method_name,
                                "original_model_file": model_file,
                                "optimized_model_file": optimized_model_path,
                                "optimal_threshold": optimal_threshold,
                                "optimization_metrics": optimal_metrics,
                                "default_threshold_metrics": {
                                    "f1_score": float(f1_score(y_test, model.predict(X_test))),
                                    "precision": float(precision_score(y_test, model.predict(X_test))),
                                    "recall": float(recall_score(y_test, model.predict(X_test))),
                                    "accuracy": float(accuracy_score(y_test, model.predict(X_test)))
                                }
                            }
                            
                            file_results["models_optimized"].append(model_result)
                            
                            # Track aggregates
                            all_optimal_thresholds.append(optimal_threshold)
                            all_optimized_f1_scores.append(optimal_metrics['f1_score'])
                            all_optimized_precisions.append(optimal_metrics['precision'])
                            all_optimized_recalls.append(optimal_metrics['recall'])
                            all_optimized_accuracies.append(optimal_metrics['accuracy'])
                            
                            if optimal_metrics['f1_score'] > best_f1:
                                best_f1 = optimal_metrics['f1_score']
                                best_file = f"{file_identifier}_{method_name}"
                            
                            file_successful = True
                            
                            print(f"        ✓ {method_name}: Threshold {optimal_threshold:.3f}, F1 {optimal_metrics['f1_score']:.3f}")
                            
                        except Exception as e:
                            print(f"        ✗ Error processing {method_name} model: {str(e)}")
                            continue
                    
                    if file_successful:
                        successful_optimizations += len(file_results["models_optimized"])
                        optimization_results["file_results"].append(file_results)
                    else:
                        failed_optimizations += 1
                        file_results["optimization_successful"] = False
                        file_results["error_message"] = "No models were successfully optimized"
                        optimization_results["file_results"].append(file_results)
                    
                except Exception as e:
                    print(f"    threshold_optimization_xgboost: Error processing {train_filename}: {str(e)}")
                    failed_optimizations += 1
                    continue
            
            # Calculate aggregate metrics
            if successful_optimizations > 0:
                optimization_results["optimization_overview"]["successful_optimizations"] = successful_optimizations
                optimization_results["optimization_overview"]["failed_optimizations"] = failed_optimizations
                optimization_results["optimization_overview"]["total_models_optimized"] = successful_optimizations
                
                optimization_results["aggregate_metrics"] = {
                    "average_optimal_threshold": float(np.mean(all_optimal_thresholds)),
                    "average_optimized_f1": float(np.mean(all_optimized_f1_scores)),
                    "average_optimized_precision": float(np.mean(all_optimized_precisions)),
                    "average_optimized_recall": float(np.mean(all_optimized_recalls)),
                    "average_optimized_accuracy": float(np.mean(all_optimized_accuracies)),
                    "best_performing_file": {
                        "file_identifier": best_file,
                        "f1_score": float(best_f1)
                    } if best_file else None
                }
                
                # Save consolidated results
                results_file = os.path.join(output_dir, "xgboost_threshold_optimization_results.json")
                with open(results_file, 'w') as f:
                    json.dump(optimization_results, f, indent=2)
                
                print(f"    threshold_optimization_xgboost: Optimized {successful_optimizations} XGBoost models")
                print(f"    threshold_optimization_xgboost: Average optimal threshold: {np.mean(all_optimal_thresholds):.3f}")
                print(f"    threshold_optimization_xgboost: Average optimized F1: {np.mean(all_optimized_f1_scores):.3f}")
                print(f"    threshold_optimization_xgboost: Results saved to: {output_dir}")
                
                return {
                    "success": True,
                    "models_optimized": successful_optimizations,
                    "average_optimal_threshold": float(np.mean(all_optimal_thresholds)),
                    "average_optimized_f1": float(np.mean(all_optimized_f1_scores)),
                    "average_optimized_precision": float(np.mean(all_optimized_precisions)),
                    "average_optimized_recall": float(np.mean(all_optimized_recalls)),
                    "average_optimized_accuracy": float(np.mean(all_optimized_accuracies)),
                    "output_directory": output_dir,
                    "results_file": results_file,
                    "target_feature": DEFAULT_TARGET_FEATURE
                }
            else:
                error_msg = "No XGBoost models were successfully optimized"
                print(f"    threshold_optimization_xgboost: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"threshold_optimization_xgboost failed: {str(e)}"
            print(f"    threshold_optimization_xgboost: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

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
                    create_correlation_plot(
                        correlations, 
                        filename,
                        correlation_output_dir,
                        f"Point-Biserial Correlation: {DEFAULT_TARGET_FEATURE} vs Weather Features\nFile: {filename}"
                    )
                    
                    # Create distribution plots for this file
                    create_distribution_plots(
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
                create_combined_correlation_analysis(correlation_results, correlation_output_dir)
            
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
                    create_non_weather_correlation_plot(
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
                create_combined_non_weather_correlation_analysis(correlation_results, output_dir)
            
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

    def select_time_features(self, data=None, original_file_path=None):
        """
        Select time features based on configuration - keep original or sin/cos features.
        
        This method checks if the dataset has all required time feature columns and allows
        the user to choose between:
        - USE_SIN_COS_APPROACH = True: Keep only sin/cos versions (drop original categorical versions)  
        - USE_SIN_COS_APPROACH = False: Keep only month, hour, day_of_week (drop sin/cos versions)
        
        The selection is controlled by the USE_SIN_COS_APPROACH constant in const.py.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            The dataframe to process. If None, attempts to load from merged_training_ready files.
        original_file_path : str, optional 
            Path to the original file for preserving naming patterns.
            
        Returns:
        --------
        dict
            Results of the time feature selection including success status and selection info
        """
        try:
            approach = "sin/cos features" if USE_SIN_COS_APPROACH else "original features"
            print(f"    select_time_features: Starting time feature selection...")
            print(f"    select_time_features: Configuration: USE_SIN_COS_APPROACH = {USE_SIN_COS_APPROACH} ({approach})")
            
            # Load data if not provided
            if data is None:
                merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
                merged_data_pattern = os.path.join(merged_training_ready_dir, "merged_data_*.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                # Filter out train/test files
                merged_data_files = [f for f in merged_data_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
                
                if not merged_data_files:
                    error_msg = "No merged data files found for time feature selection"
                    print(f"    select_time_features: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg
                    }
                
                # Use the most recent merged file
                input_file_path = max(merged_data_files, key=os.path.getmtime)
                print(f"    select_time_features: Loading data from {os.path.basename(input_file_path)}")
                
                df = pd.read_csv(input_file_path)
                original_file_path = input_file_path
            else:
                df = data.copy()
                print(f"    select_time_features: Processing provided dataframe with {len(df)} rows")
            
            if df.empty:
                error_msg = "Cannot process empty dataframe"
                print(f"    select_time_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"    select_time_features: Dataset shape: {df.shape}")
            
            # Define required time feature columns directly in code
            required_columns = [
                'month', 'month_sin', 'month_cos',
                'hour', 'hour_sin', 'hour_cos', 
                'day_of_week', 'day_week_sin', 'day_week_cos'
            ]
            
            original_time_features = ['month', 'hour', 'day_of_week']
            sincos_time_features = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'day_week_sin', 'day_week_cos']
            
            # Check if all required time feature columns are present
            missing_columns = []
            existing_columns = []
            
            for col in required_columns:
                if col in df.columns:
                    existing_columns.append(col)
                else:
                    missing_columns.append(col)
            
            print(f"    select_time_features: Found {len(existing_columns)}/{len(required_columns)} required time feature columns")
            print(f"    select_time_features: Existing columns: {existing_columns}")
            
            if missing_columns:
                print(f"    select_time_features: Missing columns: {missing_columns}")
                print(f"    select_time_features: WARNING - Some time feature columns are missing")
            
            # Determine which columns to keep/drop based on configuration
            columns_to_drop = []
            columns_to_keep = []
            
            if USE_SIN_COS_APPROACH:
                # Keep sin/cos versions, drop original categorical features
                for col in sincos_time_features:
                    if col in df.columns:
                        columns_to_keep.append(col)
                for col in original_time_features:
                    if col in df.columns:
                        columns_to_drop.append(col)
                print(f"    select_time_features: Keeping cyclical sin/cos features")
                
            else:
                # Keep original categorical features, drop sin/cos versions
                for col in original_time_features:
                    if col in df.columns:
                        columns_to_keep.append(col)
                for col in sincos_time_features:
                    if col in df.columns:
                        columns_to_drop.append(col)
                print(f"    select_time_features: Keeping original categorical time features")
            
            # Apply column selection
            original_shape = df.shape
            
            if columns_to_drop:
                print(f"    select_time_features: Dropping {len(columns_to_drop)} columns: {columns_to_drop}")
                df_selected = df.drop(columns=columns_to_drop, errors='ignore')
                actually_dropped = [col for col in columns_to_drop if col in df.columns]
                print(f"    select_time_features: Successfully dropped: {actually_dropped}")
            else:
                df_selected = df.copy()
                print(f"    select_time_features: No columns to drop")
            
            final_shape = df_selected.shape
            
            print(f"    select_time_features: Shape change: {original_shape} → {final_shape}")
            print(f"    select_time_features: Kept time feature columns: {columns_to_keep}")
            
            # Always save back to the merged_training_ready directory
            merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(merged_training_ready_dir, exist_ok=True)
            
            # Determine the output file path
            if original_file_path and os.path.exists(original_file_path):
                # Use the original file path if it was provided and exists
                output_path = original_file_path
                print(f"    select_time_features: Updating original file: {os.path.basename(output_path)}")
            else:
                # Find the most recent merged_data_*.csv file in merged_training_ready directory
                merged_data_pattern = os.path.join(merged_training_ready_dir, "merged_data_*.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                # Filter out train/test files to get only the main merged files
                merged_data_files = [f for f in merged_data_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
                
                if merged_data_files:
                    # Use the most recent merged file
                    output_path = max(merged_data_files, key=os.path.getmtime)
                    print(f"    select_time_features: Updating most recent merged file: {os.path.basename(output_path)}")
                else:
                    # Fallback: create a new merged_data.csv file
                    output_path = os.path.join(merged_training_ready_dir, "merged_data.csv")
                    print(f"    select_time_features: Creating new file: {os.path.basename(output_path)}")
            
            # Ensure we're saving to the correct merged_training_ready directory
            if not output_path.startswith(merged_training_ready_dir):
                print(f"    select_time_features: Warning - Output path is not in merged_training_ready directory")
                print(f"    select_time_features: Expected directory: {merged_training_ready_dir}")
                print(f"    select_time_features: Actual path: {output_path}")
            
            # Save the processed dataset back to the merged_training_ready directory
            try:
                df_selected.to_csv(output_path, index=False)
                print(f"    select_time_features: Successfully saved updated dataset to:")
                print(f"    select_time_features: → {output_path}")
                print(f"    select_time_features: → File size: {len(df_selected):,} rows, {len(df_selected.columns)} columns")
            except Exception as save_error:
                error_msg = f"Failed to save dataset to {output_path}: {str(save_error)}"
                print(f"    select_time_features: {error_msg}")
                raise Exception(error_msg)
            
            # Create summary of the selection
            selection_summary = {
                "use_sin_cos_approach": USE_SIN_COS_APPROACH,
                "approach_description": approach,
                "original_columns": len(existing_columns),
                "columns_kept": len(columns_to_keep),
                "columns_dropped": len(columns_to_drop),
                "kept_columns": columns_to_keep,
                "dropped_columns": columns_to_drop,
                "missing_required_columns": missing_columns
            }
            
            # Prepare success result
            result = {
                "success": True,
                "data": df_selected,
                "output_path": output_path,
                "original_shape": original_shape,
                "final_shape": final_shape,
                "selection_summary": selection_summary,
                "message": f"Time feature selection completed: {approach}, kept {len(columns_to_keep)} columns, dropped {len(columns_to_drop)} columns"
            }
            
            print(f"    select_time_features: Completed successfully - {result['message']}")
            
            return result
            
        except Exception as e:
            error_msg = f"select_time_features failed: {str(e)}"
            print(f"    select_time_features: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
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
                    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Add CV score to metrics
                    metrics['best_cv_score'] = best_cv_score
                    metrics['optimized_metric_name'] = SCORE_METRIC
                    
                    # Generate and save confusion matrix
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    conf_matrix_result = save_confusion_matrix(
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
                    optimal_thresholds = find_optimal_thresholds(
                        y_test, y_proba, fpr, tpr, thresholds, precision, recall, pr_thresholds
                    )
                    
                    # Use the metric specified in config
                    optimization_metric = THRESHOLD_OPTIMIZATION_CONFIG["optimization_metric"]
                    optimal_threshold = optimal_thresholds[optimization_metric]

                    # Make predictions with optimal threshold
                    y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
                    
                    # Calculate comprehensive metrics using the private method
                    metrics = calculate_classification_metrics(y_test, y_pred_optimized, y_proba)

                    # Generate and save confusion matrix
                    conf_matrix = confusion_matrix(y_test, y_pred_optimized)
                    conf_matrix_result = save_confusion_matrix(
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
                        plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold, 
                                        file_identifier, output_dir)
                    
                    if THRESHOLD_OPTIMIZATION_CONFIG["plot_precision_recall"]:
                        plot_precision_recall_curve(precision, recall, pr_auc, optimal_threshold,
                                                        file_identifier, output_dir)
                    
                    # Plot threshold analysis
                    plot_threshold_analysis(thresholds, fpr, tpr, precision[:-1], recall[:-1],
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
                plot_threshold_summary(all_optimal_thresholds, all_optimized_f1_scores, output_dir)
                
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
                "aggregate_metrics": {}
            }
            
            # Initialize tracking variables
            successful_generations = 0
            failed_generations = 0
            file_identifiers = []
            augmentation_ratios = []
            total_original_samples = 0
            total_synthetic_samples = 0
            total_final_samples = 0
            overall_original_counter = Counter()
            overall_final_counter = Counter()
            
            # Process each training file
            for train_file in train_files:
                try:
                    # Extract file identifier
                    train_filename = os.path.basename(train_file)
                    file_identifier = train_filename.replace("merged_data_", "").replace("_train_scaled.csv", "")
                    file_identifiers.append(file_identifier)
                    
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
                    
                    # ===== NEW: NaN Detection and Reporting =====
                    print(f"    generate_borderline_smote_data: Checking for NaN values in {file_identifier}...")
                    
                    # Check for NaN values in features (X_train)
                    nan_columns = []
                    nan_info = {}
                    total_samples = len(X_train)
                    
                    for column in X_train.columns:
                        nan_count = X_train[column].isna().sum()
                        if nan_count > 0:
                            nan_percentage = (nan_count / total_samples) * 100
                            nan_columns.append(column)
                            nan_info[column] = {
                                'count': nan_count,
                                'percentage': round(nan_percentage, 2)
                            }
                            print(f"    generate_borderline_smote_data: Column '{column}': {nan_count:,} NaN values ({nan_percentage:.2f}%)")
                    
                    # Check for NaN values in target (y_train)
                    target_nan_count = y_train.isna().sum()
                    if target_nan_count > 0:
                        target_nan_percentage = (target_nan_count / total_samples) * 100
                        nan_info[DEFAULT_TARGET_FEATURE] = {
                            'count': target_nan_count,
                            'percentage': round(target_nan_percentage, 2)
                        }
                        print(f"    generate_borderline_smote_data: Target feature '{DEFAULT_TARGET_FEATURE}': {target_nan_count:,} NaN values ({target_nan_percentage:.2f}%)")
                    
                    # Report NaN summary
                    if nan_columns or target_nan_count > 0:
                        total_nan_columns = len(nan_columns) + (1 if target_nan_count > 0 else 0)
                        total_columns = len(X_train.columns) + 1  # +1 for target
                        print(f"    generate_borderline_smote_data: NaN SUMMARY for {file_identifier}:")
                        print(f"    generate_borderline_smote_data: - Total columns with NaN: {total_nan_columns}/{total_columns}")
                        print(f"    generate_borderline_smote_data: - Feature columns with NaN: {len(nan_columns)}/{len(X_train.columns)}")
                        if target_nan_count > 0:
                            print(f"    generate_borderline_smote_data: - Target column has NaN: YES")
                        
                        # Skip this file due to NaN values
                        error_msg = f"Input X contains NaN. BorderlineSMOTE does not accept missing values encoded as NaN natively."
                        print(f"    generate_borderline_smote_data: Error processing {train_file}: {error_msg}")
                        print(f"    generate_borderline_smote_data: Columns with NaN values: {', '.join(nan_columns) if nan_columns else 'None in features'}")
                        if target_nan_count > 0:
                            print(f"    generate_borderline_smote_data: Target feature also contains NaN values")
                        print(f"    generate_borderline_smote_data: Consider using imputation or dropping NaN values before applying BorderlineSMOTE")
                        failed_generations += 1
                        continue
                    else:
                        print(f"    generate_borderline_smote_data: ✓ No NaN values found in {file_identifier}")
                    # ===== END NaN Detection and Reporting =====
                    
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
                    
                    for class_label, count in original_distribution.items():
                        percentage = (count / original_sample_count) * 100
                        original_distribution_detailed[str(class_label)] = {
                            "count": count,
                            "percentage": round(percentage, 2)
                        }
                    
                    for class_label, count in final_distribution.items():
                        percentage = (count / final_sample_count) * 100
                        final_distribution_detailed[str(class_label)] = {
                            "count": count,
                            "percentage": round(percentage, 2)
                        }
                    
                    # Store file-specific results
                    file_result = {
                        "file_identifier": file_identifier,
                        "original_sample_count": original_sample_count,
                        "final_sample_count": final_sample_count,
                        "synthetic_samples_added": synthetic_samples_added,
                        "augmentation_ratio": round(augmentation_ratio, 3),
                        "original_distribution": original_distribution_detailed,
                        "final_distribution": final_distribution_detailed,
                        "output_file": output_filename
                    }
                    smote_results["file_results"].append(file_result)
                    
                    # Update aggregate tracking
                    successful_generations += 1
                    augmentation_ratios.append(augmentation_ratio)
                    total_original_samples += original_sample_count
                    total_synthetic_samples += synthetic_samples_added
                    total_final_samples += final_sample_count
                    overall_original_counter.update(original_distribution)
                    overall_final_counter.update(final_distribution)
                    
                except Exception as e:
                    print(f"    generate_borderline_smote_data: Error processing {train_file}: {str(e)}")
                    failed_generations += 1
                    continue
            
            # Final results compilation
            if successful_generations > 0:
                # Calculate overall distributions with percentages
                overall_original_detailed = {}
                overall_final_detailed = {}
                
                for class_label, count in overall_original_counter.items():
                    percentage = (count / total_original_samples) * 100
                    overall_original_detailed[str(class_label)] = {
                        "count": count,
                        "percentage": round(percentage, 2)
                    }
                
                for class_label, count in overall_final_counter.items():
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
                plot_smote_summary(smote_results, output_dir)
                
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

    def train_decision_tree_with_borderline_smote_data(self):
        """
        Train Decision Tree classifiers using BorderlineSMOTE-augmented training data.
        
        This method trains decision tree models using synthetic training data generated by BorderlineSMOTE,
        while testing on the original (non-augmented) test data. It uses RandomizedSearchCV for 
        hyperparameter optimization and generates comprehensive evaluation metrics.
        
        Training Data Source:
        - data/output/borderline_smote_synthetic/borderline_smote_data_*_train_augmented.csv
        
        Test Data Source:
        - data/output/4-merged_scaled_training_ready/merged_data_*_test_scaled.csv
        
        Returns:
        --------
        dict
            A summary of the training results, including model performance metrics and comparison
            to baseline (non-SMOTE) performance.
        """
        try:
            print(f"    train_decision_tree_with_borderline_smote_data: Starting Decision Tree training with BorderlineSMOTE data...")
            
            # Check if target feature is a classification problem
            if DEFAULT_TARGET_FEATURE not in CLASSIFICATION_PROBLEM:
                error_msg = f"Target feature '{DEFAULT_TARGET_FEATURE}' is not a classification problem. Expected one of: {CLASSIFICATION_PROBLEM}"
                print(f"    train_decision_tree_with_borderline_smote_data: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"    train_decision_tree_with_borderline_smote_data: Using target feature '{DEFAULT_TARGET_FEATURE}' for classification")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, "data/output/decision_tree_borderline_smote")
            os.makedirs(output_dir, exist_ok=True)
            
            # Find BorderlineSMOTE augmented training files
            smote_data_dir = os.path.join(self.project_root, BORDERLINE_SMOTE_OUTPUT_FOLDER)
            train_pattern = os.path.join(smote_data_dir, "borderline_smote_data_*_train_augmented.csv")
            train_files = glob.glob(train_pattern)
            
            # Find original test files
            scaled_data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test_scaled.csv")
            test_files = glob.glob(test_pattern)
            
            if not train_files:
                error_msg = f"No BorderlineSMOTE training files found matching pattern: {train_pattern}"
                print(f"    train_decision_tree_with_borderline_smote_data: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            if not test_files:
                error_msg = f"No test files found matching pattern: {test_pattern}"
                print(f"    train_decision_tree_with_borderline_smote_data: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"    train_decision_tree_with_borderline_smote_data: Found {len(train_files)} SMOTE training files and {len(test_files)} test files")
            
            # Initialize results structure
            training_results = {
                "training_overview": {
                    "training_completed": datetime.now().isoformat(),
                    "target_feature": DEFAULT_TARGET_FEATURE,
                    "score_metric": SCORE_METRIC,
                    "total_files_processed": 0,
                    "successful_trainings": 0,
                    "failed_trainings": 0,
                    "borderline_smote_config": BORDERLINE_SMOTE_CONFIG
                },
                "hyperparameter_config": {
                    "param_distributions": format_param_distributions_for_json(DECISION_TREE_PARAM_DISTRIBUTIONS),
                    "n_iterations": RANDOM_SEARCH_ITERATIONS,
                    "cv_folds": RANDOM_SEARCH_CV_FOLDS,
                    "random_state": 42
                },
                "file_results": [],
                "aggregate_metrics": {
                    "average_test_accuracy": 0.0,
                    "average_test_f1": 0.0,
                    "average_test_precision": 0.0,
                    "average_test_recall": 0.0,
                    "average_test_roc_auc": 0.0,
                    "average_cv_score": 0.0
                }
            }
            
            successful_files = 0
            total_train_samples = 0
            total_test_samples = 0
            all_file_results = []
            
            # Process each file pair
            for train_file in train_files:
                try:
                    # Extract file identifier from SMOTE training file
                    train_filename = os.path.basename(train_file)
                    # Extract identifier from: borderline_smote_data_*_train_augmented.csv
                    file_identifier = train_filename.replace('borderline_smote_data_', '').replace('_train_augmented.csv', '')
                    
                    # Find corresponding test file
                    test_file = None
                    expected_test_filename = f"merged_data_{file_identifier}_test_scaled.csv"
                    for tf in test_files:
                        if os.path.basename(tf) == expected_test_filename:
                            test_file = tf
                            break
                    
                    if not test_file:
                        print(f"    train_decision_tree_with_borderline_smote_data: No matching test file found for {train_filename}")
                        continue
                    
                    print(f"    train_decision_tree_with_borderline_smote_data: Processing {file_identifier}")
                    
                    # Load SMOTE-augmented training data
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                    
                    if DEFAULT_TARGET_FEATURE not in train_df.columns:
                        print(f"    train_decision_tree_with_borderline_smote_data: Target feature '{DEFAULT_TARGET_FEATURE}' not found in {train_filename}")
                        continue
                    
                    if DEFAULT_TARGET_FEATURE not in test_df.columns:
                        print(f"    train_decision_tree_with_borderline_smote_data: Target feature '{DEFAULT_TARGET_FEATURE}' not found in test file")
                        continue
                    
                    # Prepare features and targets
                    y_train = train_df[DEFAULT_TARGET_FEATURE]
                    X_train = train_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    y_test = test_df[DEFAULT_TARGET_FEATURE]
                    X_test = test_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    
                    # Ensure feature consistency
                    common_features = X_train.columns.intersection(X_test.columns)
                    X_train = X_train[common_features]
                    X_test = X_test[common_features]
                    
                    print(f"    train_decision_tree_with_borderline_smote_data: Training samples: {len(X_train):,}, Test samples: {len(X_test):,}")
                    print(f"    train_decision_tree_with_borderline_smote_data: Features: {len(common_features)}")
                    
                    # Check class distribution
                    train_distribution = y_train.value_counts()
                    test_distribution = y_test.value_counts()
                    print(f"    train_decision_tree_with_borderline_smote_data: SMOTE Train distribution: {dict(train_distribution)}")
                    print(f"    train_decision_tree_with_borderline_smote_data: Test distribution: {dict(test_distribution)}")
                    
                    # Set up cross-validation strategy
                    cv_strategy = StratifiedKFold(
                        n_splits=RANDOM_SEARCH_CV_FOLDS,
                        shuffle=True,
                        random_state=42
                    )
                    
                    # Initialize Decision Tree and RandomizedSearchCV
                    dt = DecisionTreeClassifier(random_state=42)
                    
                    print(f"    train_decision_tree_with_borderline_smote_data: Starting RandomizedSearchCV with {RANDOM_SEARCH_ITERATIONS} iterations and {RANDOM_SEARCH_CV_FOLDS}-fold CV...")
                    
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
                    
                    # Fit RandomizedSearchCV on SMOTE-augmented data
                    random_search.fit(X_train, y_train)
                    
                    best_params = random_search.best_params_
                    best_cv_score = random_search.best_score_
                    
                    print(f"    train_decision_tree_with_borderline_smote_data: Best CV Score ({SCORE_METRIC}): {best_cv_score:.4f}")
                    print(f"    train_decision_tree_with_borderline_smote_data: Best Parameters: {best_params}")
                    
                    # Train final model with best parameters
                    best_dt = DecisionTreeClassifier(**best_params, random_state=42)
                    best_dt.fit(X_train, y_train)
                    
                    # Make predictions on original test data
                    y_pred = best_dt.predict(X_test)
                    y_pred_proba = best_dt.predict_proba(X_test)
                    
                    # Calculate comprehensive metrics
                    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Add CV score to metrics
                    metrics['best_cv_score'] = best_cv_score
                    metrics['optimized_metric_name'] = SCORE_METRIC
                    
                    # Generate and save confusion matrix
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    conf_matrix_result = save_confusion_matrix(
                        conf_matrix, y_test, y_pred, f"{file_identifier}_smote", output_dir
                    )
                    
                    # Get feature importance
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': best_dt.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    print(f"    train_decision_tree_with_borderline_smote_data: Test {SCORE_METRIC}: {metrics.get(SCORE_METRIC, 'N/A'):.4f}")
                    print(f"    train_decision_tree_with_borderline_smote_data: Test Accuracy: {metrics['accuracy']:.4f}")
                    print(f"    train_decision_tree_with_borderline_smote_data: Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                    
                    # Save model
                    model_filename = f"decision_tree_smote_model_{file_identifier}.joblib"
                    model_path = os.path.join(output_dir, model_filename)
                    joblib.dump(best_dt, model_path)
                    
                    # Save feature importance
                    importance_filename = f"feature_importance_smote_{file_identifier}.csv"
                    importance_path = os.path.join(output_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    
                    print(f"    train_decision_tree_with_borderline_smote_data: Saved model, feature importance, and confusion matrix for {file_identifier}")
                    
                    # Prepare individual file result for consolidated results
                    file_result = {
                        "file_identifier": file_identifier,
                        "train_file": os.path.basename(train_file),
                        "test_file": os.path.basename(test_file),
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "features": len(common_features),
                        "smote_train_distribution": dict(train_distribution),
                        "test_distribution": dict(test_distribution),
                        "best_cv_score": best_cv_score,
                        "best_parameters": best_params,
                        "test_metrics": metrics,
                        "model_path": model_filename,
                        "feature_importance_path": importance_filename,
                        "confusion_matrix_path": conf_matrix_result.get("confusion_matrix_filename", ""),
                        "top_5_features": feature_importance.head(5).to_dict('records')
                    }
                    
                    all_file_results.append(file_result)
                    successful_files += 1
                    total_train_samples += len(X_train)
                    total_test_samples += len(X_test)
                    
                    print(f"    train_decision_tree_with_borderline_smote_data: Successfully processed {file_identifier}")
                    
                except Exception as e:
                    print(f"    train_decision_tree_with_borderline_smote_data: Error processing {os.path.basename(train_file)}: {str(e)}")
                    continue
            
            if successful_files > 0:
                # Update results structure
                training_results["training_overview"]["total_files_processed"] = len(train_files)
                training_results["training_overview"]["successful_trainings"] = successful_files
                training_results["training_overview"]["failed_trainings"] = len(train_files) - successful_files
                training_results["file_results"] = all_file_results
                
                # Calculate aggregate metrics
                all_accuracies = [result["test_metrics"]["accuracy"] for result in all_file_results]
                all_f1_scores = [result["test_metrics"]["f1"] for result in all_file_results]
                all_roc_aucs = [result["test_metrics"]["roc_auc"] if result["test_metrics"]["roc_auc"] is not None else 0.0 for result in all_file_results]
                all_cv_scores = [result["best_cv_score"] for result in all_file_results]
                
                # Extract precision and recall from class-specific metrics (for binary classification)
                all_precisions = []
                all_recalls = []
                
                for result in all_file_results:
                    test_metrics = result["test_metrics"]
                    # For binary classification, get metrics for class 1 (positive class)
                    if "class_1_precision" in test_metrics:
                        all_precisions.append(test_metrics["class_1_precision"])
                        all_recalls.append(test_metrics["class_1_recall"])
                    elif "class_True_precision" in test_metrics:
                        all_precisions.append(test_metrics["class_True_precision"])
                        all_recalls.append(test_metrics["class_True_recall"])
                    else:
                        # Fallback: calculate precision and recall from other available metrics
                        # Use balanced accuracy as a proxy if specific class metrics are not available
                        all_precisions.append(test_metrics.get("balanced_accuracy", 0.0))
                        all_recalls.append(test_metrics.get("balanced_accuracy", 0.0))
                
                training_results["aggregate_metrics"] = {
                    "average_test_accuracy": float(np.mean(all_accuracies)),
                    "average_test_f1": float(np.mean(all_f1_scores)),
                    "average_test_precision": float(np.mean(all_precisions)) if all_precisions else 0.0,
                    "average_test_recall": float(np.mean(all_recalls)) if all_recalls else 0.0,
                    "average_test_roc_auc": float(np.mean([x for x in all_roc_aucs if x is not None and x > 0])) if any(x for x in all_roc_aucs if x is not None and x > 0) else 0.0,
                    "average_cv_score": float(np.mean(all_cv_scores)),
                    "std_test_accuracy": float(np.std(all_accuracies)),
                    "std_test_f1": float(np.std(all_f1_scores)),
                    "std_cv_score": float(np.std(all_cv_scores))
                }
                
                # Save consolidated results
                consolidated_filename = "decision_tree_borderline_smote_training_results.json"
                consolidated_path = os.path.join(output_dir, consolidated_filename)
                
                with open(consolidated_path, 'w') as f:
                    json.dump(training_results, f, indent=2, default=str)
                
                print(f"    train_decision_tree_with_borderline_smote_data: Training completed successfully!")
                print(f"    train_decision_tree_with_borderline_smote_data: Processed {successful_files} files successfully")
                
                # Get the appropriate metric name for display
                score_metric_display = SCORE_METRIC.replace('_', ' ').title()
                if SCORE_METRIC in ['f1', 'roc_auc']:
                    avg_metric_key = f"average_test_{SCORE_METRIC}"
                else:
                    avg_metric_key = "average_test_f1"  # Default fallback
                
                print(f"    train_decision_tree_with_borderline_smote_data: Average Test {score_metric_display}: {training_results['aggregate_metrics'].get(avg_metric_key, 0.0):.4f}")
                print(f"    train_decision_tree_with_borderline_smote_data: Average Test Accuracy: {training_results['aggregate_metrics']['average_test_accuracy']:.4f}")
                print(f"    train_decision_tree_with_borderline_smote_data: Average CV Score: {training_results['aggregate_metrics']['average_cv_score']:.4f}")
                print(f"    train_decision_tree_with_borderline_smote_data: Results saved to: {output_dir}")
                
                return {
                    "success": True,
                    "files_processed": successful_files,
                    "total_train_samples": total_train_samples,
                    "total_test_samples": total_test_samples,
                    "output_directory": output_dir,
                    "target_feature": DEFAULT_TARGET_FEATURE,
                    "score_metric": SCORE_METRIC,
                    "consolidated_results_file": consolidated_filename,
                    "average_test_accuracy": training_results['aggregate_metrics']['average_test_accuracy'],
                    "average_test_f1": training_results['aggregate_metrics']['average_test_f1'],
                    "average_test_precision": training_results['aggregate_metrics']['average_test_precision'],
                    "average_test_recall": training_results['aggregate_metrics']['average_test_recall'],
                    "average_cv_score": training_results['aggregate_metrics']['average_cv_score']
                }
            else:
                error_msg = "No files were successfully processed"
                print(f"    train_decision_tree_with_borderline_smote_data: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"Error in Decision Tree training with BorderlineSMOTE data: {str(e)}"
            print(f"    train_decision_tree_with_borderline_smote_data: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def threshold_optimization_decision_tree_borderline_smote(self):
        """
        Optimize decision tree classification thresholds for models trained with BorderlineSMOTE data.
        
        This method loads trained decision tree models from the BorderlineSMOTE training stage,
        analyzes ROC curves to find optimal thresholds, and saves optimized models with new thresholds.
        The optimization uses the original test data (not SMOTE-augmented) to find optimal thresholds.
        
        Input Data Sources:
        - Models: data/output/decision_tree_borderline_smote/
        - Test Data: data/output/4-merged_scaled_training_ready/merged_data_*_test_scaled.csv
        
        Output:
        - Optimized models and thresholds saved to: data/output/decision_tree_borderline_smote_threshold_optimized/
        
        Returns:
        --------
        dict
            A summary of the threshold optimization results including optimal thresholds and performance metrics.
        """
        try:
            print(f"    threshold_optimization_decision_tree_borderline_smote: Starting threshold optimization for BorderlineSMOTE models...")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, "data/output/decision_tree_borderline_smote_threshold_optimized")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if BorderlineSMOTE decision tree results exist from previous stage
            smote_dt_output_dir = os.path.join(self.project_root, "data/output/decision_tree_borderline_smote")
            smote_dt_results_file = os.path.join(smote_dt_output_dir, "decision_tree_borderline_smote_training_results.json")
            
            if not os.path.exists(smote_dt_results_file):
                error_msg = "BorderlineSMOTE Decision tree training results not found. Run train_decision_tree_with_borderline_smote_data stage first."
                print(f"    threshold_optimization_decision_tree_borderline_smote: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Load BorderlineSMOTE decision tree training results
            with open(smote_dt_results_file, 'r') as f:
                smote_dt_results = json.load(f)
            
            print(f"    threshold_optimization_decision_tree_borderline_smote: Found {len(smote_dt_results.get('file_results', []))} trained SMOTE models")
            
            # Find original test files (not SMOTE-augmented)
            scaled_data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test_scaled.csv")
            test_files = glob.glob(test_pattern)
            
            if not test_files:
                error_msg = f"Original test files not found in {scaled_data_dir}"
                print(f"    threshold_optimization_decision_tree_borderline_smote: {error_msg}")
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
                    "failed_optimizations": 0,
                    "smote_models_processed": True
                },
                "threshold_optimization_config": THRESHOLD_OPTIMIZATION_CONFIG,
                "borderline_smote_config": BORDERLINE_SMOTE_CONFIG,
                "file_results": [],
                "aggregate_metrics": {
                    "average_optimal_threshold": 0.0,
                    "average_optimized_f1": 0.0,
                    "average_optimized_precision": 0.0, 
                    "average_optimized_recall": 0.0,
                    "average_optimized_accuracy": 0.0,
                    "best_performing_file": None
                }
            }
            
            successful_optimizations = 0
            failed_optimizations = 0
            all_optimal_thresholds = []
            all_optimized_f1_scores = []
            all_optimized_precisions = []
            all_optimized_recalls = []
            all_optimized_accuracies = []
            best_f1 = -1
            best_file = None
            
            # Process each trained SMOTE model
            for file_result in smote_dt_results.get('file_results', []):
                try:
                    file_identifier = file_result.get('file_identifier')
                    if not file_identifier:
                        print(f"    threshold_optimization_decision_tree_borderline_smote: Warning - No file identifier found in result. Skipping.")
                        continue
                    
                    # Find corresponding test file
                    test_filename = f"merged_data_{file_identifier}_test_scaled.csv"
                    test_file = os.path.join(scaled_data_dir, test_filename)
                    
                    if not os.path.exists(test_file):
                        print(f"    threshold_optimization_decision_tree_borderline_smote: Warning - No test file for {file_identifier}. Skipping.")
                        failed_optimizations += 1
                        continue
                    
                    # Find corresponding trained SMOTE model
                    model_file = os.path.join(smote_dt_output_dir, f"decision_tree_smote_model_{file_identifier}.joblib")
                    
                    if not os.path.exists(model_file):
                        print(f"    threshold_optimization_decision_tree_borderline_smote: Warning - No SMOTE model file for {file_identifier}. Skipping.")
                        failed_optimizations += 1
                        continue
                    
                    print(f"    threshold_optimization_decision_tree_borderline_smote: Optimizing threshold for SMOTE model {file_identifier}")
                    
                    # Load the trained SMOTE model
                    model = joblib.load(model_file)
                    
                    # Load original test data (not SMOTE-augmented)
                    test_df = pd.read_csv(test_file)
                    
                    if DEFAULT_TARGET_FEATURE not in test_df.columns:
                        print(f"    threshold_optimization_decision_tree_borderline_smote: Target feature '{DEFAULT_TARGET_FEATURE}' not found in {test_filename}")
                        failed_optimizations += 1
                        continue
                    
                    # Prepare test features and target
                    y_test = test_df[DEFAULT_TARGET_FEATURE]
                    X_test = test_df.drop(columns=[DEFAULT_TARGET_FEATURE])
                    
                    # Extract and save feature importance from the SMOTE-trained model
                    feature_importance = pd.DataFrame({
                        'feature': X_test.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_filename = f"smote_threshold_optimized_feature_importance_{file_identifier}.csv"
                    importance_path = os.path.join(output_dir, importance_filename)
                    feature_importance.to_csv(importance_path, index=False)
                    
                    print(f"    threshold_optimization_decision_tree_borderline_smote: Saved feature importance for SMOTE model {file_identifier}")
                    
                    # Get probability predictions for positive class
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Calculate Precision-Recall curve
                    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
                    pr_auc = average_precision_score(y_test, y_proba)
                    
                    # Find optimal threshold based on different metrics
                    optimal_thresholds = find_optimal_thresholds(
                        y_test, y_proba, fpr, tpr, thresholds, precision, recall, pr_thresholds
                    )
                    
                    # Use the metric specified in config
                    optimization_metric = THRESHOLD_OPTIMIZATION_CONFIG["optimization_metric"]
                    optimal_threshold = optimal_thresholds[optimization_metric]
                    
                    # Make predictions with optimal threshold
                    y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
                    
                    # Calculate comprehensive metrics using the private method
                    metrics = calculate_classification_metrics(y_test, y_pred_optimized, y_proba)
                    
                    # Generate and save confusion matrix
                    conf_matrix = confusion_matrix(y_test, y_pred_optimized)
                    conf_matrix_result = save_confusion_matrix(
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
                    
                    # Calculate individual metrics for tracking
                    optimized_precision = precision_score(y_test, y_pred_optimized)
                    optimized_recall = recall_score(y_test, y_pred_optimized)
                    optimized_f1 = f1_score(y_test, y_pred_optimized)
                    optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
                    
                    # Create and save plots with SMOTE prefix
                    if THRESHOLD_OPTIMIZATION_CONFIG["plot_roc_curve"]:
                        plot_roc_curve_smote(fpr, tpr, roc_auc, optimal_threshold, 
                                        file_identifier, output_dir)
                    
                    if THRESHOLD_OPTIMIZATION_CONFIG["plot_precision_recall"]:
                        plot_precision_recall_curve_smote(precision, recall, pr_auc, optimal_threshold,
                                                        file_identifier, output_dir)
                    
                    # Plot threshold analysis for SMOTE models
                    plot_threshold_analysis_smote(thresholds, fpr, tpr, precision[:-1], recall[:-1],
                                                file_identifier, output_dir)
                    
                    # Save optimized SMOTE model if configured
                    if THRESHOLD_OPTIMIZATION_CONFIG["save_optimized_models"]:
                        optimized_model_data = {
                            "original_smote_model": model,
                            "optimal_threshold": optimal_threshold,
                            "optimization_metric": optimization_metric,
                            "roc_auc": roc_auc,
                            "pr_auc": pr_auc,
                            "borderline_smote_config": BORDERLINE_SMOTE_CONFIG
                        }
                        optimized_model_file = os.path.join(output_dir, f"optimized_smote_decision_tree_{file_identifier}.joblib")
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
                        "comprehensive_metrics": metrics,
                        "feature_importance_file": importance_filename,
                        "confusion_matrix": conf_matrix_result,
                        "plots_generated": {
                            "roc_curve": f"smote_roc_curve_{file_identifier}.png",
                            "precision_recall_curve": f"smote_precision_recall_curve_{file_identifier}.png",
                            "threshold_analysis": f"smote_threshold_analysis_{file_identifier}.png"
                        }
                    }
                    
                    optimization_results["file_results"].append(file_result)
                    
                    # Track metrics for aggregation
                    all_optimal_thresholds.append(optimal_threshold)
                    all_optimized_f1_scores.append(optimized_f1)
                    all_optimized_precisions.append(optimized_precision)
                    all_optimized_recalls.append(optimized_recall)
                    all_optimized_accuracies.append(optimized_accuracy)
                    
                    # Track best performing model
                    if optimized_f1 > best_f1:
                        best_f1 = optimized_f1
                        best_file = file_identifier
                    
                    successful_optimizations += 1
                    print(f"    threshold_optimization_decision_tree_borderline_smote: Successfully optimized {file_identifier} - F1: {optimized_f1:.4f}, Threshold: {optimal_threshold:.3f}")
                    
                except Exception as e:
                    print(f"    threshold_optimization_decision_tree_borderline_smote: Failed to process {file_identifier}: {str(e)}")
                    failed_optimizations += 1
                    optimization_results["file_results"].append({
                        "file_identifier": file_identifier if 'file_identifier' in locals() else "unknown",
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
                    "average_optimized_accuracy": float(np.mean(all_optimized_accuracies)),
                    "best_performing_file": {
                        "file_identifier": best_file,
                        "f1_score": float(best_f1)
                    } if best_file else None
                })
                
                # Save consolidated results
                results_file = os.path.join(output_dir, "borderline_smote_threshold_optimization_results.json")
                with open(results_file, 'w') as f:
                    json.dump(optimization_results, f, indent=2)
                
                # Create summary plot showing all optimal thresholds
                plot_threshold_summary_smote(all_optimal_thresholds, all_optimized_f1_scores, output_dir)
                
                print(f"    threshold_optimization_decision_tree_borderline_smote: Completed successfully!")
                print(f"    threshold_optimization_decision_tree_borderline_smote: Optimized {successful_optimizations} SMOTE models")
                print(f"    threshold_optimization_decision_tree_borderline_smote: Average optimal threshold: {np.mean(all_optimal_thresholds):.3f}")
                print(f"    threshold_optimization_decision_tree_borderline_smote: Average optimized F1: {np.mean(all_optimized_f1_scores):.3f}")
                print(f"    threshold_optimization_decision_tree_borderline_smote: Results saved to: {output_dir}")
                
                return {
                    "success": True,
                    "models_optimized": successful_optimizations,
                    "average_optimal_threshold": float(np.mean(all_optimal_thresholds)),
                    "average_optimized_f1": float(np.mean(all_optimized_f1_scores)),
                    "average_optimized_precision": float(np.mean(all_optimized_precisions)),
                    "average_optimized_recall": float(np.mean(all_optimized_recalls)),
                    "average_optimized_accuracy": float(np.mean(all_optimized_accuracies)),
                    "output_directory": output_dir,
                    "results_file": results_file,
                    "target_feature": DEFAULT_TARGET_FEATURE
                }
            else:
                error_msg = "No SMOTE models were successfully optimized"
                print(f"    threshold_optimization_decision_tree_borderline_smote: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"threshold_optimization_decision_tree_borderline_smote failed: {str(e)}"
            print(f"    threshold_optimization_decision_tree_borderline_smote: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def drop_nan_columns(self, data=None, input_path=None, save_results=True):
        """
        Drop columns that contain at least one NaN value from the merged dataset.
        Also drops the 'causes_related_to_weather' column as it's no longer needed for training.

        This method analyzes the merged dataset and removes any columns that have
        at least one NaN/null value, ensuring a clean dataset for model training.
        Additionally removes the 'causes_related_to_weather' column since it's typically
        used for filtering purposes and not needed in the final training dataset.
        Provides detailed reporting on which columns were dropped and why.

        Parameters:
        -----------
        data : pandas.DataFrame, optional
            DataFrame to process. If None, will look for merged data files.
        input_path : str, optional
            Specific file path to process. If None, auto-discovers merged data files.
        save_results : bool, optional
            Whether to save the cleaned dataset to disk. Defaults to True.
            
        Returns:
        --------
        dict
            Results of the NaN column cleanup operation including success status and metadata
        """
        try:
            print(f"    drop_nan_columns: Starting NaN column cleanup operation...")
            
            # Determine data source
            df = None
            original_file_path = None
            
            if data is not None:
                # Use provided DataFrame
                df = data.copy()
                print(f"    drop_nan_columns: Using provided DataFrame with shape {df.shape}")
            elif input_path is not None:
                # Use specific file path
                df = pd.read_csv(input_path)
                original_file_path = input_path
                print(f"    drop_nan_columns: Loaded data from {input_path} with shape {df.shape}")
            else:
                # Auto-discover merged data files
                merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
                merged_data_pattern = os.path.join(merged_training_ready_dir, "merged_data_*.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                # Filter out train/test files
                merged_data_files = [f for f in merged_data_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
                
                if not merged_data_files:
                    error_msg = "No merged data files found for NaN column cleanup"
                    print(f"    drop_nan_columns: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "columns_dropped": 0
                    }
                
                # Use the most recent merged file
                original_file_path = max(merged_data_files, key=os.path.getmtime)
                df = pd.read_csv(original_file_path)
                print(f"    drop_nan_columns: Auto-discovered and loaded {os.path.basename(original_file_path)} with shape {df.shape}")
            
            if df.empty:
                error_msg = "Dataset is empty - cannot perform NaN column cleanup"
                print(f"    drop_nan_columns: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "columns_dropped": 0
                }
            
            # Analyze NaN values in each column
            print(f"    drop_nan_columns: Analyzing NaN values across {len(df.columns)} columns...")
            
            # Get detailed NaN statistics
            nan_stats = {}
            columns_to_drop = []
            columns_to_keep = []
            
            # Track specific columns to drop
            specific_drops = []
            
            for column in df.columns:
                nan_count = df[column].isnull().sum()
                nan_percentage = (nan_count / len(df)) * 100
                
                nan_stats[column] = {
                    'nan_count': int(nan_count),
                    'nan_percentage': round(nan_percentage, 2),
                    'total_rows': len(df)
                }
                
                # Check if column should be dropped due to NaN values
                if nan_count > 0:
                    columns_to_drop.append(column)
                    print(f"      → Will DROP '{column}': {nan_count:,} NaN values ({nan_percentage:.2f}%)")
                # Check if it's the causes_related_to_weather column (drop regardless of NaN status)
                elif column == 'causes_related_to_weather':
                    columns_to_drop.append(column)
                    specific_drops.append(column)
                    print(f"      → Will DROP '{column}': Explicitly removed (not needed for training)")
                else:
                    columns_to_keep.append(column)
            
            print(f"    drop_nan_columns: Analysis complete")
            print(f"      • Columns with NaN values: {len([c for c in columns_to_drop if c not in specific_drops])}")
            print(f"      • Columns dropped for other reasons: {len(specific_drops)}")
            print(f"      • Clean columns to keep: {len(columns_to_keep)}")
            
            if len(columns_to_drop) == 0:
                print(f"    drop_nan_columns: No columns to drop - dataset is already clean!")
                
                # Determine output path
                if original_file_path:
                    output_path = original_file_path
                else:
                    merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
                    os.makedirs(merged_training_ready_dir, exist_ok=True)
                    
                    # Look for existing merged files to preserve naming pattern
                    merged_data_pattern = os.path.join(merged_training_ready_dir, "merged_data_*.csv")
                    merged_data_files = glob.glob(merged_data_pattern)
                    merged_data_files = [f for f in merged_data_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
                    
                    if merged_data_files:
                        output_path = max(merged_data_files, key=os.path.getmtime)
                    else:
                        output_path = os.path.join(merged_training_ready_dir, "merged_data.csv")
                
                result = {
                    "success": True,
                    "data": df,  # Return original data since no changes needed
                    "output_path": output_path,
                    "original_shape": df.shape,
                    "final_shape": df.shape,
                    "columns_dropped": 0,
                    "columns_kept": len(df.columns),
                    "dropped_column_names": [],
                    "kept_column_names": list(df.columns),
                    "nan_analysis": nan_stats,
                    "message": "No columns to drop - dataset is already clean"
                }
                
                if save_results and original_file_path:
                    # Still save a summary even if no changes were made
                    save_nan_cleanup_summary(result, original_file_path, self.project_root)
                
                return result
            
            if len(columns_to_keep) == 0:
                error_msg = "All columns would be dropped - cannot proceed with dataset"
                print(f"    drop_nan_columns: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "columns_dropped": len(columns_to_drop),
                    "nan_analysis": nan_stats
                }
            
            # Drop columns with NaN values and specific columns
            original_shape = df.shape
            df_cleaned = df[columns_to_keep].copy()
            final_shape = df_cleaned.shape
            
            print(f"    drop_nan_columns: Dropped {len(columns_to_drop)} columns total:")
            print(f"      • {len([c for c in columns_to_drop if c not in specific_drops])} columns with NaN values")
            print(f"      • {len(specific_drops)} columns dropped for training optimization")
            print(f"      • Dataset shape: {original_shape} → {final_shape}")
            
            # Determine output path for cleaned dataset - MODIFIED SECTION
            if original_file_path:
                # Save cleaned dataset back to the original merged file
                output_path = original_file_path
                print(f"    drop_nan_columns: Will overwrite original file: {os.path.basename(output_path)}")
            else:
                # Auto-discover existing merged files to preserve naming pattern
                merged_training_ready_dir = os.path.join(self.project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
                os.makedirs(merged_training_ready_dir, exist_ok=True)
                
                merged_data_pattern = os.path.join(merged_training_ready_dir, "merged_data_*.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                
                # Filter out train/test files
                merged_data_files = [f for f in merged_data_files if not (f.endswith('_train.csv') or f.endswith('_test.csv'))]
                
                if merged_data_files:
                    # Use the most recent merged file to maintain original naming pattern
                    output_path = max(merged_data_files, key=os.path.getmtime)
                    print(f"    drop_nan_columns: Will overwrite existing merged file: {os.path.basename(output_path)}")
                else:
                    # Fallback: create a generic merged file (should rarely happen)
                    output_path = os.path.join(merged_training_ready_dir, "merged_data.csv")
                    print(f"    drop_nan_columns: Will create new file: {os.path.basename(output_path)}")
            
            # Save cleaned dataset
            if save_results:
                df_cleaned.to_csv(output_path, index=False)
                print(f"    drop_nan_columns: Saved cleaned dataset back to {os.path.basename(output_path)}")
            
            # Prepare detailed result
            result = {
                "success": True,
                "data": df_cleaned,
                "output_path": output_path,
                "original_shape": original_shape,
                "final_shape": final_shape,
                "columns_dropped": len(columns_to_drop),
                "columns_kept": len(columns_to_keep),
                "dropped_column_names": columns_to_drop,
                "kept_column_names": columns_to_keep,
                "nan_analysis": nan_stats,
                "specific_drops": specific_drops,
                "message": f"Successfully dropped {len(columns_to_drop)} columns ({len([c for c in columns_to_drop if c not in specific_drops])} with NaN, {len(specific_drops)} for optimization)"
            }
            
            # Save detailed summary
            if save_results:
                save_nan_cleanup_summary(result, original_file_path or "provided_data", self.project_root)
            
            print(f"    drop_nan_columns: NaN column cleanup completed successfully")
            
            return result
            
        except Exception as e:
            error_msg = f"drop_nan_columns failed: {str(e)}"
            print(f"    drop_nan_columns: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "columns_dropped": 0
            }

    def train_xgboost_selected_features(self):
        """
        Train XGBoost models using only selected weather features from SELECTED_WEATHER_FEATURES.
        
        This method:
        1. Loads the scaled training/test data files
        2. Filters weather features to keep only those specified in SELECTED_WEATHER_FEATURES
        3. Trains XGBoost models with hyperparameter optimization
        4. Evaluates and saves the trained models
        
        The method processes files from:
        - data/output/4-merged_scaled_training_ready/merged_data_*_train_scaled.csv
        - data/output/4-merged_scaled_training_ready/merged_data_*_test_scaled.csv
        
        Returns:
        --------
        dict
            Training results summary including model performance metrics
        """
        try:
            print(f"    train_xgboost_selected_features: Starting XGBoost training with selected weather features...")
            print(f"    train_xgboost_selected_features: Selected features: {SELECTED_WEATHER_FEATURES}")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, XGBOOST_SELECTED_FEATURES_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"    train_xgboost_selected_features: Output directory: {output_dir}")
            
            # Find training and test files
            train_pattern = os.path.join(self.project_root, "data/output/4-merged_scaled_training_ready/merged_data_*_train_scaled.csv")
            test_pattern = os.path.join(self.project_root, "data/output/4-merged_scaled_training_ready/merged_data_*_test_scaled.csv")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if not train_files or not test_files:
                error_msg = f"Training or test files not found. Train files: {len(train_files)}, Test files: {len(test_files)}"
                print(f"    train_xgboost_selected_features: {error_msg}")
                return {"success": False, "error": error_msg}
            
            print(f"    train_xgboost_selected_features: Found {len(train_files)} training files and {len(test_files)} test files")
            
            # Determine problem type
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM
            
            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_xgboost_selected_features: {error_msg}")
                return {"success": False, "error": error_msg}
            
            problem_type = "classification" if is_classification else "regression"
            print(f"    train_xgboost_selected_features: Detected {problem_type} problem for target '{target_feature}'")
            
            # Initialize results structure
            training_results = {
                "training_overview": {
                    "training_completed": datetime.now().isoformat(),
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "selected_weather_features": SELECTED_WEATHER_FEATURES,
                    "total_files_processed": 0,
                    "successful_trainings": 0,
                    "failed_trainings": 0,
                    "hyperparameter_search_iterations": RANDOM_SEARCH_ITERATIONS,
                    "cross_validation_folds": RANDOM_SEARCH_CV_FOLDS
                },
                "xgboost_config": {
                    "param_distributions": format_param_distributions_for_json(XGBOOST_PARAM_DISTRIBUTIONS),
                    "methods_config": XGBOOST_METHODS_CONFIG.get(problem_type, [])
                },
                "file_results": [],
                "aggregate_metrics": {}
            }
            
            successful_trainings = 0
            failed_trainings = 0
            all_cv_scores = []
            all_test_scores = []
            
            # Memory tracking
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            print(f"    train_xgboost_selected_features: Initial memory usage: {initial_memory:.2f} MB")
            
            # Process each training/test file pair
            for train_file in train_files:
                try:
                    # Extract file identifier
                    train_filename = os.path.basename(train_file)
                    identifier_match = re.search(r'merged_data_(.+?)_train_scaled\.csv', train_filename)
                    
                    if not identifier_match:
                        print(f"      Warning: Could not extract identifier from {train_filename}")
                        failed_trainings += 1
                        continue
                    
                    identifier = identifier_match.group(1)
                    test_filename = f"merged_data_{identifier}_test_scaled.csv"
                    test_file = os.path.join(os.path.dirname(train_file), test_filename)
                    
                    if not os.path.exists(test_file):
                        print(f"      Warning: Test file not found: {test_filename}")
                        failed_trainings += 1
                        continue
                    
                    print(f"\n      Processing: {identifier}")
                    print(f"        Train file: {train_filename}")
                    print(f"        Test file: {test_filename}")
                    
                    # Load datasets
                    print(f"        Loading datasets...")
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                    
                    if train_df.empty or test_df.empty:
                        print(f"        Warning: Empty dataset found. Skipping {identifier}")
                        failed_trainings += 1
                        continue
                    
                    print(f"        Train shape: {train_df.shape}")
                    print(f"        Test shape: {test_df.shape}")
                    
                    # Check target feature
                    if target_feature not in train_df.columns or target_feature not in test_df.columns:
                        print(f"        Warning: Target feature '{target_feature}' not found. Skipping {identifier}")
                        failed_trainings += 1
                        continue
                    
                    # Filter features: keep selected weather features + non-weather features
                    print(f"        Filtering features...")
                    
                    # Identify available selected weather features
                    available_selected_weather = [col for col in SELECTED_WEATHER_FEATURES if col in train_df.columns]
                    print(f"        Available selected weather features ({len(available_selected_weather)}): {available_selected_weather}")

                    # MODIFIED LOGIC: Handle empty SELECTED_WEATHER_FEATURES
                    if not SELECTED_WEATHER_FEATURES:
                        print(f"        SELECTED_WEATHER_FEATURES is empty - training with NO weather features")
                        available_selected_weather = []
                    elif not available_selected_weather:
                        print(f"        Warning: None of the selected weather features found in dataset. Training with NO weather features for {identifier}")
                        available_selected_weather = []
                    
                    # Identify non-weather features (excluding target and index columns)
                    all_weather_cols = [col for col in ALL_WEATHER_FEATURES if col in train_df.columns]
                    non_weather_cols = [col for col in train_df.columns 
                                    if col not in all_weather_cols 
                                    and col != target_feature
                                    and not col.startswith('Unnamed')
                                    and col != 'index']
                    
                    print(f"        Non-weather features ({len(non_weather_cols)}): {non_weather_cols}")
                    
                    # Create feature set: selected weather features + non-weather features
                    feature_columns = available_selected_weather + non_weather_cols
                    print(f"        Final feature set ({len(feature_columns)}): {len(available_selected_weather)} weather + {len(non_weather_cols)} non-weather")
                    
                    # Prepare training data
                    X_train = train_df[feature_columns]
                    y_train = train_df[target_feature]
                    X_test = test_df[feature_columns]
                    y_test = test_df[target_feature]
                    
                    # Handle missing values
                    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
                        print(f"        Warning: Missing values detected. Filling with median/mode...")
                        
                        # Fill numerical features with median
                        numerical_features = X_train.select_dtypes(include=[np.number]).columns
                        for col in numerical_features:
                            median_val = X_train[col].median()
                            X_train[col].fillna(median_val, inplace=True)
                            X_test[col].fillna(median_val, inplace=True)
                        
                        # Fill categorical features with mode
                        categorical_features = X_train.select_dtypes(include=['object']).columns
                        for col in categorical_features:
                            mode_val = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else 'Unknown'
                            X_train[col].fillna(mode_val, inplace=True)
                            X_test[col].fillna(mode_val, inplace=True)
                    
                    # Prepare sample weights if configured
                    sample_weights = None
                    if WEIGHT_DELAY_COLUMN != 'NONE' and WEIGHT_DELAY_COLUMN in train_df.columns:
                        print(f"        Applying sample weights based on '{WEIGHT_DELAY_COLUMN}'...")
                        sample_weights = train_df[WEIGHT_DELAY_COLUMN].abs() + 1  # Add 1 to avoid zero weights
                        
                        if is_classification:
                            sample_weights = np.clip(sample_weights, 1, MAX_SAMPLE_WEIGHT_CLASSIFICATION)
                        else:
                            sample_weights = np.clip(sample_weights, 1, MAX_SAMPLE_WEIGHT_REGRESSION)
                    
                    # Configure cross-validation and scoring
                    if is_classification:
                        cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                        scoring_metric = SCORE_METRIC
                        base_model = xgb.XGBClassifier(
                            random_state=42,
                            n_jobs=1,
                            eval_metric='logloss'
                        )
                    else:
                        cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                        scoring_metric = 'neg_mean_squared_error'
                        base_model = xgb.XGBRegressor(
                            random_state=42,
                            n_jobs=1,
                            eval_metric='rmse'
                        )
                    
                    # Perform hyperparameter optimization
                    print(f"        Starting RandomizedSearchCV with {RANDOM_SEARCH_ITERATIONS} iterations...")
                    
                    randomized_search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
                        n_iter=RANDOM_SEARCH_ITERATIONS,
                        scoring=scoring_metric,
                        cv=cv_splitter,
                        random_state=42,
                        n_jobs=1,
                        verbose=1
                    )
                    
                    # Fit with sample weights if available
                    if sample_weights is not None:
                        randomized_search.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        randomized_search.fit(X_train, y_train)
                    
                    best_model = randomized_search.best_estimator_
                    best_params = randomized_search.best_params_
                    best_cv_score = randomized_search.best_score_
                    
                    print(f"        Best CV Score: {best_cv_score:.4f}")
                    print(f"        Best Parameters: {best_params}")
                    
                    # Make predictions on test set
                    y_pred = best_model.predict(X_test)
                    
                    # Calculate test metrics
                    if is_classification:
                        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred
                        
                        test_accuracy = accuracy_score(y_test, y_pred)
                        test_f1 = f1_score(y_test, y_pred, average='weighted')
                        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        test_recall = recall_score(y_test, y_pred, average='weighted')
                        test_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
                        
                        print(f"        Test Accuracy: {test_accuracy:.4f}")
                        print(f"        Test F1: {test_f1:.4f}")
                        print(f"        Test Precision: {test_precision:.4f}")
                        print(f"        Test Recall: {test_recall:.4f}")
                        print(f"        Test AUC: {test_auc:.4f}")
                        
                        test_score = test_f1  # Use F1 as primary metric for classification
                    else:
                        test_mse = mean_squared_error(y_test, y_pred)
                        test_rmse = np.sqrt(test_mse)
                        test_mae = mean_absolute_error(y_test, y_pred)
                        test_r2 = r2_score(y_test, y_pred)
                        
                        print(f"        Test RMSE: {test_rmse:.4f}")
                        print(f"        Test MAE: {test_mae:.4f}")
                        print(f"        Test R²: {test_r2:.4f}")
                        
                        test_score = -test_rmse  # Use negative RMSE for consistency with CV scoring
                    
                    # Save model and results
                    model_filename = f"xgboost_selected_features_{identifier}.joblib"
                    model_path = os.path.join(output_dir, model_filename)
                    joblib.dump(best_model, model_path)
                    print(f"        Model saved: {model_filename}")
                    
                    # Store detailed results
                    file_result = {
                        "file_identifier": identifier,
                        "train_file": train_filename,
                        "test_file": test_filename,
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "features_used": len(feature_columns),
                        "selected_weather_features": available_selected_weather,
                        "non_weather_features": non_weather_cols,
                        "best_cv_score": float(best_cv_score),
                        "best_parameters": best_params,
                        "model_path": model_path
                    }
                    
                    # Add problem-specific metrics
                    if is_classification:
                        file_result.update({
                            "test_accuracy": float(test_accuracy),
                            "test_f1": float(test_f1),
                            "test_precision": float(test_precision),
                            "test_recall": float(test_recall),
                            "test_auc": float(test_auc)
                        })
                    else:
                        file_result.update({
                            "test_rmse": float(test_rmse),
                            "test_mae": float(test_mae),
                            "test_r2": float(test_r2)
                        })
                    
                    training_results["file_results"].append(file_result)
                    all_cv_scores.append(best_cv_score)
                    all_test_scores.append(test_score)
                    successful_trainings += 1
                    
                    # Memory check
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"        Current memory usage: {current_memory:.2f} MB")
                    
                except Exception as e:
                    print(f"        Error processing {train_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    failed_trainings += 1
                    continue
            
            # Calculate aggregate metrics
            training_results["training_overview"]["total_files_processed"] = len(train_files)
            training_results["training_overview"]["successful_trainings"] = successful_trainings
            training_results["training_overview"]["failed_trainings"] = failed_trainings
            
            if successful_trainings > 0:
                training_results["aggregate_metrics"] = {
                    "average_cv_score": float(np.mean(all_cv_scores)),
                    "std_cv_score": float(np.std(all_cv_scores)),
                    "average_test_score": float(np.mean(all_test_scores)),
                    "std_test_score": float(np.std(all_test_scores))
                }
                
                # Calculate problem-specific aggregate metrics
                if is_classification:
                    all_test_accuracies = [r["test_accuracy"] for r in training_results["file_results"]]
                    all_test_f1s = [r["test_f1"] for r in training_results["file_results"]]
                    all_test_precisions = [r["test_precision"] for r in training_results["file_results"]]
                    all_test_recalls = [r["test_recall"] for r in training_results["file_results"]]
                    all_test_aucs = [r["test_auc"] for r in training_results["file_results"]]
                    
                    training_results["aggregate_metrics"].update({
                        "average_test_accuracy": float(np.mean(all_test_accuracies)),
                        "average_test_f1": float(np.mean(all_test_f1s)),
                        "average_test_precision": float(np.mean(all_test_precisions)),
                        "average_test_recall": float(np.mean(all_test_recalls)),
                        "average_test_auc": float(np.mean(all_test_aucs))
                    })
                else:
                    all_test_rmses = [r["test_rmse"] for r in training_results["file_results"]]
                    all_test_maes = [r["test_mae"] for r in training_results["file_results"]]
                    all_test_r2s = [r["test_r2"] for r in training_results["file_results"]]
                    
                    training_results["aggregate_metrics"].update({
                        "average_test_rmse": float(np.mean(all_test_rmses)),
                        "average_test_mae": float(np.mean(all_test_maes)),
                        "average_test_r2": float(np.mean(all_test_r2s))
                    })
            
            # Save results summary
            results_file = os.path.join(output_dir, "xgboost_selected_features_training_results.json")
            with open(results_file, 'w') as f:
                json.dump(training_results, f, indent=2)
            
            # Generate feature importance report
            generate_feature_importance_report(training_results, output_dir)
            
            if successful_trainings > 0:
                print(f"    train_xgboost_selected_features: Trained {successful_trainings} XGBoost models successfully")
                print(f"    train_xgboost_selected_features: Average CV Score: {np.mean(all_cv_scores):.4f}")
                
                if is_classification:
                    print(f"    train_xgboost_selected_features: Average Test F1: {training_results['aggregate_metrics']['average_test_f1']:.4f}")
                    print(f"    train_xgboost_selected_features: Average Test Accuracy: {training_results['aggregate_metrics']['average_test_accuracy']:.4f}")
                else:
                    print(f"    train_xgboost_selected_features: Average Test RMSE: {training_results['aggregate_metrics']['average_test_rmse']:.4f}")
                    print(f"    train_xgboost_selected_features: Average Test R²: {training_results['aggregate_metrics']['average_test_r2']:.4f}")
                
                print(f"    train_xgboost_selected_features: Results saved to: {output_dir}")
                
                return {
                    "success": True,
                    "models_trained": successful_trainings,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "selected_weather_features": SELECTED_WEATHER_FEATURES,
                    "average_cv_score": float(np.mean(all_cv_scores)),
                    "output_directory": output_dir,
                    "results_file": results_file,
                    **training_results["aggregate_metrics"]
                }
            else:
                error_msg = "No XGBoost models with selected features were successfully trained"
                print(f"    train_xgboost_selected_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"train_xgboost_selected_features failed: {str(e)}"
            print(f"    train_xgboost_selected_features: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def correlation_analysis_by_station(self, csv_files=None):
        """
        Performs comprehensive correlation analysis on merged scaled training data.
        Enhanced with scatter plot for Snow depth vs trainDelayed analysis.
        
        Two types of analysis are performed:
        1. Original analysis: Feature-to-feature correlations for delayed trains only
        2. New analysis: Correlation between trainDelayed (binary) and all features
        3. NEW: Scatter plot visualization of Snow depth vs trainDelayed
        
        This method finds all merged scaled training data files and performs comprehensive
        correlation analysis to understand feature relationships and their impact on delays.
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (currently not used - method discovers files automatically)
            
        Returns:
        --------
        dict
            Results of the comprehensive correlation analysis including both analyses
        """
        try:
            print(f"    correlation_analysis_by_station: Starting comprehensive correlation analysis...")
            
            # Create output directories
            correlation_output_dir = os.path.join(self.project_root, "data/output/correlation_analysis_comprehensive")
            delayed_only_dir = os.path.join(correlation_output_dir, "delayed_trains_only")
            target_correlation_dir = os.path.join(correlation_output_dir, "target_correlations")
            scatter_plots_dir = os.path.join(correlation_output_dir, "scatter_plots")
            
            os.makedirs(correlation_output_dir, exist_ok=True)
            os.makedirs(delayed_only_dir, exist_ok=True)
            os.makedirs(target_correlation_dir, exist_ok=True)
            os.makedirs(scatter_plots_dir, exist_ok=True)
            
            # Find all merged scaled training data files
            merged_data_pattern = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*_train_scaled.csv")
            merged_data_files = glob.glob(merged_data_pattern)
            
            # Fallback pattern
            if not merged_data_files:
                user_specified_pattern = os.path.join(self.project_root, "data/output/3-merged_training_ready/merged_data_*.csv")
                merged_data_files = glob.glob(user_specified_pattern)
                print(f"    correlation_analysis_by_station: Using user-specified path pattern")
            
            if not merged_data_files:
                error_msg = "No merged data files found for correlation analysis"
                print(f"    correlation_analysis_by_station: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    correlation_analysis_by_station: Found {len(merged_data_files)} merged data files")
            for file_path in merged_data_files:
                print(f"      - {os.path.basename(file_path)}")
            
            # Initialize storage for results
            delayed_correlation_results = []
            target_correlation_results = []
            scatter_plot_results = []
            total_files_processed = 0
            
            # Process each merged data file
            for file_path in merged_data_files:
                try:
                    filename = os.path.basename(file_path)
                    print(f"    correlation_analysis_by_station: Processing {filename}...")
                    
                    # Read the merged dataset
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"    correlation_analysis_by_station: Warning - File {filename} is empty, skipping...")
                        continue
                    
                    total_rows = len(df)
                    print(f"      Total rows in dataset: {total_rows:,}")
                    
                    # Get numeric columns for correlation analysis
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Remove target and non-relevant columns from numeric analysis
                    columns_to_exclude = [DEFAULT_TARGET_FEATURE, 'trainDelayed', 'cancelled']
                    numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]
                    
                    if not numeric_columns:
                        print(f"      No numeric columns found for correlation analysis in {filename}, skipping...")
                        continue
                    
                    print(f"      Numeric columns for analysis: {len(numeric_columns)}")
                    
                    # ==================================================================
                    # PART 1: Original Analysis - Feature-to-Feature Correlations (Delayed Only)
                    # ==================================================================
                    
                    # Filter to delayed trains only
                    if DEFAULT_TARGET_FEATURE in df.columns:
                        # Handle different target types
                        if DEFAULT_TARGET_FEATURE in ['trainDelayed', 'cancelled']:
                            delayed_df = df[df[DEFAULT_TARGET_FEATURE] == True].copy()
                        else:
                            # For numeric targets, consider delayed if > 0
                            delayed_df = df[df[DEFAULT_TARGET_FEATURE] > 0].copy()
                    else:
                        print(f"      Warning: Target feature '{DEFAULT_TARGET_FEATURE}' not found in {filename}")
                        delayed_df = pd.DataFrame()
                    
                    if not delayed_df.empty:
                        delayed_count = len(delayed_df)
                        delayed_percentage = (delayed_count / total_rows) * 100
                        
                        print(f"      Delayed trains: {delayed_count:,} ({delayed_percentage:.1f}%)")
                        
                        # Check if we have enough delayed samples for meaningful correlation
                        if delayed_count < 10:
                            print(f"      Too few delayed trains ({delayed_count}) for correlation analysis. Skipping delayed-only analysis.")
                        else:
                            feature_data = delayed_df[numeric_columns]
                            
                            # Calculate Pearson correlation matrix for delayed trains
                            delayed_correlation_matrix = feature_data.corr()
                            
                            print(f"    correlation_analysis_by_station: Calculated {delayed_correlation_matrix.shape[0]}x{delayed_correlation_matrix.shape[1]} correlation matrix for delayed trains")
                            
                            # Store results for delayed trains analysis
                            delayed_file_result = {
                                'filename': filename,
                                'total_original_rows': total_rows,
                                'delayed_rows': delayed_count,
                                'delayed_percentage': delayed_percentage,
                                'features_analyzed': list(delayed_correlation_matrix.columns),
                                'correlation_matrix': delayed_correlation_matrix,
                                'feature_count': len(delayed_correlation_matrix.columns)
                            }
                            
                            delayed_correlation_results.append(delayed_file_result)
                            
                            # Create and save correlation plot for delayed trains
                            self._create_delayed_trains_correlation_plot(
                                delayed_correlation_matrix,
                                delayed_file_result,
                                delayed_only_dir
                            )
                    
                    # ==================================================================
                    # PART 2: New Analysis - Target Variable Correlations (All Data)
                    # ==================================================================
                    
                    if DEFAULT_TARGET_FEATURE in df.columns:
                        print(f"    correlation_analysis_by_station: Analyzing correlations with {DEFAULT_TARGET_FEATURE}...")
                        
                        # Calculate correlations between target and all numeric features
                        target_correlations = {}
                        
                        for feature in numeric_columns:
                            try:
                                if feature in df.columns and not df[feature].isna().all():
                                    if DEFAULT_TARGET_FEATURE in ['trainDelayed', 'cancelled']:
                                        # Point-biserial correlation for binary target
                                        correlation = self._calculate_point_biserial_correlation(
                                            df[DEFAULT_TARGET_FEATURE].astype(int), 
                                            df[feature]
                                        )
                                        corr_type = "point_biserial"
                                    else:
                                        # Pearson correlation for continuous target
                                        correlation = df[DEFAULT_TARGET_FEATURE].corr(df[feature])
                                        corr_type = "pearson"
                                    
                                    target_correlations[feature] = {
                                        'correlation': correlation,
                                        'type': corr_type
                                    }
                            except Exception as e:
                                print(f"        Warning: Could not calculate correlation for {feature}: {e}")
                                target_correlations[feature] = {
                                    'correlation': np.nan,
                                    'type': 'error'
                                }
                        
                        # Store target correlation results
                        target_proportion = None
                        if DEFAULT_TARGET_FEATURE in ['trainDelayed', 'cancelled']:
                            target_proportion = df[DEFAULT_TARGET_FEATURE].mean()
                        
                        target_file_result = {
                            'filename': filename,
                            'total_samples': total_rows,
                            'target_proportion': target_proportion,
                            'available_features': list(numeric_columns),
                            'target_correlations': target_correlations,
                            'features_analyzed': len([f for f, info in target_correlations.items() 
                                                    if not pd.isna(info['correlation'])])
                        }
                        
                        target_correlation_results.append(target_file_result)
                        
                        # Create and save target correlation plot
                        self._create_target_correlation_plot(
                            target_correlations,
                            target_file_result,
                            target_correlation_dir
                        )
                    
                    # ==================================================================
                    # PART 3: NEW - Snow Depth vs trainDelayed Scatter Plot
                    # ==================================================================
                    
                    # Create scatter plot for Snow depth vs trainDelayed
                    scatter_result = self._create_snow_depth_scatter_plot(
                        df, filename, scatter_plots_dir
                    )
                    if scatter_result:
                        scatter_plot_results.append(scatter_result)
                    
                    total_files_processed += 1
                    print(f"      ✓ Completed analysis for {filename}")
                    
                except Exception as e:
                    print(f"      Error processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if total_files_processed == 0:
                error_msg = "No files were successfully processed"
                print(f"    correlation_analysis_by_station: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            # Create combined analyses for delayed correlations
            if delayed_correlation_results:
                self._create_combined_delayed_analysis(delayed_correlation_results, delayed_only_dir)
            
            if target_correlation_results:
                self._create_combined_target_analysis(target_correlation_results, target_correlation_dir)
            
            # Save comprehensive summary
            summary_path = os.path.join(correlation_output_dir, "comprehensive_correlation_summary.txt")
            self._save_comprehensive_summary(
                delayed_correlation_results, 
                target_correlation_results, 
                summary_path
            )
            
            print(f"    correlation_analysis_by_station: Analysis completed for {total_files_processed} files")
            print(f"    correlation_analysis_by_station: Results saved to {correlation_output_dir}")
            
            # Return comprehensive results
            result = {
                "success": True,
                "processed_files": total_files_processed,
                "output_path": correlation_output_dir,
                "summary_path": summary_path,
                "delayed_correlation_results": delayed_correlation_results,
                "target_correlation_results": target_correlation_results,
                "scatter_plot_results": scatter_plot_results,
                "total_trains_analyzed": sum(r['total_original_rows'] for r in delayed_correlation_results) if delayed_correlation_results else 0,
                "total_delayed_trains": sum(r['delayed_rows'] for r in delayed_correlation_results) if delayed_correlation_results else 0,
                "analysis_types": ["Feature-to-Feature (Delayed Only)", "Target-to-Feature (All Data)", "Snow Depth Scatter Plot"],
                "message": f"Successfully completed comprehensive correlation analysis for {total_files_processed} files"
            }
            
            print(f"    correlation_analysis_by_station: Completed successfully - {total_files_processed} files analyzed")
            
            return result
            
        except Exception as e:
            error_msg = f"correlation_analysis_by_station failed: {str(e)}"
            print(f"    correlation_analysis_by_station: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }
    
    def _calculate_point_biserial_correlation(self, binary_var, continuous_var):
        """
        Calculate Point-Biserial correlation coefficient between binary and continuous variables.
        
        Parameters:
        -----------
        binary_var : array-like
            Binary variable (0/1)
        continuous_var : array-like
            Continuous variable
            
        Returns:
        --------
        float
            Point-Biserial correlation coefficient
        """
        from scipy import stats
        
        # Remove NaN values
        mask = ~(pd.isna(binary_var) | pd.isna(continuous_var))
        binary_clean = binary_var[mask]
        continuous_clean = continuous_var[mask]
        
        if len(binary_clean) < 2:
            return np.nan
        
        # Calculate Point-Biserial correlation
        correlation, _ = stats.pointbiserialr(binary_clean, continuous_clean)
        
        return correlation

    def _calculate_phi_coefficient(self, binary_var1, binary_var2):
        """
        Calculate Phi coefficient (correlation between two binary variables).
        
        Parameters:
        -----------
        binary_var1 : array-like
            First binary variable (0/1)
        binary_var2 : array-like
            Second binary variable (0/1)
            
        Returns:
        --------
        float
            Phi coefficient
        """
        # Remove NaN values
        mask = ~(pd.isna(binary_var1) | pd.isna(binary_var2))
        var1_clean = binary_var1[mask]
        var2_clean = binary_var2[mask]
        
        if len(var1_clean) < 2:
            return np.nan
        
        # Create contingency table
        contingency = pd.crosstab(var1_clean, var2_clean)
        
        if contingency.shape != (2, 2):
            return np.nan
        
        # Calculate Phi coefficient
        n = contingency.sum().sum()
        chi2 = ((contingency.iloc[0,0] * contingency.iloc[1,1] - contingency.iloc[0,1] * contingency.iloc[1,0]) ** 2 * n) / \
            (contingency.iloc[0,:].sum() * contingency.iloc[1,:].sum() * 
                contingency.iloc[:,0].sum() * contingency.iloc[:,1].sum())
        
        phi = np.sqrt(chi2 / n)
        
        # Determine sign
        if contingency.iloc[0,0] * contingency.iloc[1,1] < contingency.iloc[0,1] * contingency.iloc[1,0]:
            phi = -phi
        
        return phi

    def _create_target_correlation_plot(self, target_correlations, file_result, output_dir):
        """
        Create visualization for correlations between target variable and all features.
        
        Parameters:
        -----------
        target_correlations : dict
            Dictionary of feature correlations with target
        file_result : dict
            Results for this specific file
        output_dir : str
            Directory to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract correlation values and feature names
            features = []
            correlations = []
            correlation_types = []
            
            for feature, corr_info in target_correlations.items():
                if not pd.isna(corr_info['correlation']):
                    features.append(feature)
                    correlations.append(corr_info['correlation'])
                    correlation_types.append(corr_info['type'])
            
            if not features:
                print(f"      No valid correlations to plot for {file_result['filename']}")
                return
            
            # Sort by absolute correlation value
            sorted_indices = sorted(range(len(correlations)), 
                                key=lambda i: abs(correlations[i]), 
                                reverse=True)
            
            features = [features[i] for i in sorted_indices]
            correlations = [correlations[i] for i in sorted_indices]
            correlation_types = [correlation_types[i] for i in sorted_indices]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Subplot 1: Bar chart of correlations
            colors = ['green' if c > 0 else 'red' for c in correlations]
            bars = ax1.barh(range(len(features)), correlations, color=colors, alpha=0.7)
            
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features, fontsize=8)
            ax1.set_xlabel('Correlation with trainDelayed', fontsize=10)
            ax1.set_title(f'Feature Correlations with Train Delays\n{file_result["filename"]}\n'
                        f'Total: {file_result["total_rows"]:,} trains, '
                        f'Delayed: {file_result["delayed_rows"]:,} ({file_result["delayed_percentage"]:.1f}%)',
                        fontsize=12, pad=20)
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax1.grid(True, alpha=0.3)
            
            # Add correlation values on bars
            for i, (bar, corr) in enumerate(zip(bars, correlations)):
                width = bar.get_width()
                ax1.text(width + 0.005 if width > 0 else width - 0.005, 
                        bar.get_y() + bar.get_height()/2,
                        f'{corr:.3f}', 
                        ha='left' if width > 0 else 'right',
                        va='center', fontsize=7)
            
            # Subplot 2: Top positive and negative correlations
            top_n = min(10, len(correlations))
            top_positive = [(f, c) for f, c in zip(features, correlations) if c > 0][:top_n]
            top_negative = [(f, c) for f, c in zip(features, correlations) if c < 0][:top_n]
            
            # Create heatmap-style visualization
            heatmap_data = []
            heatmap_labels = []
            
            for feat, corr in top_positive[:5]:
                heatmap_data.append(corr)
                heatmap_labels.append(f"{feat[:20]}...")
            
            for feat, corr in top_negative[:5]:
                heatmap_data.append(corr)
                heatmap_labels.append(f"{feat[:20]}...")
            
            if heatmap_data:
                heatmap_matrix = np.array(heatmap_data).reshape(-1, 1)
                
                im = ax2.imshow(heatmap_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                ax2.set_yticks(range(len(heatmap_labels)))
                ax2.set_yticklabels(heatmap_labels, fontsize=8)
                ax2.set_xticks([])
                ax2.set_title('Top Positive & Negative Correlations', fontsize=12)
                
                # Add text annotations
                for i, val in enumerate(heatmap_data):
                    ax2.text(0, i, f'{val:.3f}', ha='center', va='center',
                            color='white' if abs(val) > 0.5 else 'black', fontsize=9)
                
                # Add colorbar
                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Save plot
            base_filename = os.path.splitext(file_result['filename'])[0]
            plot_path = os.path.join(output_dir, f"{base_filename}_target_correlations.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      Saved target correlation plot: {base_filename}_target_correlations.png")
            
        except Exception as e:
            print(f"      Error creating target correlation plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_combined_target_analysis(self, target_correlation_results, output_dir):
        """
        Create combined analysis visualization for target correlations across all files.
        
        Parameters:
        -----------
        target_correlation_results : list
            List of target correlation results from all files
        output_dir : str
            Directory to save the combined analysis
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("    correlation_analysis_by_station: Creating combined target correlation analysis...")
            
            # Aggregate correlations across all files
            all_correlations = {}
            
            for result in target_correlation_results:
                for feature, corr_info in result['target_correlations'].items():
                    if feature not in all_correlations:
                        all_correlations[feature] = []
                    
                    if not pd.isna(corr_info['correlation']):
                        all_correlations[feature].append(corr_info['correlation'])
            
            # Calculate average correlations
            avg_correlations = {}
            for feature, corr_list in all_correlations.items():
                if corr_list:
                    avg_correlations[feature] = {
                        'mean': np.mean(corr_list),
                        'std': np.std(corr_list),
                        'min': np.min(corr_list),
                        'max': np.max(corr_list),
                        'count': len(corr_list)
                    }
            
            if not avg_correlations:
                print("    correlation_analysis_by_station: No correlations to combine")
                return
            
            # Sort by absolute mean correlation
            sorted_features = sorted(avg_correlations.keys(), 
                                key=lambda x: abs(avg_correlations[x]['mean']), 
                                reverse=True)
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(18, 10))
            
            # Create grid layout
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Top: bar chart
            ax2 = fig.add_subplot(gs[1, 0])  # Bottom left: distribution
            ax3 = fig.add_subplot(gs[1, 1])  # Bottom right: summary stats
            
            # Ax1: Bar chart with error bars
            top_features = sorted_features[:20]  # Show top 20 features
            means = [avg_correlations[f]['mean'] for f in top_features]
            stds = [avg_correlations[f]['std'] for f in top_features]
            
            colors = ['green' if m > 0 else 'red' for m in means]
            bars = ax1.barh(range(len(top_features)), means, xerr=stds, 
                        color=colors, alpha=0.7, capsize=3)
            
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features, fontsize=9)
            ax1.set_xlabel('Average Correlation with trainDelayed', fontsize=11)
            ax1.set_title(f'Top Feature Correlations with Train Delays (Averaged Across {len(target_correlation_results)} Files)',
                        fontsize=13, pad=20)
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax1.grid(True, alpha=0.3)
            
            # Ax2: Distribution of correlations
            positive_corrs = [avg_correlations[f]['mean'] for f in sorted_features 
                            if avg_correlations[f]['mean'] > 0]
            negative_corrs = [avg_correlations[f]['mean'] for f in sorted_features 
                            if avg_correlations[f]['mean'] < 0]
            
            ax2.hist(positive_corrs, bins=20, alpha=0.7, color='green', label='Positive')
            ax2.hist(negative_corrs, bins=20, alpha=0.7, color='red', label='Negative')
            ax2.set_xlabel('Correlation Value', fontsize=10)
            ax2.set_ylabel('Number of Features', fontsize=10)
            ax2.set_title('Distribution of Feature Correlations', fontsize=11)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Ax3: Summary statistics table
            summary_data = {
                'Metric': ['Total Features', 'Positive Correlations', 'Negative Correlations',
                        'Strongest Positive', 'Strongest Negative', 'Average |Correlation|'],
                'Value': [
                    len(sorted_features),
                    len(positive_corrs),
                    len(negative_corrs),
                    f"{max(means):.3f} ({sorted_features[means.index(max(means))]})",
                    f"{min(means):.3f} ({sorted_features[means.index(min(means))]})",
                    f"{np.mean([abs(m) for m in means]):.3f}"
                ]
            }
            
            # Create table
            ax3.axis('tight')
            ax3.axis('off')
            table = ax3.table(cellText=[[k, v] for k, v in zip(summary_data['Metric'], summary_data['Value'])],
                            colLabels=['Metric', 'Value'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            ax3.set_title('Summary Statistics', fontsize=11, pad=20)
            
            plt.suptitle('Combined Target Correlation Analysis', fontsize=15, y=1.02)
            
            # Save plot
            plot_path = os.path.join(output_dir, "combined_target_correlation_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    correlation_analysis_by_station: Saved combined target analysis to combined_target_correlation_analysis.png")
            
            # Save detailed CSV
            csv_data = []
            for feature in sorted_features:
                stats = avg_correlations[feature]
                csv_data.append({
                    'Feature': feature,
                    'Mean_Correlation': stats['mean'],
                    'Std_Dev': stats['std'],
                    'Min_Correlation': stats['min'],
                    'Max_Correlation': stats['max'],
                    'Files_Count': stats['count']
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_dir, "target_correlations_summary.csv")
            csv_df.to_csv(csv_path, index=False)
            print(f"    correlation_analysis_by_station: Saved correlation summary to target_correlations_summary.csv")
            
        except Exception as e:
            print(f"    correlation_analysis_by_station: Error creating combined target analysis: {e}")
            import traceback
            traceback.print_exc()

    def _save_comprehensive_summary(self, delayed_results, target_results, summary_path):
        """
        Save comprehensive text summary of both correlation analyses.
        
        Parameters:
        -----------
        delayed_results : list
            Results from delayed-only correlation analysis
        target_results : list
            Results from target correlation analysis
        summary_path : str
            Path to save the summary file
        """
        try:
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("COMPREHENSIVE CORRELATION ANALYSIS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
                f.write(f"Files Processed: {len(delayed_results)}\n\n")
                
                # Part 1: Delayed Trains Analysis Summary
                if delayed_results:
                    f.write("-" * 40 + "\n")
                    f.write("PART 1: DELAYED TRAINS CORRELATION ANALYSIS\n")
                    f.write("-" * 40 + "\n\n")
                    
                    total_trains = sum(r['total_original_rows'] for r in delayed_results)
                    total_delayed = sum(r['delayed_rows'] for r in delayed_results)
                    
                    f.write(f"Total Trains Analyzed: {total_trains:,}\n")
                    f.write(f"Total Delayed Trains: {total_delayed:,} ({total_delayed/total_trains*100:.2f}%)\n\n")
                    
                    f.write("File-by-file Results:\n")
                    for result in delayed_results:
                        f.write(f"\n  File: {result['filename']}\n")
                        f.write(f"    Total trains: {result['total_original_rows']:,}\n")
                        f.write(f"    Delayed trains: {result['delayed_rows']:,} ({result['delayed_percentage']:.1f}%)\n")
                        f.write(f"    Features analyzed: {result['feature_count']}\n")
                
                # Part 2: Target Correlation Analysis Summary
                if target_results:
                    f.write("\n" + "-" * 40 + "\n")
                    f.write("PART 2: TARGET VARIABLE CORRELATION ANALYSIS\n")
                    f.write("-" * 40 + "\n\n")
                    
                    f.write("This analysis measures correlations between trainDelayed (binary) and all features.\n")
                    f.write("Methods used:\n")
                    f.write("  • Point-Biserial: For continuous features\n")
                    f.write("  • Phi Coefficient: For binary features\n\n")
                    
                    # Aggregate top correlations across all files
                    all_correlations = {}
                    for result in target_results:
                        for feature, corr_info in result['target_correlations'].items():
                            if not pd.isna(corr_info['correlation']):
                                if feature not in all_correlations:
                                    all_correlations[feature] = []
                                all_correlations[feature].append(corr_info['correlation'])
                    
                    # Calculate averages and find top correlations
                    avg_correlations = {f: np.mean(corrs) for f, corrs in all_correlations.items()}
                    sorted_corrs = sorted(avg_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    f.write("Top 10 Features Most Correlated with Train Delays:\n")
                    for i, (feature, corr) in enumerate(sorted_corrs[:10], 1):
                        f.write(f"  {i:2d}. {feature}: {corr:+.4f}\n")
                    
                    f.write("\nFile-by-file Target Correlation Results:\n")
                    for result in target_results:
                        f.write(f"\n  File: {result['filename']}\n")
                        f.write(f"    Features analyzed: {result['features_analyzed']}\n")
                        
                        # Find strongest correlations in this file
                        file_corrs = [(f, info['correlation']) 
                                    for f, info in result['target_correlations'].items() 
                                    if not pd.isna(info['correlation'])]
                        file_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        if file_corrs:
                            f.write(f"    Strongest correlation: {file_corrs[0][0]} ({file_corrs[0][1]:+.4f})\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF SUMMARY\n")
                f.write("=" * 80 + "\n")
            
            print(f"    correlation_analysis_by_station: Saved comprehensive summary to {os.path.basename(summary_path)}")
            
        except Exception as e:
            print(f"    correlation_analysis_by_station: Error saving summary: {e}")
            import traceback
            traceback.print_exc()

    def _create_delayed_trains_correlation_plot(self, correlation_matrix, file_result, output_dir):
        """
        Create correlation heatmap visualization for delayed trains analysis.
        (This is the original method, kept as-is)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, 
                    annot=False,  # Don't annotate all values
                    fmt='.2f',
                    cmap='coolwarm',
                    vmin=-1, vmax=1,
                    center=0,
                    square=True,
                    mask=mask,
                    cbar_kws={"shrink": 0.8})
            
            # Customize plot
            plt.title(f'Feature Correlations in Delayed Trains\n{file_result["filename"]}\n'
                    f'Delayed Trains: {file_result["delayed_rows"]:,} ({file_result["delayed_percentage"]:.1f}%)',
                    fontsize=14, pad=20)
            
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save plot
            base_filename = os.path.splitext(file_result['filename'])[0]
            plot_path = os.path.join(output_dir, f"{base_filename}_delayed_correlations.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      Saved delayed trains correlation plot: {base_filename}_delayed_correlations.png")
            
            # Also save the correlation matrix as CSV
            csv_path = os.path.join(output_dir, f"{base_filename}_correlation_matrix.csv")
            correlation_matrix.to_csv(csv_path)
            
        except Exception as e:
            print(f"      Error creating correlation plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_combined_delayed_analysis(self, correlation_results, output_dir):
        """
        Create combined analysis for delayed trains correlations across all files.
        (This is the original method, kept as-is)
        """
        try:
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("    correlation_analysis_by_station: Creating combined delayed trains analysis...")
            
            # Find common features across all files
            common_features = None
            for result in correlation_results:
                features = set(result['features_analyzed'])
                if common_features is None:
                    common_features = features
                else:
                    common_features = common_features.intersection(features)
            
            if not common_features or len(common_features) < 2:
                print("    correlation_analysis_by_station: Not enough common features for combined analysis")
                return
            
            common_features = sorted(list(common_features))
            print(f"    correlation_analysis_by_station: Found {len(common_features)} common features across all files")
            
            # Calculate average correlation matrix
            n_features = len(common_features)
            avg_correlation_matrix = np.zeros((n_features, n_features))
            correlation_counts = np.zeros((n_features, n_features))
            
            for result in correlation_results:
                corr_matrix = result['correlation_matrix']
                for i, feat1 in enumerate(common_features):
                    for j, feat2 in enumerate(common_features):
                        if feat1 in corr_matrix.columns and feat2 in corr_matrix.columns:
                            value = corr_matrix.loc[feat1, feat2]
                            if not pd.isna(value):
                                avg_correlation_matrix[i, j] += value
                                correlation_counts[i, j] += 1
            
            # Calculate averages
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_correlation_matrix = np.divide(avg_correlation_matrix, correlation_counts)
                avg_correlation_matrix[correlation_counts == 0] = np.nan
            
            # Convert to DataFrame
            avg_corr_df = pd.DataFrame(avg_correlation_matrix, 
                                    index=common_features, 
                                    columns=common_features)
            
            # Create visualization
            plt.figure(figsize=(14, 12))
            
            # Create heatmap
            mask = np.triu(np.ones_like(avg_corr_df, dtype=bool))
            sns.heatmap(avg_corr_df,
                    annot=False,
                    fmt='.2f',
                    cmap='coolwarm',
                    vmin=-1, vmax=1,
                    center=0,
                    square=True,
                    mask=mask,
                    cbar_kws={"shrink": 0.8})
            
            # Customize plot
            total_delayed = sum(r['delayed_rows'] for r in correlation_results)
            total_files = len(correlation_results)
            
            plt.title(f'Average Feature Correlations in Delayed Trains\n'
                    f'Combined Analysis Across {total_files} Files\n'
                    f'Total Delayed Trains: {total_delayed:,}',
                    fontsize=16, pad=20)
            
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, "combined_delayed_trains_correlation_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    correlation_analysis_by_station: Saved combined analysis to combined_delayed_trains_correlation_analysis.png")
            
            # Save average correlation matrix as CSV
            csv_path = os.path.join(output_dir, "average_correlation_matrix_delayed_trains.csv")
            avg_corr_df.to_csv(csv_path)
            print(f"    correlation_analysis_by_station: Saved average correlation matrix to average_correlation_matrix_delayed_trains.csv")
            
        except Exception as e:
            print(f"    correlation_analysis_by_station: Error creating combined analysis: {e}")
            import traceback
            traceback.print_exc()

    def snow_depth_delay_analysis(self, csv_files=None):
        """
        Advanced analysis of Snow Depth vs Train Delays (Continuous Values)
        
        This pipeline stage analyzes the relationship between snow depth and continuous delay values
        (differenceInMinutes_eachStation_offset) using comprehensive scatter plot visualizations
        and statistical analysis to understand how snow conditions affect train punctuality.
        
        Key Analyses Performed:
        1. Scatter plot: Snow depth vs differenceInMinutes_eachStation_offset
        2. Statistical correlation analysis (Pearson and Spearman)
        3. Delay distribution analysis by snow depth ranges
        4. Seasonal/temporal analysis of snow-delay relationships
        5. Weather condition severity impact assessment
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (currently not used - method discovers files automatically)
            
        Returns:
        --------
        dict
            Results of the snow depth delay analysis including scatter plots and statistics
        """
        try:
            print(f"    snow_depth_delay_analysis: Starting comprehensive snow depth vs delay analysis...")
            
            # Create output directories
            analysis_output_dir = os.path.join(self.project_root, "data/output/snow_depth_delay_analysis")
            scatter_plots_dir = os.path.join(analysis_output_dir, "scatter_plots")
            statistical_analysis_dir = os.path.join(analysis_output_dir, "statistical_analysis")
            seasonal_analysis_dir = os.path.join(analysis_output_dir, "seasonal_analysis")
            
            os.makedirs(analysis_output_dir, exist_ok=True)
            os.makedirs(scatter_plots_dir, exist_ok=True)
            os.makedirs(statistical_analysis_dir, exist_ok=True)
            os.makedirs(seasonal_analysis_dir, exist_ok=True)
            
            # Find all merged training data files
            merged_data_pattern = os.path.join(self.project_root, "data/output/3-merged_training_ready/merged_data_*_train.csv")
            merged_data_files = glob.glob(merged_data_pattern)
            
            # Fallback to scaled data if available
            if not merged_data_files:
                scaled_pattern = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*_train_scaled.csv")
                merged_data_files = glob.glob(scaled_pattern)
                print(f"    snow_depth_delay_analysis: Using scaled data files")
            
            if not merged_data_files:
                error_msg = "No merged data files found for snow depth delay analysis"
                print(f"    snow_depth_delay_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    snow_depth_delay_analysis: Found {len(merged_data_files)} data files to process")
            
            analysis_results = []
            total_files_processed = 0
            combined_data = []
            
            # Process each file
            for file_path in merged_data_files:
                filename = os.path.basename(file_path)
                print(f"      Processing {filename}...")
                
                try:
                    # Load data
                    df = pd.read_csv(file_path)
                    
                    # Check required columns
                    required_columns = ['Snow depth', 'differenceInMinutes_eachStation_offset']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        print(f"        Warning: Missing columns {missing_columns} in {filename}, skipping...")
                        continue
                    
                    # Clean data - remove NaN values
                    clean_df = df[['Snow depth', 'differenceInMinutes_eachStation_offset']].dropna()
                    clean_df = clean_df[clean_df['differenceInMinutes_eachStation_offset'] >= 5]
                    
                    if len(clean_df) == 0:
                        print(f"        Warning: No valid data after cleaning in {filename}, skipping...")
                        continue
                    
                    # Perform file-specific analysis
                    file_analysis = self._analyze_snow_depth_delays_single_file(
                        clean_df, filename, scatter_plots_dir, statistical_analysis_dir
                    )
                    
                    if file_analysis:
                        analysis_results.append(file_analysis)
                        # Add file identifier to data for combined analysis
                        clean_df['source_file'] = filename
                        filtered_df = clean_df[clean_df['differenceInMinutes_eachStation_offset'] >= 5]
                        combined_data.append(clean_df)
                    
                    total_files_processed += 1
                    print(f"      ✓ Completed analysis for {filename}")
                    
                except Exception as e:
                    print(f"      Error processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if total_files_processed == 0:
                error_msg = "No files were successfully processed"
                print(f"    snow_depth_delay_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            # Combined analysis across all files
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                combined_analysis = self._create_combined_snow_depth_analysis(
                    combined_df, analysis_output_dir, seasonal_analysis_dir
                )
            
            # Create comprehensive summary
            summary_path = os.path.join(analysis_output_dir, "snow_depth_delay_analysis_summary.txt")
            self._save_snow_depth_analysis_summary(analysis_results, combined_analysis if combined_data else None, summary_path)
            
            print(f"    snow_depth_delay_analysis: Analysis completed for {total_files_processed} files")
            print(f"    snow_depth_delay_analysis: Results saved to {analysis_output_dir}")
            
            # Return comprehensive results
            result = {
                "success": True,
                "processed_files": total_files_processed,
                "output_path": analysis_output_dir,
                "summary_path": summary_path,
                "file_analyses": analysis_results,
                "combined_analysis": combined_analysis if combined_data else None,
                "total_records_analyzed": sum(r['valid_records'] for r in analysis_results) if analysis_results else 0,
                "analysis_types": ["Scatter Plots", "Statistical Correlations", "Delay Distribution Analysis", "Seasonal Analysis"],
                "message": f"Successfully completed snow depth vs delay analysis for {total_files_processed} files"
            }
            
            print(f"    snow_depth_delay_analysis: Completed successfully - {total_files_processed} files analyzed")
            
            return result
            
        except Exception as e:
            error_msg = f"snow_depth_delay_analysis failed: {str(e)}"
            print(f"    snow_depth_delay_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "processed_files": 0
            }

    def _analyze_snow_depth_delays_single_file(self, df, filename, scatter_dir, stats_dir):
        """
        Analyze snow depth vs delays for a single file.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Clean DataFrame with Snow depth and differenceInMinutes_eachStation_offset columns
        filename : str
            Name of the source file
        scatter_dir : str
            Directory to save scatter plots
        stats_dir : str
            Directory to save statistical analysis
            
        Returns:
        --------
        dict
            Analysis results for this file
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import pearsonr, spearmanr
            
            snow_depth = df['Snow depth']
            delay_minutes = df['differenceInMinutes_eachStation_offset']
            
            # Basic statistics
            basic_stats = {
                'valid_records': len(df),
                'snow_depth_stats': {
                    'mean': float(snow_depth.mean()),
                    'std': float(snow_depth.std()),
                    'min': float(snow_depth.min()),
                    'max': float(snow_depth.max()),
                    'median': float(snow_depth.median())
                },
                'delay_stats': {
                    'mean': float(delay_minutes.mean()),
                    'std': float(delay_minutes.std()),
                    'min': float(delay_minutes.min()),
                    'max': float(delay_minutes.max()),
                    'median': float(delay_minutes.median())
                }
            }
            
            # Correlation analysis
            try:
                pearson_corr, pearson_p = pearsonr(snow_depth, delay_minutes)
                spearman_corr, spearman_p = spearmanr(snow_depth, delay_minutes)
            except:
                pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
            
            correlation_stats = {
                'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else None,
                'pearson_p_value': float(pearson_p) if not np.isnan(pearson_p) else None,
                'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else None,
                'spearman_p_value': float(spearman_p) if not np.isnan(spearman_p) else None,
            }
            
            # Create comprehensive scatter plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Main scatter plot
            ax1.scatter(snow_depth, delay_minutes, alpha=0.6, s=20, c='steelblue', edgecolors='none')
            ax1.set_xlabel('Snow Depth (cm)')
            ax1.set_ylabel('Train Delay (minutes)')
            ax1.set_title(f'Snow Depth vs Train Delays\n{filename}')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            try:
                z = np.polyfit(snow_depth, delay_minutes, 1)
                p = np.poly1d(z)
                ax1.plot(snow_depth, p(snow_depth), "r--", alpha=0.8, linewidth=2)
            except:
                pass
            
            # Add correlation info to plot
            corr_text = f'Pearson r: {pearson_corr:.3f} (p={pearson_p:.3f})\n'
            corr_text += f'Spearman ρ: {spearman_corr:.3f} (p={spearman_p:.3f})'
            ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Hexbin plot for density
            hb = ax2.hexbin(snow_depth, delay_minutes, gridsize=20, cmap='Blues', alpha=0.7)
            ax2.set_xlabel('Snow Depth (cm)')
            ax2.set_ylabel('Train Delay (minutes)')
            ax2.set_title('Density Plot: Snow Depth vs Delays')
            plt.colorbar(hb, ax=ax2, label='Count')
            
            # Snow depth distribution
            ax3.hist(snow_depth, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax3.set_xlabel('Snow Depth (cm)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Snow Depth Values')
            ax3.axvline(snow_depth.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {snow_depth.mean():.1f}')
            ax3.legend()
            
            # Delay distribution
            ax4.hist(delay_minutes, bins=30, alpha=0.7, color='salmon', edgecolor='black')
            ax4.set_xlabel('Train Delay (minutes)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Train Delays')
            ax4.axvline(delay_minutes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {delay_minutes.mean():.1f}')
            ax4.legend()
            
            plt.tight_layout()
            
            # Save plot
            safe_filename = filename.replace('.csv', '').replace(' ', '_')
            plot_filename = f"snow_depth_delay_analysis_{safe_filename}.png"
            plot_path = os.path.join(scatter_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create delay ranges analysis
            delay_ranges_analysis = self._analyze_delay_by_snow_ranges(df)
            
            # Save detailed statistics
            stats_filename = f"snow_depth_stats_{safe_filename}.txt"
            stats_path = os.path.join(stats_dir, stats_filename)
            self._save_single_file_stats(basic_stats, correlation_stats, delay_ranges_analysis, stats_path, filename)
            
            print(f"        ✓ Created scatter plot: {plot_filename}")
            print(f"        ✓ Saved statistics: {stats_filename}")
            
            return {
                'filename': filename,
                'plot_path': plot_path,
                'stats_path': stats_path,
                'basic_stats': basic_stats,
                'correlations': correlation_stats,
                'delay_ranges_analysis': delay_ranges_analysis,
                'valid_records': len(df)
            }
            
        except Exception as e:
            print(f"        Error in single file analysis for {filename}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_delay_by_snow_ranges(self, df):
        """
        Analyze delay patterns across different snow depth ranges.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with Snow depth and delay data
            
        Returns:
        --------
        dict
            Analysis of delays by snow depth ranges
        """
        try:
            snow_depth = df['Snow depth']
            delay_minutes = df['differenceInMinutes_eachStation_offset']
            
            # Define snow depth ranges
            ranges = [
                ('No Snow', 0, 0),
                ('Light Snow', 0.1, 5),
                ('Moderate Snow', 5.1, 15),
                ('Heavy Snow', 15.1, 30),
                ('Extreme Snow', 30.1, float('inf'))
            ]
            
            range_analysis = {}
            
            for range_name, min_val, max_val in ranges:
                if max_val == float('inf'):
                    mask = snow_depth > min_val
                else:
                    mask = (snow_depth >= min_val) & (snow_depth <= max_val)
                
                range_data = delay_minutes[mask]
                
                if len(range_data) > 0:
                    range_analysis[range_name] = {
                        'count': len(range_data),
                        'mean_delay': float(range_data.mean()),
                        'median_delay': float(range_data.median()),
                        'std_delay': float(range_data.std()),
                        'max_delay': float(range_data.max()),
                        'min_delay': float(range_data.min()),
                        'percentage_of_total': float(len(range_data) / len(df) * 100)
                    }
                else:
                    range_analysis[range_name] = {
                        'count': 0,
                        'mean_delay': 0,
                        'median_delay': 0,
                        'std_delay': 0,
                        'max_delay': 0,
                        'min_delay': 0,
                        'percentage_of_total': 0
                    }
            
            return range_analysis
            
        except Exception as e:
            print(f"        Error in range analysis: {e}")
            return {}

    def _create_combined_snow_depth_analysis(self, combined_df, output_dir, seasonal_dir):
        """
        Create combined analysis across all files.
        
        Parameters:
        -----------
        combined_df : pandas.DataFrame
            Combined data from all files
        output_dir : str
            Main output directory
        seasonal_dir : str
            Directory for seasonal analysis
            
        Returns:
        --------
        dict
            Combined analysis results
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import pearsonr, spearmanr

            combined_df = combined_df[combined_df['differenceInMinutes_eachStation_offset'] >= 5]
            
            snow_depth = combined_df['Snow depth']
            delay_minutes = combined_df['differenceInMinutes_eachStation_offset']
            
            # Overall correlation
            pearson_corr, pearson_p = pearsonr(snow_depth, delay_minutes)
            spearman_corr, spearman_p = spearmanr(snow_depth, delay_minutes)
            
            # Create comprehensive combined plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
            
            # Main scatter with sample
            if len(combined_df) > 10000:
                sample_df = combined_df.sample(n=10000, random_state=42)
                sample_snow = sample_df['Snow depth']
                sample_delay = sample_df['differenceInMinutes_eachStation_offset']
            else:
                sample_snow = snow_depth
                sample_delay = delay_minutes
            
            ax1.scatter(sample_snow, sample_delay, alpha=0.5, s=15, c='darkblue', edgecolors='none')
            ax1.set_xlabel('Snow Depth (cm)')
            ax1.set_ylabel('Train Delay (minutes)')
            ax1.set_title(f'Combined Analysis: Snow Depth vs Train Delays\n(Total Records: {len(combined_df):,})')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            
            # Trend line
            try:
                z = np.polyfit(snow_depth, delay_minutes, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(snow_depth.min(), snow_depth.max(), 100)
                ax1.plot(x_trend, p(x_trend), "r-", linewidth=2, alpha=0.8)
            except:
                pass
            
            # Correlation info
            corr_text = f'Pearson r: {pearson_corr:.4f} (p={pearson_p:.4f})\n'
            corr_text += f'Spearman ρ: {spearman_corr:.4f} (p={spearman_p:.4f})\n'
            corr_text += f'Total Records: {len(combined_df):,}'
            ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
            # Box plot by snow depth ranges
            combined_df['snow_range'] = pd.cut(combined_df['Snow depth'], 
                                            bins=[-0.1, 0, 5, 15, 30, 100], 
                                            labels=['No Snow', 'Light\n(0-5cm)', 'Moderate\n(5-15cm)', 'Heavy\n(15-30cm)', 'Extreme\n(30cm+)'])
            
            box_data = [group['differenceInMinutes_eachStation_offset'].values 
            for name, group in combined_df.groupby('snow_range', observed=True) if len(group) > 0]
            box_labels = [name for name, group in combined_df.groupby('snow_range', observed=True) if len(group) > 0]
            
            ax2.boxplot(box_data, labels=box_labels)
            ax2.set_ylabel('Train Delay (minutes)')
            ax2.set_title('Delay Distribution by Snow Depth Ranges')
            ax2.tick_params(axis='x', rotation=45)
            
            # Delay frequency by snow presence
            no_snow_delays = delay_minutes[snow_depth == 0]
            with_snow_delays = delay_minutes[snow_depth > 0]
            
            ax3.hist([no_snow_delays, with_snow_delays], bins=50, alpha=0.7, 
                    label=[f'No Snow (n={len(no_snow_delays):,})', f'With Snow (n={len(with_snow_delays):,})'],
                    color=['lightblue', 'orange'])
            ax3.set_xlabel('Train Delay (minutes)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Delay Distribution: No Snow vs With Snow')
            ax3.legend()
            
            # Snow depth distribution
            ax4.hist(snow_depth, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax4.set_xlabel('Snow Depth (cm)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Overall Snow Depth Distribution')
            ax4.axvline(snow_depth.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {snow_depth.mean():.1f} cm')
            ax4.legend()
            
            plt.tight_layout()
            
            # Save combined plot
            combined_plot_path = os.path.join(output_dir, "combined_snow_depth_delay_analysis.png")
            plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Statistical summary
            combined_analysis = {
                'total_records': len(combined_df),
                'overall_correlations': {
                    'pearson_correlation': float(pearson_corr),
                    'pearson_p_value': float(pearson_p),
                    'spearman_correlation': float(spearman_corr),
                    'spearman_p_value': float(spearman_p),
                },
                'snow_depth_summary': {
                    'mean': float(snow_depth.mean()),
                    'std': float(snow_depth.std()),
                    'median': float(snow_depth.median()),
                    'min': float(snow_depth.min()),
                    'max': float(snow_depth.max()),
                    'records_with_snow': int((snow_depth > 0).sum()),
                    'percentage_with_snow': float((snow_depth > 0).mean() * 100)
                },
                'delay_summary': {
                    'overall_mean_delay': float(delay_minutes.mean()),
                    'no_snow_mean_delay': float(no_snow_delays.mean()) if len(no_snow_delays) > 0 else 0,
                    'with_snow_mean_delay': float(with_snow_delays.mean()) if len(with_snow_delays) > 0 else 0,
                    'delay_difference': float(with_snow_delays.mean() - no_snow_delays.mean()) if len(no_snow_delays) > 0 and len(with_snow_delays) > 0 else 0
                },
                'plot_path': combined_plot_path
            }
            
            print(f"        ✓ Created combined analysis plot")
            
            return combined_analysis
            
        except Exception as e:
            print(f"        Error in combined analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_single_file_stats(self, basic_stats, correlation_stats, ranges_analysis, stats_path, filename):
        """Save detailed statistics for a single file."""
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"Snow Depth vs Train Delay Analysis - {filename}\n")
                f.write("=" * 60 + "\n\n")
                
                # Basic statistics
                f.write("BASIC STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Valid Records: {basic_stats['valid_records']:,}\n\n")
                
                f.write("Snow Depth Statistics:\n")
                snow_stats = basic_stats['snow_depth_stats']
                f.write(f"  Mean: {snow_stats['mean']:.2f} cm\n")
                f.write(f"  Std Dev: {snow_stats['std']:.2f} cm\n")
                f.write(f"  Median: {snow_stats['median']:.2f} cm\n")
                f.write(f"  Range: {snow_stats['min']:.2f} - {snow_stats['max']:.2f} cm\n\n")
                
                f.write("Train Delay Statistics:\n")
                delay_stats = basic_stats['delay_stats']
                f.write(f"  Mean: {delay_stats['mean']:.2f} minutes\n")
                f.write(f"  Std Dev: {delay_stats['std']:.2f} minutes\n")
                f.write(f"  Median: {delay_stats['median']:.2f} minutes\n")
                f.write(f"  Range: {delay_stats['min']:.2f} - {delay_stats['max']:.2f} minutes\n\n")
                
                # Correlation statistics
                f.write("CORRELATION ANALYSIS\n")
                f.write("-" * 20 + "\n")
                if correlation_stats['pearson_correlation'] is not None:
                    f.write(f"Pearson Correlation: r = {correlation_stats['pearson_correlation']:.4f} (p = {correlation_stats['pearson_p_value']:.4f})\n")
                if correlation_stats['spearman_correlation'] is not None:
                    f.write(f"Spearman Correlation: ρ = {correlation_stats['spearman_correlation']:.4f} (p = {correlation_stats['spearman_p_value']:.4f})\n")
                f.write("\n")
                
                # Interpretation
                if correlation_stats['pearson_correlation'] is not None:
                    corr_val = abs(correlation_stats['pearson_correlation'])
                    if corr_val < 0.1:
                        interpretation = "Very weak correlation"
                    elif corr_val < 0.3:
                        interpretation = "Weak correlation"
                    elif corr_val < 0.5:
                        interpretation = "Moderate correlation"
                    elif corr_val < 0.7:
                        interpretation = "Strong correlation"
                    else:
                        interpretation = "Very strong correlation"
                        
                    direction = "positive" if correlation_stats['pearson_correlation'] > 0 else "negative"
                    f.write(f"Interpretation: {interpretation} ({direction})\n\n")
                
                # Range analysis
                f.write("DELAY ANALYSIS BY SNOW DEPTH RANGES\n")
                f.write("-" * 40 + "\n")
                for range_name, stats in ranges_analysis.items():
                    f.write(f"{range_name}:\n")
                    f.write(f"  Count: {stats['count']:,} ({stats['percentage_of_total']:.1f}%)\n")
                    if stats['count'] > 0:
                        f.write(f"  Mean Delay: {stats['mean_delay']:.2f} minutes\n")
                        f.write(f"  Median Delay: {stats['median_delay']:.2f} minutes\n")
                        f.write(f"  Std Dev: {stats['std_delay']:.2f} minutes\n")
                        f.write(f"  Range: {stats['min_delay']:.2f} - {stats['max_delay']:.2f} minutes\n")
                    f.write("\n")
                    
        except Exception as e:
            print(f"        Error saving stats: {e}")

    def _save_snow_depth_analysis_summary(self, file_analyses, combined_analysis, summary_path):
        """Save comprehensive summary of the snow depth delay analysis."""
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE SNOW DEPTH vs TRAIN DELAY ANALYSIS SUMMARY\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Overview
                f.write("ANALYSIS OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Files processed: {len(file_analyses)}\n")
                f.write(f"Total records analyzed: {sum(a['valid_records'] for a in file_analyses):,}\n\n")
                
                # File-by-file summary
                f.write("FILE-BY-FILE RESULTS\n")
                f.write("-" * 25 + "\n")
                for analysis in file_analyses:
                    f.write(f"File: {analysis['filename']}\n")
                    f.write(f"  Records: {analysis['valid_records']:,}\n")
                    if analysis['correlations']['pearson_correlation'] is not None:
                        f.write(f"  Pearson r: {analysis['correlations']['pearson_correlation']:.4f}\n")
                        f.write(f"  Spearman ρ: {analysis['correlations']['spearman_correlation']:.4f}\n")
                    f.write("\n")
                
                # Combined results
                if combined_analysis:
                    f.write("COMBINED ANALYSIS RESULTS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Total Combined Records: {combined_analysis['total_records']:,}\n")
                    f.write(f"Records with Snow: {combined_analysis['snow_depth_summary']['records_with_snow']:,} ")
                    f.write(f"({combined_analysis['snow_depth_summary']['percentage_with_snow']:.1f}%)\n\n")
                    
                    f.write("Overall Correlations:\n")
                    f.write(f"  Pearson r: {combined_analysis['overall_correlations']['pearson_correlation']:.4f} ")
                    f.write(f"(p = {combined_analysis['overall_correlations']['pearson_p_value']:.6f})\n")
                    f.write(f"  Spearman ρ: {combined_analysis['overall_correlations']['spearman_correlation']:.4f} ")
                    f.write(f"(p = {combined_analysis['overall_correlations']['spearman_p_value']:.6f})\n\n")
                    
                    f.write("Key Findings:\n")
                    delay_diff = combined_analysis['delay_summary']['delay_difference']
                    if delay_diff > 0:
                        f.write(f"  • Trains experience {delay_diff:.2f} minutes MORE delay on average when snow is present\n")
                    elif delay_diff < 0:
                        f.write(f"  • Trains experience {abs(delay_diff):.2f} minutes LESS delay on average when snow is present\n")
                    else:
                        f.write(f"  • No significant difference in delays between snowy and non-snowy conditions\n")
                    
                    f.write(f"  • Mean delay without snow: {combined_analysis['delay_summary']['no_snow_mean_delay']:.2f} minutes\n")
                    f.write(f"  • Mean delay with snow: {combined_analysis['delay_summary']['with_snow_mean_delay']:.2f} minutes\n")
                    f.write(f"  • Average snow depth: {combined_analysis['snow_depth_summary']['mean']:.2f} cm\n")
                    f.write(f"  • Maximum snow depth: {combined_analysis['snow_depth_summary']['max']:.2f} cm\n\n")
                    
                    # **NEW SECTION: Heavy Snow Delay Impact Analysis**
                    f.write("HEAVY SNOW DELAY IMPACT ANALYSIS (Snow > 15 cm)\n")
                    f.write("-" * 50 + "\n")
                    
                    # Calculate percentage of delayed trains when snow > 15 cm
                    # We need to recreate the combined dataset or get it from the analysis
                    try:
                        # If we need to recreate the data from file analyses, we can do it here
                        # For now, let's assume we have access to the combined data through a different approach
                        
                        # Try to get the data from the combined analysis or calculate from file analyses
                        heavy_snow_stats = self._calculate_heavy_snow_delay_percentage(file_analyses)
                        
                        if heavy_snow_stats['total_trains'] > 0:
                            f.write(f"Analysis Criteria:\n")
                            f.write(f"  • Snow Depth Threshold: > 15.0 cm\n")
                            f.write(f"  • Delay Threshold: ≥ 5 minutes\n\n")
                            
                            f.write(f"Results:\n")
                            f.write(f"  • Total trains in heavy snow conditions: {heavy_snow_stats['total_trains']:,}\n")
                            f.write(f"  • Delayed trains (≥5 min): {heavy_snow_stats['delayed_trains']:,}\n")
                            f.write(f"  • Percentage of trains delayed: {heavy_snow_stats['delay_percentage']:.1f}%\n")
                            f.write(f"  • Percentage of trains on-time: {heavy_snow_stats['on_time_percentage']:.1f}%\n")
                            f.write(f"  • Average delay in heavy snow: {heavy_snow_stats['avg_delay']:.2f} minutes\n")
                            f.write(f"  • Maximum delay recorded: {heavy_snow_stats['max_delay']:.2f} minutes\n\n")
                            
                            # Impact assessment
                            if heavy_snow_stats['delay_percentage'] >= 80:
                                impact_level = "SEVERE"
                                impact_desc = "Heavy snow significantly disrupts train operations"
                            elif heavy_snow_stats['delay_percentage'] >= 60:
                                impact_level = "HIGH" 
                                impact_desc = "Heavy snow causes substantial delays"
                            elif heavy_snow_stats['delay_percentage'] >= 40:
                                impact_level = "MODERATE"
                                impact_desc = "Heavy snow has noticeable impact on punctuality"
                            elif heavy_snow_stats['delay_percentage'] >= 20:
                                impact_level = "LOW"
                                impact_desc = "Heavy snow has minimal impact on delays"
                            else:
                                impact_level = "MINIMAL"
                                impact_desc = "Heavy snow shows little correlation with delays"
                                
                            f.write(f"Impact Assessment: {impact_level}\n")
                            f.write(f"  {impact_desc}\n\n")
                            
                        else:
                            f.write(f"No records found with snow depth > 15 cm in the analyzed data.\n\n")
                            
                    except Exception as e:
                        f.write(f"Unable to calculate heavy snow delay statistics: {str(e)}\n\n")
                
                # Conclusions
                f.write("ANALYSIS CONCLUSIONS\n")
                f.write("-" * 22 + "\n")
                if combined_analysis:
                    corr_val = abs(combined_analysis['overall_correlations']['pearson_correlation'])
                    p_val = combined_analysis['overall_correlations']['pearson_p_value']
                    
                    if p_val < 0.001:
                        significance = "highly significant"
                    elif p_val < 0.01:
                        significance = "very significant"
                    elif p_val < 0.05:
                        significance = "significant"
                    else:
                        significance = "not statistically significant"
                    
                    f.write(f"The correlation between snow depth and train delays is {significance} (p = {p_val:.6f}).\n")
                    
                    if corr_val < 0.1:
                        f.write("The relationship is very weak, suggesting snow depth has minimal impact on delays.\n")
                    elif corr_val < 0.3:
                        f.write("The relationship is weak, suggesting snow depth has some but limited impact on delays.\n")
                    elif corr_val < 0.5:
                        f.write("The relationship is moderate, suggesting snow depth has a noticeable impact on delays.\n")
                    else:
                        f.write("The relationship is strong, suggesting snow depth has a significant impact on delays.\n")
                    
                    direction = combined_analysis['overall_correlations']['pearson_correlation']
                    if direction > 0:
                        f.write("Higher snow depth is associated with longer train delays.\n")
                    else:
                        f.write("Higher snow depth is associated with shorter train delays (unexpected result).\n")
                
                f.write(f"\nDetailed plots and statistics are available in the output directories.\n")
                        
        except Exception as e:
            print(f"        Error saving summary: {e}")

    def _calculate_heavy_snow_delay_percentage(self, file_analyses):
        """
        Calculate delay percentage statistics for heavy snow conditions (>15 cm) from file analyses.
        
        Parameters:
        -----------
        file_analyses : list
            List of file analysis results
            
        Returns:
        --------
        dict
            Statistics for heavy snow delay analysis
        """
        try:
            import pandas as pd
            import glob
            import os
            
            # We need to reload the data to calculate the heavy snow statistics
            # Find all merged training data files (same logic as in main analysis)
            merged_data_pattern = os.path.join(self.project_root, "data/output/3-merged_training_ready/merged_data_*_train.csv")
            merged_data_files = glob.glob(merged_data_pattern)
            
            # Fallback to scaled data if available
            if not merged_data_files:
                from config.const import MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                scaled_pattern = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*_train_scaled.csv")
                merged_data_files = glob.glob(scaled_pattern)
            
            if not merged_data_files:
                return {
                    'total_trains': 0,
                    'delayed_trains': 0,
                    'delay_percentage': 0.0,
                    'on_time_percentage': 0.0,
                    'avg_delay': 0.0,
                    'max_delay': 0.0
                }
            
            # Combine data from all files for heavy snow analysis
            heavy_snow_data = []
            

            for file_path in merged_data_files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Clean and prepare data (same logic as in main analysis)
                    required_columns = ['Snow depth', 'differenceInMinutes_eachStation_offset']
                    if not all(col in df.columns for col in required_columns):
                        continue
                        
                    # Remove rows with missing values in required columns
                    clean_df = df[required_columns].dropna()
                    
                    if len(clean_df) == 0:
                        continue
                    
                    # Filter for heavy snow conditions (> 15 cm)
                    heavy_snow_mask = clean_df['Snow depth'] > 15
                    heavy_snow_subset = clean_df[heavy_snow_mask]
                    
                    if len(heavy_snow_subset) > 0:
                        heavy_snow_data.append(heavy_snow_subset)
                        
                except Exception as e:
                    print(f"        Warning: Could not process {file_path} for heavy snow analysis: {e}")
                    continue
            
            if not heavy_snow_data:
                return {
                    'total_trains': 0,
                    'delayed_trains': 0,
                    'delay_percentage': 0.0,
                    'on_time_percentage': 0.0,
                    'avg_delay': 0.0,
                    'max_delay': 0.0
                }
            
            # Combine all heavy snow data
            combined_heavy_snow = pd.concat(heavy_snow_data, ignore_index=True)
            
            # Calculate delay statistics
            delays = combined_heavy_snow['differenceInMinutes_eachStation_offset']
            total_trains = len(delays)
            delayed_trains = (delays >= 5).sum()  # Trains delayed ≥5 minutes
            delay_percentage = float(delayed_trains / total_trains * 100) if total_trains > 0 else 0.0
            on_time_percentage = 100.0 - delay_percentage
            avg_delay = float(delays.mean()) if total_trains > 0 else 0.0
            max_delay = float(delays.max()) if total_trains > 0 else 0.0
            
            return {
                'total_trains': int(total_trains),
                'delayed_trains': int(delayed_trains),
                'delay_percentage': delay_percentage,
                'on_time_percentage': on_time_percentage,
                'avg_delay': avg_delay,
                'max_delay': max_delay
            }
            
        except Exception as e:
            print(f"        Error calculating heavy snow delay percentage: {e}")
            return {
                'total_trains': 0,
                'delayed_trains': 0,
                'delay_percentage': 0.0,
                'on_time_percentage': 0.0,
                'avg_delay': 0.0,
                'max_delay': 0.0
            }

    def numeric_correlation_analysis(self, csv_files=None):
        """
        Comprehensive Numeric Correlation Analysis against differenceInMinutes_eachStation_offset
        
        This pipeline stage analyzes the correlations between differenceInMinutes_eachStation_offset 
        (target variable) and all other numeric features in the dataset, focusing only on delayed trains
        (differenceInMinutes_eachStation_offset > 5). It provides comprehensive statistical analysis 
        and visualizations including correlation matrices, scatter plots, and feature importance ranking.
        
        Key Analyses Performed:
        1. Correlation matrix heatmap for all numeric features vs target (delayed trains only)
        2. Individual scatter plots for top correlated features
        3. Statistical correlation analysis (Pearson and Spearman)
        4. Feature correlation ranking and importance
        5. Distribution analysis of correlations
        6. Combined analysis across all files
        
        Parameters:
        -----------
        csv_files : list, optional
            List of CSV file paths (currently not used - method discovers files automatically)
            
        Returns:
        --------
        dict
            Results of the numeric correlation analysis including plots and statistics
        """
        try:
            print(f"    numeric_correlation_analysis: Starting comprehensive numeric correlation analysis...")
            
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
            
            # Find all merged scaled training data files
            merged_data_pattern = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, "merged_data_*_train_scaled.csv")
            merged_data_files = glob.glob(merged_data_pattern)
            
            # Fallback to non-scaled data if scaled not available
            if not merged_data_files:
                merged_data_pattern = os.path.join(self.project_root, "data/output/3-merged_training_ready", "merged_data_*_train.csv")
                merged_data_files = glob.glob(merged_data_pattern)
                print(f"    numeric_correlation_analysis: Using non-scaled data files")
            
            if not merged_data_files:
                error_msg = "No merged data files found for numeric correlation analysis"
                print(f"    numeric_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }
            
            print(f"    numeric_correlation_analysis: Found {len(merged_data_files)} data files to process")
            
            analysis_results = []
            total_files_processed = 0
            combined_data = []
            all_correlation_results = {}
            target_column = 'differenceInMinutes_eachStation_offset'
            
            # Import required libraries
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import pearsonr, spearmanr
            import numpy as np
            import json
            
            # Process each file
            for file_path in merged_data_files:
                filename = os.path.basename(file_path)
                print(f"      Processing {filename}...")
                
                try:
                    # Load data
                    df = pd.read_csv(file_path)
                    
                    # Check if target column exists
                    if target_column not in df.columns:
                        print(f"        Warning: Target column '{target_column}' not found in {filename}, skipping...")
                        continue
                    
                    # Extract all numeric columns (excluding target)
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Remove target column and any index columns from features
                    feature_columns = [col for col in numeric_columns 
                                    if col != target_column 
                                    and not col.startswith('Unnamed') 
                                    and col != 'index']
                    
                    if not feature_columns:
                        print(f"        Warning: No numeric features found in {filename}, skipping...")
                        continue
                    
                    print(f"        Found {len(feature_columns)} numeric features to analyze")
                    print(f"        Features: {feature_columns}")
                    
                    # Clean data - remove NaN values
                    analysis_df = df[[target_column] + feature_columns].dropna()
                    
                    # Filter for delayed trains only (differenceInMinutes_eachStation_offset > 5)
                    analysis_df = analysis_df[analysis_df[target_column] > 5]
                    
                    if len(analysis_df) == 0:
                        print(f"        Warning: No valid data after cleaning and filtering for delays > 5 min in {filename}, skipping...")
                        continue
                    
                    print(f"        Valid records for analysis (delayed trains > 5 min): {len(analysis_df):,}")
                    
                    # Calculate correlations for this file
                    file_correlations = {}
                    pearson_correlations = {}
                    spearman_correlations = {}
                    
                    target_data = analysis_df[target_column]
                    
                    # Calculate correlations with each feature
                    for feature in feature_columns:
                        feature_data = analysis_df[feature]
                        
                        # Skip if feature has no variance
                        if feature_data.std() == 0:
                            continue
                        
                        try:
                            # Pearson correlation
                            pearson_corr, pearson_p = pearsonr(target_data, feature_data)
                            pearson_correlations[feature] = {
                                'correlation': pearson_corr,
                                'p_value': pearson_p
                            }
                            
                            # Spearman correlation
                            spearman_corr, spearman_p = spearmanr(target_data, feature_data)
                            spearman_correlations[feature] = {
                                'correlation': spearman_corr,
                                'p_value': spearman_p
                            }
                            
                            file_correlations[feature] = {
                                'pearson': pearson_corr,
                                'spearman': spearman_corr,
                                'pearson_p_value': pearson_p,
                                'spearman_p_value': spearman_p
                            }
                            
                        except Exception as e:
                            print(f"          Warning: Could not calculate correlation for {feature}: {e}")
                            continue
                    
                    # Sort features by absolute Pearson correlation
                    sorted_features = sorted(file_correlations.items(), 
                                        key=lambda x: abs(x[1]['pearson']), 
                                        reverse=True)
                    
                    print(f"        Top 5 correlated features (Pearson):")
                    for i, (feature, corr_data) in enumerate(sorted_features[:5]):
                        print(f"          {i+1}. {feature}: {corr_data['pearson']:.4f}")
                    
                    # Create correlation matrix heatmap for this file
                    correlation_matrix_data = []
                    feature_names = []
                    
                    for feature in feature_columns:
                        if feature in file_correlations:
                            correlation_matrix_data.append(file_correlations[feature]['pearson'])
                            feature_names.append(feature)
                    
                    if correlation_matrix_data:
                        # Create correlation heatmap
                        plt.figure(figsize=(12, 8))
                        
                        # Create data for heatmap (single row showing correlations with target)
                        heatmap_data = np.array(correlation_matrix_data).reshape(1, -1)
                        
                        # Create heatmap
                        sns.heatmap(heatmap_data, 
                                annot=True, 
                                fmt='.3f',
                                cmap='RdBu_r', 
                                center=0,
                                xticklabels=feature_names,
                                yticklabels=[f'{target_column}'],
                                cbar_kws={'label': 'Pearson Correlation'})
                        
                        plt.title(f'Numeric Features Correlation with {target_column} (Delayed Trains > 5min)\n{filename}', 
                                fontsize=14, fontweight='bold')
                        plt.xlabel('Features', fontsize=12)
                        plt.ylabel('Target Variable', fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        # Save heatmap
                        heatmap_path = os.path.join(correlation_plots_dir, f"correlation_heatmap_{filename.replace('.csv', '.png')}")
                        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"        Saved correlation heatmap to {os.path.basename(heatmap_path)}")
                    
                    # Create scatter plots for top 6 correlated features
                    top_features = [item[0] for item in sorted_features[:6]]
                    
                    if top_features:
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                        fig.suptitle(f'Top Correlated Features vs {target_column} (Delayed Trains > 5min)\n{filename}', 
                                fontsize=16, fontweight='bold')
                        
                        axes = axes.ravel()
                        
                        for i, feature in enumerate(top_features):
                            if i >= 6:
                                break
                                
                            ax = axes[i]
                            
                            # Create scatter plot with sample if data is large
                            plot_df = analysis_df
                            if len(analysis_df) > 5000:
                                plot_df = analysis_df.sample(n=5000, random_state=42)
                            
                            ax.scatter(plot_df[feature], plot_df[target_column], 
                                    alpha=0.6, s=20, c='darkblue', edgecolors='none')
                            
                            # Add trend line
                            z = np.polyfit(plot_df[feature], plot_df[target_column], 1)
                            p = np.poly1d(z)
                            ax.plot(plot_df[feature], p(plot_df[feature]), "r--", alpha=0.8, linewidth=2)
                            
                            correlation_value = file_correlations[feature]['pearson']
                            ax.set_xlabel(feature, fontsize=10)
                            ax.set_ylabel(target_column, fontsize=10)
                            ax.set_title(f'{feature}\nPearson: {correlation_value:.4f}', 
                                    fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                        
                        # Hide empty subplots
                        for i in range(len(top_features), 6):
                            axes[i].set_visible(False)
                        
                        plt.tight_layout()
                        
                        # Save scatter plots
                        scatter_path = os.path.join(scatter_plots_dir, f"top_correlations_scatter_{filename.replace('.csv', '.png')}")
                        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"        Saved scatter plots to {os.path.basename(scatter_path)}")
                    
                    # Create feature ranking visualization
                    if sorted_features:
                        plt.figure(figsize=(12, 8))
                        
                        features_list = [item[0] for item in sorted_features[:15]]  # Top 15
                        correlations_list = [item[1]['pearson'] for item in sorted_features[:15]]
                        
                        # Create horizontal bar plot
                        bars = plt.barh(range(len(features_list)), correlations_list, 
                                    color=['red' if x < 0 else 'blue' for x in correlations_list])
                        
                        plt.yticks(range(len(features_list)), features_list)
                        plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
                        plt.title(f'Feature Correlation Ranking vs {target_column} (Delayed Trains > 5min)\n{filename}', 
                                fontsize=14, fontweight='bold')
                        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                        plt.grid(True, axis='x', alpha=0.3)
                        
                        # Add correlation values as text on bars
                        for i, (bar, corr) in enumerate(zip(bars, correlations_list)):
                            plt.text(corr + (0.01 if corr >= 0 else -0.01), i, 
                                    f'{corr:.3f}', va='center', 
                                    ha='left' if corr >= 0 else 'right', fontsize=9)
                        
                        plt.tight_layout()
                        
                        # Save ranking plot
                        ranking_path = os.path.join(feature_ranking_dir, f"feature_ranking_{filename.replace('.csv', '.png')}")
                        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"        Saved feature ranking to {os.path.basename(ranking_path)}")
                    
                    # Save statistical analysis to JSON
                    file_analysis_stats = {
                        'filename': filename,
                        'total_records': len(analysis_df),
                        'total_features_analyzed': len(feature_columns),
                        'target_variable': target_column,
                        'target_statistics': {
                            'mean': float(target_data.mean()),
                            'std': float(target_data.std()),
                            'min': float(target_data.min()),
                            'max': float(target_data.max()),
                            'median': float(target_data.median())
                        },
                        'correlation_analysis': {
                            'pearson_correlations': {k: {
                                'correlation': float(v['correlation']),
                                'p_value': float(v['p_value'])
                            } for k, v in pearson_correlations.items()},
                            'spearman_correlations': {k: {
                                'correlation': float(v['correlation']),
                                'p_value': float(v['p_value'])
                            } for k, v in spearman_correlations.items()}
                        },
                        'top_correlated_features': [
                            {
                                'feature': feature,
                                'pearson_correlation': float(corr_data['pearson']),
                                'spearman_correlation': float(corr_data['spearman']),
                                'pearson_p_value': float(corr_data['pearson_p_value']),
                                'spearman_p_value': float(corr_data['spearman_p_value'])
                            }
                            for feature, corr_data in sorted_features[:10]
                        ]
                    }
                    
                    # Save individual file statistics
                    stats_path = os.path.join(statistical_analysis_dir, f"stats_{filename.replace('.csv', '.json')}")
                    with open(stats_path, 'w') as f:
                        json.dump(file_analysis_stats, f, indent=2)
                    
                    print(f"        Saved statistical analysis to {os.path.basename(stats_path)}")
                    
                    # Store results for combined analysis
                    analysis_results.append(file_analysis_stats)
                    combined_data.append(analysis_df)
                    all_correlation_results[filename] = file_correlations
                    total_files_processed += 1
                    
                except Exception as e:
                    print(f"        Error processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Create combined analysis if we have data from multiple files
            if len(combined_data) > 1 and total_files_processed > 0:
                print(f"      Creating combined analysis across {total_files_processed} files...")
                
                try:
                    # Combine all data
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    print(f"      Combined dataset shape: {combined_df.shape}")
                    
                    # Get all numeric features
                    all_numeric_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
                    feature_columns = [col for col in all_numeric_features 
                                    if col != target_column 
                                    and not col.startswith('Unnamed') 
                                    and col != 'index']
                    
                    # Calculate combined correlations
                    combined_correlations = {}
                    target_data = combined_df[target_column]
                    
                    for feature in feature_columns:
                        if feature in combined_df.columns:
                            feature_data = combined_df[feature]
                            
                            if feature_data.std() > 0:  # Skip zero variance features
                                try:
                                    pearson_corr, pearson_p = pearsonr(target_data, feature_data)
                                    spearman_corr, spearman_p = spearmanr(target_data, feature_data)
                                    
                                    combined_correlations[feature] = {
                                        'pearson': pearson_corr,
                                        'spearman': spearman_corr,
                                        'pearson_p_value': pearson_p,
                                        'spearman_p_value': spearman_p
                                    }
                                except:
                                    continue
                    
                    # Sort by absolute correlation
                    sorted_combined_features = sorted(combined_correlations.items(), 
                                                key=lambda x: abs(x[1]['pearson']), 
                                                reverse=True)
                    
                    print(f"      Top 10 correlated features (Combined Analysis):")
                    for i, (feature, corr_data) in enumerate(sorted_combined_features[:10]):
                        print(f"        {i+1}. {feature}: {corr_data['pearson']:.4f}")
                    
                    # Create combined correlation heatmap
                    if combined_correlations:
                        plt.figure(figsize=(16, 10))
                        
                        # Prepare data for comprehensive heatmap
                        feature_names = list(combined_correlations.keys())
                        correlation_values = [combined_correlations[f]['pearson'] for f in feature_names]
                        
                        # Create correlation matrix (features vs target)
                        heatmap_data = np.array(correlation_values).reshape(1, -1)
                        
                        sns.heatmap(heatmap_data, 
                                annot=True, 
                                fmt='.3f',
                                cmap='RdBu_r', 
                                center=0,
                                xticklabels=feature_names,
                                yticklabels=[f'{target_column}'],
                                cbar_kws={'label': 'Pearson Correlation'})
                        
                        plt.title(f'Combined Numeric Features Correlation Analysis (Delayed Trains > 5min)\nTarget: {target_column} (N={len(combined_df):,})', 
                                fontsize=16, fontweight='bold')
                        plt.xlabel('Features', fontsize=12)
                        plt.ylabel('Target Variable', fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        # Save combined heatmap
                        combined_heatmap_path = os.path.join(correlation_plots_dir, "combined_correlation_heatmap.png")
                        plt.savefig(combined_heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"      Saved combined correlation heatmap")
                    
                    # Create combined feature ranking
                    if sorted_combined_features:
                        plt.figure(figsize=(14, 10))
                        
                        top_features = sorted_combined_features[:20]  # Top 20
                        features_list = [item[0] for item in top_features]
                        correlations_list = [item[1]['pearson'] for item in top_features]
                        
                        bars = plt.barh(range(len(features_list)), correlations_list, 
                                    color=['red' if x < 0 else 'darkblue' for x in correlations_list])
                        
                        plt.yticks(range(len(features_list)), features_list)
                        plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
                        plt.title(f'Combined Feature Correlation Ranking (Delayed Trains > 5min)\nTarget: {target_column} (N={len(combined_df):,})', 
                                fontsize=14, fontweight='bold')
                        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                        plt.grid(True, axis='x', alpha=0.3)
                        
                        # Add correlation values
                        for i, (bar, corr) in enumerate(zip(bars, correlations_list)):
                            plt.text(corr + (0.01 if corr >= 0 else -0.01), i, 
                                    f'{corr:.3f}', va='center', 
                                    ha='left' if corr >= 0 else 'right', fontsize=9)
                        
                        plt.tight_layout()
                        
                        combined_ranking_path = os.path.join(feature_ranking_dir, "combined_feature_ranking.png")
                        plt.savefig(combined_ranking_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"      Saved combined feature ranking")
                    
                    # Save combined statistical analysis
                    combined_stats = {
                        'analysis_type': 'combined_numeric_correlation_analysis_delayed_trains',
                        'data_filter': 'differenceInMinutes_eachStation_offset > 5',
                        'files_analyzed': total_files_processed,
                        'total_combined_records': len(combined_df),
                        'target_variable': target_column,
                        'total_features_analyzed': len(feature_columns),
                        'target_statistics': {
                            'mean': float(target_data.mean()),
                            'std': float(target_data.std()),
                            'min': float(target_data.min()),
                            'max': float(target_data.max()),
                            'median': float(target_data.median())
                        },
                        'correlation_summary': {
                            'strongest_positive_correlation': {
                                'feature': sorted_combined_features[0][0],
                                'correlation': float(sorted_combined_features[0][1]['pearson'])
                            } if sorted_combined_features and sorted_combined_features[0][1]['pearson'] > 0 else None,
                            'strongest_negative_correlation': {
                                'feature': min(sorted_combined_features, key=lambda x: x[1]['pearson'])[0],
                                'correlation': float(min(sorted_combined_features, key=lambda x: x[1]['pearson'])[1]['pearson'])
                            } if sorted_combined_features else None,
                            'features_with_strong_correlation': len([f for f, c in combined_correlations.items() if abs(c['pearson']) > 0.5]),
                            'features_with_moderate_correlation': len([f for f, c in combined_correlations.items() if 0.3 < abs(c['pearson']) <= 0.5]),
                            'features_with_weak_correlation': len([f for f, c in combined_correlations.items() if abs(c['pearson']) <= 0.3])
                        },
                        'detailed_correlations': {
                            feature: {
                                'pearson_correlation': float(corr_data['pearson']),
                                'spearman_correlation': float(corr_data['spearman']),
                                'pearson_p_value': float(corr_data['pearson_p_value']),
                                'spearman_p_value': float(corr_data['spearman_p_value'])
                            }
                            for feature, corr_data in combined_correlations.items()
                        }
                    }
                    
                    combined_stats_path = os.path.join(statistical_analysis_dir, "combined_correlation_analysis.json")
                    with open(combined_stats_path, 'w') as f:
                        json.dump(combined_stats, f, indent=2)
                    
                    print(f"      Saved combined statistical analysis")
                    
                except Exception as e:
                    print(f"      Error in combined analysis: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Create summary report
            summary_stats = {
                'success': True,
                'processed_files': total_files_processed,
                'total_files_found': len(merged_data_files),
                'analysis_types': [
                    'correlation_heatmaps',
                    'scatter_plots',
                    'feature_ranking',
                    'statistical_analysis'
                ],
                'output_path': analysis_output_dir,
                'target_variable': target_column
            }
            
            if analysis_results:
                summary_stats['individual_file_results'] = len(analysis_results)
                
            if len(combined_data) > 1:
                summary_stats['combined_analysis'] = True
                summary_stats['total_combined_records'] = len(pd.concat(combined_data, ignore_index=True))
            
            print(f"    numeric_correlation_analysis: Successfully completed analysis")
            print(f"      Files processed: {total_files_processed}/{len(merged_data_files)}")
            print(f"      Analysis outputs saved to: {analysis_output_dir}")
            
            return summary_stats
            
        except Exception as e:
            print(f"    numeric_correlation_analysis: Error - {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "processed_files": 0
            }