from contextlib import contextmanager
from datetime import datetime
import glob
import os
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
    ALL_WEATHER_FEATURES,
    BORDERLINE_SMOTE_CONFIG,
    BORDERLINE_SMOTE_OUTPUT_FOLDER,
    CLASSIFICATION_PROBLEM,
    DATA_FILE_PREFIX_FOR_TRAINING,
    DECISION_TREE_PARAM_DISTRIBUTIONS,
    DECISION_TREE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER,
    IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    BOOLEAN_FEATURES,
    MAX_SAMPLE_WEIGHT_REGRESSION,
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
    SELECTED_WEATHER_FEATURES,
    STRONG_INDICATORS,
    TARGET_STATION_CODE,
    TEST_SIZE,
    THRESHOLD_OPTIMIZATION_CONFIG,
    TRAIN_DELAY_MINUTES,
    TRAIN_DELAYED_TARGET_COLUMN,
    TRAINING_READY_OUTPUT_FOLDER,
    USE_SIN_COS_APPROACH,
    VALID_TARGET_FEATURES,
    VALID_TRAIN_PREDICTION_FEATURES,
    WEATHER_COLS_TO_MERGE,
    WEATHER_MISSING_THRESHOLD,
    WEIGHT_DELAY_COLUMN,
    XGBOOST_METHODS_CONFIG,
    XGBOOST_PARAM_DISTRIBUTIONS,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    DEFAULT_TARGET_FEATURE,
    MAX_SAMPLE_WEIGHT_CLASSIFICATION,
    XGBOOST_SELECTED_FEATURES_OUTPUT_FOLDER,
    XGBOOST_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER,
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
                    print(f"      ✓ Average CV Score: {xgboost_result.get('average_cv_score', 0):.4f}")
                    
                    # Print problem-specific metrics
                    if xgboost_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {xgboost_result.get('average_test_f1', 0):.4f}")
                        print(f"      ✓ Average Test Accuracy: {xgboost_result.get('average_test_accuracy', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {xgboost_result.get('average_test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {xgboost_result.get('average_test_r2', 0):.4f}")
                        
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

    def train_xgboost_with_randomized_search_cv(self):
        """
        Train XGBoost models with hyperparameter optimization using RandomizedSearchCV.
        
        This method trains XGBoost models on the scaled training data with comprehensive
        hyperparameter tuning. It automatically detects whether the problem is classification 
        or regression based on the target feature.
        
        Input Data Sources:
        - Training Files: D:/Dev/Railway-FMI-Data_training/data/output/4-merged_scaled_training_ready/merged*data_train_scaled.csv
        - Test Files: D:/Dev/Railway-FMI-Data_training/data/output/4-merged_scaled_training_ready/merged*data_test_scaled.csv
        
        Output:
        - Trained XGBoost models saved to: data/output/xgboost_randomized_search/
        
        Returns:
        --------
        dict
            A summary of the XGBoost training results including model performance metrics.
        """
        try:
            print(f"    train_xgboost_with_randomized_search_cv: Starting XGBoost training with hyperparameter optimization...")
            
            # Create output directory
            output_dir = os.path.join(self.project_root, "data/output/xgboost_randomized_search")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"    train_xgboost_with_randomized_search_cv: Output directory: {output_dir}")
            
            # Find training and test files using relative paths
            train_pattern = os.path.join(self.project_root, "data/output/4-merged_scaled_training_ready/merged_data_*_train_scaled.csv")
            test_pattern = os.path.join(self.project_root, "data/output/4-merged_scaled_training_ready/merged_data_*_test_scaled.csv")
            
            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            
            if not train_files or not test_files:
                error_msg = f"Training or test files not found. Train files: {len(train_files)}, Test files: {len(test_files)}"
                print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            print(f"    train_xgboost_with_randomized_search_cv: Found {len(train_files)} training files and {len(test_files)} test files")
            
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
            
            # Initialize results structure
            training_results = {
                "training_overview": {
                    "training_completed": datetime.now().isoformat(),
                    "problem_type": problem_type,
                    "target_feature": target_feature,
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
            best_score = -float('inf')
            best_file = None
            
            # Memory tracking
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            print(f"    train_xgboost_with_randomized_search_cv: Initial memory usage: {initial_memory:.2f} MB")
            
            # Process each training/test file pair
            for train_file in train_files:
                try:
                    # Extract file identifier from train file name
                    train_filename = os.path.basename(train_file)
                    # Extract identifier (e.g., "2023_01" from "merged_data_2023_01_train_scaled.csv")
                    identifier_match = re.search(r'merged_data_(.+?)_train_scaled\.csv', train_filename)
                    if not identifier_match:
                        print(f"    train_xgboost_with_randomized_search_cv: Warning - Could not extract identifier from {train_filename}")
                        continue
                    
                    file_identifier = identifier_match.group(1)
                    
                    # Find corresponding test file
                    test_filename = train_filename.replace('_train_scaled.csv', '_test_scaled.csv')
                    test_file = None
                    for tf in test_files:
                        if os.path.basename(tf) == test_filename:
                            test_file = tf
                            break
                    
                    if not test_file or not os.path.exists(test_file):
                        print(f"    train_xgboost_with_randomized_search_cv: Warning - No corresponding test file for {train_filename}")
                        failed_trainings += 1
                        continue
                    
                    print(f"    train_xgboost_with_randomized_search_cv: Processing {file_identifier}")
                    
                    # Load training and test data
                    print(f"      Loading training data from {train_file}")
                    train_df = pd.read_csv(train_file)
                    
                    print(f"      Loading test data from {test_file}")
                    test_df = pd.read_csv(test_file)
                    
                    # Check if target feature exists
                    if target_feature not in train_df.columns:
                        print(f"      Warning - Target feature '{target_feature}' not found in {file_identifier}")
                        failed_trainings += 1
                        continue
                    
                    # Prepare features and target
                    feature_columns = [col for col in train_df.columns if col != target_feature]
                    
                    X_train = train_df[feature_columns]
                    y_train = train_df[target_feature]
                    X_test = test_df[feature_columns]
                    y_test = test_df[target_feature]
                    
                    print(f"      Dataset info - Train: {X_train.shape}, Test: {X_test.shape}")
                    print(f"      Target distribution - Train: {y_train.value_counts().to_dict() if is_classification else f'Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}'}")
                    
                    # Handle missing values
                    X_train = X_train.fillna(0)
                    X_test = X_test.fillna(0)
                    
                    # Calculate sample weights if enabled
                    sample_weights = None
                    if WEIGHT_DELAY_COLUMN != 'NONE' and WEIGHT_DELAY_COLUMN in train_df.columns:
                        delays = train_df[WEIGHT_DELAY_COLUMN].fillna(0)
                        sample_weights = np.ones(len(y_train))
                        
                        if is_classification:
                            # Higher weights for delayed samples
                            delayed_idx = (delays > 0)
                            if np.any(delayed_idx):
                                mean_delay = delays[delayed_idx].mean()
                                sample_weights[delayed_idx] = np.minimum(
                                    MAX_SAMPLE_WEIGHT_CLASSIFICATION, 
                                    1 + delays[delayed_idx] / mean_delay
                                )
                        else:
                            # Higher weights for larger target values
                            sample_weights = np.minimum(
                                MAX_SAMPLE_WEIGHT_REGRESSION,
                                1.0 + np.abs(y_train) / (np.abs(y_train).mean() * 2)
                            )
                        
                        print(f"      Sample weights range: [{sample_weights.min():.2f} - {sample_weights.max():.2f}]")
                    
                    # Set up cross-validation strategy
                    if is_classification:
                        cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                        scoring_metric = 'f1'
                    else:
                        cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                        scoring_metric = 'neg_mean_squared_error'
                    
                    # Initialize XGBoost model based on problem type
                    if is_classification:
                        base_model = xgb.XGBClassifier(
                            random_state=42,
                            n_jobs=1,  # Prevent memory issues
                            eval_metric='logloss'
                        )
                    else:
                        base_model = xgb.XGBRegressor(
                            random_state=42,
                            n_jobs=1,  # Prevent memory issues
                            eval_metric='rmse'
                        )
                    
                    # Perform hyperparameter optimization
                    print(f"      Starting RandomizedSearchCV with {RANDOM_SEARCH_ITERATIONS} iterations...")
                    
                    randomized_search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
                        n_iter=RANDOM_SEARCH_ITERATIONS,
                        scoring=scoring_metric,
                        cv=cv_splitter,
                        random_state=42,
                        n_jobs=1,  # Prevent memory issues
                        verbose=2
                    )
                    
                    # Fit with sample weights if available
                    if sample_weights is not None:
                        randomized_search.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        randomized_search.fit(X_train, y_train)
                    
                    best_model = randomized_search.best_estimator_
                    best_params = randomized_search.best_params_
                    best_cv_score = randomized_search.best_score_
                    
                    print(f"      Best CV Score: {best_cv_score:.4f}")
                    print(f"      Best Parameters: {best_params}")
                    
                    # Make predictions on test set
                    y_pred = best_model.predict(X_test)
                    
                    # Calculate test metrics
                    if is_classification:
                        test_accuracy = accuracy_score(y_test, y_pred)
                        test_f1 = f1_score(y_test, y_pred, average='weighted')
                        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        # Calculate confusion matrix
                        conf_matrix = confusion_matrix(y_test, y_pred)
                        
                        if hasattr(best_model, 'predict_proba'):
                            y_proba = best_model.predict_proba(X_test)
                            if y_proba.shape[1] == 2:  # Binary classification
                                test_auc = roc_auc_score(y_test, y_proba[:, 1])
                            else:
                                test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                        else:
                            test_auc = None
                        
                        test_score = test_f1  # Use F1 as primary metric for classification
                        
                        print(f"      Classification Results:")
                        print(f"        Accuracy: {test_accuracy:.4f}")
                        print(f"        F1 Score: {test_f1:.4f}")
                        print(f"        Precision: {test_precision:.4f}")
                        print(f"        Recall: {test_recall:.4f}")
                        if test_auc is not None:
                            print(f"        ROC AUC: {test_auc:.4f}")
                        
                        # Save confusion matrix
                        conf_matrix_result = save_confusion_matrix(
                            conf_matrix, 
                            y_test, 
                            y_pred, 
                            file_identifier, 
                            output_dir
                        )
                        
                        file_metrics = {
                            "accuracy": float(test_accuracy),
                            "f1_score": float(test_f1),
                            "precision": float(test_precision),
                            "recall": float(test_recall),
                            "roc_auc": float(test_auc) if test_auc is not None else None,
                            "confusion_matrix_file": conf_matrix_result
                        }
                        
                    else:
                        test_mse = mean_squared_error(y_test, y_pred)
                        test_rmse = np.sqrt(test_mse)
                        test_mae = mean_absolute_error(y_test, y_pred)
                        test_r2 = r2_score(y_test, y_pred)
                        
                        test_score = -test_mse  # Use negative MSE (higher is better)
                        
                        print(f"      Regression Results:")
                        print(f"        RMSE: {test_rmse:.4f}")
                        print(f"        MAE: {test_mae:.4f}")
                        print(f"        R² Score: {test_r2:.4f}")
                        print(f"        MSE: {test_mse:.4f}")
                        
                        file_metrics = {
                            "rmse": float(test_rmse),
                            "mae": float(test_mae),
                            "r2_score": float(test_r2),
                            "mse": float(test_mse)
                        }
                    
                    # Track best performing model
                    if test_score > best_score:
                        best_score = test_score
                        best_file = file_identifier
                    
                    # Save trained model
                    model_filename = f"xgboost_model_{file_identifier}.joblib"
                    model_path = os.path.join(output_dir, model_filename)
                    joblib.dump(best_model, model_path)
                    
                    # Save feature importance
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance_df = pd.DataFrame({
                            'feature': feature_columns,
                            'importance': best_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        importance_filename = f"feature_importance_{file_identifier}.csv"
                        importance_path = os.path.join(output_dir, importance_filename)
                        feature_importance_df.to_csv(importance_path, index=False)
                    
                    # Store results
                    file_result = {
                        "file_identifier": file_identifier,
                        "train_file": train_file,
                        "test_file": test_file,
                        "training_successful": True,
                        "best_cv_score": float(best_cv_score),
                        "best_parameters": best_params,
                        "test_metrics": file_metrics,
                        "model_file": model_path,
                        "feature_importance_file": importance_path if hasattr(best_model, 'feature_importances_') else None,
                        "dataset_info": {
                            "train_samples": int(len(X_train)),
                            "test_samples": int(len(X_test)),
                            "num_features": int(len(feature_columns)),
                            "target_feature": target_feature
                        }
                    }
                    
                    training_results["file_results"].append(file_result)
                    all_cv_scores.append(best_cv_score)
                    all_test_scores.append(test_score)
                    successful_trainings += 1
                    
                    print(f"      ✓ Successfully trained XGBoost model for {file_identifier}")
                    
                    # Memory cleanup
                    del X_train, X_test, y_train, y_test, train_df, test_df
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"      Memory usage: {current_memory:.2f} MB")
                    
                except Exception as e:
                    error_msg = f"Failed to train XGBoost for {file_identifier if 'file_identifier' in locals() else 'unknown'}: {str(e)}"
                    print(f"      ✗ {error_msg}")
                    
                    training_results["file_results"].append({
                        "file_identifier": file_identifier if 'file_identifier' in locals() else "unknown",
                        "train_file": train_file,
                        "training_successful": False,
                        "error": str(e)
                    })
                    failed_trainings += 1
                    continue
            
            # Finalize results
            if successful_trainings > 0:
                training_results["training_overview"].update({
                    "total_files_processed": successful_trainings + failed_trainings,
                    "successful_trainings": successful_trainings,
                    "failed_trainings": failed_trainings
                })
                
                # Calculate aggregate metrics
                if is_classification:
                    avg_f1_scores = [result["test_metrics"]["f1_score"] for result in training_results["file_results"] if result["training_successful"]]
                    avg_accuracies = [result["test_metrics"]["accuracy"] for result in training_results["file_results"] if result["training_successful"]]
                    
                    training_results["aggregate_metrics"] = {
                        "average_cv_score": float(np.mean(all_cv_scores)),
                        "average_test_f1": float(np.mean(avg_f1_scores)),
                        "average_test_accuracy": float(np.mean(avg_accuracies)),
                        "best_performing_file": {
                            "file_identifier": best_file,
                            "test_score": float(best_score)
                        } if best_file else None
                    }
                else:
                    avg_rmse_scores = [result["test_metrics"]["rmse"] for result in training_results["file_results"] if result["training_successful"]]
                    avg_r2_scores = [result["test_metrics"]["r2_score"] for result in training_results["file_results"] if result["training_successful"]]
                    
                    training_results["aggregate_metrics"] = {
                        "average_cv_score": float(np.mean(all_cv_scores)),
                        "average_test_rmse": float(np.mean(avg_rmse_scores)),
                        "average_test_r2": float(np.mean(avg_r2_scores)),
                        "best_performing_file": {
                            "file_identifier": best_file,
                            "test_score": float(best_score)
                        } if best_file else None
                    }
                
                # Save consolidated results
                results_file = os.path.join(output_dir, "xgboost_randomized_search_training_results.json")
                with open(results_file, 'w') as f:
                    json.dump(training_results, f, indent=2)
                
                final_memory = process.memory_info().rss / 1024 / 1024
                print(f"    train_xgboost_with_randomized_search_cv: Final memory usage: {final_memory:.2f} MB")
                print(f"    train_xgboost_with_randomized_search_cv: Completed successfully!")
                print(f"    train_xgboost_with_randomized_search_cv: Trained {successful_trainings} XGBoost models")
                print(f"    train_xgboost_with_randomized_search_cv: Average CV Score: {np.mean(all_cv_scores):.4f}")
                
                if is_classification:
                    print(f"    train_xgboost_with_randomized_search_cv: Average Test F1: {training_results['aggregate_metrics']['average_test_f1']:.4f}")
                else:
                    print(f"    train_xgboost_with_randomized_search_cv: Average Test RMSE: {training_results['aggregate_metrics']['average_test_rmse']:.4f}")
                
                print(f"    train_xgboost_with_randomized_search_cv: Results saved to: {output_dir}")
                
                return {
                    "success": True,
                    "models_trained": successful_trainings,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "average_cv_score": float(np.mean(all_cv_scores)),
                    "output_directory": output_dir,
                    "results_file": results_file,
                    **training_results["aggregate_metrics"]
                }
            else:
                error_msg = "No XGBoost models were successfully trained"
                print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"train_xgboost_with_randomized_search_cv failed: {str(e)}"
            print(f"    train_xgboost_with_randomized_search_cv: {error_msg}")
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
