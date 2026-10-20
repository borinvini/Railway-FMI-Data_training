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
import warnings

# joblib/loky's resource_tracker warns at interpreter shutdown when Windows
# has already removed a worker's temp memmap folder; the workers still ran
# to completion, so these are noise rather than a sign of a real leak. The
# tracker runs in its own spawned subprocess, which only inherits filters
# via PYTHONWARNINGS (a plain filterwarnings() call here would not reach it).
_RESOURCE_TRACKER_FILTER = "ignore:resource_tracker:UserWarning:joblib.externals.loky.backend.resource_tracker"
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: .*",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.resource_tracker",
)
if _RESOURCE_TRACKER_FILTER not in os.environ.get("PYTHONWARNINGS", ""):
    os.environ["PYTHONWARNINGS"] = ",".join(
        filter(None, [os.environ.get("PYTHONWARNINGS", ""), _RESOURCE_TRACKER_FILTER])
    )
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, BayesianRidge
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import lightgbm as lgb
import shap

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score,
    f1_score, r2_score, confusion_matrix
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.cv_utils import (filled_path, groups_path, load_filled_flags, load_train_groups,
                          mae_baseline, make_cv, make_minutes_scorer, neg_pos_ratio,
                          residual_quantiles, to_minutes)
from src.day_split import split_days
from src import exploration_plots as xplots
from src.data_exploration import (WIND_COLUMN, association_by_year, association_table, bin_codes,
                                  data_summary, drift_table, monthly_series, profile_by,
                                  recover_readable_variables, spearman_matrix, variable_names,
                                  wind_sector_labels, MONTH_NAMES)
from src.feature_analysis import (drop_group_ablation, feature_groups, fit_xgboost,
                                  grouped_permutation_importance, make_score_fn,
                                  protected_columns, spearman_clusters)
from src.file_utils import (
    save_dataframe_to_parquet,
)

from config.const import (
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    DATA_ROOT,
)

from config.const_preprocessing import (
    GROUP_KEY_COLUMN,
    VALID_TARGET_FEATURES,
    TARGET_STATION_CODE,
    LABEL_COLUMNS,
    PREPROCESSED_OUTPUT_FOLDER,
    DATA_FILE_PREFIX_FOR_TRAINING,
    DEFAULT_TARGET_FEATURE,
    ALL_WEATHER_FEATURES,
    SKEWED_WEATHER_FEATURES,
    TRAINING_READY_OUTPUT_FOLDER,
    REGRESSION_PROBLEM,
    CLASSIFICATION_PROBLEM,
    TRAIN_DELAY_MINUTES,
)

from config.const_training import (
    MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER,
    FILTER_LOWER_QUANTILE,
    FILTER_UPPER_QUANTILE,
    REGRESSION_FILTER_TRAIN_ONLY,
    REGRESSION_CANCELLED_POLICY,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    RANDOM_FOREST_PARAM_DISTRIBUTIONS,
    RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION,
    LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
    ELASTICNET_PARAM_DISTRIBUTIONS,
    NAIVE_BAYES_OUTPUT_FOLDER,
    NAIVE_BAYES_PARAM_DISTRIBUTIONS,
    BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
    SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER,
    SHAP_CORRELATION_TOP_N_PAIRS,
    SHAP_CORRELATION_TOP_N_FEATURES,
    DATA_EXPLORATION_OUTPUT_FOLDER,
    SPEARMAN_CLUSTER_ABS_CORR,
    EXPLORATION_N_BINS,
    EXPLORATION_WIND_SECTORS,
    EXPLORATION_MIN_CELL_COUNT,
    EXPLORATION_ANNOTATE_MAX,
    EXPLORATION_MAX_PANELS,
    MODEL_FEATURE_IMPORTANCE_OUTPUT_FOLDER,
    PERMUTATION_REPEATS,
    ABLATION_NOISE_SEEDS,
    NOISE_FLOOR_FACTOR,
    ABLATION_EXTRA_SETS,
    FEATURE_ANALYSIS_XGB_PARAMS,
    MERGED_TRAINING_READY_OUTPUT_FOLDER,
    FCLL_SEQUENCES_OUTPUT_FOLDER,
    FCLL_NET_OUTPUT_FOLDER,
    FCLL_H_TRAINS,
    FCLL_Z_STATIONS,
    TEST_SIZE,
    RANDOM_STATE,
    HOLDOUT_LAST_YEAR,
    HOLDOUT_YEAR_COLUMN,
    MONTH_KEY_COLUMN,
    SCORE_METRIC,
    REGRESSION_SCORE_METRIC,
    REGRESSION_OBJECTIVE,
    REGRESSION_LOG_TARGET,
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
    SEARCH_N_JOBS,
    MODEL_N_JOBS,
)


class TrainingPipeline:
    def __init__(self):
        """
        Initialize the TrainingPipeline class with default values.
        """
        # Get script directory and project root for file operations
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = DATA_ROOT or os.path.dirname(self.script_dir)
        self.output_dir = os.path.join(self.project_root, OUTPUT_FOLDER)
        self.preprocessed_dir = os.path.join(self.project_root, PREPROCESSED_OUTPUT_FOLDER)
        self.random_forest_dir = os.path.join(self.project_root, RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
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
                    # build_fcll_sequences reads raw events, not this frame, so it
                    # needs the scenario's column list rather than the frame itself.
                    self.selected_columns = cols_selection_result.get("selected_columns", [])
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

        if state_machine.get("build_fcll_sequences", False):
            try:
                print(f"    → build_fcll_sequences")
                # SELECTED_COLUMNS, not the frame from select_training_cols:
                # that stage reads the merged tabular parquet, which fcll_net
                # never builds. The scenario's column list is all this needs.
                seq = self.build_fcll_sequences(
                    nf_columns=getattr(self, "selected_columns", None) or list(SELECTED_COLUMNS)
                )
                if seq and seq.get("success", False):
                    result["steps_executed"].append("build_fcll_sequences")
                    result["file_info"]["fcll_samples"] = seq.get("n_samples", 0)
                    print(f"      ✓ {seq['n_samples']:,} samples in {seq['shards']} shards")
                    result["success"] = True
                else:
                    error_msg = seq.get("error", "build_fcll_sequences returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result
            except Exception as e:
                result["errors"].append(f"build_fcll_sequences failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ build_fcll_sequences (disabled)")

        if state_machine.get("shap_correlation_analysis", False):
            try:
                print(f"    → shap_correlation_analysis")
                _use_scaled_for_shap = state_machine.get("scale_weather_features", False)
                _use_filtered_for_shap = (
                    state_machine.get("filter_delay_outliers", False)
                    and not state_machine.get("select_training_cols", False)
                )
                _shap_data_folder = (
                    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled_for_shap
                    else MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered_for_shap
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                shap_result = self.shap_correlation_analysis(
                    data_dir=os.path.join(self.project_root, _shap_data_folder)
                )

                if shap_result and shap_result.get("success", False):
                    result["steps_executed"].append("shap_correlation_analysis")
                    print(f"      ✓ Successfully computed SHAP correlation analysis")
                    print(f"      ✓ Features analyzed: {shap_result.get('num_features', 0)}")
                    most_pair = shap_result.get('most_correlated_pair')
                    if most_pair:
                        print(f"      ✓ Most correlated pair: {most_pair[0]} <-> {most_pair[1]} (corr={most_pair[2]:.3f})")
                    print(f"      ✓ Results saved to: {shap_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = shap_result.get("error", "shap_correlation_analysis returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"shap_correlation_analysis failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ shap_correlation_analysis (disabled)")

        if state_machine.get("explore_training_data", False):
            try:
                print(f"    → explore_training_data")
                # The descriptive charts read stage 502: every row in physical units with its
                # year, which 503 and 505 (split, scaled, no year) no longer carry.
                explore_result = self.explore_training_data(
                    data_dir=os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
                )

                if explore_result and explore_result.get("success", False):
                    result["steps_executed"].append("explore_training_data")
                    print(f"      ✓ Successfully explored the training data (no model fitted)")
                    print(f"      ✓ Rows: {explore_result.get('n_rows', 0):,} | variables: {explore_result.get('n_variables', 0)} | charts: {explore_result.get('n_charts', 0)}")
                    print(f"      ✓ Suggested non-redundant features: {explore_result.get('suggested_feature_count', 0)}")
                    print(f"      ✓ Results saved to: {explore_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = (explore_result or {}).get("error", "explore_training_data returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"explore_training_data failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ explore_training_data (disabled)")

        if state_machine.get("model_feature_importance_analysis", False):
            try:
                print(f"    → model_feature_importance_analysis")
                # Train/test/hold-out as the trainers see them: the scaled folder, else the
                # raw split. Never the balanced folder (resampled rows would corrupt an ablation).
                # The scaled folder also wins when it already exists, so that
                # `--stages model_feature_importance_analysis` alone, with the scaling stage off,
                # runs on the outputs of an earlier run instead of looking in 503.
                _fi_scaled_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
                _fi_data_dir = (
                    _fi_scaled_dir
                    if state_machine.get("scale_weather_features", False)
                    or glob.glob(os.path.join(_fi_scaled_dir, "merged_data_*_train.parquet"))
                    else os.path.join(self.project_root, SPLIT_DATASET_OUTPUT_FOLDER)
                )
                fi_result = self.model_feature_importance_analysis(data_dir=_fi_data_dir)

                if fi_result and fi_result.get("success", False):
                    result["steps_executed"].append("model_feature_importance_analysis")
                    print(f"      ✓ Successfully computed model feature importance analysis")
                    print(f"      ✓ Features analyzed: {fi_result.get('num_features', 0)} in {fi_result.get('num_groups', 0)} groups")
                    top_group = fi_result.get("top_group")
                    if top_group:
                        print(f"      ✓ Most important group on {fi_result.get('primary_split')}: {top_group[0]} (score drop {top_group[1]:.4f})")
                    print(f"      ✓ Results saved to: {fi_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = (fi_result or {}).get("error", "model_feature_importance_analysis returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"model_feature_importance_analysis failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ model_feature_importance_analysis (disabled)")

        if state_machine.get("train_fcll_net", False):
            try:
                print(f"    → train_fcll_net")
                fcll_result = self.train_fcll_net()

                if fcll_result and fcll_result.get("success", False):
                    result["steps_executed"].append("train_fcll_net")
                    metrics = fcll_result.get("metrics", {})
                    result["file_info"]["fcll_net_metrics"] = metrics
                    print(f"      ✓ Successfully trained FCLL-Net")
                    print(f"      ✓ Test samples: {metrics.get('n_test', 0):,}")
                    print(f"      ✓ Accuracy: {metrics.get('accuracy', 0):.4f}")
                    print(f"      ✓ F1: {metrics.get('f1', 0):.4f}")
                    print(f"      ✓ ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}")
                    print(f"      ✓ Epochs run: {metrics.get('epochs_run', 0)}")
                    print(f"      ✓ Results saved to: {fcll_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = fcll_result.get("error", "train_fcll_net returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"train_fcll_net failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_fcll_net (disabled)")

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

        if state_machine.get("train_random_forest_with_randomized_search_cv", False):
            try:
                print(f"    → train_random_forest_with_randomized_search_cv")
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
                random_forest_result = self.train_random_forest_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )

                if random_forest_result and random_forest_result.get("success", False):
                    result["steps_executed"].append("train_random_forest_with_randomized_search_cv")
                    result["file_info"]["random_forest_models_trained"] = random_forest_result.get("models_trained", 0)
                    result["file_info"]["random_forest_problem_type"] = random_forest_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained Random Forest models")
                    print(f"      ✓ Problem type: {random_forest_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {random_forest_result.get('models_trained', 0)}")
                    print(f"      ✓ Target feature: {random_forest_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Average CV Score: {random_forest_result.get('cv_score', 0):.4f}")

                    if random_forest_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {random_forest_result.get('test_f1', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {random_forest_result.get('test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {random_forest_result.get('test_r2', 0):.4f}")

                    print(f"      ✓ Results saved to: {random_forest_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = random_forest_result.get("error", "train_random_forest_with_randomized_search_cv returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"train_random_forest_with_randomized_search_cv failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_random_forest_with_randomized_search_cv (disabled)")

        if state_machine.get("train_logistic_regression_with_randomized_search_cv", False):
            try:
                print(f"    → train_logistic_regression_with_randomized_search_cv")
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
                logistic_regression_result = self.train_logistic_regression_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )

                if logistic_regression_result and logistic_regression_result.get("success", False):
                    result["steps_executed"].append("train_logistic_regression_with_randomized_search_cv")
                    result["file_info"]["logistic_regression_models_trained"] = logistic_regression_result.get("models_trained", 0)
                    result["file_info"]["logistic_regression_problem_type"] = logistic_regression_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained Logistic Regression models")
                    print(f"      ✓ Problem type: {logistic_regression_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {logistic_regression_result.get('models_trained', 0)}")
                    print(f"      ✓ Target feature: {logistic_regression_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Average CV Score: {logistic_regression_result.get('cv_score', 0):.4f}")

                    if logistic_regression_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {logistic_regression_result.get('test_f1', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {logistic_regression_result.get('test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {logistic_regression_result.get('test_r2', 0):.4f}")

                    print(f"      ✓ Results saved to: {logistic_regression_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = logistic_regression_result.get("error", "train_logistic_regression_with_randomized_search_cv returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"train_logistic_regression_with_randomized_search_cv failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_logistic_regression_with_randomized_search_cv (disabled)")

        if state_machine.get("train_naive_bayes_with_randomized_search_cv", False):
            try:
                print(f"    → train_naive_bayes_with_randomized_search_cv")
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
                naive_bayes_result = self.train_naive_bayes_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )

                if naive_bayes_result and naive_bayes_result.get("success", False):
                    result["steps_executed"].append("train_naive_bayes_with_randomized_search_cv")
                    result["file_info"]["naive_bayes_models_trained"] = naive_bayes_result.get("models_trained", 0)
                    result["file_info"]["naive_bayes_problem_type"] = naive_bayes_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained Naive Bayes models")
                    print(f"      ✓ Problem type: {naive_bayes_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {naive_bayes_result.get('models_trained', 0)}")
                    print(f"      ✓ Target feature: {naive_bayes_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Average CV Score: {naive_bayes_result.get('cv_score', 0):.4f}")

                    if naive_bayes_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {naive_bayes_result.get('test_f1', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {naive_bayes_result.get('test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {naive_bayes_result.get('test_r2', 0):.4f}")

                    print(f"      ✓ Results saved to: {naive_bayes_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = naive_bayes_result.get("error", "train_naive_bayes_with_randomized_search_cv returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"train_naive_bayes_with_randomized_search_cv failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_naive_bayes_with_randomized_search_cv (disabled)")

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

        # A numeric target is trimmed on the train rows only, after the split (stage 503):
        # bounds taken from every row would also trim test and hold-out, and RMSE / R2
        # would then be measured on a target without its extreme delays. The data is
        # still saved below so stage 502 never picks up a stale filtered file.
        defer_to_split = DEFAULT_TARGET_FEATURE in REGRESSION_PROBLEM and REGRESSION_FILTER_TRAIN_ONLY
        if defer_to_split:
            lower_bound = upper_bound = None
            lower_mask = upper_mask = pd.Series(False, index=df.index)
        else:
            lower_bound = float(df[target_col].quantile(FILTER_LOWER_QUANTILE))
            upper_bound = float(df[target_col].quantile(FILTER_UPPER_QUANTILE))

            lower_mask = df[target_col] < lower_bound
            upper_mask = df[target_col] > upper_bound

        rows_removed_lower = int(lower_mask.sum())
        rows_removed_upper = int(upper_mask.sum())

        df = df[~lower_mask & ~upper_mask].copy()
        rows_after = len(df)

        print(f"\n    filter_delay_outliers: Filtering {rows_before:,} rows...")
        if defer_to_split:
            print(f"      Numeric target '{DEFAULT_TARGET_FEATURE}': no rows removed here; the tails are cut "
                  f"from the train split only in split_dataset (REGRESSION_FILTER_TRAIN_ONLY)")
        else:
            print(f"      Lower bound (q={FILTER_LOWER_QUANTILE}): {lower_bound:.2f} min")
            print(f"      Upper bound (q={FILTER_UPPER_QUANTILE}): {upper_bound:.2f} min")
        if defer_to_split:
            pass
        elif rows_before > 0:
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
        holdout_files = glob.glob(os.path.join(data_dir, "*_holdout.parquet"))

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

        if target_col in ("trainDelayed", "trainDisrupted"):
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

        # Copy the out-of-time hold-out file unchanged too — it must never be
        # resampled (SMOTE/Tomek would distort its real-world class distribution).
        holdout_output_path = None
        if holdout_files:
            holdout_src = holdout_files[0]
            holdout_output_path = os.path.join(output_folder, os.path.basename(holdout_src))
            shutil.copy2(holdout_src, holdout_output_path)
            print(f"    balance_classes: Copied hold-out file to: {holdout_output_path}")

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
                "holdout_output_path": holdout_output_path,
            }

        if is_regression:
            # Resampling classes is meaningless for a numeric target (the old branch
            # binarised it and ran SMOTE). The train rows pass through unchanged.
            print(f"    balance_classes: '{target_col}' is a numeric target — nothing to balance, saving train unchanged")
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
                "holdout_output_path": holdout_output_path,
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
                "holdout_output_path": holdout_output_path,
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
                or c.startswith("wawa_group_")
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
            "holdout_output_path": holdout_output_path,
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

                    # Stamp each row with its source year (derived from the filename, not a
                    # per-row timestamp — no such column survives preprocessing). This gives
                    # split_dataset a real per-row year signal for the out-of-time hold-out;
                    # it is dropped again right after the split so it never becomes a feature.
                    df[HOLDOUT_YEAR_COLUMN] = year
                    df[MONTH_KEY_COLUMN] = f"{year}-{month_number:02d}"

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

    def apply_column_selection(self, df, columns):
        """Return `df` restricted to `columns`, plus the grouping key.

        The key is appended rather than required in `columns` because no scenario
        lists it: it is not a feature, and adding it to all 13 catalogues would
        invite someone to train on the date. split_dataset drops it after grouping.
        """
        keep = list(columns)
        if GROUP_KEY_COLUMN in df.columns and GROUP_KEY_COLUMN not in keep:
            keep.append(GROUP_KEY_COLUMN)
        # Same for the monthly-file key: split_dataset imputes with it, then drops it.
        if MONTH_KEY_COLUMN in df.columns and MONTH_KEY_COLUMN not in keep:
            keep.append(MONTH_KEY_COLUMN)
        # Labels: only the target may stay. The trainers use every non-target column as a
        # feature, so a second label (trainDelayed beside trainDisrupted, cancelled, ...) would
        # hand the model its own answer, whatever a feature catalogue lists. Every target-like
        # column counts, the differenceInMinutes* ones too: with a numeric target the other two
        # delay columns are the same quantity, and with a classification target they are what
        # the label was computed from.
        if DEFAULT_TARGET_FEATURE in df.columns:
            other_labels = [c for c in VALID_TARGET_FEATURES if c != DEFAULT_TARGET_FEATURE and c in keep]
            if other_labels:
                print(f"    apply_column_selection: dropping label columns other than the target "
                      f"'{DEFAULT_TARGET_FEATURE}': {other_labels}")
                keep = [c for c in keep if c not in other_labels]
            if DEFAULT_TARGET_FEATURE not in keep:
                keep.append(DEFAULT_TARGET_FEATURE)
        return df[keep]

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
                        if not f.endswith(("_train.parquet", "_val.parquet", "_train_groups.parquet",
                                           "_train_filled.parquet", "_test_filled.parquet",
                                           "_holdout_filled.parquet",
                                           "_test.parquet", "_holdout.parquet"))
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

            # Force-keep the hold-out year column even if not explicitly selected, so
            # split_dataset can carve out the out-of-time hold-out downstream. It is
            # dropped again right after the split — never reaches balancing/scaling/training.
            if HOLDOUT_LAST_YEAR and HOLDOUT_YEAR_COLUMN in column_names and HOLDOUT_YEAR_COLUMN not in selected_columns:
                selected_columns.append(HOLDOUT_YEAR_COLUMN)
                print(f"    select_training_cols: Force-keeping '{HOLDOUT_YEAR_COLUMN}' for the out-of-time hold-out split")

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
            df_filtered = self.apply_column_selection(df, selected_columns).copy()
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

    def build_fcll_sequences(self, input_dir=None, output_dir=None, nf_columns=None,
                             target_station=None, h=FCLL_H_TRAINS, z=FCLL_Z_STATIONS):
        """Build FCLL-Net sequence tensors, one shard per input month.

        Streams: each month is read, converted and written before the next is
        opened. The raw table is ~518k rows per month over 96 months; holding all
        of it would need far more than the 64G the batch script requests.
        """
        from src.fcll_sequences import assemble_sequences, derive_station_visits

        input_dir = input_dir or os.path.join(self.project_root, INPUT_FOLDER)
        output_dir = output_dir or os.path.join(self.project_root, FCLL_SEQUENCES_OUTPUT_FOLDER)
        target_station = target_station or TARGET_STATION_CODE
        nf_columns = nf_columns or []
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(input_dir, "matched_data_flat_*.parquet")))
        if not files:
            return {"success": False, "error": f"no input parquet in {input_dir}"}

        total = 0
        kept_columns = None
        for path in files:
            month = os.path.basename(path).replace("matched_data_flat_", "").replace(".parquet", "")
            df = pd.read_parquet(path)
            visits = derive_station_visits(df, carry_columns=nf_columns)
            available = [c for c in nf_columns if c in visits.columns]
            if kept_columns is None:
                kept_columns = available
            elif available != kept_columns:
                # Shards are concatenated later; a month with a different NF width
                # would produce a ragged array at load time instead of here.
                return {"success": False, "error":
                        f"{month} has NF columns {available}, expected {kept_columns}"}
            out = assemble_sequences(visits, target_station=target_station,
                                     nf_columns=available, h=h, z=z)
            np.savez_compressed(os.path.join(output_dir, f"fcll_{month}.npz"), **out)
            total += len(out["label"])
            print(f"      fcll_{month}: {len(out['label']):,} samples")
            del df, visits, out

        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"h": h, "z": z, "n_samples": total, "columns": kept_columns,
                       "target_station": target_station}, fh, indent=2)
        return {"success": True, "n_samples": total, "shards": len(files)}

    def train_fcll_net(self, data_dir=None):
        """Train FCLL-Net on the sequences built by stage 506."""
        import torch

        from src.fcll_trainer import load_shards, train_fcll

        data_dir = data_dir or os.path.join(self.project_root, FCLL_SEQUENCES_OUTPUT_FOLDER)
        output_dir = os.path.join(self.project_root, FCLL_NET_OUTPUT_FOLDER)
        os.makedirs(output_dir, exist_ok=True)
        try:
            data = load_shards(data_dir)
        except FileNotFoundError as exc:
            return {"success": False, "error": str(exc)}

        try:
            result = train_fcll(data)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        torch.save(result["model"].state_dict(), os.path.join(output_dir, "fcll_net.pt"))
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(result["metrics"], fh, indent=2)
        with open(os.path.join(output_dir, "val_loss_history.json"), "w", encoding="utf-8") as fh:
            json.dump(result["history"], fh, indent=2)
        return {"success": True, "metrics": result["metrics"],
                "output_directory": output_dir}

    def split_by_group(self, df, group_column, random_state=RANDOM_STATE):
        """Split `df` into train/test so no group value spans both splits.

        There is no validation split: it was written but never read by the tabular
        models (they validate with day-grouped CV inside train, see src/cv_utils.py).
        Groups are shuffled and partitioned by src/day_split.split_days, NOT
        stratified; the base rate (23.5% overall) is stable enough across days that
        random day assignment holds it, which test_the_base_rate_survives_the_split
        checks.
        """
        train_groups, test_groups = split_days(df[group_column].unique(), seed=random_state)
        select = lambda wanted: df[df[group_column].isin(wanted)]
        return select(train_groups), select(test_groups)

    @staticmethod
    def fit_train_medians(train_df):
        """Medians of the numeric feature columns of the TRAIN split.

        Targets (VALID_TARGET_FEATURES), the grouping key and the hold-out year
        are never imputed. A column that is entirely NaN in train has no median to
        borrow, so it raises instead of inventing a value.
        """
        skip = set(VALID_TARGET_FEATURES) | {GROUP_KEY_COLUMN, HOLDOUT_YEAR_COLUMN, MONTH_KEY_COLUMN}
        numeric = train_df.select_dtypes(include="number").columns
        medians = {}
        empty = []
        for col in numeric:
            if col in skip:
                continue
            value = train_df[col].median()
            if pd.isna(value):
                empty.append(col)
            else:
                medians[col] = float(value)
        if empty:
            raise ValueError(f"Columns with no non-missing value in the train split: {empty}")
        return medians

    @staticmethod
    def weather_columns(frame):
        """Weather columns of `frame`: base readings plus the 12h/24h/72h aggregates.

        The same selection scale_weather_features uses.
        """
        window_patterns = ('(12h', '(24h', '(72h')
        return [c for c in frame.columns
                if c in ALL_WEATHER_FEATURES or any(p in c for p in window_patterns)]

    @staticmethod
    def _calendar_month(key):
        """Calendar month (1-12) parsed from a "YYYY-MM" key Series; NaN when unreadable."""
        return pd.to_numeric(key.astype("string").str[5:7], errors="coerce")

    @classmethod
    def fit_month_medians(cls, train_df):
        """Weather medians per monthly file and per calendar month, from the TRAIN split only.

        Returns {"year_month": {"2023-01": {col: median}}, "calendar_month": {1: {col: median}}},
        or {} when the frame carries no MONTH_KEY_COLUMN. A group with no train value for a
        column is left out, so apply_medians falls through to the next level.
        """
        if MONTH_KEY_COLUMN not in train_df.columns:
            return {}
        skip = set(VALID_TARGET_FEATURES) | {GROUP_KEY_COLUMN, HOLDOUT_YEAR_COLUMN, MONTH_KEY_COLUMN}
        numeric = set(train_df.select_dtypes(include="number").columns)
        columns = [c for c in cls.weather_columns(train_df) if c in numeric and c not in skip]
        if not columns:
            return {}
        key = train_df[MONTH_KEY_COLUMN]

        def rows(grouped, name):
            return {name(k): {c: float(v) for c, v in row.items() if pd.notna(v)}
                    for k, row in grouped.iterrows()}

        by_year_month = train_df[columns].groupby(key).median()
        by_calendar = train_df[columns].groupby(cls._calendar_month(key)).median()
        return {"year_month": rows(by_year_month, str),
                "calendar_month": rows(by_calendar, int)}

    @classmethod
    def apply_medians(cls, frame, medians, month_medians=None):
        """Fill NaN in `frame` from the train medians; other columns are untouched.

        A weather cell takes the median of its own monthly file first, then of its
        calendar month over all train years (the hold-out year has no train rows of
        its own), then the global train median of the column.
        """
        out = frame
        if month_medians and MONTH_KEY_COLUMN in frame.columns:
            key = frame[MONTH_KEY_COLUMN]
            calendar = cls._calendar_month(key)
            by_year_month = month_medians.get("year_month", {})
            by_calendar = month_medians.get("calendar_month", {})
            columns = ({c for v in by_year_month.values() for c in v}
                       | {c for v in by_calendar.values() for c in v})
            out = frame.copy(deep=False)
            for col in sorted(columns & set(out.columns)):
                missing = out[col].isna()
                if not missing.any():
                    continue
                own = key.map({k: v[col] for k, v in by_year_month.items() if col in v})
                fallback = calendar.map({m: v[col] for m, v in by_calendar.items() if col in v})
                out[col] = out[col].where(~missing, own.fillna(fallback))
        known = {col: v for col, v in medians.items() if col in out.columns}
        return out.fillna(value=known)

    @staticmethod
    def imputation_report(medians, month_medians):
        """The fitted imputer as a JSON-ready dict (saved beside the splits and the scaler)."""
        month_medians = month_medians or {}
        return {
            "key": MONTH_KEY_COLUMN if month_medians else None,
            "global": medians,
            "per_calendar_month": {str(m): v for m, v in
                                   sorted(month_medians.get("calendar_month", {}).items())},
            "per_year_month": dict(sorted(month_medians.get("year_month", {}).items())),
        }

    @staticmethod
    def trim_train_tails(train, target_column):
        """Cut the delay tails of a NUMERIC target from the train rows only.

        The bounds are the FILTER_LOWER/UPPER_QUANTILE of the train split's own target, so
        test and hold-out never influence them and keep every real delay. Returns
        (train, info); `info` is None when the target is not numeric or the option is off.
        """
        if not (REGRESSION_FILTER_TRAIN_ONLY and target_column in REGRESSION_PROBLEM
                and target_column in train.columns):
            return train, None
        lower = float(train[target_column].quantile(FILTER_LOWER_QUANTILE))
        upper = float(train[target_column].quantile(FILTER_UPPER_QUANTILE))
        keep = train[target_column].between(lower, upper) | train[target_column].isna()
        info = {"target": target_column, "lower_bound": lower, "upper_bound": upper,
                "train_rows_before": int(len(train)), "train_rows_removed": int((~keep).sum())}
        print(f"      Train-only delay filter on '{target_column}': bounds {lower:.2f} .. {upper:.2f} min, "
              f"removed {info['train_rows_removed']:,} of {info['train_rows_before']:,} train rows "
              f"(test and hold-out untouched)")
        return train[keep], info

    @staticmethod
    def cancelled_fill_value(train, target_column):
        """Delay given to a train with no delay under REGRESSION_CANCELLED_POLICY = 'cap'.

        The FILTER_UPPER_QUANTILE bound of the real delays of the TRAIN rows (NaN are skipped),
        so test and hold-out never influence it. None when the policy is 'drop' or the target is
        not numeric.
        """
        if not (REGRESSION_CANCELLED_POLICY == 'cap' and target_column in REGRESSION_PROBLEM
                and target_column in train.columns):
            return None
        return float(train[target_column].quantile(FILTER_UPPER_QUANTILE))

    @staticmethod
    def drop_missing_target(frame, target_column):
        """'drop' policy: remove the rows of one split that have no delay (numeric target only).

        Applied after the day split, never before, so which days are train, test or hold-out
        does not depend on REGRESSION_CANCELLED_POLICY. Returns (frame, n_dropped).
        """
        if not (REGRESSION_CANCELLED_POLICY == 'drop' and target_column in REGRESSION_PROBLEM
                and target_column in frame.columns):
            return frame, 0
        keep = frame[target_column].notna()
        return frame[keep], int((~keep).sum())

    @staticmethod
    def fill_missing_target(frame, target_column, value):
        """Fill NaN targets of `frame` with `value`; returns (frame, flags).

        `flags` is a bool array, True where the target was filled in (row-aligned with
        `frame`), or None when `value` is None and the frame is returned untouched.
        """
        if value is None:
            return frame, None
        missing = frame[target_column].isna().to_numpy()
        frame = frame.copy()
        frame[target_column] = frame[target_column].fillna(value)
        return frame, missing

    def write_splits(self, df, output_dir, identifier, group_column=GROUP_KEY_COLUMN,
                     random_state=RANDOM_STATE, basename=None, target_column=None):
        """Split and write train/test parquet, dropping the grouping key.

        The operating day of each train row goes to `<basename>_train_groups.parquet`,
        row-aligned with the train file, so the trainers can run day-grouped CV
        without the key ever being a feature.

        Missing numeric features are filled with medians of the TRAIN split only
        (stage 12 no longer imputes), so no statistic sees val/test days. Weather
        columns use the median of their own monthly file, then of their calendar
        month, then the global train median (the all-year Air temperature median at
        OL is +2.9 C, January's is -6.3 C). The month key is dropped from the output.

        `basename` exists because split_dataset names its outputs after the source
        file it found, not after an identifier it chose.
        """
        os.makedirs(output_dir, exist_ok=True)
        basename = basename or f"merged_data_{identifier}"
        train, test = self.split_by_group(df, group_column=group_column,
                                          random_state=random_state)
        train, dropped_train = self.drop_missing_target(train, target_column)
        test, dropped_test = self.drop_missing_target(test, target_column)
        # The fill value uses the same train quantile as the tail cut, so it is taken before the cut.
        cap_value = self.cancelled_fill_value(train, target_column)
        # Before medians and before the day sidecar, so both describe the rows that are kept.
        train, outlier_filter = self.trim_train_tails(train, target_column)
        train, train_flags = self.fill_missing_target(train, target_column, cap_value)
        test, test_flags = self.fill_missing_target(test, target_column, cap_value)
        flags = {"train": train_flags, "test": test_flags}
        train_features = train.drop(columns=[group_column])
        medians = self.fit_train_medians(train_features)
        month_medians = self.fit_month_medians(train_features)
        if not month_medians:
            print(f"      ⚠️  No '{MONTH_KEY_COLUMN}' column: weather NaN get the global train median")
        written = {}
        frames = {}
        for name, part in (("train", train), ("test", test)):
            days = part[group_column].reset_index(drop=True)
            part = self.apply_medians(part.drop(columns=[group_column]), medians, month_medians)
            part = part.drop(columns=[MONTH_KEY_COLUMN], errors="ignore")
            path = os.path.join(output_dir, f"{basename}_{name}.parquet")
            part.to_parquet(path, index=False)
            written[name] = {"path": path, "rows": len(part)}
            frames[name] = part
            if name == "train":
                days.to_frame(group_column).to_parquet(groups_path(path), index=False)
            if flags[name] is not None:
                pd.DataFrame({"filled": flags[name]}).to_parquet(filled_path(path), index=False)
        return {"success": True, "files": written, "frames": frames,
                "medians": medians, "month_medians": month_medians,
                "train_outlier_filter": outlier_filter,
                "dropped_missing_target": {"train": dropped_train, "test": dropped_test},
                "cancelled_fill": None if cap_value is None else {
                    "value": cap_value, "filled_train": int(train_flags.sum()),
                    "filled_test": int(test_flags.sum())}}

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
            # _val and _holdout are outputs of a previous run, not sources. Without
            # this the second run of the stage sees three source files and aborts.
            split_suffixes = ('_train.parquet', '_val.parquet', '_train_groups.parquet',
                              '_train_filled.parquet', '_test_filled.parquet', '_holdout_filled.parquet',
                              '_test.parquet', '_holdout.parquet')
            source_parquet_files = [f for f in all_parquet_files if not f.endswith(split_suffixes)]

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
                    # Cancelled trains have no delay. The label of a numeric target cannot be
                    # NaN: the fit, the shap stage and every metric fail on it.
                    n_missing = int(df[target_column].isna().sum())
                    if n_missing:
                        print(f"      {n_missing:,} rows have no '{target_column}' (cancelled trains). "
                              f"REGRESSION_CANCELLED_POLICY='{REGRESSION_CANCELLED_POLICY}' is applied after the "
                              f"split, so the train/test days are the same under both policies")
            else:
                print(f"      No target column detected - using simple random split")

            # --- Out-of-time hold-out: carve out the most recent calendar year BEFORE
            # the regular train/test split, so it is never involved in training or
            # tuning — it exists only to measure performance on genuinely new/future
            # data. This also fixes temporal leakage in the train/test split itself,
            # since it now only ever draws from earlier years. ---
            holdout_df = None
            holdout_year = None
            dev_df = df
            if HOLDOUT_LAST_YEAR:
                if HOLDOUT_YEAR_COLUMN in df.columns:
                    year_series = df[HOLDOUT_YEAR_COLUMN]
                    holdout_year = int(year_series.max())
                    holdout_mask = year_series == holdout_year
                    holdout_df = df[holdout_mask].drop(columns=[HOLDOUT_YEAR_COLUMN])
                    dev_df = df[~holdout_mask].drop(columns=[HOLDOUT_YEAR_COLUMN])
                    print(f"      Out-of-time hold-out: year {holdout_year} -> {len(holdout_df):,} rows reserved "
                          f"(never trained/tuned on)")
                    print(f"      Remaining for train/test split: {len(dev_df):,} rows (years < {holdout_year})")
                    # Recompute stratify on dev_df only — hold-out rows must not affect the split
                    if is_classification and target_column:
                        stratify = dev_df[target_column]
                else:
                    print(f"      HOLDOUT_LAST_YEAR is enabled but '{HOLDOUT_YEAR_COLUMN}' column was not found "
                          f"— skipping hold-out (falling back to a 2-way train/test split)")

            # Generate output filenames
            base_name = csv_filename.replace('.parquet', '')
            train_filename = f"{base_name}_train.parquet"
            test_filename = f"{base_name}_test.parquet"
            holdout_filename = f"{base_name}_holdout.parquet" if holdout_df is not None else None

            split_output_dir = output_dir if output_dir is not None else merged_training_ready_dir
            os.makedirs(split_output_dir, exist_ok=True)

            # Files of an earlier run that this run may not rewrite: the old validation
            # split, and the day sidecar (stale once the rows of train change).
            for stale in (f"{base_name}_val.parquet", os.path.basename(groups_path(train_filename)),
                          f"{base_name}_train_filled.parquet", f"{base_name}_test_filled.parquet",
                          f"{base_name}_holdout_filled.parquet"):
                stale_path = os.path.join(split_output_dir, stale)
                if os.path.exists(stale_path):
                    os.remove(stale_path)

            # Day-grouped two-way split. A random row split leaks for any sequence
            # model — a test row's preceding trains land in the training set — and it
            # leaks weather and knock-on delay for the tabular models too, since one
            # operating day's rows are far from independent.
            grouped = GROUP_KEY_COLUMN in dev_df.columns
            cancelled_fill = None
            dropped_missing = {"train": 0, "test": 0, "holdout": 0}
            if grouped:
                split = self.write_splits(
                    dev_df, output_dir=split_output_dir, identifier=base_name,
                    group_column=GROUP_KEY_COLUMN, random_state=random_state,
                    basename=base_name, target_column=target_column,
                )
                train_outlier_filter = split["train_outlier_filter"]
                cancelled_fill = split["cancelled_fill"]
                dropped_missing.update(split["dropped_missing_target"])
                train_df = split["frames"]["train"]
                test_df = split["frames"]["test"]
                medians = split["medians"]
                month_medians = split["month_medians"]
                n_days = dev_df[GROUP_KEY_COLUMN].nunique()
                dev_df = dev_df.drop(columns=[GROUP_KEY_COLUMN, MONTH_KEY_COLUMN], errors="ignore")
                print(f"      Day-grouped split on '{GROUP_KEY_COLUMN}' ({n_days:,} operating days)")
            else:
                # No grouping key: the source predates the key, or is a fixture.
                # Fall back to the original stratified 2-way split rather than
                # inventing groups, but say so loudly — this path leaks.
                print(f"      ⚠️  '{GROUP_KEY_COLUMN}' not found — falling back to a random row split")
                train_df, test_df = train_test_split(
                    dev_df,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify
                )
                train_df, dropped_missing["train"] = self.drop_missing_target(train_df, target_column)
                test_df, dropped_missing["test"] = self.drop_missing_target(test_df, target_column)
                cap_value = self.cancelled_fill_value(train_df, target_column)
                train_df, train_outlier_filter = self.trim_train_tails(train_df, target_column)
                train_df, train_flags = self.fill_missing_target(train_df, target_column, cap_value)
                test_df, test_flags = self.fill_missing_target(test_df, target_column, cap_value)
                if cap_value is not None:
                    cancelled_fill = {"value": cap_value, "filled_train": int(train_flags.sum()),
                                      "filled_test": int(test_flags.sum())}
                medians = self.fit_train_medians(train_df)
                month_medians = self.fit_month_medians(train_df)
                train_df = self.apply_medians(train_df, medians, month_medians)
                test_df = self.apply_medians(test_df, medians, month_medians)
                train_df = train_df.drop(columns=[MONTH_KEY_COLUMN], errors="ignore")
                test_df = test_df.drop(columns=[MONTH_KEY_COLUMN], errors="ignore")
                train_df.to_parquet(os.path.join(split_output_dir, train_filename), index=False)
                test_df.to_parquet(os.path.join(split_output_dir, test_filename), index=False)
                for flag_name, flag_values in ((train_filename, train_flags), (test_filename, test_flags)):
                    if flag_values is not None:
                        pd.DataFrame({"filled": flag_values}).to_parquet(
                            filled_path(os.path.join(split_output_dir, flag_name)), index=False)

            print(f"      Split results:")
            print(f"        Train set: {len(train_df):,} samples ({(len(train_df) / len(dev_df)) * 100:.1f}%)")
            print(f"        Test set: {len(test_df):,} samples ({(len(test_df) / len(dev_df)) * 100:.1f}%)")

            with open(os.path.join(split_output_dir, "imputation_medians.json"), "w", encoding="utf-8") as f:
                json.dump(self.imputation_report(medians, month_medians), f, indent=2)
            print(f"      Imputed {len(medians)} numeric columns with train medians "
                  f"({len(month_medians.get('year_month', {}))} monthly files for weather columns)")

            print(f"      Saved train set to: {train_filename}")
            print(f"      Saved test set to: {test_filename}")

            if holdout_df is not None:
                holdout_path = os.path.join(split_output_dir, holdout_filename)
                if GROUP_KEY_COLUMN in holdout_df.columns:
                    holdout_df = holdout_df.drop(columns=[GROUP_KEY_COLUMN])
                holdout_df, dropped_missing["holdout"] = self.drop_missing_target(holdout_df, target_column)
                holdout_df, holdout_flags = self.fill_missing_target(
                    holdout_df, target_column, cancelled_fill["value"] if cancelled_fill else None)
                holdout_df = self.apply_medians(holdout_df, medians, month_medians)
                holdout_df = holdout_df.drop(columns=[MONTH_KEY_COLUMN], errors="ignore")
                holdout_df.to_parquet(holdout_path, index=False)
                if holdout_flags is not None:
                    pd.DataFrame({"filled": holdout_flags}).to_parquet(filled_path(holdout_path), index=False)
                    cancelled_fill["filled_holdout"] = int(holdout_flags.sum())
                print(f"      Saved hold-out set to: {holdout_filename}")
            
            if sum(dropped_missing.values()):
                print(f"      Dropped rows with no '{target_column}' (cancelled trains): "
                      f"{dropped_missing['train']:,} train, {dropped_missing['test']:,} test, "
                      f"{dropped_missing['holdout']:,} hold-out")
            if cancelled_fill:
                print(f"      Cancelled trains (no delay) filled with {cancelled_fill['value']:.2f} min, the train bound: "
                      f"{cancelled_fill['filled_train']:,} train, {cancelled_fill['filled_test']:,} test, "
                      f"{cancelled_fill.get('filled_holdout', 0):,} hold-out rows (flags saved beside the splits)")

            # Calculate class distribution after split for classification problems
            class_distribution_train = {}
            class_distribution_test = {}
            class_distribution_holdout = {}

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

                # Hold-out set distribution
                if holdout_df is not None:
                    holdout_value_counts = holdout_df[target_column].value_counts()
                    for class_value, count in holdout_value_counts.items():
                        class_distribution_holdout[class_value] = {
                            'count': int(count),
                            'percentage': (count / len(holdout_df)) * 100
                        }

            # Prepare result data
            result_data = {
                'filename': csv_filename,
                'original_rows': len(df),
                'dev_rows': len(dev_df),
                'holdout_rows': len(holdout_df) if holdout_df is not None else 0,
                'holdout_year': holdout_year,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'train_file': train_filename,
                'test_file': test_filename,
                'grouped_by': GROUP_KEY_COLUMN if grouped else None,
                'train_outlier_filter': train_outlier_filter,
                'cancelled_fill': cancelled_fill,
                'dropped_missing_target': dropped_missing,
                'holdout_file': holdout_filename,
                'test_size_requested': test_size,
                'test_size_actual': len(test_df) / len(dev_df),
                'target_column': target_column,
                'is_classification': is_classification
            }

            # Add class distribution data for classification problems
            if is_classification:
                result_data['class_distribution_before'] = class_distribution_before
                result_data['class_distribution_train'] = class_distribution_train
                result_data['class_distribution_test'] = class_distribution_test
                if holdout_df is not None:
                    result_data['class_distribution_holdout'] = class_distribution_holdout

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
                f.write(f"Stratify column: {target_column}\n")
                if holdout_df is not None:
                    f.write(f"Out-of-time hold-out year: {holdout_year} (never trained/tuned on)\n")
                f.write("\n")

                f.write(f"Dataset Information:\n")
                f.write(f"  Original rows: {result_data['original_rows']:,}\n")
                f.write(f"  Rows used for train/test split: {result_data['dev_rows']:,}\n")
                f.write(f"  Columns: {df.shape[1]}\n")
                f.write(f"  Problem type: {'Classification' if is_classification else 'Regression'}\n\n")

                f.write(f"Split Results:\n")
                f.write(f"  Grouped by: {result_data['grouped_by'] or 'NONE (random row split)'}\n")
                if cancelled_fill:
                    f.write(f"  Cancelled trains filled with {cancelled_fill['value']:.2f} min (train bound): "
                            f"{cancelled_fill['filled_train']:,} train, {cancelled_fill['filled_test']:,} test, "
                            f"{cancelled_fill.get('filled_holdout', 0):,} hold-out rows\n")
                if train_outlier_filter:
                    f.write(f"  Train-only delay filter on {train_outlier_filter['target']}: "
                            f"{train_outlier_filter['lower_bound']:.2f} .. {train_outlier_filter['upper_bound']:.2f} min, "
                            f"removed {train_outlier_filter['train_rows_removed']:,} of "
                            f"{train_outlier_filter['train_rows_before']:,} train rows "
                            f"(test and hold-out untouched)\n")
                f.write(f"  Train set: {result_data['train_rows']:,} samples ({(result_data['train_rows'] / result_data['dev_rows']) * 100:.1f}%)\n")
                f.write(f"  Test set: {result_data['test_rows']:,} samples ({(result_data['test_rows'] / result_data['dev_rows']) * 100:.1f}%)\n")
                f.write(f"  Actual test size: {result_data['test_size_actual']:.3f}\n")
                if holdout_df is not None:
                    f.write(f"  Hold-out set (year {holdout_year}): {result_data['holdout_rows']:,} samples "
                            f"({(result_data['holdout_rows'] / result_data['original_rows']) * 100:.1f}% of original)\n")
                f.write("\n")

                f.write(f"Output Files:\n")
                f.write(f"  Train: {result_data['train_file']}\n")
                f.write(f"  Test: {result_data['test_file']}\n")
                if holdout_filename:
                    f.write(f"  Hold-out: {holdout_filename}\n")
                f.write("\n")

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

                    # Hold-out set
                    if holdout_df is not None:
                        f.write(f"\n  Hold-out Set - year {holdout_year} ({result_data['holdout_rows']:,} samples):\n")
                        for class_value, stats in class_distribution_holdout.items():
                            f.write(f"    Class {class_value}: {stats['count']:,} samples ({stats['percentage']:.2f}%)\n")

            print(f"    split_dataset: Summary saved to {summary_filename}")

            # Return success result
            result = {
                "success": True,
                "processed_files": 1,
                "total_train_rows": len(train_df),
                "total_test_rows": len(test_df),
                "total_holdout_rows": len(holdout_df) if holdout_df is not None else 0,
                "holdout_year": holdout_year,
                "test_size": test_size,
                "split_details": [result_data],
                "summary_path": summary_path,
                "message": f"Successfully split dataset {csv_filename} into train/test"
                           f"{'/holdout' if holdout_df is not None else ''} sets"
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
            holdout_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_holdout.parquet")

            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)
            # Out-of-time hold-out is optional (HOLDOUT_LAST_YEAR toggle / missing year column
            # upstream) — scale it the same way as test if present, but never error if absent.
            holdout_files = glob.glob(holdout_pattern)
            
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

            holdout_file_path = holdout_files[0] if holdout_files else None
            holdout_filename = os.path.basename(holdout_file_path) if holdout_file_path else None

            print(f"    scale_weather_features: Processing file pair:")
            print(f"      Train: {train_filename}")
            print(f"      Test: {test_filename}")
            if holdout_filename:
                print(f"      Hold-out: {holdout_filename}")

            # Load datasets
            try:
                train_df = pd.read_parquet(train_file_path)
                test_df = pd.read_parquet(test_file_path)
                holdout_df = pd.read_parquet(holdout_file_path) if holdout_file_path else None
            except Exception as e:
                error_msg = f"Error loading datasets: {str(e)}"
                print(f"    scale_weather_features: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "processed_files": 0
                }

            # The day sidecar from stage 503 travels with the train rows (scaling keeps
            # their order) so the trainers can run day-grouped CV from this folder. A
            # stale copy is removed first; without a source (resampled train, or an
            # ungrouped fallback split) the trainers fall back to row-wise CV, loudly.
            sidecar_src = groups_path(train_file_path)
            sidecar_dst = groups_path(os.path.join(scaled_training_ready_dir, train_filename))
            same_folder = os.path.abspath(sidecar_src) == os.path.abspath(sidecar_dst)
            if not same_folder and os.path.exists(sidecar_dst):
                os.remove(sidecar_dst)
            if not same_folder and os.path.exists(sidecar_src):
                shutil.copy2(sidecar_src, sidecar_dst)
                print(f"    scale_weather_features: Copied operating-day groups to: {os.path.basename(sidecar_dst)}")

            # Flags of rows whose delay was filled in (REGRESSION_CANCELLED_POLICY = 'cap').
            for split_src, split_name in ((train_file_path, train_filename), (test_file_path, test_filename),
                                          (holdout_file_path, holdout_filename)):
                if not split_src:
                    continue
                flags_src = filled_path(split_src)
                flags_dst = filled_path(os.path.join(scaled_training_ready_dir, split_name))
                if os.path.abspath(flags_src) == os.path.abspath(flags_dst):
                    continue
                if os.path.exists(flags_dst):
                    os.remove(flags_dst)
                if os.path.exists(flags_src):
                    shutil.copy2(flags_src, flags_dst)
                    print(f"    scale_weather_features: Copied filled-row flags to: {os.path.basename(flags_dst)}")

            print(f"    scale_weather_features: Train dataset shape: {train_df.shape}")
            print(f"    scale_weather_features: Test dataset shape: {test_df.shape}")
            if holdout_df is not None:
                print(f"    scale_weather_features: Hold-out dataset shape: {holdout_df.shape}")
            
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
                scaled_holdout_filename = holdout_filename  # already .parquet, or None

                scaled_train_path = os.path.join(scaled_training_ready_dir, scaled_train_filename)
                scaled_test_path = os.path.join(scaled_training_ready_dir, scaled_test_filename)

                try:
                    # Copy train file
                    shutil.copy2(train_file_path, scaled_train_path)
                    print(f"    scale_weather_features: ✓ Copied train file to: {scaled_train_filename}")

                    # Copy test file
                    shutil.copy2(test_file_path, scaled_test_path)
                    print(f"    scale_weather_features: ✓ Copied test file to: {scaled_test_filename}")

                    # Copy hold-out file, if present
                    if holdout_file_path:
                        scaled_holdout_path = os.path.join(scaled_training_ready_dir, scaled_holdout_filename)
                        shutil.copy2(holdout_file_path, scaled_holdout_path)
                        print(f"    scale_weather_features: ✓ Copied hold-out file to: {scaled_holdout_filename}")

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
                    "original_holdout_file": holdout_filename,
                    "scaled_train_file": scaled_train_filename,
                    "scaled_test_file": scaled_test_filename,
                    "scaled_holdout_file": scaled_holdout_filename,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "holdout_rows": len(holdout_df) if holdout_df is not None else 0,
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
                    "holdout_rows": len(holdout_df) if holdout_df is not None else 0,
                    "weather_features_scaled": [],  # Empty list
                    "scaling_summary": no_scaling_result,
                    "output_directory": scaled_training_ready_dir,
                    "message": f"No weather features found. Files copied as-is: {train_filename}, {test_filename}"
                               + (f", {holdout_filename}" if holdout_filename else ""),
                    "scaling_applied": False  # Flag to indicate no scaling was applied
                }
            
            # EXISTING LOGIC: Continue with normal scaling when weather features are found
            print(f"    scale_weather_features: Found {len(available_weather_features)} weather features to scale")
            print(f"      Weather features: {available_weather_features}")
            
            # Extract weather features for scaling
            train_weather_features = train_df[available_weather_features].copy()
            test_weather_features = test_df[available_weather_features].copy()
            holdout_weather_features = (
                holdout_df[available_weather_features].copy() if holdout_df is not None else None
            )

            # Zero-inflated/right-skewed features get a log1p pre-transform (deterministic,
            # not fit on data) before RobustScaler. These features are physically
            # non-negative, but raw FMI data uses negative sentinels (e.g. -1) for
            # "not applicable" readings; log1p(-1) is -inf, so clip to 0 first.
            if skewed_cols:
                print(f"    scale_weather_features: Applying log1p to {len(skewed_cols)} skewed feature(s): {skewed_cols}")
                train_weather_features[skewed_cols] = np.log1p(train_weather_features[skewed_cols].clip(lower=0))
                test_weather_features[skewed_cols] = np.log1p(test_weather_features[skewed_cols].clip(lower=0))
                if holdout_weather_features is not None:
                    holdout_weather_features[skewed_cols] = np.log1p(holdout_weather_features[skewed_cols].clip(lower=0))

            # Create and fit scaler on training data only
            scaler = RobustScaler()
            scaler.fit(train_weather_features)

            # Save the fitted scaler (plus which columns need log1p) so it can be
            # applied to new data at inference time
            scaler_filename = "weather_scaler.joblib"
            scaler_path = os.path.join(scaled_training_ready_dir, scaler_filename)

            # The imputer was fitted at the split (stage 503); keep it beside the
            # scaler, which must run after it on new data.
            imputer = None
            imputation_path = os.path.join(
                self.project_root, SPLIT_DATASET_OUTPUT_FOLDER, "imputation_medians.json")
            if os.path.isfile(imputation_path):
                with open(imputation_path, encoding="utf-8") as f:
                    imputer = json.load(f)
                shutil.copyfile(
                    imputation_path,
                    os.path.join(scaled_training_ready_dir, "imputation_medians.json"))
                print(f"    scale_weather_features: ✓ Imputation medians copied beside the scaler")
            else:
                print(f"    scale_weather_features: ⚠️  {imputation_path} not found; "
                      f"the scaler artifact carries no imputer")

            joblib.dump(
                {
                    "scaler": scaler,
                    "weather_features": available_weather_features,
                    "skewed_features": skewed_cols,
                    "imputer": imputer,
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

            # Transform the out-of-time hold-out with the SAME fitted scaler (never
            # re-fit) — it must be scaled exactly like production/new data would be.
            holdout_scaled_df = None
            if holdout_df is not None:
                holdout_weather_scaled = scaler.transform(holdout_weather_features)
                holdout_scaled_df = holdout_df.copy()
                holdout_scaled_df[available_weather_features] = holdout_weather_scaled

            # Generate output filenames
            scaled_train_filename = train_filename  # already .parquet
            scaled_test_filename = test_filename    # already .parquet
            scaled_holdout_filename = holdout_filename  # already .parquet, or None

            scaled_train_path = os.path.join(scaled_training_ready_dir, scaled_train_filename)
            scaled_test_path = os.path.join(scaled_training_ready_dir, scaled_test_filename)

            # Save scaled datasets
            train_scaled_df.to_parquet(scaled_train_path, index=False)
            test_scaled_df.to_parquet(scaled_test_path, index=False)

            print(f"    scale_weather_features: ✓ Scaled train dataset saved to: {scaled_train_filename}")
            print(f"    scale_weather_features: ✓ Scaled test dataset saved to: {scaled_test_filename}")

            if holdout_scaled_df is not None:
                scaled_holdout_path = os.path.join(scaled_training_ready_dir, scaled_holdout_filename)
                holdout_scaled_df.to_parquet(scaled_holdout_path, index=False)
                print(f"    scale_weather_features: ✓ Scaled hold-out dataset saved to: {scaled_holdout_filename}")

            # Prepare scaling result
            scaling_result = {
                "original_train_file": train_filename,
                "original_test_file": test_filename,
                "original_holdout_file": holdout_filename,
                "scaled_train_file": scaled_train_filename,
                "scaled_test_file": scaled_test_filename,
                "scaled_holdout_file": scaled_holdout_filename,
                "scaler_file": scaler_filename,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "holdout_rows": len(holdout_df) if holdout_df is not None else 0,
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
                if holdout_filename:
                    f.write(f"  Hold-out file: {holdout_filename} ({len(holdout_df):,} rows)\n")
                f.write(f"  Scaled train file: {scaled_train_filename}\n")
                f.write(f"  Scaled test file: {scaled_test_filename}\n")
                if scaled_holdout_filename:
                    f.write(f"  Scaled hold-out file: {scaled_holdout_filename}\n")
                f.write("\n")
                
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
                "holdout_rows": len(holdout_df) if holdout_df is not None else 0,
                "weather_features_scaled": available_weather_features,
                "skewed_features_log1p": skewed_cols,
                "scaler_path": scaler_path,
                "scaling_summary": scaling_result,
                "output_directory": scaled_training_ready_dir,
                "message": f"Successfully scaled weather features for train/test"
                           f"{'/holdout' if holdout_df is not None else ''} file"
                           f"{'s' if holdout_df is not None else ' pair'}: {train_filename}, {test_filename}"
                           + (f", {holdout_filename}" if holdout_filename else ""),
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

    def _save_figure_png_and_pdf(self, fig, png_path, pdf_path):
        """
        Save a matplotlib figure to both PNG and PDF, then close it.

        matplotlib's PDF backend triggers a Pillow API deprecation warning
        ('mode' parameter is deprecated) that matplotlib hasn't yet updated
        for. Suppress this known noisy warning without silencing legitimate
        ones.
        """
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"'mode' parameter is deprecated",
                category=DeprecationWarning,
                module=r"matplotlib\.backends\.backend_pdf",
            )
            fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)

    def _save_confusion_matrix_plot(self, cm, class_labels, title, png_path):
        """
        Render a confusion matrix (list of lists or ndarray) as an annotated
        heatmap PNG — used for the best model's confusion matrix in every
        train_* method (raw classes for classification, on-time/delayed for
        the regression branch's binarized threshold).
        """
        cm_arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_arr, cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm_arr.shape[1]),
            yticks=np.arange(cm_arr.shape[0]),
            xticklabels=class_labels,
            yticklabels=class_labels,
            ylabel='True label',
            xlabel='Predicted label',
            title=title
        )
        thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 0.5
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, format(cm_arr[i, j], 'd'), ha='center', va='center',
                        color='white' if cm_arr[i, j] > thresh else 'black')
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def shap_correlation_analysis(self, data_dir=None):
        """
        Diagnostic-only stage: fits a plain default XGBoost model, computes
        SHAP values on the training data, and reports how strongly each pair
        of features' SHAP contributions correlate with each other.

        Nothing is dropped or modified based on the result — this stage only
        reports which features carry redundant explanatory signal via a full
        correlation matrix CSV plus three chart pairs (PNG + PDF each): the
        full feature x feature heatmap, a bar chart of the top correlated
        feature pairs, and a smaller annotated sub-heatmap of the features
        most involved in strong correlations.
        """
        try:
            print(f"    shap_correlation_analysis: Starting SHAP correlation analysis...")

            output_dir = os.path.join(self.project_root, SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            print(f"    shap_correlation_analysis: Output directory: {output_dir}")

            if data_dir is None:
                data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)

            train_pattern = os.path.join(data_dir, "merged_data_*_train.parquet")
            train_files = glob.glob(train_pattern)

            if not train_files:
                error_msg = f"Training file not found in {data_dir}. Train files: {len(train_files)}"
                print(f"    shap_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            if len(train_files) != 1:
                error_msg = f"Expected exactly one train file. Found {len(train_files)} train files"
                print(f"    shap_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            train_file = train_files[0]
            print(f"    shap_correlation_analysis: Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM

            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    shap_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    shap_correlation_analysis: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            problem_type = "classification" if is_classification else "regression"
            print(f"    shap_correlation_analysis: Detected {problem_type} problem for target '{target_feature}'")

            feature_columns = [col for col in train_df.columns if col != target_feature]
            X_train = train_df[feature_columns]
            y_train = train_df[target_feature]

            print(f"    shap_correlation_analysis: Dataset info - Train: {X_train.shape}")
            print(f"    shap_correlation_analysis: Fitting plain default XGBoost ({problem_type})...")

            if is_classification:
                model = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=SEARCH_N_JOBS)
            else:
                model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=SEARCH_N_JOBS)

            model.fit(X_train, y_train)

            print(f"    shap_correlation_analysis: Computing SHAP values...")
            explainer = shap.TreeExplainer(model)
            raw_shap_values = explainer.shap_values(X_train)

            if isinstance(raw_shap_values, list):
                shap_array = raw_shap_values[-1]
            elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
                shap_array = raw_shap_values[:, :, -1]
            else:
                shap_array = raw_shap_values

            print(f"    shap_correlation_analysis: Computing SHAP correlation matrix...")
            corr_matrix = np.corrcoef(shap_array, rowvar=False)
            corr_df = pd.DataFrame(corr_matrix, index=feature_columns, columns=feature_columns)

            csv_path = os.path.join(output_dir, "shap_correlation_matrix.csv")
            corr_df.to_csv(csv_path)
            print(f"    shap_correlation_analysis: Saved correlation matrix to {csv_path}")

            num_features = len(feature_columns)

            # Deduplicated, off-diagonal feature pairs ranked by |correlation|.
            # Reused for both the console summary (most_correlated_pair) and
            # the top-pairs bar chart, so the "most correlated" answer is
            # always consistent between them.
            feature_names_arr = np.array(feature_columns)
            triu_i, triu_j = np.triu_indices(num_features, k=1)
            pairs_df = pd.DataFrame({
                "feature_1": feature_names_arr[triu_i],
                "feature_2": feature_names_arr[triu_j],
                "correlation": corr_df.values[triu_i, triu_j],
            })
            pairs_df = pairs_df.reindex(
                pairs_df["correlation"].abs().sort_values(ascending=False, na_position="last").index
            ).reset_index(drop=True)

            most_correlated_pair = None
            if len(pairs_df) > 0 and not pd.isna(pairs_df.loc[0, "correlation"]):
                top_row = pairs_df.loc[0]
                most_correlated_pair = (
                    top_row["feature_1"],
                    top_row["feature_2"],
                    float(top_row["correlation"]),
                )

            # --- Chart 1: full feature x feature heatmap ---
            fig_size = max(10, num_features * 0.15)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            im = ax.imshow(corr_df.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(num_features))
            ax.set_xticklabels(feature_columns, rotation=90, fontsize=6)
            ax.set_yticks(range(num_features))
            ax.set_yticklabels(feature_columns, fontsize=6)
            fig.colorbar(im, ax=ax, label='SHAP value correlation')
            ax.set_title('SHAP Value Correlation Between Features')
            fig.tight_layout()

            png_path = os.path.join(output_dir, "shap_correlation_heatmap.png")
            pdf_path = os.path.join(output_dir, "shap_correlation_heatmap.pdf")
            self._save_figure_png_and_pdf(fig, png_path, pdf_path)
            print(f"    shap_correlation_analysis: Saved full heatmap to {png_path} and {pdf_path}")

            # --- Chart 2: top-N most correlated feature pairs (bar chart) ---
            top_pairs_df = pairs_df.dropna(subset=["correlation"]).head(
                min(SHAP_CORRELATION_TOP_N_PAIRS, len(pairs_df))
            )
            pair_labels = [
                f"{row.feature_1} <-> {row.feature_2}" for row in top_pairs_df.itertuples()
            ]
            pair_values = top_pairs_df["correlation"].to_numpy()

            fig2, ax2 = plt.subplots(figsize=(10, max(6, len(pair_labels) * 0.35)))
            if len(pair_labels) > 0:
                y_pos = np.arange(len(pair_labels))
                bar_colors = plt.cm.coolwarm((pair_values + 1) / 2)
                ax2.barh(y_pos, pair_values, color=bar_colors)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(pair_labels, fontsize=8)
                ax2.invert_yaxis()
                ax2.axvline(0, color='black', linewidth=0.8)
            ax2.set_xlim(-1, 1)
            ax2.set_xlabel('SHAP value correlation')
            ax2.set_title(f'Top {len(pair_labels)} Most Correlated Feature Pairs (by SHAP value)')
            fig2.tight_layout()

            top_pairs_png_path = os.path.join(output_dir, "shap_correlation_top_pairs.png")
            top_pairs_pdf_path = os.path.join(output_dir, "shap_correlation_top_pairs.pdf")
            self._save_figure_png_and_pdf(fig2, top_pairs_png_path, top_pairs_pdf_path)
            print(f"    shap_correlation_analysis: Saved top-pairs chart to {top_pairs_png_path} and {top_pairs_pdf_path}")

            # --- Chart 3: sub-heatmap of the features most involved in strong correlations ---
            corr_no_diag = corr_df.copy()
            np.fill_diagonal(corr_no_diag.values, np.nan)
            max_abs_corr_per_feature = corr_no_diag.abs().max(axis=1)
            top_k = min(SHAP_CORRELATION_TOP_N_FEATURES, num_features)
            top_feature_names = max_abs_corr_per_feature.sort_values(
                ascending=False, na_position="last"
            ).head(top_k).index.tolist()
            sub_corr_df = corr_df.loc[top_feature_names, top_feature_names]

            fig3, ax3 = plt.subplots(figsize=(max(8, top_k * 0.6), max(8, top_k * 0.6)))
            im3 = ax3.imshow(sub_corr_df.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax3.set_xticks(range(top_k))
            ax3.set_xticklabels(top_feature_names, rotation=90, fontsize=8)
            ax3.set_yticks(range(top_k))
            ax3.set_yticklabels(top_feature_names, fontsize=8)
            for i in range(top_k):
                for j in range(top_k):
                    val = sub_corr_df.values[i, j]
                    if not np.isnan(val):
                        ax3.text(
                            j, i, f"{val:.2f}", ha='center', va='center', fontsize=6,
                            color='white' if abs(val) > 0.5 else 'black',
                        )
            fig3.colorbar(im3, ax=ax3, label='SHAP value correlation')
            ax3.set_title(f'Top {top_k} Features by Strongest SHAP Correlation')
            fig3.tight_layout()

            top_features_png_path = os.path.join(output_dir, "shap_correlation_top_features_heatmap.png")
            top_features_pdf_path = os.path.join(output_dir, "shap_correlation_top_features_heatmap.pdf")
            self._save_figure_png_and_pdf(fig3, top_features_png_path, top_features_pdf_path)
            print(f"    shap_correlation_analysis: Saved top-features heatmap to {top_features_png_path} and {top_features_pdf_path}")

            print(f"    shap_correlation_analysis: Analysis completed successfully!")

            return {
                "success": True,
                "output_directory": output_dir,
                "num_features": num_features,
                "target_feature": target_feature,
                "problem_type": problem_type,
                "most_correlated_pair": most_correlated_pair,
                "csv_path": csv_path,
                "png_path": png_path,
                "pdf_path": pdf_path,
                "top_pairs_png_path": top_pairs_png_path,
                "top_pairs_pdf_path": top_pairs_pdf_path,
                "top_features_heatmap_png_path": top_features_png_path,
                "top_features_heatmap_pdf_path": top_features_pdf_path,
            }

        except Exception as e:
            error_msg = f"shap_correlation_analysis failed: {str(e)}"
            print(f"    shap_correlation_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def explore_training_data(self, data_dir=None):
        """
        Diagnostic-only stage 701: charts and tables that describe the data itself. No model.

        Reads the stage 502 file (every row in physical units, with its year) and writes to
        DATA_EXPLORATION_OUTPUT_FOLDER up to 13 charts (target distribution and drift over
        time; delay by hour, weekday, month and wind direction; delay against each weather
        variable; feature distributions; Spearman heatmap and dendrogram; how strongly each
        variable is tied to the target, per year; which variables look different in the
        hold-out year), the tables behind them, a summary.json and suggested_features.txt
        (Spearman clusters on the development years, --columns-file format). Hour, month,
        weekday, day of month and wind direction are recovered from their sin/cos pairs so
        the charts use real units. Nothing is dropped or modified; no model is fitted.
        """
        label = "explore_training_data"
        try:
            print(f"    {label}: Starting data exploration (no model is fitted)...")
            output_dir = os.path.join(self.project_root, DATA_EXPLORATION_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            if data_dir is None:
                data_dir = os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)

            files = glob.glob(os.path.join(data_dir, "merged_data_*.parquet"))
            if len(files) != 1:
                error_msg = (f"Expected exactly one stage 502 file in {data_dir}. Found {len(files)} "
                             f"(run select_training_cols first)")
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}

            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            if not (is_classification or target_feature in REGRESSION_PROBLEM):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}
            problem_type = "classification" if is_classification else "regression"

            df = pd.read_parquet(files[0])
            if target_feature not in df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}
            rows_in_file = len(df)
            df = df[df[target_feature].notna()].reset_index(drop=True)
            if df.empty:
                error_msg = f"Every value of '{target_feature}' is missing"
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}

            service = {HOLDOUT_YEAR_COLUMN, GROUP_KEY_COLUMN, MONTH_KEY_COLUMN}
            feature_columns = [c for c in df.columns if c != target_feature and c not in service]
            readable, meta = recover_readable_variables(df[feature_columns])
            names = variable_names(meta)
            if not names:
                error_msg = "No numeric feature with any variation to describe"
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}

            y = df[target_feature].astype(float).to_numpy()
            years = df[HOLDOUT_YEAR_COLUMN] if HOLDOUT_YEAR_COLUMN in df.columns else None
            hold_year = int(years.max()) if (HOLDOUT_LAST_YEAR and years is not None and years.nunique() > 1) else None
            hold_mask = (years == hold_year).to_numpy() if hold_year is not None else np.zeros(len(df), dtype=bool)
            dev_mask = ~hold_mask
            print(f"    {label}: {len(df):,} rows ({rows_in_file - len(df):,} without a target dropped), "
                  f"{len(names)} variables, hold-out year: {hold_year if hold_year is not None else 'none'}")

            codes = bin_codes(readable, meta, EXPLORATION_N_BINS, EXPLORATION_WIND_SECTORS)
            assoc = association_table(readable, meta, codes, y)
            assoc_by_year = association_by_year(codes, y, years) if (years is not None and years.nunique() > 1) else None
            drift = drift_table(readable, meta, hold_mask) if hold_year is not None else None
            numeric_features = df[feature_columns].select_dtypes("number")
            corr_ordered, _ = spearman_matrix(numeric_features)
            # The clusters, and the list suggested from them, look at the development years only.
            clustering = spearman_clusters(numeric_features[dev_mask], protected_columns(feature_groups(numeric_features.columns)),
                                           SPEARMAN_CLUSTER_ABS_CORR)

            # --- tables ---
            data_summary(df[feature_columns + [target_feature]]).to_csv(os.path.join(output_dir, "data_summary.csv"), index=False)
            assoc.to_csv(os.path.join(output_dir, "association_with_target.csv"), index=False)
            if assoc_by_year is not None:
                assoc_by_year.to_csv(os.path.join(output_dir, "association_by_year.csv"), index_label="variable")
            if drift is not None:
                drift.to_csv(os.path.join(output_dir, "drift_holdout.csv"), index=False)
            corr_ordered.to_csv(os.path.join(output_dir, "spearman_correlation_matrix.csv"))
            clustering["clusters"].to_csv(os.path.join(output_dir, "spearman_clusters.csv"), index=False)
            suggested_path = os.path.join(output_dir, "suggested_features.txt")
            with open(suggested_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("# Suggested non-redundant features from stage 701 (Spearman clusters, development years).\n")
                f.write(f"# One column per cluster of continuous columns with |rho| >= {SPEARMAN_CLUSTER_ABS_CORR};\n")
                f.write("# sin/cos pairs and one-hot families are kept whole. Pass it with --columns-file.\n")
                f.write(f"{target_feature}\n")
                for col in clustering["suggested"]:
                    f.write(f"{col}\n")

            # --- charts ---
            charts_written, charts_skipped = [], {}

            def emit(stem, figure, why):
                if figure is None:
                    charts_skipped[stem] = why
                    return
                self._save_figure_png_and_pdf(figure, os.path.join(output_dir, f"{stem}.png"),
                                              os.path.join(output_dir, f"{stem}.pdf"))
                charts_written.append(stem)

            ylabel = target_feature if is_classification else "delay (minutes)"
            emit("01_target_distribution", xplots.plot_target_distribution(
                y[dev_mask], y[hold_mask], is_classification, DELAY_THRESHOLD_MINUTES, ylabel), "no data")
            if MONTH_KEY_COLUMN in df.columns:
                emit("02_target_over_time", xplots.plot_target_over_time(
                    monthly_series(df[MONTH_KEY_COLUMN], y, is_classification, DELAY_THRESHOLD_MINUTES),
                    is_classification, DELAY_THRESHOLD_MINUTES, hold_year, ylabel), "fewer than 3 months")
            else:
                charts_skipped["02_target_over_time"] = f"no {MONTH_KEY_COLUMN} column"

            thr = DELAY_THRESHOLD_MINUTES
            if "hour" in names:
                prof = profile_by(codes["hour"], y, is_classification, thr)
                emit("03_delay_by_hour", xplots.plot_profile_line(
                    prof, [f"{int(k):02d}" for k in prof["key"]], "hour of day (scheduled arrival)", "Delay by hour of day",
                    "How to read: top = how often trains scheduled in that hour are late (line, with a 95% band) over how many trains "
                    "there are (grey bars). Bottom = how big the delays are: median, 75th and 90th percentile in minutes.",
                    is_classification, ylabel, thr), "no hour variable")
            else:
                charts_skipped["03_delay_by_hour"] = "no hour variable"
            if "hour" in names and "weekday" in names:
                cell = pd.DataFrame({"d": codes["weekday"], "h": codes["hour"], "y": y}).dropna()
                cell["hit"] = cell["y"] if is_classification else (cell["y"] > thr).astype(float)
                agg = cell.groupby(["d", "h"])["hit"]
                share = agg.mean().unstack("h").reindex(index=range(1, 8), columns=range(24))
                counts = agg.size().unstack("h").reindex(index=range(1, 8), columns=range(24)).fillna(0)
                emit("04_delay_by_weekday_and_hour", xplots.plot_weekday_hour_heatmap(
                    share, counts, EXPLORATION_MIN_CELL_COUNT, thr, is_classification), "no hour/weekday")
            else:
                charts_skipped["04_delay_by_weekday_and_hour"] = "no hour/weekday variable"
            if "month" in names:
                prof = profile_by(codes["month"], y, is_classification, thr)
                emit("05_delay_by_month", xplots.plot_profile_line(
                    prof, [MONTH_NAMES[int(k) - 1] for k in prof["key"]], "calendar month", "Delay by month (seasonality)",
                    "How to read: top = how often trains are late in each calendar month (line, with a 95% band) over the number of "
                    "trains (grey bars). Bottom = how big the delays are: median, 75th and 90th percentile in minutes.",
                    is_classification, ylabel, thr), "no month variable")
            else:
                charts_skipped["05_delay_by_month"] = "no month variable"
            if WIND_COLUMN in names:
                prof = profile_by(codes[WIND_COLUMN], y, is_classification, thr)
                calm_row = prof[prof["key"] < 0]
                calm = {"n": float(calm_row["n"].iloc[0]), "share": float(calm_row["share"].iloc[0])} if len(calm_row) else None
                emit("06_delay_by_wind_direction", xplots.plot_wind_polar(
                    prof[prof["key"] >= 0], wind_sector_labels(EXPLORATION_WIND_SECTORS), calm, thr, is_classification),
                     "no sectors")
            else:
                charts_skipped["06_delay_by_wind_direction"] = "no wind direction variable"

            continuous = [n for n in names if meta[n]["kind"] == "continuous"]
            rho = assoc.set_index("variable")["spearman_rho"]
            panel_names = sorted(continuous, key=lambda n: -abs(rho.get(n, 0) if pd.notna(rho.get(n, np.nan)) else 0))[:EXPLORATION_MAX_PANELS]
            panels = []
            for n in panel_names:
                prof = profile_by(codes[n], y, is_classification, thr)
                x_mid = readable[n].groupby(codes[n]).median()
                prof["x"] = prof["key"].map(x_mid)
                panels.append({"name": n, "x": prof["x"].to_numpy(), "share": prof["share"].to_numpy(),
                               "low": prof["share_low"].to_numpy(), "high": prof["share_high"].to_numpy(),
                               "rho": float(rho.get(n, np.nan))})
            emit("07_delay_vs_weather", xplots.plot_binned_panels(panels, thr, is_classification), "no continuous variable")

            items = []
            for n in continuous[:EXPLORATION_MAX_PANELS]:
                col = readable[n]
                notes = []
                if (col == -1).mean() >= 0.01:
                    notes.append(f"-1: {(col == -1).mean():.0%}")
                if (col == 0).mean() >= 0.30:
                    notes.append(f"0: {(col == 0).mean():.0%}")
                items.append({"name": n, "dev": col[dev_mask].to_numpy(), "hold": col[hold_mask].to_numpy(), "notes": notes})
            emit("08_feature_distributions", xplots.plot_distribution_panels(items), "no continuous variable")
            emit("09_spearman_heatmap", xplots.plot_spearman_heatmap(corr_ordered, EXPLORATION_ANNOTATE_MAX), "fewer than 2 variables")
            emit("10_spearman_dendrogram", xplots.plot_dendrogram(clustering, SPEARMAN_CLUSTER_ABS_CORR),
                 "fewer than 2 continuous variables")
            emit("11_association_with_target", xplots.plot_association(assoc, ylabel), "no association could be computed")
            emit("12_association_by_year", xplots.plot_association_by_year(assoc_by_year, hold_year), "fewer than 2 years")
            emit("13_drift_holdout_vs_development", xplots.plot_drift(drift), "no hold-out year")

            summary = {
                "source_file": os.path.basename(files[0]), "target_feature": target_feature, "problem_type": problem_type,
                "model_fitted": False, "rows_in_file": int(rows_in_file), "rows_used": int(len(df)),
                "rows_dropped_missing_target": int(rows_in_file - len(df)),
                "years": sorted(int(v) for v in years.unique()) if years is not None else [],
                "hold_out_year": hold_year, "n_variables": len(names),
                "variables": {n: meta[n]["kind"] for n in names},
                "recovered_from_sin_cos": {n: meta[n]["sources"] for n in names if meta[n]["kind"] == "circular"},
                "constant_columns": meta["__constant__"],
                "charts_written": charts_written, "charts_skipped": charts_skipped,
                "top_association_with_target": [
                    {"variable": r.variable, "eta2": float(r.eta2)} for r in assoc.dropna(subset=["eta2"]).head(5).itertuples()],
                "top_drift": [{"variable": r.variable, "ks": float(r.ks)} for r in drift.head(5).itertuples()] if drift is not None else [],
                "spearman_clusters_with_several_members": (
                    clustering["clusters"].query("cluster_size > 1").groupby("cluster")["feature"].apply(list).tolist()
                    if len(clustering["clusters"]) else []),
                "suggested_feature_count": len(clustering["suggested"]),
                "settings": {"n_bins": EXPLORATION_N_BINS, "wind_sectors": EXPLORATION_WIND_SECTORS,
                             "spearman_cluster_abs_corr": SPEARMAN_CLUSTER_ABS_CORR,
                             "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES},
            }
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=float)

            print(f"    {label}: {len(charts_written)} charts written, {len(charts_skipped)} skipped")
            return {
                "success": True,
                "output_directory": output_dir,
                "n_rows": int(len(df)),
                "n_variables": len(names),
                "n_charts": len(charts_written),
                "charts": charts_written,
                "charts_skipped": charts_skipped,
                "target_feature": target_feature,
                "problem_type": problem_type,
                "hold_out_year": hold_year,
                "suggested_feature_count": len(clustering["suggested"]),
                "suggested_features_path": suggested_path,
                "summary_path": summary_path,
            }

        except Exception as e:
            error_msg = f"explore_training_data failed: {str(e)}"
            print(f"    {label}: {error_msg}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": error_msg}

    def model_feature_importance_analysis(self, data_dir=None):
        """
        Diagnostic-only stage 702 (off by default): which feature groups a MODEL relies on.

        Fits a plain XGBoost on the train split (defaults plus subsampling, no search; the
        tuned model does not exist yet at this point of the run) and writes, to
        MODEL_FEATURE_IMPORTANCE_OUTPUT_FOLDER:
          * grouped permutation importance on the test and hold-out sets (a sin/cos pair,
            the weather-scenario one-hots and the wawa groups are shuffled together),
          * a drop-group ablation (refit without each group and each named set), with a
            noise floor from refitting the full model with several seeds.
        The model-free description of the data (distributions, Spearman clusters, drift)
        is stage 701, explore_training_data. Nothing is dropped or modified here.
        """
        label = "model_feature_importance_analysis"
        try:
            print(f"    {label}: Starting model-based feature importance analysis...")
            output_dir = os.path.join(self.project_root, MODEL_FEATURE_IMPORTANCE_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            if data_dir is None:
                data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)

            split_files = {}
            for split in ("train", "test"):
                files = glob.glob(os.path.join(data_dir, f"merged_data_*_{split}.parquet"))
                if len(files) != 1:
                    error_msg = f"Expected exactly one {split} file in {data_dir}. Found {len(files)} {split} files"
                    print(f"    {label}: {error_msg}")
                    return {"success": False, "error": error_msg}
                split_files[split] = files[0]

            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            if not (is_classification or target_feature in REGRESSION_PROBLEM):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}
            problem_type = "classification" if is_classification else "regression"

            train_df = pd.read_parquet(split_files["train"])
            test_df = pd.read_parquet(split_files["test"])
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}

            feature_columns = [col for col in train_df.columns if col != target_feature]
            if not feature_columns:
                error_msg = "No feature columns besides the target"
                print(f"    {label}: {error_msg}")
                return {"success": False, "error": error_msg}
            X_train, y_train = train_df[feature_columns], train_df[target_feature]
            evals = {"test": (test_df[feature_columns], test_df[target_feature])}
            X_holdout, y_holdout = self._load_holdout_set(data_dir, feature_columns, target_feature, label)
            if X_holdout is not None:
                evals["holdout"] = (X_holdout, y_holdout)
            primary = "holdout" if "holdout" in evals else "test"
            print(f"    {label}: {problem_type} target '{target_feature}', {len(feature_columns)} features, "
                  f"train {len(X_train):,} rows, scored on {sorted(evals)} (charts use '{primary}')")

            # Same fit as the XGBoost trainer: minutes, or log1p(y + shift) when asked.
            metric = SCORE_METRIC if is_classification else REGRESSION_SCORE_METRIC
            y_shift, y_train_fit = None, y_train.to_numpy()
            if not is_classification and REGRESSION_LOG_TARGET:
                y_shift = max(0.0, -float(y_train.min())) + 1.0
                y_train_fit = np.log1p(y_train.to_numpy() + y_shift)
            scale_pos_weight = neg_pos_ratio(y_train) if is_classification else None

            def fit_fn(X, y_fit, seed):
                return fit_xgboost(X, y_fit, seed, is_classification=is_classification,
                                   objective=REGRESSION_OBJECTIVE, n_jobs=SEARCH_N_JOBS,
                                   params=FEATURE_ANALYSIS_XGB_PARAMS, scale_pos_weight=scale_pos_weight)

            score_fn = make_score_fn(is_classification, metric, y_shift)
            groups = feature_groups(feature_columns)
            multi = {name: cols for name, cols in groups.items() if len(cols) > 1}
            print(f"    {label}: {len(groups)} groups ({len(multi)} multi-column: {sorted(multi)}); metric '{metric}'")

            # --- 1) grouped permutation importance ---
            print(f"    {label}: Grouped permutation importance ({PERMUTATION_REPEATS} repeats)...")
            base_model = fit_fn(X_train, y_train_fit, RANDOM_STATE)
            perm_parts = []
            for split, (X_eval, y_eval) in evals.items():
                part = grouped_permutation_importance(
                    base_model, X_eval, y_eval, groups, score_fn, PERMUTATION_REPEATS,
                    RANDOM_STATE, NOISE_FLOOR_FACTOR)
                part.insert(0, "split", split)
                perm_parts.append(part)
            perm_df = pd.concat(perm_parts, ignore_index=True)
            perm_csv = os.path.join(output_dir, "permutation_importance.csv")
            perm_df.to_csv(perm_csv, index=False)

            # --- 2) drop-group ablation ---
            print(f"    {label}: Drop-group ablation ({len(groups)} groups + named sets, "
                  f"{ABLATION_NOISE_SEEDS} seeds for the noise floor)...")
            ablation_df, baseline, noise = drop_group_ablation(
                fit_fn, X_train, y_train_fit, evals, groups, ABLATION_EXTRA_SETS, score_fn,
                ABLATION_NOISE_SEEDS, RANDOM_STATE, NOISE_FLOOR_FACTOR)
            ablation_csv = os.path.join(output_dir, "ablation.csv")
            ablation_df.to_csv(ablation_csv, index=False)

            # --- charts ---
            def bar_chart(names, values, errors, grey, title, xlabel, png, pdf, band=None):
                fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.32)))
                y_pos = np.arange(len(names))
                colors = ["lightgrey" if g else "tab:blue" for g in grey]
                ax.barh(y_pos, values, xerr=errors, color=colors, ecolor="black", capsize=2)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color="black", linewidth=0.8)
                if band is not None and not np.isnan(band):
                    ax.axvspan(-band, band, color="orange", alpha=0.2, label="noise floor")
                    ax.legend(loc="lower right", fontsize=8)
                ax.set_xlabel(xlabel)
                ax.set_title(title)
                fig.tight_layout()
                self._save_figure_png_and_pdf(fig, png, pdf)

            unit = "minutes" if (not is_classification and metric in ("mae", "rmse")) else "score"
            top = perm_df[perm_df["split"] == primary].head(25)
            perm_png = os.path.join(output_dir, "permutation_importance_holdout.png")
            perm_pdf = os.path.join(output_dir, "permutation_importance_holdout.pdf")
            bar_chart(top["group"], top["importance_mean"], top["importance_std"], top["within_noise"],
                      f"Grouped permutation importance on the {primary} set ({metric}; grey = within noise)",
                      f"Score drop when the group is shuffled ({unit}; higher = the model uses it)",
                      perm_png, perm_pdf)
            abl = ablation_df.sort_values(f"{primary}_change", ascending=False).head(25)
            abl_png = os.path.join(output_dir, "ablation_holdout.png")
            abl_pdf = os.path.join(output_dir, "ablation_holdout.pdf")
            bar_chart(abl["dropped"], abl[f"{primary}_change"], None, abl[f"{primary}_within_noise"],
                      f"Drop-group ablation on the {primary} set ({metric}; grey = within noise)",
                      f"Score lost when the group is removed ({unit}; positive = the group helps)",
                      abl_png, abl_pdf, band=NOISE_FLOOR_FACTOR * noise[primary])

            # --- summary ---
            ranked = perm_df[perm_df["split"] == primary]
            summary = {
                "target_feature": target_feature, "problem_type": problem_type, "metric": metric,
                "objective": None if is_classification else REGRESSION_OBJECTIVE,
                "log_target": bool(not is_classification and REGRESSION_LOG_TARGET),
                "model": "plain XGBoost (defaults + " + json.dumps(FEATURE_ANALYSIS_XGB_PARAMS) + "), no search",
                "n_features": len(feature_columns), "n_train_rows": int(len(X_train)),
                "splits_scored": sorted(evals), "primary_split": primary,
                "permutation_repeats": PERMUTATION_REPEATS, "noise_seeds": ABLATION_NOISE_SEEDS,
                "noise_floor_factor": NOISE_FLOOR_FACTOR, "baseline_score": baseline, "noise_std": noise,
                "groups": groups,
                "top_permutation_groups": [
                    {"group": r.group, "importance": float(r.importance_mean), "within_noise": bool(r.within_noise)}
                    for r in ranked.head(5).itertuples()],
            }
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=float)

            print(f"    {label}: Analysis completed successfully!")
            return {
                "success": True,
                "output_directory": output_dir,
                "num_features": len(feature_columns),
                "num_groups": len(groups),
                "target_feature": target_feature,
                "problem_type": problem_type,
                "primary_split": primary,
                "top_group": (ranked.iloc[0]["group"], float(ranked.iloc[0]["importance_mean"])) if len(ranked) else None,
                "summary_path": summary_path,
                "permutation_csv": perm_csv,
                "ablation_csv": ablation_csv,
                "permutation_png_path": perm_png,
                "ablation_png_path": abl_png,
            }

        except Exception as e:
            error_msg = f"model_feature_importance_analysis failed: {str(e)}"
            print(f"    {label}: {error_msg}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": error_msg}

    def _load_holdout_set(self, scaled_data_dir, feature_columns, target_feature, method_label):
        """
        Load the optional out-of-time hold-out file (*_holdout.parquet), if one was
        produced by split_dataset (HOLDOUT_LAST_YEAR). Shared by every train_* method
        so each also scores the hold-out — never fits/tunes on it — alongside test.

        Returns (X_holdout, y_holdout) or (None, None) if no hold-out file exists.
        """
        holdout_pattern = os.path.join(scaled_data_dir, "merged_data_*_holdout.parquet")
        holdout_files = glob.glob(holdout_pattern)
        if not holdout_files:
            print(f"    {method_label}: No hold-out file found in {scaled_data_dir} — skipping hold-out evaluation")
            return None, None
        if len(holdout_files) > 1:
            print(f"    {method_label}: Multiple hold-out files found ({len(holdout_files)}) — skipping hold-out evaluation")
            return None, None

        holdout_file = holdout_files[0]
        print(f"      Loading hold-out data from {holdout_file}")
        holdout_df = pd.read_parquet(holdout_file)
        if target_feature not in holdout_df.columns:
            print(f"    {method_label}: Target feature '{target_feature}' not found in hold-out file — skipping")
            return None, None

        X_holdout = holdout_df[feature_columns]
        y_holdout = holdout_df[target_feature]
        print(f"      Hold-out set: {X_holdout.shape}")
        return X_holdout, y_holdout

    def _score_predictions(self, y_true, y_pred, is_classification):
        """
        Compute the same metric set used for final test evaluation, given raw-scale
        (already inverse-transformed) predictions and true values. Shared so the
        out-of-time hold-out is scored identically to the test set in every train_*
        method — see the final_test_* blocks this mirrors.
        """
        metrics = {}
        if is_classification:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
            metrics['precision'] = precision_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted', zero_division=0)
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            denom = np.sum(np.abs(y_true))
            metrics['wmape'] = (np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else 0.0

            y_true_binary = (y_true > DELAY_THRESHOLD_MINUTES).astype(int)
            y_pred_binary = (y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
            metrics['bin_precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics['bin_recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics['bin_f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics['bin_accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
            metrics['bin_confusion_matrix'] = confusion_matrix(y_true_binary, y_pred_binary).tolist()
        return metrics

    def _evaluate_holdout(self, model, X_holdout, y_holdout, is_classification, y_shift, method_label):
        """
        Score the already-trained best_model on the out-of-time hold-out — no fit(),
        no tuning. Mirrors the final_test_* metric computation (including the
        log1p/y_shift inverse transform for regression targets) so hold-out and test
        metrics are directly comparable.
        """
        y_pred = model.predict(X_holdout)
        if not is_classification:
            y_pred = to_minutes(y_pred, y_shift)

        metrics = self._score_predictions(y_holdout, y_pred, is_classification)

        if is_classification and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_holdout)[:, 1]
            metrics['auc'] = roc_auc_score(y_holdout, y_pred_proba) if len(np.unique(y_holdout)) > 1 else 0.0

        # Not printed here — the caller prints a formatted "Hold-out Summary" block
        # (mirroring "Training Summary") once these metrics are in hand.
        return metrics

    @staticmethod
    def _save_regression_diagnostics(y_true, y_pred, title, path):
        """Predicted-vs-actual (hexbin) and residual histogram for a numeric target, in minutes."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        resid = y_true - y_pred
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(np.quantile(y_true, 0.995), np.quantile(y_pred, 0.995)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        hb = ax1.hexbin(y_true, y_pred, gridsize=60, extent=(lo, hi, lo, hi), bins='log', mincnt=1, cmap='viridis')
        ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='perfect prediction')
        ax1.set_xlabel('Actual delay (min)')
        ax1.set_ylabel('Predicted delay (min)')
        ax1.set_title('Predicted vs actual (log colour scale)')
        ax1.legend(loc='upper left')
        fig.colorbar(hb, ax=ax1, label='rows')

        r_lo, r_hi = np.quantile(resid, [0.005, 0.995])
        ax2.hist(resid, bins=80, range=(float(r_lo), float(r_hi)), color='tab:blue', alpha=0.8)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1)
        ax2.axvline(float(np.median(resid)), color='black', linewidth=1, label=f'median {np.median(resid):.2f} min')
        ax2.set_xlabel('Actual - predicted (min); negative = predicted too late')
        ax2.set_ylabel('rows')
        ax2.set_title('Residuals (0.5% - 99.5% range shown)')
        ax2.legend()

        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)

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

            # Out-of-time hold-out (optional) — never trained/tuned on, scored later
            X_holdout, y_holdout = self._load_holdout_set(
                scaled_data_dir, feature_columns, target_feature, "train_xgboost_with_randomized_search_cv"
            )

            # Rows whose delay was filled in (REGRESSION_CANCELLED_POLICY = 'cap'); None otherwise.
            test_filled = load_filled_flags(test_file, len(test_df))
            holdout_filled = None
            if X_holdout is not None:
                holdout_files = glob.glob(os.path.join(scaled_data_dir, "merged_data_*_holdout.parquet"))
                if len(holdout_files) == 1:
                    holdout_filled = load_filled_flags(holdout_files[0], len(X_holdout))
            
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
            
            # Day-grouped CV: rows of one operating day never sit in both the fitting and
            # the validation fold (see src/cv_utils.py). None without a sidecar file.
            groups = load_train_groups(train_file, len(train_df))
            if groups is not None:
                print(f"      Day-grouped CV: {len(set(groups)):,} operating days in the train split")

            # Set up cross-validation strategy
            y_train_log = None  # assigned in regression branch below
            y_shift = 0.0       # assigned in regression branch below
            if is_classification:
                cv_splitter = make_cv(True, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                # Class weight instead of SMOTE: negatives per positive of the train split.
                spw = neg_pos_ratio(y_train)
                print(f"      scale_pos_weight = {spw:.3f} (train negatives / positives)")
                base_model = xgb.XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                    scale_pos_weight=spw,
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = make_cv(False, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = xgb.XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                    objective=REGRESSION_OBJECTIVE,
                )
                print(f"      Objective: {REGRESSION_OBJECTIVE}")
                if REGRESSION_LOG_TARGET:
                    # Shift target so all values are positive, then log-transform to compress
                    # the extreme right tail (max 963 min) that deflates R² artificially.
                    y_shift = max(0.0, -float(y_train.min())) + 1.0
                    y_train_log = np.log1p(y_train + y_shift)
                    print(f"      Log-transform target: shift={y_shift:.2f}, "
                          f"log range=[{y_train_log.min():.3f}, {y_train_log.max():.3f}] "
                          f"(original [{y_train.min():.1f}, {y_train.max():.1f}])")
                else:
                    # y_shift None = the model is fitted and predicts in minutes (to_minutes).
                    y_shift = None
                    print("      Target fitted on the raw minutes (no log transform)")
                # Score in minutes whatever the fit, so the tuning score and the reported
                # test / hold-out metrics are the same quantity.
                scoring_metric = make_minutes_scorer(y_shift, REGRESSION_SCORE_METRIC)
                print(f"      Tuning metric: {REGRESSION_SCORE_METRIC} (minutes)")

            # Define iteration range: 10 to 100 (step=10)
            iteration_values = list(range(10, RANDOM_SEARCH_ITERATIONS + 1, 10))
            print(f"      Testing iteration values: {iteration_values}")

            # Initialize tracking lists
            iteration_results = []
            test_f1_scores = []
            cv_scores = []
            test_accuracy_scores = []
            test_precision_scores = []
            test_recall_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_r2_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []
            holdout_metrics_per_iteration = []

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
                    n_jobs=SEARCH_N_JOBS,
                    verbose=0
                )
                
                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train if (is_classification or y_shift is None) else y_train_log
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, groups=groups, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit, groups=groups)

                # Get best model for current iteration count
                current_best_model = randomized_search.best_estimator_
                current_cv_score = randomized_search.best_score_

                # Make predictions on test set; inverse-transform for regression
                y_pred = current_best_model.predict(X_test)
                if not is_classification:
                    y_pred = to_minutes(y_pred, y_shift)

                # Calculate test metrics
                if is_classification:
                    test_f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                    test_accuracy = accuracy_score(y_test, y_pred)
                    test_precision = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_recall    = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}")

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
                    test_bin_confusion_matrix = confusion_matrix(y_test_binary, y_pred_binary).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")

                # Score this iteration's model on the out-of-time hold-out too (tracking only —
                # model selection below stays keyed on CV score, never on hold-out performance).
                current_holdout_metrics = None
                if X_holdout is not None:
                    current_holdout_metrics = self._evaluate_holdout(
                        current_best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_xgboost_with_randomized_search_cv"
                    )
                holdout_metrics_per_iteration.append(current_holdout_metrics)

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

                if is_classification:
                    test_f1_scores.append(test_f1)
                    test_accuracy_scores.append(test_accuracy)
                    test_precision_scores.append(test_precision)
                    test_recall_scores.append(test_recall)
                else:
                    test_f1_scores.append(test_rmse)
                    test_mae_scores.append(test_mae)
                    test_wmape_scores.append(test_wmape)
                    test_r2_scores.append(test_r2)
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
                    'test_accuracy': test_accuracy_scores[-1] if is_classification else None,
                    'test_precision': test_precision_scores[-1] if is_classification else None,
                    'test_recall': test_recall_scores[-1] if is_classification else None,
                    'test_confusion_matrix': test_confusion_matrix if is_classification else None,
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_r2': test_r2_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'test_bin_confusion_matrix': test_bin_confusion_matrix if not is_classification else None,
                    'holdout_metrics': current_holdout_metrics,
                    'best_params': randomized_search.best_params_
                })

            # Calculate final metrics once on the best model found across all iterations
            print(f"      Calculating final metrics with best model (iteration {best_iteration})...")
            final_y_pred = best_model.predict(X_test)
            if not is_classification:
                final_y_pred = to_minutes(final_y_pred, y_shift)

            if is_classification:
                final_test_accuracy = accuracy_score(y_test, final_y_pred)
                final_test_f1 = f1_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                final_test_precision = precision_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_confusion_matrix = confusion_matrix(y_test, final_y_pred).tolist()

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
                final_baseline = mae_baseline(y_train, y_test, final_test_mae)
                final_residuals = residual_quantiles(y_test, final_y_pred)
                final_running = None
                if test_filled is not None and test_filled.any():
                    keep = ~test_filled
                    y_run, p_run = np.asarray(y_test)[keep], np.asarray(final_y_pred)[keep]
                    final_running = self._score_predictions(y_run, p_run, False)
                    final_running.update(mae_baseline(y_train, y_run, final_running['mae']))
                    final_running['n_rows'] = int(keep.sum())
                    final_running['n_filled_excluded'] = int(test_filled.sum())

                y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
                final_y_pred_binary = (final_y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
                final_bin_precision = precision_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_recall    = recall_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_f1        = f1_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_accuracy  = accuracy_score(y_test_binary, final_y_pred_binary)
                final_bin_confusion_matrix = confusion_matrix(y_test_binary, final_y_pred_binary).tolist()

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

                # Save confusion matrix plot for the best model
                cm_for_plot = final_confusion_matrix if is_classification else final_bin_confusion_matrix
                cm_labels = [str(c) for c in sorted(np.unique(y_test))] if is_classification else ['On-time', 'Delayed']
                cm_title = f'XGBoost Confusion Matrix - Best Model ({problem_type.title()})\nDataset: {file_identifier} | Best Iteration: {best_iteration}'
                cm_plot_filename = os.path.join(output_dir, f'xgboost_confusion_matrix_{file_identifier}.png')
                self._save_confusion_matrix_plot(cm_for_plot, cm_labels, cm_title, cm_plot_filename)
                print(f"      Confusion matrix plot saved to: {cm_plot_filename}")

                if not is_classification:
                    diag_filename = os.path.join(output_dir, f'xgboost_residuals_{file_identifier}.png')
                    self._save_regression_diagnostics(
                        y_test, final_y_pred,
                        f'XGBoost regression, test split | Dataset: {file_identifier} | Best Iteration: {best_iteration}',
                        diag_filename)
                    print(f"      Regression diagnostics plot saved to: {diag_filename}")

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
                        "test_auc": float(final_test_auc),
                        "confusion_matrix": final_confusion_matrix
                    }
                else:
                    results["final_metrics"] = {
                        "test_rmse": float(final_test_rmse),
                        "test_mae": float(final_test_mae),
                        "test_wmape": float(final_test_wmape),
                        "test_r2": float(final_test_r2),
                        "baseline_mae_train_median": final_baseline["baseline_mae"],
                        "mae_skill": final_baseline["mae_skill"],
                        "residual_quantiles": final_residuals,
                        "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES,
                        "test_bin_precision": float(final_bin_precision),
                        "test_bin_recall": float(final_bin_recall),
                        "test_bin_f1": float(final_bin_f1),
                        "test_bin_accuracy": float(final_bin_accuracy),
                        "test_bin_confusion_matrix": final_bin_confusion_matrix
                    }

                if not is_classification and final_running is not None:
                    results["final_metrics_running_trains"] = final_running

                # Add iteration-wise metrics summary
                if is_classification:
                    results["iteration_metrics_summary"] = {
                        "f1_values": [float(x) for x in test_f1_scores],
                        "accuracy_values": [float(x) for x in test_accuracy_scores],
                        "precision_values": [float(x) for x in test_precision_scores],
                        "recall_values": [float(x) for x in test_recall_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_f1": float(max(test_f1_scores)),
                        "best_accuracy": float(max(test_accuracy_scores)),
                        "best_precision": float(max(test_precision_scores)),
                        "best_recall": float(max(test_recall_scores)),
                        "average_f1": float(np.mean(test_f1_scores)),
                        "average_accuracy": float(np.mean(test_accuracy_scores)),
                        "average_precision": float(np.mean(test_precision_scores)),
                        "average_recall": float(np.mean(test_recall_scores))
                    }
                else:
                    results["iteration_metrics_summary"] = {
                        "rmse_values": [float(x) for x in test_f1_scores],
                        "mae_values": [float(x) for x in test_mae_scores],
                        "wmape_values": [float(x) for x in test_wmape_scores],
                        "r2_values": [float(x) for x in test_r2_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_rmse": float(min(test_f1_scores)),
                        "best_mae": float(min(test_mae_scores)),
                        "best_wmape": float(min(test_wmape_scores)),
                        "best_r2": float(max(test_r2_scores)),
                        "average_rmse": float(np.mean(test_f1_scores)),
                        "average_mae": float(np.mean(test_mae_scores)),
                        "average_wmape": float(np.mean(test_wmape_scores)),
                        "average_r2": float(np.mean(test_r2_scores)),
                        "bin_f1_values": [float(x) for x in test_bin_f1_scores],
                        "bin_accuracy_values": [float(x) for x in test_bin_accuracy_scores],
                        "bin_precision_values": [float(x) for x in test_bin_precision_scores],
                        "bin_recall_values": [float(x) for x in test_bin_recall_scores]
                    }

                # Add out-of-time hold-out metrics (never trained/tuned on) if available
                if X_holdout is not None:
                    results["holdout_metrics"] = self._evaluate_holdout(
                        best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_xgboost_with_randomized_search_cv"
                    )
                    results["dataset_info"]["holdout_samples"] = len(X_holdout)

                    if not is_classification:
                        h_pred = to_minutes(best_model.predict(X_holdout), y_shift)
                        results["holdout_metrics"].update(
                            mae_baseline(y_train, y_holdout, results["holdout_metrics"]["mae"]))
                        results["holdout_metrics"]["residual_quantiles"] = residual_quantiles(y_holdout, h_pred)
                        if holdout_filled is not None and holdout_filled.any():
                            keep = ~holdout_filled
                            y_run, p_run = np.asarray(y_holdout)[keep], np.asarray(h_pred)[keep]
                            hold_running = self._score_predictions(y_run, p_run, False)
                            hold_running.update(mae_baseline(y_train, y_run, hold_running['mae']))
                            hold_running['n_rows'] = int(keep.sum())
                            hold_running['n_filled_excluded'] = int(holdout_filled.sum())
                            results["holdout_metrics_running_trains"] = hold_running
                        self._save_regression_diagnostics(
                            y_holdout, h_pred,
                            f'XGBoost regression, hold-out year | Dataset: {file_identifier} | Best Iteration: {best_iteration}',
                            os.path.join(output_dir, f'xgboost_residuals_holdout_{file_identifier}.png'))

                    # Per-iteration hold-out metrics plus a summary mirroring
                    # iteration_metrics_summary above.
                    results["holdout_iteration_analysis"] = holdout_metrics_per_iteration
                    valid_holdout = [h for h in holdout_metrics_per_iteration if h is not None]
                    if valid_holdout:
                        if is_classification:
                            h_acc = [float(h['accuracy']) for h in valid_holdout]
                            h_f1 = [float(h['f1']) for h in valid_holdout]
                            h_prec = [float(h['precision']) for h in valid_holdout]
                            h_rec = [float(h['recall']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "accuracy_values": h_acc,
                                "f1_values": h_f1,
                                "precision_values": h_prec,
                                "recall_values": h_rec,
                                "best_accuracy": float(max(h_acc)),
                                "best_f1": float(max(h_f1)),
                                "best_precision": float(max(h_prec)),
                                "best_recall": float(max(h_rec)),
                                "average_accuracy": float(np.mean(h_acc)),
                                "average_f1": float(np.mean(h_f1)),
                                "average_precision": float(np.mean(h_prec)),
                                "average_recall": float(np.mean(h_rec))
                            }
                        else:
                            h_rmse = [float(h['rmse']) for h in valid_holdout]
                            h_mae = [float(h['mae']) for h in valid_holdout]
                            h_wmape = [float(h['wmape']) for h in valid_holdout]
                            h_r2 = [float(h['r2']) for h in valid_holdout]
                            h_bin_f1 = [float(h['bin_f1']) for h in valid_holdout]
                            h_bin_prec = [float(h['bin_precision']) for h in valid_holdout]
                            h_bin_rec = [float(h['bin_recall']) for h in valid_holdout]
                            h_bin_acc = [float(h['bin_accuracy']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "rmse_values": h_rmse,
                                "mae_values": h_mae,
                                "wmape_values": h_wmape,
                                "r2_values": h_r2,
                                "best_r2": float(max(h_r2)),
                                "average_r2": float(np.mean(h_r2)),
                                "best_rmse": float(min(h_rmse)),
                                "best_mae": float(min(h_mae)),
                                "best_wmape": float(min(h_wmape)),
                                "average_rmse": float(np.mean(h_rmse)),
                                "average_mae": float(np.mean(h_mae)),
                                "average_wmape": float(np.mean(h_wmape)),
                                "bin_f1_values": h_bin_f1,
                                "bin_precision_values": h_bin_prec,
                                "bin_recall_values": h_bin_rec,
                                "bin_accuracy_values": h_bin_acc
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
            print(f"        {'F1' if is_classification else 'RMSE'} Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
            if is_classification:
                print(f"        Final Accuracy: {final_test_accuracy:.4f}  F1: {final_test_f1:.4f}  Precision: {final_test_precision:.4f}  Recall: {final_test_recall:.4f}  AUC: {final_test_auc:.4f}")
            if not is_classification:
                print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                print(f"        Baseline MAE (always the train median): {final_baseline['baseline_mae']:.4f} min  MAE skill: {final_baseline['mae_skill']:.3f}")
                if final_running is not None:
                    print(f"        Running trains only (n={final_running['n_rows']:,}, {final_running['n_filled_excluded']:,} filled rows excluded): "
                          f"RMSE: {final_running['rmse']:.4f}  MAE: {final_running['mae']:.4f}  R²: {final_running['r2']:.4f}  "
                          f"MAE skill: {final_running['mae_skill']:.3f}")
                print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min; "
                      f"{'filled cancelled rows count as delayed' if final_running is not None else 'cancelled trains are not in a numeric run'}):")
                print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")

            if results.get("holdout_metrics"):
                hm = results["holdout_metrics"]
                print(f"      Hold-out Summary (out-of-time, never trained/tuned on):")
                if is_classification:
                    print(f"        Accuracy: {hm['accuracy']:.4f}  F1: {hm['f1']:.4f}  Precision: {hm['precision']:.4f}  Recall: {hm['recall']:.4f}  AUC: {hm.get('auc', 0):.4f}")
                else:
                    print(f"        RMSE: {hm['rmse']:.4f}  MAE: {hm['mae']:.4f}  R²: {hm['r2']:.4f}  WMAPE: {hm['wmape']:.2f}%")
                    print(f"        Baseline MAE (always the train median): {hm['baseline_mae']:.4f} min  MAE skill: {hm['mae_skill']:.3f}")
                    hr = results.get("holdout_metrics_running_trains")
                    if hr:
                        print(f"        Running trains only (n={hr['n_rows']:,}, {hr['n_filled_excluded']:,} filled rows excluded): "
                              f"RMSE: {hr['rmse']:.4f}  MAE: {hr['mae']:.4f}  R²: {hr['r2']:.4f}  MAE skill: {hr['mae_skill']:.3f}")
                    print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                    print(f"          Precision: {hm['bin_precision']:.4f}  Recall: {hm['bin_recall']:.4f}  F1: {hm['bin_f1']:.4f}  Accuracy: {hm['bin_accuracy']:.4f}")

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

            # Out-of-time hold-out (optional) — never trained/tuned on, scored later
            X_holdout, y_holdout = self._load_holdout_set(
                scaled_data_dir, feature_columns, target_feature, "train_lightgbm_with_randomized_search_cv"
            )

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

            # Day-grouped CV: rows of one operating day never sit in both the fitting and
            # the validation fold (see src/cv_utils.py). None without a sidecar file.
            groups = load_train_groups(train_file, len(train_df))
            if groups is not None:
                print(f"      Day-grouped CV: {len(set(groups)):,} operating days in the train split")

            # Set up cross-validation strategy
            y_train_log = None  # assigned in regression branch below
            y_shift = 0.0       # assigned in regression branch below
            if is_classification:
                cv_splitter = make_cv(True, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                # Class weight instead of SMOTE: negatives per positive of the train split.
                spw = neg_pos_ratio(y_train)
                print(f"      scale_pos_weight = {spw:.3f} (train negatives / positives)")
                base_model = lgb.LGBMClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                    subsample_freq=1,
                    verbose=-1,
                    scale_pos_weight=spw,
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = make_cv(False, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = lgb.LGBMRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
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
            test_accuracy_scores = []
            test_precision_scores = []
            test_recall_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []
            holdout_metrics_per_iteration = []

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
                    n_jobs=SEARCH_N_JOBS,
                    verbose=0
                )

                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train_log if not is_classification else y_train
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, groups=groups, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit, groups=groups)

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
                    test_precision = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_recall    = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}")

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
                    test_bin_confusion_matrix = confusion_matrix(y_test_binary, y_pred_binary).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")

                # Score this iteration's model on the out-of-time hold-out too (tracking only —
                # model selection below stays keyed on CV score, never on hold-out performance).
                current_holdout_metrics = None
                if X_holdout is not None:
                    current_holdout_metrics = self._evaluate_holdout(
                        current_best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_lightgbm_with_randomized_search_cv"
                    )
                holdout_metrics_per_iteration.append(current_holdout_metrics)

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

                if is_classification:
                    test_f1_scores.append(test_f1)
                    test_accuracy_scores.append(test_accuracy)
                    test_precision_scores.append(test_precision)
                    test_recall_scores.append(test_recall)
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
                    'test_accuracy': test_accuracy_scores[-1] if is_classification else None,
                    'test_precision': test_precision_scores[-1] if is_classification else None,
                    'test_recall': test_recall_scores[-1] if is_classification else None,
                    'test_confusion_matrix': test_confusion_matrix if is_classification else None,
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'test_bin_confusion_matrix': test_bin_confusion_matrix if not is_classification else None,
                    'holdout_metrics': current_holdout_metrics,
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
                final_test_precision = precision_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_confusion_matrix = confusion_matrix(y_test, final_y_pred).tolist()

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
                final_bin_confusion_matrix = confusion_matrix(y_test_binary, final_y_pred_binary).tolist()

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

                # Save confusion matrix plot for the best model
                cm_for_plot = final_confusion_matrix if is_classification else final_bin_confusion_matrix
                cm_labels = [str(c) for c in sorted(np.unique(y_test))] if is_classification else ['On-time', 'Delayed']
                cm_title = f'LightGBM Confusion Matrix - Best Model ({problem_type.title()})\nDataset: {file_identifier} | Best Iteration: {best_iteration}'
                cm_plot_filename = os.path.join(output_dir, f'lightgbm_confusion_matrix_{file_identifier}.png')
                self._save_confusion_matrix_plot(cm_for_plot, cm_labels, cm_title, cm_plot_filename)
                print(f"      Confusion matrix plot saved to: {cm_plot_filename}")

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
                        "test_auc": float(final_test_auc),
                        "confusion_matrix": final_confusion_matrix
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
                        "test_bin_accuracy": float(final_bin_accuracy),
                        "test_bin_confusion_matrix": final_bin_confusion_matrix
                    }

                # Add iteration-wise metrics summary
                if is_classification:
                    results["iteration_metrics_summary"] = {
                        "f1_values": [float(x) for x in test_f1_scores],
                        "accuracy_values": [float(x) for x in test_accuracy_scores],
                        "precision_values": [float(x) for x in test_precision_scores],
                        "recall_values": [float(x) for x in test_recall_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_f1": float(max(test_f1_scores)),
                        "best_accuracy": float(max(test_accuracy_scores)),
                        "best_precision": float(max(test_precision_scores)),
                        "best_recall": float(max(test_recall_scores)),
                        "average_f1": float(np.mean(test_f1_scores)),
                        "average_accuracy": float(np.mean(test_accuracy_scores)),
                        "average_precision": float(np.mean(test_precision_scores)),
                        "average_recall": float(np.mean(test_recall_scores))
                    }
                else:
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

                # Add out-of-time hold-out metrics (never trained/tuned on) if available
                if X_holdout is not None:
                    results["holdout_metrics"] = self._evaluate_holdout(
                        best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_lightgbm_with_randomized_search_cv"
                    )
                    results["dataset_info"]["holdout_samples"] = len(X_holdout)

                    # Per-iteration hold-out metrics plus a summary mirroring
                    # iteration_metrics_summary above.
                    results["holdout_iteration_analysis"] = holdout_metrics_per_iteration
                    valid_holdout = [h for h in holdout_metrics_per_iteration if h is not None]
                    if valid_holdout:
                        if is_classification:
                            h_acc = [float(h['accuracy']) for h in valid_holdout]
                            h_f1 = [float(h['f1']) for h in valid_holdout]
                            h_prec = [float(h['precision']) for h in valid_holdout]
                            h_rec = [float(h['recall']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "accuracy_values": h_acc,
                                "f1_values": h_f1,
                                "precision_values": h_prec,
                                "recall_values": h_rec,
                                "best_accuracy": float(max(h_acc)),
                                "best_f1": float(max(h_f1)),
                                "best_precision": float(max(h_prec)),
                                "best_recall": float(max(h_rec)),
                                "average_accuracy": float(np.mean(h_acc)),
                                "average_f1": float(np.mean(h_f1)),
                                "average_precision": float(np.mean(h_prec)),
                                "average_recall": float(np.mean(h_rec))
                            }
                        else:
                            h_rmse = [float(h['rmse']) for h in valid_holdout]
                            h_mae = [float(h['mae']) for h in valid_holdout]
                            h_wmape = [float(h['wmape']) for h in valid_holdout]
                            h_bin_f1 = [float(h['bin_f1']) for h in valid_holdout]
                            h_bin_prec = [float(h['bin_precision']) for h in valid_holdout]
                            h_bin_rec = [float(h['bin_recall']) for h in valid_holdout]
                            h_bin_acc = [float(h['bin_accuracy']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "rmse_values": h_rmse,
                                "mae_values": h_mae,
                                "wmape_values": h_wmape,
                                "best_rmse": float(min(h_rmse)),
                                "best_mae": float(min(h_mae)),
                                "best_wmape": float(min(h_wmape)),
                                "average_rmse": float(np.mean(h_rmse)),
                                "average_mae": float(np.mean(h_mae)),
                                "average_wmape": float(np.mean(h_wmape)),
                                "bin_f1_values": h_bin_f1,
                                "bin_precision_values": h_bin_prec,
                                "bin_recall_values": h_bin_rec,
                                "bin_accuracy_values": h_bin_acc
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
                print(f"        F1 Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
                if is_classification:
                    print(f"        Final Accuracy: {final_test_accuracy:.4f}  F1: {final_test_f1:.4f}  Precision: {final_test_precision:.4f}  Recall: {final_test_recall:.4f}  AUC: {final_test_auc:.4f}")
                if not is_classification:
                    print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                    print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                    print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")

                if results.get("holdout_metrics"):
                    hm = results["holdout_metrics"]
                    print(f"      Hold-out Summary (out-of-time, never trained/tuned on):")
                    if is_classification:
                        print(f"        Accuracy: {hm['accuracy']:.4f}  F1: {hm['f1']:.4f}  Precision: {hm['precision']:.4f}  Recall: {hm['recall']:.4f}  AUC: {hm.get('auc', 0):.4f}")
                    else:
                        print(f"        RMSE: {hm['rmse']:.4f}  MAE: {hm['mae']:.4f}  R²: {hm['r2']:.4f}  WMAPE: {hm['wmape']:.2f}%")
                        print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                        print(f"          Precision: {hm['bin_precision']:.4f}  Recall: {hm['bin_recall']:.4f}  F1: {hm['bin_f1']:.4f}  Accuracy: {hm['bin_accuracy']:.4f}")

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

    def train_random_forest_with_randomized_search_cv(self, data_dir=None):
        """
        Random Forest training method that tests different iteration counts and plots performance curve.

        This method:
        1. Tests RANDOM_SEARCH_ITERATIONS from 10 to 100 (step=10)
        2. Tracks test F1 binary scores for each iteration count
        3. Plots test_f1_binary vs n_iter curve
        4. Saves the best model and performance metrics
        """

        try:
            print(f"    train_random_forest_with_randomized_search_cv: Starting Random Forest training with iteration analysis...")

            # Create output directory
            output_dir = os.path.join(self.project_root, RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)

            print(f"    train_random_forest_with_randomized_search_cv: Output directory: {output_dir}")

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
                print(f"    train_random_forest_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Check if we have exactly one train/test file pair
            if len(train_files) != 1 or len(test_files) != 1:
                error_msg = f"Expected exactly one train and one test file. Found {len(train_files)} train files and {len(test_files)} test files"
                print(f"    train_random_forest_with_randomized_search_cv: {error_msg}")
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
                print(f"    train_random_forest_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            file_identifier = identifier_match.group(1)
            print(f"    train_random_forest_with_randomized_search_cv: Processing dataset: {file_identifier}")

            # Determine problem type based on target feature
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM

            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_random_forest_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            problem_type = "classification" if is_classification else "regression"
            print(f"    train_random_forest_with_randomized_search_cv: Detected {problem_type} problem for target '{target_feature}'")

            # Load training and test data
            print(f"      Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            print(f"      Loading test data from {test_file}")
            test_df = pd.read_parquet(test_file)

            # Check if target feature exists
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    train_random_forest_with_randomized_search_cv: {error_msg}")
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

            # Out-of-time hold-out (optional) — never trained/tuned on, scored later
            X_holdout, y_holdout = self._load_holdout_set(
                scaled_data_dir, feature_columns, target_feature, "train_random_forest_with_randomized_search_cv"
            )

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

            # Day-grouped CV: rows of one operating day never sit in both the fitting and
            # the validation fold (see src/cv_utils.py). None without a sidecar file.
            groups = load_train_groups(train_file, len(train_df))
            if groups is not None:
                print(f"      Day-grouped CV: {len(set(groups)):,} operating days in the train split")

            # Set up cross-validation strategy
            y_train_log = None  # assigned in regression branch below
            y_shift = 0.0       # assigned in regression branch below
            if is_classification:
                cv_splitter = make_cv(True, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                    class_weight='balanced',  # instead of SMOTE
                )
                param_distributions=RANDOM_FOREST_PARAM_DISTRIBUTIONS
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = make_cv(False, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                )
                param_distributions = RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION
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
            test_accuracy_scores = []
            test_precision_scores = []
            test_recall_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []
            holdout_metrics_per_iteration = []

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
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=RANDOM_STATE,
                    n_jobs=SEARCH_N_JOBS,
                    verbose=0
                )

                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train_log if not is_classification else y_train
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, groups=groups, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit, groups=groups)

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
                    test_precision = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_recall    = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}")

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
                    test_bin_confusion_matrix = confusion_matrix(y_test_binary, y_pred_binary).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")

                # Score this iteration's model on the out-of-time hold-out too (tracking only —
                # model selection below stays keyed on CV score, never on hold-out performance).
                current_holdout_metrics = None
                if X_holdout is not None:
                    current_holdout_metrics = self._evaluate_holdout(
                        current_best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_random_forest_with_randomized_search_cv"
                    )
                holdout_metrics_per_iteration.append(current_holdout_metrics)

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

                if is_classification:
                    test_f1_scores.append(test_f1)
                    test_accuracy_scores.append(test_accuracy)
                    test_precision_scores.append(test_precision)
                    test_recall_scores.append(test_recall)
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
                    'test_accuracy': test_accuracy_scores[-1] if is_classification else None,
                    'test_precision': test_precision_scores[-1] if is_classification else None,
                    'test_recall': test_recall_scores[-1] if is_classification else None,
                    'test_confusion_matrix': test_confusion_matrix if is_classification else None,
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'test_bin_confusion_matrix': test_bin_confusion_matrix if not is_classification else None,
                    'holdout_metrics': current_holdout_metrics,
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
                final_test_precision = precision_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_confusion_matrix = confusion_matrix(y_test, final_y_pred).tolist()

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
                final_bin_confusion_matrix = confusion_matrix(y_test_binary, final_y_pred_binary).tolist()

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

            plt.title(f'Random Forest Performance vs RandomizedSearch Iterations\nDataset: {file_identifier}',
                    fontsize=16, fontweight='bold', pad=20)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                    fontsize=12, framealpha=0.9)

            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'random_forest_iteration_analysis_{file_identifier}.png')
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
                plt.title(f'Random Forest Feature Importance - Best Model ({problem_type.title()})\n'
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

                importance_plot_filename = os.path.join(output_dir, f'random_forest_feature_importance_{file_identifier}.png')
                plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
                print(f"      Feature importance plot saved to: {importance_plot_filename}")
                plt.close()

                importance_csv_filename = os.path.join(output_dir, f'random_forest_feature_importance_{file_identifier}.csv')
                importance_df.sort_values('importance', ascending=False).to_csv(importance_csv_filename, index=False)
                print(f"      Feature importance data saved to: {importance_csv_filename}")

                # Save confusion matrix plot for the best model
                cm_for_plot = final_confusion_matrix if is_classification else final_bin_confusion_matrix
                cm_labels = [str(c) for c in sorted(np.unique(y_test))] if is_classification else ['On-time', 'Delayed']
                cm_title = f'Random Forest Confusion Matrix - Best Model ({problem_type.title()})\nDataset: {file_identifier} | Best Iteration: {best_iteration}'
                cm_plot_filename = os.path.join(output_dir, f'random_forest_confusion_matrix_{file_identifier}.png')
                self._save_confusion_matrix_plot(cm_for_plot, cm_labels, cm_title, cm_plot_filename)
                print(f"      Confusion matrix plot saved to: {cm_plot_filename}")

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
                        "test_auc": float(final_test_auc),
                        "confusion_matrix": final_confusion_matrix
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
                        "test_bin_accuracy": float(final_bin_accuracy),
                        "test_bin_confusion_matrix": final_bin_confusion_matrix
                    }

                # Add iteration-wise metrics summary
                if is_classification:
                    results["iteration_metrics_summary"] = {
                        "f1_values": [float(x) for x in test_f1_scores],
                        "accuracy_values": [float(x) for x in test_accuracy_scores],
                        "precision_values": [float(x) for x in test_precision_scores],
                        "recall_values": [float(x) for x in test_recall_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_f1": float(max(test_f1_scores)),
                        "best_accuracy": float(max(test_accuracy_scores)),
                        "best_precision": float(max(test_precision_scores)),
                        "best_recall": float(max(test_recall_scores)),
                        "average_f1": float(np.mean(test_f1_scores)),
                        "average_accuracy": float(np.mean(test_accuracy_scores)),
                        "average_precision": float(np.mean(test_precision_scores)),
                        "average_recall": float(np.mean(test_recall_scores))
                    }
                else:
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

                # Add out-of-time hold-out metrics (never trained/tuned on) if available
                if X_holdout is not None:
                    results["holdout_metrics"] = self._evaluate_holdout(
                        best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_random_forest_with_randomized_search_cv"
                    )
                    results["dataset_info"]["holdout_samples"] = len(X_holdout)

                    # Per-iteration hold-out metrics plus a summary mirroring
                    # iteration_metrics_summary above.
                    results["holdout_iteration_analysis"] = holdout_metrics_per_iteration
                    valid_holdout = [h for h in holdout_metrics_per_iteration if h is not None]
                    if valid_holdout:
                        if is_classification:
                            h_acc = [float(h['accuracy']) for h in valid_holdout]
                            h_f1 = [float(h['f1']) for h in valid_holdout]
                            h_prec = [float(h['precision']) for h in valid_holdout]
                            h_rec = [float(h['recall']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "accuracy_values": h_acc,
                                "f1_values": h_f1,
                                "precision_values": h_prec,
                                "recall_values": h_rec,
                                "best_accuracy": float(max(h_acc)),
                                "best_f1": float(max(h_f1)),
                                "best_precision": float(max(h_prec)),
                                "best_recall": float(max(h_rec)),
                                "average_accuracy": float(np.mean(h_acc)),
                                "average_f1": float(np.mean(h_f1)),
                                "average_precision": float(np.mean(h_prec)),
                                "average_recall": float(np.mean(h_rec))
                            }
                        else:
                            h_rmse = [float(h['rmse']) for h in valid_holdout]
                            h_mae = [float(h['mae']) for h in valid_holdout]
                            h_wmape = [float(h['wmape']) for h in valid_holdout]
                            h_bin_f1 = [float(h['bin_f1']) for h in valid_holdout]
                            h_bin_prec = [float(h['bin_precision']) for h in valid_holdout]
                            h_bin_rec = [float(h['bin_recall']) for h in valid_holdout]
                            h_bin_acc = [float(h['bin_accuracy']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "rmse_values": h_rmse,
                                "mae_values": h_mae,
                                "wmape_values": h_wmape,
                                "best_rmse": float(min(h_rmse)),
                                "best_mae": float(min(h_mae)),
                                "best_wmape": float(min(h_wmape)),
                                "average_rmse": float(np.mean(h_rmse)),
                                "average_mae": float(np.mean(h_mae)),
                                "average_wmape": float(np.mean(h_wmape)),
                                "bin_f1_values": h_bin_f1,
                                "bin_precision_values": h_bin_prec,
                                "bin_recall_values": h_bin_rec,
                                "bin_accuracy_values": h_bin_acc
                            }

                results_file = os.path.join(output_dir, f"random_forest_iteration_analysis_{file_identifier}.json")
                with open(results_file, 'w') as f:
                    results_str = json.loads(json.dumps(results, default=str))
                    json.dump(results_str, f, indent=2)

                print(f"      Results saved to: {results_file}")

                # Save the best model
                model_filename = os.path.join(output_dir, f'random_forest_best_model_{file_identifier}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"      Best model saved to: {model_filename}")

                # Find the index of the best iteration in the results list
                best_iteration_idx = iteration_values.index(best_iteration) if best_iteration else 0

                # Print summary
                print(f"      Training Summary:")
                print(f"        Problem Type: {problem_type}")
                print(f"        Best Iteration Count: {best_iteration}")
                print(f"        Best CV Score: {best_cv_score:.4f}")
                print(f"        F1 Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
                if is_classification:
                    print(f"        Final Accuracy: {final_test_accuracy:.4f}  F1: {final_test_f1:.4f}  Precision: {final_test_precision:.4f}  Recall: {final_test_recall:.4f}  AUC: {final_test_auc:.4f}")
                if not is_classification:
                    print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                    print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                    print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")

                if results.get("holdout_metrics"):
                    hm = results["holdout_metrics"]
                    print(f"      Hold-out Summary (out-of-time, never trained/tuned on):")
                    if is_classification:
                        print(f"        Accuracy: {hm['accuracy']:.4f}  F1: {hm['f1']:.4f}  Precision: {hm['precision']:.4f}  Recall: {hm['recall']:.4f}  AUC: {hm.get('auc', 0):.4f}")
                    else:
                        print(f"        RMSE: {hm['rmse']:.4f}  MAE: {hm['mae']:.4f}  R²: {hm['r2']:.4f}  WMAPE: {hm['wmape']:.2f}%")
                        print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                        print(f"          Precision: {hm['bin_precision']:.4f}  Recall: {hm['bin_recall']:.4f}  F1: {hm['bin_f1']:.4f}  Accuracy: {hm['bin_accuracy']:.4f}")

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
            error_msg = f"Error in train_random_forest_with_randomized_search_cv: {str(e)}"
            print(f"    train_random_forest_with_randomized_search_cv: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def train_logistic_regression_with_randomized_search_cv(self, data_dir=None):
        """
        Logistic Regression training method that tests different iteration counts and plots performance curve.

        Uses LogisticRegression for classification and ElasticNet (regularized linear
        regression) for regression, since sklearn's LogisticRegression has no direct
        regression counterpart.

        This method:
        1. Tests RANDOM_SEARCH_ITERATIONS from 10 to 100 (step=10)
        2. Tracks test F1 binary scores for each iteration count
        3. Plots test_f1_binary vs n_iter curve
        4. Saves the best model and performance metrics
        """

        try:
            print(f"    train_logistic_regression_with_randomized_search_cv: Starting Logistic Regression training with iteration analysis...")

            # Create output directory
            output_dir = os.path.join(self.project_root, REGULARIZED_REGRESSION_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)

            print(f"    train_logistic_regression_with_randomized_search_cv: Output directory: {output_dir}")

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
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Check if we have exactly one train/test file pair
            if len(train_files) != 1 or len(test_files) != 1:
                error_msg = f"Expected exactly one train and one test file. Found {len(train_files)} train files and {len(test_files)} test files"
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
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
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            file_identifier = identifier_match.group(1)
            print(f"    train_logistic_regression_with_randomized_search_cv: Processing dataset: {file_identifier}")

            # Determine problem type based on target feature
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM

            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            problem_type = "classification" if is_classification else "regression"
            print(f"    train_logistic_regression_with_randomized_search_cv: Detected {problem_type} problem for target '{target_feature}'")

            # Load training and test data
            print(f"      Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            print(f"      Loading test data from {test_file}")
            test_df = pd.read_parquet(test_file)

            # Check if target feature exists
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
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

            # Out-of-time hold-out (optional) — never trained/tuned on, scored later
            X_holdout, y_holdout = self._load_holdout_set(
                scaled_data_dir, feature_columns, target_feature, "train_logistic_regression_with_randomized_search_cv"
            )

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

            # Day-grouped CV: rows of one operating day never sit in both the fitting and
            # the validation fold (see src/cv_utils.py). None without a sidecar file.
            groups = load_train_groups(train_file, len(train_df))
            if groups is not None:
                print(f"      Day-grouped CV: {len(set(groups)):,} operating days in the train split")

            # Set up cross-validation strategy
            y_train_log = None  # assigned in regression branch below
            y_shift = 0.0       # assigned in regression branch below
            if is_classification:
                cv_splitter = make_cv(True, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = LogisticRegression(
                    solver='saga',        # only solver supporting l1/l2/elasticnet — enables the full penalty search
                    max_iter=5000,        # avoid ConvergenceWarning spam; not searched
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                    class_weight='balanced',  # instead of SMOTE
                )
                param_distributions = LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = make_cv(False, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = ElasticNet(
                    max_iter=5000,        # coordinate descent headroom; not searched
                    random_state=RANDOM_STATE,
                    # NOTE: ElasticNet does not support parallel job specification (raises TypeError)
                )
                param_distributions = ELASTICNET_PARAM_DISTRIBUTIONS
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
            test_accuracy_scores = []
            test_precision_scores = []
            test_recall_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []
            holdout_metrics_per_iteration = []

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
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=RANDOM_STATE,
                    n_jobs=SEARCH_N_JOBS,
                    verbose=0
                )

                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train_log if not is_classification else y_train
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, groups=groups, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit, groups=groups)

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
                    test_precision = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_recall    = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}")

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
                    test_bin_confusion_matrix = confusion_matrix(y_test_binary, y_pred_binary).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")

                # Score this iteration's model on the out-of-time hold-out too (tracking only —
                # model selection below stays keyed on CV score, never on hold-out performance).
                current_holdout_metrics = None
                if X_holdout is not None:
                    current_holdout_metrics = self._evaluate_holdout(
                        current_best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_logistic_regression_with_randomized_search_cv"
                    )
                holdout_metrics_per_iteration.append(current_holdout_metrics)

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

                if is_classification:
                    test_f1_scores.append(test_f1)
                    test_accuracy_scores.append(test_accuracy)
                    test_precision_scores.append(test_precision)
                    test_recall_scores.append(test_recall)
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
                    'test_accuracy': test_accuracy_scores[-1] if is_classification else None,
                    'test_precision': test_precision_scores[-1] if is_classification else None,
                    'test_recall': test_recall_scores[-1] if is_classification else None,
                    'test_confusion_matrix': test_confusion_matrix if is_classification else None,
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'test_bin_confusion_matrix': test_bin_confusion_matrix if not is_classification else None,
                    'holdout_metrics': current_holdout_metrics,
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
                final_test_precision = precision_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_confusion_matrix = confusion_matrix(y_test, final_y_pred).tolist()

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
                final_bin_confusion_matrix = confusion_matrix(y_test_binary, final_y_pred_binary).tolist()

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

            plt.title(f'Logistic Regression Performance vs RandomizedSearch Iterations\nDataset: {file_identifier}',
                    fontsize=16, fontweight='bold', pad=20)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                    fontsize=12, framealpha=0.9)

            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'logistic_regression_iteration_analysis_{file_identifier}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Performance curve saved to: {plot_filename}")

            # CREATE AND SAVE FEATURE IMPORTANCE PLOT
            results = {}
            if best_model is not None:
                print(f"      Creating feature importance plot...")

                # Linear models: use coefficients instead of tree-based importance scores.
                # Binary LogisticRegression.coef_ has shape (1, n_features); ElasticNet.coef_
                # has shape (n_features,) for a single-target regression.
                coef = best_model.coef_[0] if is_classification else best_model.coef_
                feature_importance = np.abs(coef)   # magnitude drives sorting + bar length
                feature_names = feature_columns

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance,  # |coef|, used for sort + plot
                    'coefficient': coef,                # signed — preserves direction of effect
                }).sort_values('importance', ascending=True)

                plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))

                bars = plt.barh(range(len(importance_df)), importance_df['importance'])

                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Coefficient Magnitude', fontsize=12)
                plt.title(f'Logistic Regression Coefficient Magnitude - Best Model ({problem_type.title()})\n'
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

                importance_plot_filename = os.path.join(output_dir, f'logistic_regression_feature_importance_{file_identifier}.png')
                plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
                print(f"      Feature importance plot saved to: {importance_plot_filename}")
                plt.close()

                importance_csv_filename = os.path.join(output_dir, f'logistic_regression_feature_importance_{file_identifier}.csv')
                importance_df.sort_values('importance', ascending=False).to_csv(importance_csv_filename, index=False)
                print(f"      Feature importance data saved to: {importance_csv_filename}")

                # Save confusion matrix plot for the best model
                cm_for_plot = final_confusion_matrix if is_classification else final_bin_confusion_matrix
                cm_labels = [str(c) for c in sorted(np.unique(y_test))] if is_classification else ['On-time', 'Delayed']
                cm_title = f'Logistic Regression Confusion Matrix - Best Model ({problem_type.title()})\nDataset: {file_identifier} | Best Iteration: {best_iteration}'
                cm_plot_filename = os.path.join(output_dir, f'logistic_regression_confusion_matrix_{file_identifier}.png')
                self._save_confusion_matrix_plot(cm_for_plot, cm_labels, cm_title, cm_plot_filename)
                print(f"      Confusion matrix plot saved to: {cm_plot_filename}")

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
                    "feature_importance": dict(zip(feature_columns, np.abs(coef))) if best_model else None
                }

                # Add problem-specific metrics
                if is_classification:
                    results["final_metrics"] = {
                        "test_accuracy": float(final_test_accuracy),
                        "test_f1": float(final_test_f1),
                        "test_precision": float(final_test_precision),
                        "test_recall": float(final_test_recall),
                        "test_auc": float(final_test_auc),
                        "confusion_matrix": final_confusion_matrix
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
                        "test_bin_accuracy": float(final_bin_accuracy),
                        "test_bin_confusion_matrix": final_bin_confusion_matrix
                    }

                # Add iteration-wise metrics summary
                if is_classification:
                    results["iteration_metrics_summary"] = {
                        "f1_values": [float(x) for x in test_f1_scores],
                        "accuracy_values": [float(x) for x in test_accuracy_scores],
                        "precision_values": [float(x) for x in test_precision_scores],
                        "recall_values": [float(x) for x in test_recall_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_f1": float(max(test_f1_scores)),
                        "best_accuracy": float(max(test_accuracy_scores)),
                        "best_precision": float(max(test_precision_scores)),
                        "best_recall": float(max(test_recall_scores)),
                        "average_f1": float(np.mean(test_f1_scores)),
                        "average_accuracy": float(np.mean(test_accuracy_scores)),
                        "average_precision": float(np.mean(test_precision_scores)),
                        "average_recall": float(np.mean(test_recall_scores))
                    }
                else:
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

                # Add out-of-time hold-out metrics (never trained/tuned on) if available
                if X_holdout is not None:
                    results["holdout_metrics"] = self._evaluate_holdout(
                        best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_logistic_regression_with_randomized_search_cv"
                    )
                    results["dataset_info"]["holdout_samples"] = len(X_holdout)

                    # Per-iteration hold-out metrics plus a summary mirroring
                    # iteration_metrics_summary above.
                    results["holdout_iteration_analysis"] = holdout_metrics_per_iteration
                    valid_holdout = [h for h in holdout_metrics_per_iteration if h is not None]
                    if valid_holdout:
                        if is_classification:
                            h_acc = [float(h['accuracy']) for h in valid_holdout]
                            h_f1 = [float(h['f1']) for h in valid_holdout]
                            h_prec = [float(h['precision']) for h in valid_holdout]
                            h_rec = [float(h['recall']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "accuracy_values": h_acc,
                                "f1_values": h_f1,
                                "precision_values": h_prec,
                                "recall_values": h_rec,
                                "best_accuracy": float(max(h_acc)),
                                "best_f1": float(max(h_f1)),
                                "best_precision": float(max(h_prec)),
                                "best_recall": float(max(h_rec)),
                                "average_accuracy": float(np.mean(h_acc)),
                                "average_f1": float(np.mean(h_f1)),
                                "average_precision": float(np.mean(h_prec)),
                                "average_recall": float(np.mean(h_rec))
                            }
                        else:
                            h_rmse = [float(h['rmse']) for h in valid_holdout]
                            h_mae = [float(h['mae']) for h in valid_holdout]
                            h_wmape = [float(h['wmape']) for h in valid_holdout]
                            h_bin_f1 = [float(h['bin_f1']) for h in valid_holdout]
                            h_bin_prec = [float(h['bin_precision']) for h in valid_holdout]
                            h_bin_rec = [float(h['bin_recall']) for h in valid_holdout]
                            h_bin_acc = [float(h['bin_accuracy']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "rmse_values": h_rmse,
                                "mae_values": h_mae,
                                "wmape_values": h_wmape,
                                "best_rmse": float(min(h_rmse)),
                                "best_mae": float(min(h_mae)),
                                "best_wmape": float(min(h_wmape)),
                                "average_rmse": float(np.mean(h_rmse)),
                                "average_mae": float(np.mean(h_mae)),
                                "average_wmape": float(np.mean(h_wmape)),
                                "bin_f1_values": h_bin_f1,
                                "bin_precision_values": h_bin_prec,
                                "bin_recall_values": h_bin_rec,
                                "bin_accuracy_values": h_bin_acc
                            }

                results_file = os.path.join(output_dir, f"logistic_regression_iteration_analysis_{file_identifier}.json")
                with open(results_file, 'w') as f:
                    results_str = json.loads(json.dumps(results, default=str))
                    json.dump(results_str, f, indent=2)

                print(f"      Results saved to: {results_file}")

                # Save the best model
                model_filename = os.path.join(output_dir, f'logistic_regression_best_model_{file_identifier}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"      Best model saved to: {model_filename}")

                # Find the index of the best iteration in the results list
                best_iteration_idx = iteration_values.index(best_iteration) if best_iteration else 0

                # Print summary
                print(f"      Training Summary:")
                print(f"        Problem Type: {problem_type}")
                print(f"        Best Iteration Count: {best_iteration}")
                print(f"        Best CV Score: {best_cv_score:.4f}")
                print(f"        F1 Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
                if is_classification:
                    print(f"        Final Accuracy: {final_test_accuracy:.4f}  F1: {final_test_f1:.4f}  Precision: {final_test_precision:.4f}  Recall: {final_test_recall:.4f}  AUC: {final_test_auc:.4f}")
                if not is_classification:
                    print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                    print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                    print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")

                if results.get("holdout_metrics"):
                    hm = results["holdout_metrics"]
                    print(f"      Hold-out Summary (out-of-time, never trained/tuned on):")
                    if is_classification:
                        print(f"        Accuracy: {hm['accuracy']:.4f}  F1: {hm['f1']:.4f}  Precision: {hm['precision']:.4f}  Recall: {hm['recall']:.4f}  AUC: {hm.get('auc', 0):.4f}")
                    else:
                        print(f"        RMSE: {hm['rmse']:.4f}  MAE: {hm['mae']:.4f}  R²: {hm['r2']:.4f}  WMAPE: {hm['wmape']:.2f}%")
                        print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                        print(f"          Precision: {hm['bin_precision']:.4f}  Recall: {hm['bin_recall']:.4f}  F1: {hm['bin_f1']:.4f}  Accuracy: {hm['bin_accuracy']:.4f}")

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
            error_msg = f"Error in train_logistic_regression_with_randomized_search_cv: {str(e)}"
            print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

    def train_naive_bayes_with_randomized_search_cv(self, data_dir=None):
        """
        Naive Bayes training method that tests different iteration counts and plots performance curve.

        Uses GaussianNB for classification and BayesianRidge (Bayesian linear
        regression) for regression, since sklearn's GaussianNB has no direct
        regression counterpart.

        This method:
        1. Tests RANDOM_SEARCH_ITERATIONS from 10 to 100 (step=10)
        2. Tracks test F1 binary scores for each iteration count
        3. Plots test_f1_binary vs n_iter curve
        4. Saves the best model and performance metrics
        """

        try:
            print(f"    train_naive_bayes_with_randomized_search_cv: Starting Naive Bayes training with iteration analysis...")

            # Create output directory
            output_dir = os.path.join(self.project_root, NAIVE_BAYES_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)

            print(f"    train_naive_bayes_with_randomized_search_cv: Output directory: {output_dir}")

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
                print(f"    train_naive_bayes_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Check if we have exactly one train/test file pair
            if len(train_files) != 1 or len(test_files) != 1:
                error_msg = f"Expected exactly one train and one test file. Found {len(train_files)} train files and {len(test_files)} test files"
                print(f"    train_naive_bayes_with_randomized_search_cv: {error_msg}")
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
                print(f"    train_naive_bayes_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            file_identifier = identifier_match.group(1)
            print(f"    train_naive_bayes_with_randomized_search_cv: Processing dataset: {file_identifier}")

            # Determine problem type based on target feature
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM

            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_naive_bayes_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            problem_type = "classification" if is_classification else "regression"
            print(f"    train_naive_bayes_with_randomized_search_cv: Detected {problem_type} problem for target '{target_feature}'")

            # Load training and test data
            print(f"      Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            print(f"      Loading test data from {test_file}")
            test_df = pd.read_parquet(test_file)

            # Check if target feature exists
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    train_naive_bayes_with_randomized_search_cv: {error_msg}")
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

            # Out-of-time hold-out (optional) — never trained/tuned on, scored later
            X_holdout, y_holdout = self._load_holdout_set(
                scaled_data_dir, feature_columns, target_feature, "train_naive_bayes_with_randomized_search_cv"
            )

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

            # Day-grouped CV: rows of one operating day never sit in both the fitting and
            # the validation fold (see src/cv_utils.py). None without a sidecar file.
            groups = load_train_groups(train_file, len(train_df))
            if groups is not None:
                print(f"      Day-grouped CV: {len(set(groups)):,} operating days in the train split")

            # Set up cross-validation strategy
            y_train_log = None  # assigned in regression branch below
            y_shift = 0.0       # assigned in regression branch below
            model_name = 'Naive Bayes' if is_classification else 'Bayesian Ridge'
            if is_classification:
                cv_splitter = make_cv(True, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = GaussianNB()
                param_distributions = NAIVE_BAYES_PARAM_DISTRIBUTIONS
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = make_cv(False, groups, RANDOM_SEARCH_CV_FOLDS, RANDOM_STATE)
                base_model = BayesianRidge(
                    max_iter=5000,        # coordinate ascent headroom; not searched
                )
                param_distributions = BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS
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
            test_accuracy_scores = []
            test_precision_scores = []
            test_recall_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []
            holdout_metrics_per_iteration = []

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
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=RANDOM_STATE,
                    n_jobs=SEARCH_N_JOBS,
                    verbose=0
                )

                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train_log if not is_classification else y_train
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, groups=groups, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit, groups=groups)

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
                    test_precision = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_recall    = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                    test_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}")

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
                    test_bin_confusion_matrix = confusion_matrix(y_test_binary, y_pred_binary).tolist()

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")

                # Score this iteration's model on the out-of-time hold-out too (tracking only —
                # model selection below stays keyed on CV score, never on hold-out performance).
                current_holdout_metrics = None
                if X_holdout is not None:
                    current_holdout_metrics = self._evaluate_holdout(
                        current_best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_naive_bayes_with_randomized_search_cv"
                    )
                holdout_metrics_per_iteration.append(current_holdout_metrics)

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

                if is_classification:
                    test_f1_scores.append(test_f1)
                    test_accuracy_scores.append(test_accuracy)
                    test_precision_scores.append(test_precision)
                    test_recall_scores.append(test_recall)
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
                    'test_accuracy': test_accuracy_scores[-1] if is_classification else None,
                    'test_precision': test_precision_scores[-1] if is_classification else None,
                    'test_recall': test_recall_scores[-1] if is_classification else None,
                    'test_confusion_matrix': test_confusion_matrix if is_classification else None,
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'test_bin_confusion_matrix': test_bin_confusion_matrix if not is_classification else None,
                    'holdout_metrics': current_holdout_metrics,
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
                final_test_precision = precision_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted', zero_division=0)
                final_confusion_matrix = confusion_matrix(y_test, final_y_pred).tolist()

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
                final_bin_confusion_matrix = confusion_matrix(y_test_binary, final_y_pred_binary).tolist()

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

            plt.title(f'{model_name} Performance vs RandomizedSearch Iterations\nDataset: {file_identifier}',
                    fontsize=16, fontweight='bold', pad=20)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                    fontsize=12, framealpha=0.9)

            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'naive_bayes_iteration_analysis_{file_identifier}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Performance curve saved to: {plot_filename}")

            # CREATE AND SAVE FEATURE IMPORTANCE PLOT
            results = {}
            if best_model is not None:
                print(f"      Creating feature importance plot...")

                # GaussianNB has no coef_: use a Cohen's-d-style standardized mean
                # difference between the two classes' per-feature Gaussian means
                # (theta_), scaled by the pooled per-feature standard deviation
                # (var_). Assumes binary classification, matching the binary
                # assumption already used throughout this pipeline.
                # BayesianRidge.coef_ has shape (n_features,), same as ElasticNet's.
                importance_label = 'Standardized Mean Difference' if is_classification else 'Coefficient Magnitude'
                if is_classification:
                    signed_diff = best_model.theta_[1] - best_model.theta_[0]
                    pooled_std = np.sqrt((best_model.var_[0] + best_model.var_[1]) / 2)
                    coef = signed_diff / pooled_std
                else:
                    coef = best_model.coef_
                feature_importance = np.abs(coef)   # magnitude drives sorting + bar length
                feature_names = feature_columns

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance,  # |coef|, used for sort + plot
                    'coefficient': coef,                # signed — preserves direction of effect
                }).sort_values('importance', ascending=True)

                plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))

                bars = plt.barh(range(len(importance_df)), importance_df['importance'])

                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel(importance_label, fontsize=12)
                plt.title(f'{model_name} {importance_label} - Best Model ({problem_type.title()})\n'
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

                importance_plot_filename = os.path.join(output_dir, f'naive_bayes_feature_importance_{file_identifier}.png')
                plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
                print(f"      Feature importance plot saved to: {importance_plot_filename}")
                plt.close()

                importance_csv_filename = os.path.join(output_dir, f'naive_bayes_feature_importance_{file_identifier}.csv')
                importance_df.sort_values('importance', ascending=False).to_csv(importance_csv_filename, index=False)
                print(f"      Feature importance data saved to: {importance_csv_filename}")

                # Save confusion matrix plot for the best model
                cm_for_plot = final_confusion_matrix if is_classification else final_bin_confusion_matrix
                cm_labels = [str(c) for c in sorted(np.unique(y_test))] if is_classification else ['On-time', 'Delayed']
                cm_title = f'{model_name} Confusion Matrix - Best Model ({problem_type.title()})\nDataset: {file_identifier} | Best Iteration: {best_iteration}'
                cm_plot_filename = os.path.join(output_dir, f'naive_bayes_confusion_matrix_{file_identifier}.png')
                self._save_confusion_matrix_plot(cm_for_plot, cm_labels, cm_title, cm_plot_filename)
                print(f"      Confusion matrix plot saved to: {cm_plot_filename}")

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
                    "feature_importance": dict(zip(feature_columns, np.abs(coef))) if best_model else None
                }

                # Add problem-specific metrics
                if is_classification:
                    results["final_metrics"] = {
                        "test_accuracy": float(final_test_accuracy),
                        "test_f1": float(final_test_f1),
                        "test_precision": float(final_test_precision),
                        "test_recall": float(final_test_recall),
                        "test_auc": float(final_test_auc),
                        "confusion_matrix": final_confusion_matrix
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
                        "test_bin_accuracy": float(final_bin_accuracy),
                        "test_bin_confusion_matrix": final_bin_confusion_matrix
                    }

                # Add iteration-wise metrics summary
                if is_classification:
                    results["iteration_metrics_summary"] = {
                        "f1_values": [float(x) for x in test_f1_scores],
                        "accuracy_values": [float(x) for x in test_accuracy_scores],
                        "precision_values": [float(x) for x in test_precision_scores],
                        "recall_values": [float(x) for x in test_recall_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_f1": float(max(test_f1_scores)),
                        "best_accuracy": float(max(test_accuracy_scores)),
                        "best_precision": float(max(test_precision_scores)),
                        "best_recall": float(max(test_recall_scores)),
                        "average_f1": float(np.mean(test_f1_scores)),
                        "average_accuracy": float(np.mean(test_accuracy_scores)),
                        "average_precision": float(np.mean(test_precision_scores)),
                        "average_recall": float(np.mean(test_recall_scores))
                    }
                else:
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

                # Add out-of-time hold-out metrics (never trained/tuned on) if available
                if X_holdout is not None:
                    results["holdout_metrics"] = self._evaluate_holdout(
                        best_model, X_holdout, y_holdout, is_classification, y_shift,
                        "train_naive_bayes_with_randomized_search_cv"
                    )
                    results["dataset_info"]["holdout_samples"] = len(X_holdout)

                    # Per-iteration hold-out metrics plus a summary mirroring
                    # iteration_metrics_summary above.
                    results["holdout_iteration_analysis"] = holdout_metrics_per_iteration
                    valid_holdout = [h for h in holdout_metrics_per_iteration if h is not None]
                    if valid_holdout:
                        if is_classification:
                            h_acc = [float(h['accuracy']) for h in valid_holdout]
                            h_f1 = [float(h['f1']) for h in valid_holdout]
                            h_prec = [float(h['precision']) for h in valid_holdout]
                            h_rec = [float(h['recall']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "accuracy_values": h_acc,
                                "f1_values": h_f1,
                                "precision_values": h_prec,
                                "recall_values": h_rec,
                                "best_accuracy": float(max(h_acc)),
                                "best_f1": float(max(h_f1)),
                                "best_precision": float(max(h_prec)),
                                "best_recall": float(max(h_rec)),
                                "average_accuracy": float(np.mean(h_acc)),
                                "average_f1": float(np.mean(h_f1)),
                                "average_precision": float(np.mean(h_prec)),
                                "average_recall": float(np.mean(h_rec))
                            }
                        else:
                            h_rmse = [float(h['rmse']) for h in valid_holdout]
                            h_mae = [float(h['mae']) for h in valid_holdout]
                            h_wmape = [float(h['wmape']) for h in valid_holdout]
                            h_bin_f1 = [float(h['bin_f1']) for h in valid_holdout]
                            h_bin_prec = [float(h['bin_precision']) for h in valid_holdout]
                            h_bin_rec = [float(h['bin_recall']) for h in valid_holdout]
                            h_bin_acc = [float(h['bin_accuracy']) for h in valid_holdout]
                            results["holdout_iteration_summary"] = {
                                "rmse_values": h_rmse,
                                "mae_values": h_mae,
                                "wmape_values": h_wmape,
                                "best_rmse": float(min(h_rmse)),
                                "best_mae": float(min(h_mae)),
                                "best_wmape": float(min(h_wmape)),
                                "average_rmse": float(np.mean(h_rmse)),
                                "average_mae": float(np.mean(h_mae)),
                                "average_wmape": float(np.mean(h_wmape)),
                                "bin_f1_values": h_bin_f1,
                                "bin_precision_values": h_bin_prec,
                                "bin_recall_values": h_bin_rec,
                                "bin_accuracy_values": h_bin_acc
                            }

                results_file = os.path.join(output_dir, f"naive_bayes_iteration_analysis_{file_identifier}.json")
                with open(results_file, 'w') as f:
                    results_str = json.loads(json.dumps(results, default=str))
                    json.dump(results_str, f, indent=2)

                print(f"      Results saved to: {results_file}")

                # Save the best model
                model_filename = os.path.join(output_dir, f'naive_bayes_best_model_{file_identifier}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"      Best model saved to: {model_filename}")

                # Find the index of the best iteration in the results list
                best_iteration_idx = iteration_values.index(best_iteration) if best_iteration else 0

                # Print summary
                print(f"      Training Summary:")
                print(f"        Problem Type: {problem_type}")
                print(f"        Best Iteration Count: {best_iteration}")
                print(f"        Best CV Score: {best_cv_score:.4f}")
                print(f"        F1 Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
                if is_classification:
                    print(f"        Final Accuracy: {final_test_accuracy:.4f}  F1: {final_test_f1:.4f}  Precision: {final_test_precision:.4f}  Recall: {final_test_recall:.4f}  AUC: {final_test_auc:.4f}")
                if not is_classification:
                    print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                    print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                    print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")

                if results.get("holdout_metrics"):
                    hm = results["holdout_metrics"]
                    print(f"      Hold-out Summary (out-of-time, never trained/tuned on):")
                    if is_classification:
                        print(f"        Accuracy: {hm['accuracy']:.4f}  F1: {hm['f1']:.4f}  Precision: {hm['precision']:.4f}  Recall: {hm['recall']:.4f}  AUC: {hm.get('auc', 0):.4f}")
                    else:
                        print(f"        RMSE: {hm['rmse']:.4f}  MAE: {hm['mae']:.4f}  R²: {hm['r2']:.4f}  WMAPE: {hm['wmape']:.2f}%")
                        print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                        print(f"          Precision: {hm['bin_precision']:.4f}  Recall: {hm['bin_recall']:.4f}  F1: {hm['bin_f1']:.4f}  Accuracy: {hm['bin_accuracy']:.4f}")

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
            error_msg = f"Error in train_naive_bayes_with_randomized_search_cv: {str(e)}"
            print(f"    train_naive_bayes_with_randomized_search_cv: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }
