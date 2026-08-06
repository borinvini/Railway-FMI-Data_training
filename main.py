"""Entry point for the Railway-FMI preprocessing and training pipelines.

Configuration lives in config/const*.py. Command-line flags override those
defaults and are the mechanism used by the CSC Slurm scripts in hpc/.

Import ordering matters: config modules bind their constants at import time via
`from config... import NAME`, so environment overrides must be applied before the
first config/src import. That is why those imports live inside _run().
"""
import argparse
import os
import sys

# Short model name -> trainer stage key in TRAINING_STATE_MACHINE.
MODEL_STAGES = {
    "xgboost": "train_xgboost_with_randomized_search_cv",
    "lightgbm": "train_lightgbm_with_randomized_search_cv",
    "random_forest": "train_random_forest_with_randomized_search_cv",
    "logistic_regression": "train_logistic_regression_with_randomized_search_cv",
    "naive_bayes": "train_naive_bayes_with_randomized_search_cv",
}

# Data-preparation stages every trainer depends on, so that one array task is
# self-contained. shap_correlation_analysis is excluded: it is diagnostic, not a
# dependency, and running it five times would waste allocation.
PREREQUISITE_STAGES = (
    "merge_data_files",
    "filter_delay_outliers",
    "select_training_cols",
    "split_dataset",
    "balance_classes",
    "scale_weather_features",
)


def parse_args(argv=None):
    """Parse command-line arguments. With no arguments every value is None/False,
    so the pipeline behaves exactly as it did before this layer existed."""
    parser = argparse.ArgumentParser(
        description="Run the Railway-FMI preprocessing and/or training pipeline."
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stage names to enable, overriding TRAINING_STATE_MACHINE.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(MODEL_STAGES),
        help="Run a single trainer plus its prerequisite stages. Used by hpc/train_array.sh.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Absolute path holding data/. On CSC this is /scratch/<project>/railway-fmi.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Outer CV parallelism. Defaults to SLURM_CPUS_PER_TASK, else all cores.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=None,
        help="Override RANDOM_SEARCH_ITERATIONS. Minimum 10; smaller values skip training.",
    )
    parser.add_argument(
        "--columns-file",
        default=None,
        help="Path to a one-name-per-line file of columns to keep, overriding "
             "SELECTED_COLUMNS in config/const_training.py. Used by hpc/*.sh so "
             "changing the feature set needs no commit.",
    )
    parser.add_argument(
        "--dump-columns",
        action="store_true",
        help="Print the merged dataset's candidate columns and exit, for populating "
             "SELECTED_COLUMNS. Runs no trainer.",
    )
    args = parser.parse_args(argv)

    if args.search_iterations is not None and args.search_iterations < 10:
        parser.error(
            "--search-iterations must be at least 10: the trainer iterates "
            "range(10, N+1, 10), which is empty below 10."
        )
    if args.stages and args.model:
        parser.error("--stages and --model are mutually exclusive.")
    if args.n_jobs is not None and (args.n_jobs == 0 or args.n_jobs < -1):
        parser.error(
            "--n-jobs must be -1 (use all available cores) or a positive integer."
        )
    if args.columns_file is not None:
        # Imported inside the function, never at module scope: main.py's
        # top-level imports must stay {argparse, os, sys} or config constants
        # bind before apply_env_overrides() runs. tests/test_cli.py:117 guards
        # this. columns_file itself imports nothing from config.
        from config.columns_file import load_columns
        try:
            load_columns(args.columns_file)
        except (OSError, ValueError) as exc:
            # Fail here, not in const_training at import time: a bad file then
            # costs one second instead of a queued allocation and a merge.
            parser.error(str(exc))

    return args


def apply_env_overrides(args):
    """Translate flags into the environment variables that config/ reads at import
    time. Must run before any config/src import."""
    if args.data_root is not None:
        os.environ["RAILWAY_DATA_ROOT"] = args.data_root
    if args.n_jobs is not None:
        # A dedicated variable, not SLURM_CPUS_PER_TASK: that variable is Slurm's
        # own record of the step's allocation and the batch scripts rely on it
        # independently (see hpc/*.sh SRUN_CPUS_PER_TASK exports). Overwriting it
        # here would corrupt that record for anything else reading it.
        os.environ["RAILWAY_N_JOBS"] = str(args.n_jobs)
    if args.search_iterations is not None:
        os.environ["RAILWAY_SEARCH_ITERATIONS"] = str(args.search_iterations)
    if args.columns_file is not None:
        # Stored verbatim rather than absolutised: nothing between here and the
        # config import in _run() changes the working directory, and the path as
        # typed is what appears in the run's log, matching the batch script.
        os.environ["RAILWAY_COLUMNS_FILE"] = args.columns_file


def build_state_machine(args, default):
    """Return the training state machine to execute.

    No --stages and no --model returns `default` unchanged.
    """
    if args.stages:
        requested = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in requested if s not in default]
        if unknown:
            raise ValueError(
                f"Unknown stage name(s): {unknown}. Valid stages: {sorted(default)}"
            )
        return {stage: stage in requested for stage in default}

    if args.model:
        wanted = set(PREREQUISITE_STAGES) | {MODEL_STAGES[args.model]}
        return {stage: stage in wanted for stage in default}

    return default


def dump_columns(pipeline, parquet_files):
    """Print the merged dataset's columns in a form pasteable into SELECTED_COLUMNS.

    Exists so the interactive prompt in select_training_cols never has to run under
    Slurm, where stdin is /dev/null.
    """
    merged = pipeline.merge_data_files(parquet_files)
    if not merged or not merged.get("success", False):
        print(f"✗ merge_data_files failed: {merged.get('error') if merged else 'no result'}")
        return 1

    columns = list(merged["data"].columns)
    print("\n" + "=" * 60)
    print(f"CANDIDATE COLUMNS ({len(columns)})")
    print("=" * 60)
    for index, name in enumerate(columns, start=1):
        print(f"  {index:3d}. {name}")

    print("\nPaste the ones you want into SELECTED_COLUMNS in config/const_training.py:")
    print("SELECTED_COLUMNS = [")
    for name in columns:
        print(f"    '{name}',")
    print("]")
    return 0


def _make_pipeline():
    """Construct a TrainingPipeline.

    A separate module-level function purely so tests can substitute a fake. The
    import is deferred because config constants bind at import time and
    apply_env_overrides() must run first.
    """
    from src.training_pipeline import TrainingPipeline
    return TrainingPipeline()


def _run(args):
    """Execute the preprocessing and/or training pipeline.

    Called after parse_args() and apply_env_overrides() have already run, so
    `args` holds the parsed CLI flags and any env-based overrides (data root,
    n-jobs, search iterations) are already reflected in the config constants
    imported below.
    """
    # Imported here, not at module scope: config constants are bound at import
    # time, so apply_env_overrides() must have already run.
    from src.file_utils import check_parquet_files, extract_date_range, ensure_folder_structure
    from src.training_pipeline import TrainingPipeline
    from src.preprocessing_pipeline import PreprocessingPipeline

    from config.const import (
        EXECUTE_PREPROCESSING_DATA_PIPELINE,
        EXECUTE_TRAINING_PIPELINE,
        INPUT_FOLDER,
    )

    from config.const_preprocessing import (
        DEFAULT_TARGET_FEATURE,
        REQUIRED_STATIONS,
        FILTER_TRAINS_BY_STATIONS,
        PREPROCESSING_STATE_MACHINE,
    )

    from config.const_training import (
        IMPORTANCE_THRESHOLD,
        TRAINING_STATE_MACHINE,
    )

    # STEP 1: Ensure folder structure exists BEFORE any other operations
    print("="*60)
    print("RAILWAY FMI DATA PROCESSING - INITIALIZATION")
    print("="*60)

    try:
        folder_result = ensure_folder_structure()
        print("\n✓ Folder structure initialization completed successfully!")
    except Exception as e:
        print(f"\n✗ Failed to initialize folder structure: {e}")
        print("Cannot proceed without proper folder structure. Please check permissions and try again.")
        return

    # STEP 2: Display configuration information
    print("\n" + "="*60)
    print("STARTING RAILWAY FMI DATA CHECK AND PROCESSING")
    print("="*60)

    print(f"Using target feature: '{DEFAULT_TARGET_FEATURE}'")
    print(f"Feature importance threshold: {IMPORTANCE_THRESHOLD}")
    print(f"Preprocessing pipeline execution enabled: {EXECUTE_PREPROCESSING_DATA_PIPELINE}")
    print(f"Training pipeline execution enabled: {EXECUTE_TRAINING_PIPELINE}")

    # STEP 3: Display configuration warnings if applicable
    if FILTER_TRAINS_BY_STATIONS:
        print(f"\n⚠️  WARNING: Train filtering is ENABLED!")
        print(f"Only trains passing through ALL of these stations will be processed: {', '.join(REQUIRED_STATIONS)}")
        print(f"This may significantly reduce the amount of data available for training.\n")

    # STEP 4: Check for parquet files in the input directory
    print(f"Checking for parquet files...")
    parquet_files = check_parquet_files()

    # STEP 5: Extract date range from filenames
    date_range = extract_date_range(parquet_files)

    print("\nInitial data check complete.")

    if args.dump_columns:
        return dump_columns(_make_pipeline(), parquet_files)

    if not parquet_files:
        if EXECUTE_PREPROCESSING_DATA_PIPELINE:
            print(f"\n⚠️  No parquet files found in {INPUT_FOLDER}")
            print("The folder structure has been created successfully.")
            print("Please place your parquet files in the input directory and run the script again.")
            print(f"\nNext steps:")
            print(f"  1. Add your parquet files to: {INPUT_FOLDER}")
            print(f"  2. Expected format: matched_data_YYYY_MM.parquet")
            print(f"  3. Example: matched_data_2023_12.parquet")
            print(f"  4. Then run: python main.py")
            return
        # Training-only runs read from data/output/101-preprocessed_training_ready
        # (see training_pipeline.merge_data_files), so raw input is not required.
        print(f"\n⚠️  No parquet files in {INPUT_FOLDER}; continuing because "
              f"preprocessing is disabled and training reads preprocessed output.")

    # STEP 6: Check if preprocessing pipeline execution is enabled
    if not EXECUTE_PREPROCESSING_DATA_PIPELINE:
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE EXECUTION SKIPPED")
        print("="*60)
        print("Preprocessing pipeline execution is disabled in configuration.")
        print("To enable preprocessing pipeline execution, set EXECUTE_PREPROCESSING_DATA_PIPELINE = True in config/const.py")
        print(f"Found {len(parquet_files)} parquet files ready for processing when enabled.")
    else:
        # STEP 7: Initialize and run the preprocessing pipeline
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE INITIALIZATION")
        print("="*60)

        print(f"Initializing Preprocessing Pipeline...")
        preprocessing_pipeline = PreprocessingPipeline()  # CHANGED: Use new PreprocessingPipeline class

        print(f"Preprocessing state machine configuration: {PREPROCESSING_STATE_MACHINE}")
        print(f"Starting preprocessing pipeline execution with target feature: '{DEFAULT_TARGET_FEATURE}'")

        # Run the full preprocessing pipeline on all parquet files with the default target feature
        preprocessing_results = preprocessing_pipeline.run_pipeline(  # CHANGED: Use PreprocessingPipeline instance
            parquet_files,
            target_feature=DEFAULT_TARGET_FEATURE
        )

        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE EXECUTION COMPLETED")
        print("="*60)

        # Display preprocessing summary
        if preprocessing_results:
            success_rate = (preprocessing_results.get('successful_preprocessing', 0) / preprocessing_results.get('total_files', 1)) * 100
            print(f"Preprocessing success rate: {success_rate:.1f}%")

            if preprocessing_results.get('failed_files', 0) > 0:
                print(f"⚠️  {preprocessing_results.get('failed_files', 0)} months failed preprocessing")
            else:
                print("✓ All months preprocessed successfully!")
        else:
            print("✗ Preprocessing pipeline returned no results!")

    # STEP 8: Check if training pipeline execution is enabled
    if not EXECUTE_TRAINING_PIPELINE:
        print("\n" + "="*60)
        print("TRAINING PIPELINE EXECUTION SKIPPED")
        print("="*60)
        print("Training pipeline execution is disabled in configuration.")
        print("To enable training pipeline execution, set EXECUTE_TRAINING_PIPELINE = True in config/const.py")
        return
    else:
        # STEP 9: Initialize and run the training pipeline
        print("\n" + "="*60)
        print("TRAINING PIPELINE INITIALIZATION")
        print("="*60)

        print(f"Initializing Training Pipeline...")
        training_pipeline = TrainingPipeline()

        state_machine = build_state_machine(args, TRAINING_STATE_MACHINE)

        print(f"Training state machine configuration: {state_machine}")
        print(f"Starting training pipeline execution...")

        # Run the training pipeline steps
        training_results = training_pipeline.execute_training_pipeline_steps(
            parquet_files,
            state_machine=state_machine
        )

        print("\n" + "="*60)
        print("TRAINING PIPELINE EXECUTION COMPLETED")
        print("="*60)

        # Display training summary
        if training_results and training_results.get('success', False):
            print("✓ Training pipeline completed successfully!")
            print(f"Steps executed: {', '.join(training_results.get('steps_executed', []))}")
            file_info = training_results.get('file_info', {})
            print(f"Files processed: {file_info.get('processed_files', 0)}/{file_info.get('total_files', 0)}")
            return 0
        elif training_results:
            print("✗ Training pipeline failed!")
            errors = training_results.get('errors', [])
            if errors:
                print("Errors encountered:")
                for error in errors:
                    print(f"  - {error}")
            return 1
        else:
            print("✗ Training pipeline returned no results!")
            return 1


def main(argv=None):
    args = parse_args(argv)
    apply_env_overrides(args)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
