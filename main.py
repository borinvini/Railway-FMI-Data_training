from src.file_utils import check_csv_files, extract_date_range, ensure_folder_structure
from src.training_pipeline import TrainingPipeline
from src.preprocessing_pipeline import PreprocessingPipeline  
from config.const import (
    DEFAULT_TARGET_FEATURE,
    EXECUTE_PREPROCESSING_DATA_PIPELINE, 
    EXECUTE_TRAINING_PIPELINE,
    FILTER_TRAINS_BY_STATIONS, 
    IMPORTANCE_THRESHOLD, 
    REQUIRED_STATIONS,
    INPUT_FOLDER,
    PREPROCESSING_STATE_MACHINE,  
    TRAINING_STATE_MACHINE
)


def main():
    """
    Main function to execute the script.
    """
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
    
    # STEP 4: Check for CSV files in the input directory
    print(f"Checking for CSV files...")
    csv_files = check_csv_files()
            
    # STEP 5: Extract date range from filenames
    date_range = extract_date_range(csv_files)
    
    print("\nInitial data check complete.")

    if not csv_files:
        print(f"\n⚠️  No CSV files found in {INPUT_FOLDER}")
        print("The folder structure has been created successfully.")
        print("Please place your CSV files in the input directory and run the script again.")
        print(f"\nNext steps:")
        print(f"  1. Add your CSV files to: {INPUT_FOLDER}")
        print(f"  2. Expected format: matched_data_YYYY_MM.csv")
        print(f"  3. Example: matched_data_2023_12.csv")
        print(f"  4. Then run: python main.py")
        return
    
    # STEP 6: Check if preprocessing pipeline execution is enabled
    if not EXECUTE_PREPROCESSING_DATA_PIPELINE:
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE EXECUTION SKIPPED")
        print("="*60)
        print("Preprocessing pipeline execution is disabled in configuration.")
        print("To enable preprocessing pipeline execution, set EXECUTE_PREPROCESSING_DATA_PIPELINE = True in config/const.py")
        print(f"Found {len(csv_files)} CSV files ready for processing when enabled.")
    else:
        # STEP 7: Initialize and run the preprocessing pipeline
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE INITIALIZATION")
        print("="*60)
        
        print(f"Initializing Preprocessing Pipeline...")
        preprocessing_pipeline = PreprocessingPipeline()  # CHANGED: Use new PreprocessingPipeline class
        
        print(f"Preprocessing state machine configuration: {PREPROCESSING_STATE_MACHINE}")
        print(f"Starting preprocessing pipeline execution with target feature: '{DEFAULT_TARGET_FEATURE}'")
        
        # Run the full preprocessing pipeline on all CSV files with the default target feature
        preprocessing_results = preprocessing_pipeline.run_pipeline(  # CHANGED: Use PreprocessingPipeline instance
            csv_files, 
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
        training_pipeline = TrainingPipeline()  # UNCHANGED: Still use TrainingPipeline for training operations
        
        print(f"Training state machine configuration: {TRAINING_STATE_MACHINE}")
        print(f"Starting training pipeline execution...")
        
        # Run the training pipeline steps
        training_results = training_pipeline.execute_training_pipeline_steps(
            csv_files,
            state_machine=TRAINING_STATE_MACHINE
        )
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE EXECUTION COMPLETED")
        print("="*60)
        
        # Display training summary
        if training_results:
            if training_results.get('success', False):
                print("✓ Training pipeline completed successfully!")
                print(f"Steps executed: {', '.join(training_results.get('steps_executed', []))}")
                file_info = training_results.get('file_info', {})
                print(f"Files processed: {file_info.get('processed_files', 0)}/{file_info.get('total_files', 0)}")
            else:
                print("✗ Training pipeline failed!")
                errors = training_results.get('errors', [])
                if errors:
                    print("Errors encountered:")
                    for error in errors:
                        print(f"  - {error}")
        else:
            print("✗ Training pipeline returned no results!")


if __name__ == "__main__":
    main()