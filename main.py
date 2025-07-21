import os
import argparse
from src.file_utils import check_csv_files, extract_date_range, load_and_preview_csv, ensure_folder_structure
from src.data_preprocessor import TrainingPipeline  # Import the class
from config.const import (
    DEFAULT_TARGET_FEATURE, 
    FILTER_TRAINS_BY_STATIONS, 
    IMPORTANCE_THRESHOLD, 
    REQUIRED_STATIONS,
    INPUT_FOLDER
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
    
    # STEP 2: Parse command line arguments
    parser = argparse.ArgumentParser(description='Process railway data and train models.')
    parser.add_argument('--target', 
                    choices=['differenceInMinutes', 'trainDelayed', 'cancelled', 'differenceInMinutes_offset'], 
                    default=DEFAULT_TARGET_FEATURE, 
                    help='Target feature to predict')
    parser.add_argument('--feature-importance-threshold', type=float, default=IMPORTANCE_THRESHOLD,
                        help='Threshold for selecting important features')
    parser.add_argument('--skip-important-features', action='store_true',
                        help='Skip training with important features')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STARTING RAILWAY FMI DATA CHECK AND PROCESSING")
    print("="*60)

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
        print(f"  4. Then run: python main.py --target trainDelayed")
        return
    
    # STEP 6: Initialize and run the training pipeline
    print(f"\nInitializing Training Pipeline...")
    pipeline = TrainingPipeline()
    
    print(f"Starting pipeline execution with target feature: '{args.target}'")
    
    # Run the full pipeline on all CSV files with the specified target feature
    results = pipeline.run_pipeline_data_by_month(
        csv_files, 
        target_feature=args.target
    )
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED")
    print("="*60)
    
    # Display final summary
    if results:
        success_rate = (results.get('successful_preprocessing', 0) / results.get('total_months', 1)) * 100
        print(f"Overall success rate: {success_rate:.1f}%")
        
        if results.get('failed_files', 0) > 0:
            print(f"⚠️  {results.get('failed_files', 0)} months failed processing")
        else:
            print("✓ All months processed successfully!")


if __name__ == "__main__":
    main()