import os
import argparse
from src.file_utils import check_csv_files, extract_date_range, load_and_preview_csv
from src.data_preprocessor import TrainingPipeline  # Import the class
from config.const import DEFAULT_TARGET_FEATURE, FILTER_TRAINS_BY_STATIONS, IMPORTANCE_THRESHOLD, REQUIRED_STATIONS  # Import the constant


def main():
    """
    Main function to execute the script.
    """
    # Parse command line arguments
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
    
    print("Starting Railway FMI Data Check...")

    if FILTER_TRAINS_BY_STATIONS:
        print(f"\n⚠️  WARNING: Train filtering is ENABLED!")
        print(f"Only trains passing through ALL of these stations will be processed: {', '.join(REQUIRED_STATIONS)}")
        print(f"This may significantly reduce the amount of data available for training.\n")
    
    
    # Check for CSV files
    csv_files = check_csv_files()
            
    # Extract date range from filenames
    date_range = extract_date_range(csv_files)
    
    print("\nCheck complete.")

    if not csv_files:
        print("\nNo CSV files to process. Exiting.")
        return
    
    # Initialize the TrainingPipeline class
    pipeline = TrainingPipeline()
    
    # Run the full pipeline on all CSV files with the specified target feature
    results = pipeline.run_pipeline_data_by_month(
        csv_files, 
        target_feature=args.target
    )
    
    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()