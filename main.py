import os
import argparse
from src.file_utils import check_csv_files, extract_date_range, load_and_preview_csv
from src.data_preprocessor import TrainingPipeline  # Import the class
from config.const import IMPORTANCE_THRESHOLD  # Import the constant


def main():
    """
    Main function to execute the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process railway data and train models.')
    parser.add_argument('--target', choices=['differenceInMinutes', 'trainDelayed', 'cancelled'], 
                        default='trainDelayed', help='Target feature to predict')
    parser.add_argument('--feature-importance-threshold', type=float, default=IMPORTANCE_THRESHOLD,
                        help='Threshold for selecting important features')
    parser.add_argument('--skip-important-features', action='store_true',
                        help='Skip training with important features')
    
    args = parser.parse_args()
    
    print("Starting Railway FMI Data Check...")
    
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
    
    # If we're not skipping important features training and no month has been processed yet,
    # we can train models with only important features for each month separately
    if not args.skip_important_features and results.get("successful_important_features", 0) == 0:
        print("\n" + "="*50)
        print("Starting training with important features for each month")
        print("="*50)
        
        # Get list of preprocessed files
        preprocessed_dir = os.path.join("data", "output", "preprocessed")
        if os.path.exists(preprocessed_dir):
            processed_files = [f for f in os.listdir(preprocessed_dir) 
                              if f.endswith(".csv") and not (f.endswith("_train.csv") or f.endswith("_test.csv"))]
            
            for file in processed_files:
                # Extract month_id from filename
                month_id = file.replace(f"{pipeline.DATA_FILE_PREFIX_FOR_TRAINING}", "").replace(".csv", "")
                
                print(f"\nTraining decision tree with important features for {month_id}...")
                result = pipeline.train_with_important_features(
                    month_id, 
                    importance_threshold=args.feature_importance_threshold
                )
                
                if result.get("success", False):
                    print(f"Successfully trained decision tree with important feature-s for {month_id}")
                    print(f"Selected {len(result.get('important_features', []))} important features")
                    print(f"Test accuracy: {result.get('accuracy', 0):.4f}")
                else:
                    print(f"Failed to train decision tree with important features for {month_id}")
    
    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()

# NEW