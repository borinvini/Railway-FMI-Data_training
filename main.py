import os
from src.file_utils import check_csv_files, extract_date_range, load_and_preview_csv
from src.data_preprocessor import TrainingPipeline  # Import the class


def main():
    """
    Main function to execute the script.
    """
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
    
    # Run the full pipeline on all CSV files
    results = pipeline.run_pipeline(csv_files)
    
    # Access the results if needed for additional operations
    print("\nPipeline execution completed.")
    
    # Additional operations with the results could be added here


if __name__ == "__main__":
    main()