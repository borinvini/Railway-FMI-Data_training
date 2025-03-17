import os
from src.file_utils import check_csv_files, extract_date_range, load_and_preview_csv
from src.data_preprocessor import preprocess_csv_file


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
    
    # Process all CSV files found
    print(f"\nFound {len(csv_files)} CSV files to process.")
    
    successful_files = 0
    failed_files = 0
    
    for i, input_file_path in enumerate(csv_files):
        filename = os.path.basename(input_file_path)
        print(f"\n[{i+1}/{len(csv_files)}] Processing file: {filename}")
        
        # Load and preview the CSV
        df = load_and_preview_csv(filename)
        
        if df is None:
            print(f"Error loading {filename}. Skipping to next file.")
            failed_files += 1
            continue
        
        print(f"DataFrame loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        
        # Process the CSV file
        print(f"Starting preprocessing for {filename}...")
        
        # Call the preprocessing function
        processed_df = preprocess_csv_file(input_file_path)
        
        if processed_df is not None:
            print(f"Preprocessing completed successfully for {filename}")
            print(f"Processed data contains {len(processed_df)} rows and {len(processed_df.columns)} columns")
            successful_files += 1
        else:
            print(f"Error occurred during preprocessing of {filename}")
            failed_files += 1
    
    # Summary
    print("\n" + "="*50)
    print("Processing Summary:")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed to process: {failed_files}")
    print("="*50)

if __name__ == "__main__":
    main()