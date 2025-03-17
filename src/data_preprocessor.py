import pandas as pd
import ast

from src.file_utils import generate_output_path

def preprocess_csv_file(input_file_path):
    """
    Preprocess a CSV file by extracting nested data from timeTableRows,
    keeping only essential columns, and expanding weather conditions.
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file.
        
    Returns:
    --------
    pandas.DataFrame
        The processed DataFrame that was saved to the output file.
    """
    try:
        # Generate output path and load the dataframe
        output_file_path, df = generate_output_path(input_file_path)
        
        # Extract nested data from the "timeTableRows" column
        cross_records = []
        for index, row in df.iterrows():
            ttr = row.get("timeTableRows", None)
            if ttr is None or pd.isnull(ttr):
                continue  # Skip empty values
                
            # If the value is a string, attempt to convert it
            if isinstance(ttr, str):
                try:
                    # Replace unquoted 'nan' with 'None' so ast.literal_eval can parse it
                    ttr_fixed = ttr.replace("nan", "None")
                    stops = ast.literal_eval(ttr_fixed)
                except Exception as e:
                    print(f"Error parsing timeTableRows in row {index}: {e}")
                    continue
            else:
                stops = ttr  # Assume it's already a Python object
                
            # Ensure stops is a list; if not, wrap it in a list
            if not isinstance(stops, list):
                stops = [stops]
                
            for stop in stops:
                cross_records.append(stop)
        
        print(f"Extracted {len(cross_records)} records from timeTableRows")
        
        # Create a DataFrame from the extracted dictionaries
        cross_df = pd.DataFrame(cross_records)
        
        # Rename 'weather_observations' to 'weather_conditions' if it exists
        if "weather_observations" in cross_df.columns:
            cross_df = cross_df.rename(columns={"weather_observations": "weather_conditions"})
            print("Renamed 'weather_observations' to 'weather_conditions'")
        
        # Keep only the columns: "differenceInMinutes", "cancelled", and "weather_conditions"
        expected_cols = ["differenceInMinutes", "cancelled", "weather_conditions"]
        available_cols = [col for col in expected_cols if col in cross_df.columns]
        
        if set(expected_cols).issubset(cross_df.columns):
            cross_df = cross_df[expected_cols]
            print(f"Kept only columns: {expected_cols}")
        else:
            print(f"Warning: Not all expected columns are available. Available columns:")
            print(list(cross_df.columns))
            print(f"Keeping available columns from expected list: {available_cols}")
            cross_df = cross_df[available_cols]
        
        # Expand the 'weather_conditions' dictionaries into separate columns
        if "weather_conditions" in cross_df.columns:
            weather_df = cross_df["weather_conditions"].apply(pd.Series)
            
            # Drop unwanted keys if they exist
            weather_df = weather_df.drop(columns=["closest_ems", "Present weather (auto)"], errors="ignore")
            
            # Join the expanded weather conditions back to the main DataFrame
            cross_df = cross_df.drop("weather_conditions", axis=1).join(weather_df)
            print("Expanded weather_conditions into separate columns")
        
        # Reorder columns: differenceInMinutes, cancelled, then the expanded weather columns
        base_cols = [col for col in ["differenceInMinutes", "cancelled"] if col in cross_df.columns]
        other_cols = [col for col in cross_df.columns if col not in base_cols]
        cols_order = base_cols + other_cols
        cross_df = cross_df[cols_order]
        print("Reordered columns")
        
        # Save the final DataFrame to a CSV file
        cross_df.to_csv(output_file_path, index=False)
        print(f"Successfully saved processed data to {output_file_path}")
        
        return cross_df
        
    except Exception as e:
        print(f"Error processing file {input_file_path}: {e}")
        return None