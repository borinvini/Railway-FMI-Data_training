import os
import re
import pandas as pd
from collections import defaultdict

def combine_monthly_files_by_custom_seasons(input_folder, output_folder):
    """
    Combines monthly CSV files into seasonal CSV files with custom season definitions and edge case handling.
    
    Season definitions:
    - Winter: December, January, February
    - Spring: March, April, May
    - Summer: June, July, August
    - Autumn: September, October, November

    Args:
        input_folder (str): Path to the folder containing monthly CSV files.
        output_folder (str): Path to the folder where seasonal CSV files will be saved.
    """

    # Map each month to its season and its order within the season
    # Updated season definitions:
    # Winter: Dec(1), Jan(2), Feb(3)
    # Spring: Mar(1), Apr(2), May(3)
    # Summer: Jun(1), Jul(2), Aug(3)
    # Autumn: Sep(1), Oct(2), Nov(3)
    MONTH_TO_SEASON = {
        12: ('winter', 1), 1: ('winter', 2), 2: ('winter', 3),
        3:  ('spring', 1), 4: ('spring', 2), 5: ('spring', 3),
        6:  ('summer', 1), 7: ('summer', 2), 8: ('summer', 3),
        9:  ('autumn', 1), 10: ('autumn', 2), 11: ('autumn', 3)
    }

    # Only process files matching the expected pattern: matched_data_YYYY_MM.csv
    files = [f for f in os.listdir(input_folder) if re.match(r'^matched_data_\d{4}_\d{2}\.csv$', f)]

    # Dictionary to group files by (start_year, end_year, season)
    season_groups = defaultdict(list)

    # Parse each file and assign it to the correct season group
    for file in files:
        parts = file.split('_')
        if len(parts) < 4:
            continue  # Skip files that don't match the expected pattern
        year = int(parts[2])
        month = int(parts[3][:2])
        season_info = MONTH_TO_SEASON.get(month)
        if not season_info:
            continue  # Skip if month is not in our mapping
        season, season_order = season_info

        # Determine the season's year range
        # For winter, December belongs to the current year's winter
        # January and February belong to the previous year's winter
        if season == 'winter':
            if month == 12:
                start_year = year
                end_year = year
            else:  # Jan, Feb
                start_year = year - 1
                end_year = year - 1
        else:
            start_year = year
            end_year = year

        # Store (month, filepath) for sorting and reading later
        season_groups[(start_year, end_year, season)].append((month, os.path.join(input_folder, file)))

    # Sort months within each season for correct order
    for key in season_groups:
        season_groups[key].sort()  # sorts by month

    # Find the minimum and maximum years present in the data
    years = sorted({int(f.split('_')[2]) for f in files})
    min_year, max_year = years[0], years[-1]

    # --- Handle edge cases for first and last winter ---

    # First winter: only Jan, Feb of min_year, save as matched_data_{min_year}_winter.csv
    first_winter_key = (min_year-1, min_year-1, 'winter')
    if first_winter_key in season_groups:
        # Only keep Jan, Feb if present
        months_files = [item for item in season_groups[first_winter_key] if item[0] in [1, 2]]
        if months_files:
            # Save with new key for single year, using the actual year of the data
            season_groups[(min_year, min_year, 'winter_first')] = months_files
        # Remove the old key to avoid double saving
        del season_groups[first_winter_key]

    # Last winter: only Dec of max_year, save as matched_data_{max_year}_winter.csv
    last_winter_key = (max_year, max_year, 'winter')
    if last_winter_key in season_groups:
        months_files = [item for item in season_groups[last_winter_key] if item[0] == 12]
        if months_files:
            # Save with new key for single year
            season_groups[(max_year, max_year, 'winter_last')] = months_files
        # Remove the old key to avoid double saving
        del season_groups[last_winter_key]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Combine and save each season's data
    for (start_year, end_year, season), month_file_list in season_groups.items():
        # Read and concatenate all DataFrames for this season
        dfs = [pd.read_csv(fpath) for _, fpath in month_file_list]
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Determine filename based on season and edge cases
            if season == 'winter_first':
                # Use the actual year of the data for the first winter
                filename = f"matched_data_{min_year}_winter.csv"
            elif season == 'winter_last':
                filename = f"matched_data_{max_year}_winter.csv"
            elif start_year == end_year:
                filename = f"matched_data_{start_year}_{season}.csv"
            else:
                filename = f"matched_data_{start_year}_{end_year}_{season}.csv"
            # Save the combined DataFrame to CSV
            combined_df.to_csv(os.path.join(output_folder, filename), index=False)
            print(f"Saved: {filename}")

if __name__ == "__main__":
    # Set input and output folders
    input_folder = os.path.join("data", "input")
    output_folder = os.path.join("data", "seasonal_data")
    # Run the seasonal combination function
    combine_monthly_files_by_custom_seasons(input_folder, output_folder)