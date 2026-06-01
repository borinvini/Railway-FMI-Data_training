# Pipeline Parquet Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the pipeline to accept pre-matched parquet input files (one row per train stop, weather already present and merged), removing the three pipeline steps that are now obsolete and swapping all file I/O from CSV to Parquet.

**Architecture:** Surgical changes only — delete `extract_nested_data`, `add_weather_1h_window_features`, and `merge_weather_columns` pipeline steps; rename CSV-named functions to parquet equivalents; update all read/write I/O. No logic changes to the remaining 15 preprocessing steps or 5 training steps.

**Tech Stack:** Python, pandas (`df.to_parquet` / `pd.read_parquet`), glob, regex

---

## File Map

| File | What changes |
|------|-------------|
| `config/const_preprocessing.py` | Update `PREPROCESSING_STATE_MACHINE`, delete `FOLDER_EXTRACT_NESTED_DATA`, `FOLDER_ADD_WEATHER_1H_WINDOW_FEATURES`, `FOLDER_MERGE_WEATHER_COLUMNS`, `WEATHER_COLS_TO_MERGE`, and the entire FMI window weather constants block |
| `src/preprocessing_pipeline.py` | Update imports, delete `extract_nested_data` + `add_weather_1h_window_features` + `merge_weather_columns` methods and their execution blocks, rename `save_month_df_to_csv` → `save_month_df_to_parquet` and `save_training_ready_csv` → `save_training_ready_parquet`, update all `save_dataframe_to_csv` calls, update filename regex, add upfront parquet file load |
| `src/file_utils.py` | Rename `save_dataframe_to_csv` → `save_dataframe_to_parquet`, rename `check_csv_files` → `check_parquet_files`, update `extract_date_range` regex, update `generate_output_path` |
| `main.py` | Remove EMS station mapping step + import, rename `check_csv_files` → `check_parquet_files`, rename `csv_files` → `parquet_files`, update print messages |
| `src/training_pipeline.py` | Update glob pattern, filename regex, and `pd.read_csv` in `merge_data_files` step |

---

## Task 1: Update `config/const_preprocessing.py`

**Files:**
- Modify: `config/const_preprocessing.py`

- [ ] **Step 1: Replace `PREPROCESSING_STATE_MACHINE`**

Find the current `PREPROCESSING_STATE_MACHINE` dict (starts at line 3) and replace it entirely with:

```python
PREPROCESSING_STATE_MACHINE = {
    "filter_by_target_station": True,
    "process_causes_column": True,
    "add_train_delayed_feature": True,
    "add_weather_scenarios_col": True,
    "weather_scenario_one_hot_encoder": True,
    "process_actual_time_column": True,
    "filter_columns": True,
    "convert_boolean_to_numeric": True,
    "handle_missing_values": True,
    "save_month_df_to_parquet": True,
    "convert_hour_to_sincos": True,
    "convert_month_to_sincos": True,
    "convert_dayofweek_to_sincos": True,
    "drop_original_temporal_columns": False,
    "select_target": False,
    "filter_strong_weather_causes": False,
    "remove_duplicates": True,
    "save_training_ready_parquet": True
}
```

- [ ] **Step 2: Delete the three folder constants for deleted steps**

Find and delete these three lines:
```python
FOLDER_EXTRACT_NESTED_DATA = "data/output/1-extract_nested_data"
```
```python
FOLDER_ADD_WEATHER_1H_WINDOW_FEATURES = "data/output/6-add_weather_1h_window_features"
```
```python
FOLDER_MERGE_WEATHER_COLUMNS = "data/output/5-merge_weather_columns"
```

- [ ] **Step 3: Delete `WEATHER_COLS_TO_MERGE`**

Find and delete:
```python
# List of weather features that has 2 cols and need to merge in 1 col
WEATHER_COLS_TO_MERGE = [
    "Snow depth", 
    "Precipitation amount", 
    "Precipitation intensity", 
    "Horizontal visibility", 
    "Wind speed", 
    "Gust speed"
]
```

- [ ] **Step 4: Delete the FMI window weather constants block**

Find the comment `# Path to FMI weather data for 1h window feature calculation` and delete from that line through the end of the file. The block starts at:
```python
# Path to FMI weather data for 1h window feature calculation
WINDOW_WEATHER_DATA_FOLDER = "data/input/window_weather_data"
```
And ends at the closing `}` of `WEATHER_1H_WINDOW_FEATURE_DESCRIPTIONS`. This removes ~200 lines covering all FMI window constants, the 24 `*_1H_WINDOW_*` column name constants, `WEATHER_1H_WINDOW_OUTPUT_COLS`, `ALL_1H_WINDOW_OUTPUT_COLUMNS`, `ALL_WEATHER_FEATURES.extend(...)`, and `WEATHER_1H_WINDOW_FEATURE_DESCRIPTIONS`.

- [ ] **Step 5: Verify**

```powershell
python -c "from config.const_preprocessing import PREPROCESSING_STATE_MACHINE; print(list(PREPROCESSING_STATE_MACHINE.keys()))"
```
Expected: 18-key list starting with `filter_by_target_station`, ending with `save_training_ready_parquet`. No import errors.

- [ ] **Step 6: Commit**

```bash
git add config/const_preprocessing.py
git commit -m "Remove FMI window constants, merge_weather_columns config, and deleted step entries"
```

---

## Task 2: Update imports and delete obsolete methods in `src/preprocessing_pipeline.py`

**Files:**
- Modify: `src/preprocessing_pipeline.py`

- [ ] **Step 1: Replace the import block at the top of the file**

Find the current import section (lines 14–77) — the `from src.file_utils import ...` line and the entire `from config.const_preprocessing import (...)` block — and replace with:

```python
from src.file_utils import generate_output_path, save_dataframe_to_parquet

from config.const_preprocessing import (
    FOLDER_ADD_TRAIN_DELAYED_FEATURE,
    FOLDER_ADD_WEATHER_SCENARIOS_COL,
    FOLDER_CONVERT_BOOLEAN_TO_NUMERIC,
    FOLDER_CONVERT_DAYOFWEEK_TO_SINCOS,
    FOLDER_CONVERT_HOUR_TO_SINCOS,
    FOLDER_CONVERT_MONTH_TO_SINCOS,
    FOLDER_DROP_ORIGINAL_TEMPORAL_COLUMNS,
    FOLDER_FILTER_BY_TARGET_STATION,
    FOLDER_FILTER_COLUMNS,
    FOLDER_FILTER_STRONG_WEATHER_CAUSES,
    FOLDER_HANDLE_MISSING_VALUES,
    FOLDER_PROCESS_ACTUAL_TIME_COLUMN,
    FOLDER_PROCESS_CAUSES_COLUMN,
    FOLDER_REMOVE_DUPLICATES,
    FOLDER_SELECT_TARGET,
    FOLDER_WEATHER_SCENARIO_ONE_HOT_ENCODER,
    PREPROCESSED_OUTPUT_FOLDER,
    TRAINING_READY_OUTPUT_FOLDER,
    DEFAULT_TARGET_FEATURE,
    PREPROCESSING_STATE_MACHINE,
    TARGET_STATION_CODE,
    TRAIN_DELAYED_TARGET_COLUMN,
    TRAIN_DELAY_MINUTES,
    VALID_WEATHER_SCENARIO_FEATURES,
    VALID_TARGET_FEATURES,
    VALID_TRAIN_PREDICTION_FEATURES,
    ALL_WEATHER_FEATURES,
    BOOLEAN_FEATURES,
    IMPORTANT_WEATHER_FEATURES,
    CATEGORICAL_FEATURES,
    WEATHER_MISSING_THRESHOLD,
    DATA_FILE_PREFIX_FOR_TRAINING,
)
```

- [ ] **Step 2: Delete the `extract_nested_data` method**

Find `def extract_nested_data(self, input_file_path):` (around line 969) and delete the entire method — from the `def` line through its final `return None` line (approximately line 1083, ending just before `def filter_by_target_station`).

- [ ] **Step 3: Delete the `merge_weather_columns` method**

Find `def merge_weather_columns(self, dataframe=None, month_id=None):` (around line 1499) and delete the entire method — from the `def` line through its closing `except` block (ends just before `def add_weather_1h_window_features`).

- [ ] **Step 4: Delete the `add_weather_1h_window_features` method**

Find `def add_weather_1h_window_features(self, dataframe=None, month_id=None):` (around line 1663) and delete the entire method — from the `def` line through its closing `except` block (~500 lines, ends just before the next `def`).

- [ ] **Step 5: Delete the `extract_nested_data` execution block from `execute_preprocessing_pipeline_steps`**

Find and delete the entire block:
```python
        if state_machine.get("extract_nested_data", False):
            try:
                print(f"    → extract_nested_data")
                processed_df = self.extract_nested_data(input_file_path)
                
                if processed_df is not None and not processed_df.empty:
                    # Add year information for reference
                    processed_df['data_year'] = year
                    result["data"] = processed_df
                    result["steps_executed"].append("extract_nested_data")
                    result["file_info"]["rows"] = len(processed_df)
                    result["file_info"]["columns"] = len(processed_df.columns)
                    print(f"      ✓ Extracted {len(processed_df)} rows, {len(processed_df.columns)} columns")
                else:
                    result["errors"].append("extract_nested_data returned empty data")
                    print(f"      ✗ Failed - empty result")
                    return result
                    
            except Exception as e:
                result["errors"].append(f"extract_nested_data failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ extract_nested_data (disabled)")
```

- [ ] **Step 6: Delete the `merge_weather_columns` execution block from `execute_preprocessing_pipeline_steps`**

Find and delete the entire block (approximately lines 474–499):
```python
        if state_machine.get("merge_weather_columns", False):
            if result["data"] is not None:
                try:
                    print(f"    → merge_weather_columns")
                    merged_df = self.merge_weather_columns(dataframe=result["data"], month_id=file_id)
                    
                    if merged_df is not None:
                        result["data"] = merged_df
                        result["steps_executed"].append("merge_weather_columns")
                        result["file_info"]["rows"] = len(merged_df)
                        result["file_info"]["columns"] = len(merged_df.columns)
                        print(f"      ✓ Merged weather columns for {len(merged_df)} rows")
                    else:
                        result["errors"].append("merge_weather_columns failed")
                        print(f"      ✗ Failed to merge weather columns")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"merge_weather_columns failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ merge_weather_columns (no data available)")
                result["errors"].append("merge_weather_columns skipped - no data available")
        else:
            print(f"    ⊝ merge_weather_columns (disabled)")
```

- [ ] **Step 7: Delete the `add_weather_1h_window_features` execution block from `execute_preprocessing_pipeline_steps`**

Find and delete the entire block (approximately lines 501–529):
```python
        if state_machine.get("add_weather_1h_window_features", False):
            if result["data"] is not None:
                try:
                    print(f"    → add_weather_1h_window_features")
                    weather_1h_window_df = self.add_weather_1h_window_features(
                        dataframe=result["data"], 
                        month_id=file_id
                    )
                    
                    if weather_1h_window_df is not None:
                        result["data"] = weather_1h_window_df
                        result["steps_executed"].append("add_weather_1h_window_features")
                        result["file_info"]["rows"] = len(weather_1h_window_df)
                        result["file_info"]["columns"] = len(weather_1h_window_df.columns)
                        print(f"      ✓ Added 1h window weather features: {len(weather_1h_window_df)} rows, {len(weather_1h_window_df.columns)} columns")
                    else:
                        result["errors"].append("add_weather_1h_window_features failed")
                        print(f"      ✗ Failed to add 1h window weather features")
                        return result
                        
                except Exception as e:
                    result["errors"].append(f"add_weather_1h_window_features failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ add_weather_1h_window_features (no data available)")
                result["errors"].append("add_weather_1h_window_features skipped - no data available")
        else:
            print(f"    ⊝ add_weather_1h_window_features (disabled)")
```

- [ ] **Step 8: Add upfront parquet file load in `execute_preprocessing_pipeline_steps`**

Since `extract_nested_data` no longer loads the file, the pipeline needs to load the parquet input itself. Find the `result` dict initialization block (around line 340–351) and add the following immediately after it:

```python
        # Load input parquet file
        try:
            print(f"  Loading input file: {input_file_path}")
            df = pd.read_parquet(input_file_path)
            print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            result["data"] = df
            result["file_info"]["rows"] = len(df)
            result["file_info"]["columns"] = len(df.columns)
        except Exception as e:
            result["errors"].append(f"Failed to load input file: {str(e)}")
            print(f"  ✗ Failed to load {input_file_path}: {str(e)}")
            return result
```

- [ ] **Step 9: Verify imports work**

```powershell
python -c "from src.preprocessing_pipeline import PreprocessingPipeline; print('OK')"
```
Expected: `OK` with no import errors.

- [ ] **Step 10: Commit**

```bash
git add src/preprocessing_pipeline.py
git commit -m "Remove 3 obsolete pipeline steps: extract_nested_data, merge_weather_columns, add_weather_1h_window_features"
```

---

## Task 3: Rename save methods and update all CSV calls in `src/preprocessing_pipeline.py`

**Files:**
- Modify: `src/preprocessing_pipeline.py`

- [ ] **Step 1: Update the filename regex in `run_pipeline`**

Find (around line 204):
```python
            match = re.search(r'(\d{4})_(\d{2})\.csv$', filename)
```
Replace with:
```python
            match = re.search(r'(\d{4})_(\d{2})\.parquet$', filename)
```

Also find the print statement nearby:
```python
                print(f"  {info['original_file']} -> preprocessed_data_{info['file_id']}.csv")
```
Replace with:
```python
                print(f"  {info['original_file']} -> preprocessed_data_{info['file_id']}.parquet")
```

- [ ] **Step 2: Replace all `save_dataframe_to_csv` calls with `save_dataframe_to_parquet`**

In `src/preprocessing_pipeline.py` only, replace every occurrence of:
- `save_dataframe_to_csv(` → `save_dataframe_to_parquet(`

There are approximately 12 occurrences (one per remaining pipeline step that saves intermediate output). Verify with:
```powershell
Select-String -Path "src\preprocessing_pipeline.py" -Pattern "save_dataframe_to_csv"
```
Expected: zero results after replacement.

- [ ] **Step 3: Rename `save_month_df_to_csv` method**

Find the method definition (around line 3276, now shifted due to deletions):
```python
    def save_month_df_to_csv(self, month_id, dataframe):
```
Replace the entire method with:
```python
    def save_month_df_to_parquet(self, month_id, dataframe):
        """
        Save a processed month's dataframe to a parquet file.
        """
        try:
            if dataframe is None or dataframe.empty:
                print(f"Warning: Cannot save empty dataframe for {month_id}")
                return False

            filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{month_id}.parquet"
            file_path = os.path.join(self.preprocessed_dir, filename)

            os.makedirs(self.preprocessed_dir, exist_ok=True)

            dataframe.to_parquet(file_path, index=False)
            print(f"Successfully saved dataframe to {file_path}")
            return True

        except Exception as e:
            print(f"Error saving dataframe for {month_id}: {e}")
            return False
```

- [ ] **Step 4: Rename `save_training_ready_csv` method**

Find the method definition (around line 4410, now shifted):
```python
    def save_training_ready_csv(self, month_id, dataframe):
```
Replace the entire method with:
```python
    def save_training_ready_parquet(self, month_id, dataframe):
        """
        Save a training-ready dataframe to the training-ready directory as parquet.
        """
        try:
            if dataframe is None or dataframe.empty:
                print(f"Warning: Cannot save empty dataframe for {month_id}")
                return False

            training_ready_dir = TRAINING_READY_OUTPUT_FOLDER
            filename = f"training_ready_{month_id}.parquet"
            file_path = os.path.join(training_ready_dir, filename)

            os.makedirs(training_ready_dir, exist_ok=True)

            dataframe.to_parquet(file_path, index=False)
            print(f"Successfully saved training-ready dataframe to {file_path}")
            return True

        except Exception as e:
            print(f"Error saving training-ready dataframe for {month_id}: {e}")
            return False
```

- [ ] **Step 5: Update execution blocks for the two renamed save steps**

In `execute_preprocessing_pipeline_steps`, find:
```python
        if state_machine.get("save_month_df_to_csv", False):
```
Replace with:
```python
        if state_machine.get("save_month_df_to_parquet", False):
```
Inside that block, find:
```python
                    save_success = self.save_month_df_to_csv(file_id, result["data"])
```
Replace with:
```python
                    save_success = self.save_month_df_to_parquet(file_id, result["data"])
```

Find:
```python
        if state_machine.get("save_training_ready_csv", False):
```
Replace with:
```python
        if state_machine.get("save_training_ready_parquet", False):
```
Inside that block, find:
```python
                    save_success = self.save_training_ready_csv(file_id, result["data"])
```
Replace with:
```python
                    save_success = self.save_training_ready_parquet(file_id, result["data"])
```

- [ ] **Step 6: Verify no stale CSV references remain**

```powershell
Select-String -Path "src\preprocessing_pipeline.py" -Pattern "save_dataframe_to_csv|save_month_df_to_csv|save_training_ready_csv|merge_weather_columns|extract_nested_data|add_weather_1h_window_features|WEATHER_COLS_TO_MERGE"
```
Expected: zero results.

- [ ] **Step 7: Verify import still works**

```powershell
python -c "from src.preprocessing_pipeline import PreprocessingPipeline; print('OK')"
```
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add src/preprocessing_pipeline.py
git commit -m "Rename CSV save methods to parquet, update all intermediate save calls"
```

---

## Task 4: Update `src/file_utils.py`

**Files:**
- Modify: `src/file_utils.py`

- [ ] **Step 1: Replace `save_dataframe_to_csv` with `save_dataframe_to_parquet`**

Find the function definition (around line 41):
```python
def save_dataframe_to_csv(folder_path: str, month_id: str, df: pd.DataFrame, 
                          file_prefix: str = "data") -> str:
```
Replace the entire function with:
```python
def save_dataframe_to_parquet(folder_path: str, month_id: str, df: pd.DataFrame,
                              file_prefix: str = "data") -> str:
    """
    Generic function to save a DataFrame to a parquet file in a specified folder.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{file_prefix}_{month_id}.parquet"
    file_path = folder / filename

    df.to_parquet(file_path, index=False)

    print(f"Successfully saved DataFrame to {file_path}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return str(file_path)
```

- [ ] **Step 2: Replace `check_csv_files` with `check_parquet_files`**

Find the function definition (around line 137):
```python
def check_csv_files():
```
Replace the entire function with:
```python
def check_parquet_files():
    """
    Check if there are parquet files in the data/input folder and print their names.
    Returns a list of parquet file paths found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_folder = os.path.join(project_root, INPUT_FOLDER)

    print(f"Looking for parquet files in: {input_folder}")

    parquet_files = []
    if os.path.exists(input_folder):
        parquet_files = glob.glob(os.path.join(input_folder, '*.parquet'))

    if parquet_files:
        print(f"\n✓ Success! Found {len(parquet_files)} parquet files in the input folder:")
        for file in parquet_files:
            file_size = os.path.getsize(file) / (1024 * 1024)
            print(f"  - {os.path.basename(file)} ({file_size:.1f} MB)")
    else:
        print(f"\n⚠️  No parquet files found in the input folder.")
        print(f"\nTo get started:")
        print(f"  1. Place your parquet files in: {input_folder}")
        print(f"  2. Expected file format: {DATA_FILE_PREFIX}YYYY_MM.parquet")
        print(f"  3. Example: {DATA_FILE_PREFIX}2023_12.parquet")
        print(f"  4. Then run: python main.py")

    return parquet_files
```

- [ ] **Step 3: Update `extract_date_range` regex**

Find (around line 189):
```python
    pattern = f'{DATA_FILE_PREFIX}(\\d{{4}})_(\\d{{2}})\\.csv'  # Use the constant here
```
Replace with:
```python
    pattern = f'{DATA_FILE_PREFIX}(\\d{{4}})_(\\d{{2}})\\.parquet'
```

- [ ] **Step 4: Update `generate_output_path`**

Find (around line 239):
```python
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year}_{month}.csv"
```
Replace with:
```python
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year}_{month}.parquet"
```

Find:
```python
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}processed.csv"
```
Replace with:
```python
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}processed.parquet"
```

Find:
```python
    print(f"Loading CSV file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
```
Replace with:
```python
    print(f"Loading parquet file: {input_file_path}")
    df = pd.read_parquet(input_file_path)
    print(f"Successfully loaded parquet with {len(df)} rows and {len(df.columns)} columns")
```

- [ ] **Step 5: Verify**

```powershell
python -c "from src.file_utils import check_parquet_files, save_dataframe_to_parquet; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/file_utils.py
git commit -m "Rename CSV I/O functions to parquet equivalents in file_utils"
```

---

## Task 5: Update `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update imports**

Find:
```python
from src.file_utils import check_csv_files, extract_date_range, ensure_folder_structure, find_closest_ems_stations
```
Replace with:
```python
from src.file_utils import check_parquet_files, extract_date_range, ensure_folder_structure
```

- [ ] **Step 2: Delete the EMS station mapping block (Step 1.5)**

Find and delete the entire block:
```python
    # STEP 1.5: Find closest EMS stations for each train station
    # This creates a mapping file used for weather data assignment
    print("\n" + "-" * 60)
    print("METADATA PROCESSING: Train Station to EMS Station Mapping")
    print("-" * 60)
    try:
        closest_ems_df = find_closest_ems_stations(n_closest=5)
        print("✓ Closest EMS stations mapping created successfully!")
    except FileNotFoundError as e:
        print(f"⚠️  Skipping EMS station mapping (metadata files not found):")
        print(f"   {e}")
        print("   Please ensure the following files exist:")
        print("   - data/input/metadata/metadata_train_stations.csv")
        print("   - data/input/metadata/metadata_fmi_ems_stations.csv")
    except Exception as e:
        print(f"✗ Error creating EMS station mapping: {e}")
```

- [ ] **Step 3: Update file discovery and variable name**

Find:
```python
    # STEP 4: Check for CSV files in the input directory
    print(f"Checking for CSV files...")
    csv_files = check_csv_files()
```
Replace with:
```python
    # STEP 4: Check for parquet files in the input directory
    print(f"Checking for parquet files...")
    parquet_files = check_parquet_files()
```

Find:
```python
    # STEP 5: Extract date range from filenames
    date_range = extract_date_range(csv_files)
```
Replace with:
```python
    # STEP 5: Extract date range from filenames
    date_range = extract_date_range(parquet_files)
```

- [ ] **Step 4: Update all remaining `csv_files` references and messages**

Replace every remaining `csv_files` with `parquet_files`. Then update these messages:

Find:
```python
        print(f"\n⚠️  No CSV files found in {INPUT_FOLDER}")
```
Replace with:
```python
        print(f"\n⚠️  No parquet files found in {INPUT_FOLDER}")
```

Find:
```python
        print(f"  2. Expected format: matched_data_YYYY_MM.csv")
```
Replace with:
```python
        print(f"  2. Expected format: matched_data_YYYY_MM.parquet")
```

Find:
```python
        print(f"Found {len(csv_files)} CSV files ready for processing when enabled.")
```
Replace with:
```python
        print(f"Found {len(parquet_files)} parquet files ready for processing when enabled.")
```

- [ ] **Step 5: Verify**

```powershell
python -c "import main; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "Remove EMS mapping step, switch file discovery to parquet"
```

---

## Task 6: Update `src/training_pipeline.py`

**Files:**
- Modify: `src/training_pipeline.py`

- [ ] **Step 1: Update the glob pattern for training-ready files**

Find (around line 854):
```python
            training_ready_pattern = os.path.join(self.project_root, TRAINING_READY_OUTPUT_FOLDER, "training_ready_*.csv")
```
Replace with:
```python
            training_ready_pattern = os.path.join(self.project_root, TRAINING_READY_OUTPUT_FOLDER, "training_ready_*.parquet")
```

- [ ] **Step 2: Update the filename regex**

Find (around line 882):
```python
                    month_match = re.search(r'training_ready_(\d{4})_(\d{2})\.csv$', filename)
```
Replace with:
```python
                    month_match = re.search(r'training_ready_(\d{4})_(\d{2})\.parquet$', filename)
```

- [ ] **Step 3: Update the `read_csv` call for training-ready files**

Find the `pd.read_csv` that reads from the `training_ready_pattern` loop (around line 892). Confirm it is inside the loop over `training_ready_files` — the surrounding context should be:
```python
                    df = pd.read_csv(file_path)
```
Replace with:
```python
                    df = pd.read_parquet(file_path)
```

- [ ] **Step 4: Verify**

```powershell
python -c "from src.training_pipeline import TrainingPipeline; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/training_pipeline.py
git commit -m "Update training pipeline to read training-ready parquet files"
```

---

## Task 7: Smoke Test — Import and Dead Reference Check

**Files:** No changes — verification only

- [ ] **Step 1: Verify all modules import cleanly**

```powershell
python -c "
from config.const_preprocessing import PREPROCESSING_STATE_MACHINE
from config.const_training import TRAINING_STATE_MACHINE
from src.file_utils import check_parquet_files, save_dataframe_to_parquet, extract_date_range, generate_output_path
from src.preprocessing_pipeline import PreprocessingPipeline
from src.training_pipeline import TrainingPipeline
import main
print('All imports OK')
print('Preprocessing steps:', list(PREPROCESSING_STATE_MACHINE.keys()))
"
```
Expected: `All imports OK` followed by the 18-key list starting with `filter_by_target_station`.

- [ ] **Step 2: Confirm no references to deleted symbols remain in active code**

```powershell
Select-String -Path "src\preprocessing_pipeline.py","src\file_utils.py","main.py","src\training_pipeline.py","config\const_preprocessing.py" -Pattern "WINDOW_WEATHER_DATA_FOLDER|TRAIN_STATION_EMS_METADATA_PATH|FMI_TIMESTAMP_COL|FMI_STATION_NAME_COL|WEATHER_WINDOW_MINUTES|ALL_1H_WINDOW_OUTPUT_COLUMNS|WEATHER_1H_WINDOW_FEATURES|FOLDER_EXTRACT_NESTED_DATA|FOLDER_ADD_WEATHER_1H_WINDOW_FEATURES|FOLDER_MERGE_WEATHER_COLUMNS|WEATHER_COLS_TO_MERGE|extract_nested_data|add_weather_1h_window_features|merge_weather_columns|find_closest_ems_stations|check_csv_files|save_dataframe_to_csv|save_month_df_to_csv|save_training_ready_csv"
```
Expected: zero results.

- [ ] **Step 3: Confirm parquet round-trip works**

```powershell
python -c "
import pandas as pd, tempfile
from src.file_utils import save_dataframe_to_parquet

df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
with tempfile.TemporaryDirectory() as tmp:
    path = save_dataframe_to_parquet(tmp, '2024_01', df, 'test')
    loaded = pd.read_parquet(path)
    assert list(loaded.columns) == ['a', 'b']
    assert len(loaded) == 2
print('Parquet round-trip: OK')
"
```
Expected: `Parquet round-trip: OK`

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "Pipeline restructure complete: parquet I/O, remove 3 obsolete steps"
```
