# Pipeline Restructure: Parquet + Pre-matched Data

**Date:** 2026-06-01  
**Branch:** reimplementing-project  
**Approach:** Minimal surgical changes (Approach A)

---

## Context

The input data format has changed in three ways:

1. **File format:** CSV → Parquet (`.parquet`)
2. **Row granularity:** One row per train (all stops in a nested `timeTableRows` column) → one row per train stop
3. **Weather data:** Previously computed as a 1-hour rolling window join against separate FMI files → now pre-matched and present directly in the input file

The goal is to update every pipeline stage to match these new requirements, delete stages that are now unnecessary, and make no other changes.

---

## What Gets Deleted

### Pipeline step: `extract_nested_data`
- **Reason:** The new matched data already has one row per train stop. There is no `timeTableRows` column to unpack.
- **Remove from:** `PREPROCESSING_STATE_MACHINE` dict, method body in `preprocessing_pipeline.py`, `FOLDER_EXTRACT_NESTED_DATA` constant in `const_preprocessing.py`.

### Pipeline step: `add_weather_1h_window_features`
- **Reason:** Weather is pre-matched in the input file. The separate FMI window weather files and the join logic are no longer needed.
- **Remove from:** `PREPROCESSING_STATE_MACHINE` dict, method body in `preprocessing_pipeline.py`, `FOLDER_ADD_WEATHER_1H_WINDOW_FEATURES` constant.

### FMI window weather constants block in `const_preprocessing.py`
The entire section of constants that supported the 1h window join is deleted:
- `WINDOW_WEATHER_DATA_FOLDER`
- `TRAIN_STATION_EMS_METADATA_PATH`
- `TRAIN_SCHEDULED_TIME_COL`, `TRAIN_STATION_SHORT_CODE_COL`
- `FMI_TIMESTAMP_COL`, `FMI_STATION_NAME_COL`
- `WEATHER_WINDOW_MINUTES`
- All 8 `FMI_*_COL` column name constants
- `WEATHER_1H_WINDOW_FEATURES` dict
- `NUM_WEATHER_1H_WINDOW_FEATURES`, `NUM_STATS_PER_FEATURE`, `TOTAL_1H_WINDOW_COLUMNS`
- All 24 `*_1H_WINDOW_{MIN,MAX,MEAN}` output column name constants
- `WEATHER_1H_WINDOW_OUTPUT_COLS` dict
- `ALL_1H_WINDOW_OUTPUT_COLUMNS` list
- `WEATHER_1H_WINDOW_FEATURE_DESCRIPTIONS` dict
- `ALL_WEATHER_FEATURES.extend(ALL_1H_WINDOW_OUTPUT_COLUMNS)` call

### EMS station mapping step in `main.py`
- Step 1.5 (`find_closest_ems_stations`) and its import from `file_utils` are removed. This was only needed to support the FMI weather join.

---

## What Gets Updated

### File format: CSV → Parquet (4 files affected)

#### `file_utils.py`
| Old | New |
|-----|-----|
| `save_dataframe_to_csv()` | `save_dataframe_to_parquet()` — uses `df.to_parquet(..., index=False)`, produces `.parquet` filenames |
| `check_csv_files()` — glob `*.csv` | `check_parquet_files()` — glob `*.parquet` |
| `extract_date_range()` — regex ends `\.csv` | regex ends `\.parquet` |
| `generate_output_path()` — `pd.read_csv(...)` | `pd.read_parquet(...)`, output filename uses `.parquet` |

#### `preprocessing_pipeline.py`
- All `save_dataframe_to_csv(...)` calls → `save_dataframe_to_parquet(...)`
- Filename regex in `run_pipeline()`: `(\d{4})_(\d{2})\.csv$` → `(\d{4})_(\d{2})\.parquet$`
- Pipeline step `save_month_df_to_csv` → renamed to `save_month_df_to_parquet` (state machine key + method name + method body)
- Pipeline step `save_training_ready_csv` → renamed to `save_training_ready_parquet` (state machine key + method name + method body)

#### `training_pipeline.py`
- All `pd.read_csv(...)` calls that load training-ready files → `pd.read_parquet(...)`
- All glob patterns for training-ready files: `*.csv` → `*.parquet`

#### `main.py`
- `check_csv_files()` call → `check_parquet_files()`
- User-facing print messages updated from "CSV" to "parquet" where relevant

### Import updates
- `main.py`: remove import of `find_closest_ems_stations` from `file_utils`; update `check_csv_files` → `check_parquet_files`
- `preprocessing_pipeline.py`: remove all imports of deleted FMI window weather constants

---

## What Stays Unchanged

### Preprocessing pipeline steps (16 steps, logic untouched)
`filter_by_target_station`, `process_causes_column`, `add_train_delayed_feature`, `merge_weather_columns`, `add_weather_scenarios_col`, `weather_scenario_one_hot_encoder`, `process_actual_time_column`, `filter_columns`, `convert_boolean_to_numeric`, `handle_missing_values`, `convert_hour_to_sincos`, `convert_month_to_sincos`, `convert_dayofweek_to_sincos`, `drop_original_temporal_columns`, `select_target`, `filter_strong_weather_causes`, `remove_duplicates`

### Training pipeline steps (logic untouched)
`merge_data_files`, `select_training_cols`, `split_dataset`, `scale_weather_features`, `train_xgboost_with_randomized_search_cv`

### All config constants not related to FMI window weather
Weather scenario definitions, delay thresholds, target feature selection, SMOTE config, XGBoost hyperparameters, boolean/temporal feature lists, etc.

### Output folder numbering
Folder name constants keep their current numbering. Gaps where steps 1 and 6 used to be are left as-is.

---

## File Summary

| File | Change type |
|------|-------------|
| `config/const_preprocessing.py` | Delete FMI window constants block, delete two folder constants |
| `src/preprocessing_pipeline.py` | Delete 2 methods, rename 2 methods, update all CSV I/O calls, update filename regex, update imports |
| `src/file_utils.py` | Rename 4 functions, swap CSV I/O for parquet throughout |
| `main.py` | Remove EMS mapping step, update function call names, update print messages |
| `src/training_pipeline.py` | Swap CSV reads for parquet reads, update glob patterns |

---

## Success Criteria

- `python main.py` runs without error against `matched_data_2024_01.parquet` style input files
- All 16 remaining preprocessing steps execute in order and produce `.parquet` output files
- No reference to `timeTableRows`, `window_weather_data`, or FMI window constants remains in any active code path
- Training pipeline reads the parquet output and runs to completion
