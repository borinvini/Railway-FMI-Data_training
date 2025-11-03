# Pipeline State Machine Configuration
# Dictionary containing method names in execution order with enable/disable flags
PREPROCESSING_STATE_MACHINE = {
    # Data extraction and preprocessing methods
    "extract_nested_data": True,
    "filter_by_target_station": True,
    "process_causes_column": True,
    "add_train_delayed_feature": True,
    "merge_weather_columns": True,
    "add_weather_scenarios_col": True,
    "weather_scenario_one_hot_encoder": True,
    "process_actual_time_column": True,
    "filter_columns": True,
    "convert_boolean_to_numeric": True,
    "handle_missing_values": True,
    "save_month_df_to_csv": True,
    "convert_hour_to_sincos": True,
    "convert_month_to_sincos": True,
    "convert_dayofweek_to_sincos": True,
    "drop_original_temporal_columns": False,
    "select_target": False, 
    "filter_strong_weather_causes": False,
    "remove_duplicates": True,
    "save_training_ready_csv": True
}

FOLDER_EXTRACT_NESTED_DATA = "data/output/1-extract_nested_data"
FOLDER_FILTER_BY_TARGET_STATION = "data/output/2-filter_by_target_station"
FOLDER_PROCESS_CAUSES_COLUMN = "data/output/3-process_causes_column"
FOLDER_ADD_TRAIN_DELAYED_FEATURE = "data/output/4-add_train_delayed_feature"
FOLDER_MERGE_WEATHER_COLUMNS = "data/output/5-merge_weather_columns"
FOLDER_ADD_WEATHER_SCENARIOS_COL = "data/output/6-add_weather_scenarios_col"
FOLDER_WEATHER_SCENARIO_ONE_HOT_ENCODER = "data/output/7-weather_scenario_one_hot_encoder"

FOLDER_PROCESS_ACTUAL_TIME_COLUMN = "data/output/8-process_actual_time_column"
FOLDER_FILTER_COLUMNS = "data/output/9-filter_columns"
FOLDER_CONVERT_BOOLEAN_TO_NUMERIC = "data/output/10-convert_boolean_to_numeric"
FOLDER_HANDLE_MISSING_VALUES = "data/output/11-handle_missing_values"
FOLDER_CONVERT_HOUR_TO_SINCOS = "data/output/12-convert_hour_to_sincos"
FOLDER_CONVERT_MONTH_TO_SINCOS = "data/output/13-convert_month_to_sincos"
FOLDER_CONVERT_DAYOFWEEK_TO_SINCOS = "data/output/14-convert_dayofweek_to_sincos"
FOLDER_DROP_ORIGINAL_TEMPORAL_COLUMNS = "data/output/15-drop_original_temporal_columns"
FOLDER_SELECT_TARGET = "data/output/16-select_target"
FOLDER_FILTER_STRONG_WEATHER_CAUSES = "data/output/17-filter_strong_weather_causes"
FOLDER_REMOVE_DUPLICATES = "data/output/18-remove_duplicates"

DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
PREPROCESSED_OUTPUT_FOLDER = "data/output/100-preprocessed"
TRAINING_READY_OUTPUT_FOLDER = "data/output/101-preprocessed_training_ready"

# Weather column missing value threshold (drop columns with more missing values than this %)
WEATHER_MISSING_THRESHOLD = 30.0

# Target feature to use for prediction
DEFAULT_TARGET_FEATURE = 'differenceInMinutes_eachStation_offset'  
# Possible values: 'differenceInMinutes', 'differenceInMinutes_offset', 
# 'differenceInMinutes_eachStation_offset', 'trainDelayed', 'cancelled'

# Target column to use for calculating trainDelayed feature
TRAIN_DELAYED_TARGET_COLUMN = 'differenceInMinutes_eachStation_offset'
# Possible values: 'differenceInMinutes', 'differenceInMinutes_offset', 'differenceInMinutes_eachStation_offset'

# Valid target features for selection
VALID_TARGET_FEATURES = [
    'differenceInMinutes', 
    'differenceInMinutes_offset', 
    'differenceInMinutes_eachStation_offset', 
    'trainDelayed', 
    'cancelled'
]

CLASSIFICATION_PROBLEM = ['trainDelayed', 'cancelled']
REGRESSION_PROBLEM = ['differenceInMinutes', 'differenceInMinutes_offset', 'differenceInMinutes_eachStation_offset']

# Station short code to filter data for - only exact matches will be kept
TARGET_STATION_CODE = 'OL'  # Example: 'OL', 'HKI', 'ROI', etc.

# Time feature selection configuration
# True = keep sin/cos features (drop original), False = keep original features (drop sin/cos)
USE_SIN_COS_APPROACH = True

TEMPORAL_SINCOS_FEATURES = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'day_week_sin', 'day_week_cos']

# Valid prediction features (non-target features used for training)
VALID_TRAIN_PREDICTION_FEATURES = ["trainStopping", "commercialStop","month","hour","day_of_week", "day_of_month", "causes_related_to_weather", "train_id", "weather_scenario"]

# Target features that are categoricals for classification problems
CATEGORIAL_TARGET_FEATURES = ['trainDelayed', 'cancelled']

# Boolean features that need to be converted to numeric
BOOLEAN_FEATURES = ['trainStopping', 'commercialStop']

# Multi category features
CATEGORICAL_FEATURES = ["month","hour","day_of_week","causes", "causes_related_to_weather"]

# Set to True to drop trainStopping and commercialStop from training
DROP_TRAIN_FEATURES = True

# Set to True to drop trainStopping and commercialStop from training
delay = True

# Value to be considered as a delay (in minutes)
# Long-distance trains: 5 min
# Short trains: 2-3 min
TRAIN_DELAY_MINUTES = 5

ALL_WEATHER_FEATURES = [
    'Air temperature', 
    'Wind speed', 
    'Gust speed', 
    'Wind direction', 
    'Relative humidity', 
    'Dew-point temperature',
    'Precipitation amount',
    'Precipitation intensity',
    'Snow depth',
    'Pressure (msl)',
    'Horizontal visibility',
    'Cloud amount'
]

IMPORTANT_WEATHER_FEATURES = [
    'Air temperature', 
    'Wind speed', 
    'Gust speed', 
    'Relative humidity', 
    'Precipitation amount',
    'Precipitation intensity',
    'Snow depth',
    'Pressure (msl)',
    'Horizontal visibility',
    'Cloud amount'
]


# List of weather features that has 2 cols and need to merge in 1 col
WEATHER_COLS_TO_MERGE = [
    "Snow depth", 
    "Precipitation amount", 
    "Precipitation intensity", 
    "Horizontal visibility", 
    "Wind speed", 
    "Gust speed"
]

SELECTED_WEATHER_FEATURES = [
    'Air temperature', 
    'Wind speed', 
    'Gust speed', 
    'Snow depth',
]



# Train filtering configuration
FILTER_TRAINS_BY_STATIONS = False  # Set to True to filter trains by required stations
REQUIRED_STATIONS = ['HKI', 'OL', 'ROI']  # Trains must pass through ALL of these stations


# Weather Indicator Categories for Causes Analysis
# Used to create 'causes_related_to_weather' column from nested 'causes' data
# Extracts 'detailedCategoryCode' and assigns weather likelihood scores:
# - Score 3 (Strong): Direct weather-related delays (I1, I2)
# - Score 2 (Possible): May be weather-influenced (A1, K1, O1, P1, S1, S2, T2, T3, V3)
# - Score 1 (Weak): Other non-empty category codes
# - Score 0 (None): Empty/missing cause data

# Strong weather delay indicators (score: 3)
STRONG_INDICATORS = {'I1', 'I2'}

# Possible weather delay indicators (score: 2)  
POSSIBLE_INDICATORS = {'A1', 'K1', 'O1', 'P1', 'S1', 'S2', 'T2', 'T3', 'V3'}
