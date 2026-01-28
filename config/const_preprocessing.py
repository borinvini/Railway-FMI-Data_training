# Pipeline State Machine Configuration
# Dictionary containing method names in execution order with enable/disable flags
PREPROCESSING_STATE_MACHINE = {
    # Data extraction and preprocessing methods
    "extract_nested_data": True,
    "filter_by_target_station": True,
    "process_causes_column": True,
    "add_train_delayed_feature": True,
    "merge_weather_columns": True,
    "add_weather_1h_window_features": True,
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
FOLDER_ADD_WEATHER_1H_WINDOW_FEATURES = "data/output/6-add_weather_1h_window_features"
FOLDER_ADD_WEATHER_SCENARIOS_COL = "data/output/7-add_weather_scenarios_col"
FOLDER_WEATHER_SCENARIO_ONE_HOT_ENCODER = "data/output/8-weather_scenario_one_hot_encoder"

FOLDER_PROCESS_ACTUAL_TIME_COLUMN = "data/output/8-process_actual_time_column"
FOLDER_FILTER_COLUMNS = "data/output/8-filter_columns"
FOLDER_CONVERT_BOOLEAN_TO_NUMERIC = "data/output/11-convert_boolean_to_numeric"
FOLDER_HANDLE_MISSING_VALUES = "data/output/12-handle_missing_values"
FOLDER_CONVERT_HOUR_TO_SINCOS = "data/output/13-convert_hour_to_sincos"
FOLDER_CONVERT_MONTH_TO_SINCOS = "data/output/14-convert_month_to_sincos"
FOLDER_CONVERT_DAYOFWEEK_TO_SINCOS = "data/output/15-convert_dayofweek_to_sincos"
FOLDER_DROP_ORIGINAL_TEMPORAL_COLUMNS = "data/output/16-drop_original_temporal_columns"
FOLDER_SELECT_TARGET = "data/output/17-select_target"
FOLDER_FILTER_STRONG_WEATHER_CAUSES = "data/output/18-filter_strong_weather_causes"
FOLDER_REMOVE_DUPLICATES = "data/output/19-remove_duplicates"

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
CATEGORICAL_FEATURES = [
    "month",
    "hour",
    "day_of_week",
    "causes",
    "causes_related_to_weather",
    # Weather scenario one-hot encoded features
    "weather_scenario_Normal_Clear",
    "weather_scenario_Blizzard",
    "weather_scenario_Heavy_Snow",
    "weather_scenario_Extreme_Cold",
    "weather_scenario_Heavy_Rain",
    "weather_scenario_Freezing_Rain",
    "weather_scenario_Black_Ice",
    "weather_scenario_Dense_Fog",
    "weather_scenario_High_Winds",
    "weather_scenario_Extreme_Heat"
]

# Define weather scenario feature columns
VALID_WEATHER_SCENARIO_FEATURES = [
    'weather_scenario_Normal_Clear',
    'weather_scenario_Blizzard',
    'weather_scenario_Heavy_Snow',
    'weather_scenario_Extreme_Cold',
    'weather_scenario_Heavy_Rain',
    'weather_scenario_Freezing_Rain',
    'weather_scenario_Black_Ice',
    'weather_scenario_Dense_Fog',
    'weather_scenario_High_Winds',
    'weather_scenario_Extreme_Heat'
]

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


# Path to FMI weather data for 1h window feature calculation
WINDOW_WEATHER_DATA_FOLDER = "data/input/window_weather_data"

# Path to train station to EMS station mapping metadata
TRAIN_STATION_EMS_METADATA_PATH = "data/input/metadata/train_station_closest_ems.csv"

# Train data column names for matching
TRAIN_SCHEDULED_TIME_COL = "scheduledTime"
TRAIN_STATION_SHORT_CODE_COL = "stationShortCode"

# FMI Weather data column names
FMI_TIMESTAMP_COL = "timestamp"
FMI_STATION_NAME_COL = "station_name"

# Time window configuration (in minutes)
WEATHER_WINDOW_MINUTES = 60

# FMI column names for all 8 weather features
FMI_AIR_TEMPERATURE_COL = "Air temperature"
FMI_WIND_SPEED_COL = "Wind speed"
FMI_RELATIVE_HUMIDITY_COL = "Relative humidity"
FMI_PRECIPITATION_INTENSITY_COL = "Precipitation intensity"
FMI_SNOW_DEPTH_COL = "Snow depth"
FMI_PRESSURE_MSL_COL = "Pressure (msl)"
FMI_HORIZONTAL_VISIBILITY_COL = "Horizontal visibility"
FMI_CLOUD_AMOUNT_COL = "Cloud amount"

# -----------------------------------------------------------------------------
# Weather Features Configuration for 1h Window Processing
# Dictionary mapping feature key -> FMI column name
# This allows easy iteration and maintains consistency
# -----------------------------------------------------------------------------
WEATHER_1H_WINDOW_FEATURES = {
    "air_temperature": FMI_AIR_TEMPERATURE_COL,
    "wind_speed": FMI_WIND_SPEED_COL,
    "relative_humidity": FMI_RELATIVE_HUMIDITY_COL,
    "precipitation_intensity": FMI_PRECIPITATION_INTENSITY_COL,
    "snow_depth": FMI_SNOW_DEPTH_COL,
    "pressure_msl": FMI_PRESSURE_MSL_COL,
    "horizontal_visibility": FMI_HORIZONTAL_VISIBILITY_COL,
    "cloud_amount": FMI_CLOUD_AMOUNT_COL,
}

# Number of weather features for 1h window processing
NUM_WEATHER_1H_WINDOW_FEATURES = len(WEATHER_1H_WINDOW_FEATURES)

# Number of statistics per feature (min, max, mean)
NUM_STATS_PER_FEATURE = 3

# Total number of new columns (8 features × 3 stats = 24)
TOTAL_1H_WINDOW_COLUMNS = NUM_WEATHER_1H_WINDOW_FEATURES * NUM_STATS_PER_FEATURE

# -----------------------------------------------------------------------------
# Output Column Names for 1h Window Statistics
# Format: {feature_key}_1h_window_{statistic}
# -----------------------------------------------------------------------------

# Air Temperature (°C)
AIR_TEMP_1H_WINDOW_MIN = "air_temperature_1h_window_min"
AIR_TEMP_1H_WINDOW_MAX = "air_temperature_1h_window_max"
AIR_TEMP_1H_WINDOW_MEAN = "air_temperature_1h_window_mean"

# Wind Speed (m/s)
WIND_SPEED_1H_WINDOW_MIN = "wind_speed_1h_window_min"
WIND_SPEED_1H_WINDOW_MAX = "wind_speed_1h_window_max"
WIND_SPEED_1H_WINDOW_MEAN = "wind_speed_1h_window_mean"

# Relative Humidity (%)
HUMIDITY_1H_WINDOW_MIN = "relative_humidity_1h_window_min"
HUMIDITY_1H_WINDOW_MAX = "relative_humidity_1h_window_max"
HUMIDITY_1H_WINDOW_MEAN = "relative_humidity_1h_window_mean"

# Precipitation Intensity (mm/h)
PRECIP_INTENSITY_1H_WINDOW_MIN = "precipitation_intensity_1h_window_min"
PRECIP_INTENSITY_1H_WINDOW_MAX = "precipitation_intensity_1h_window_max"
PRECIP_INTENSITY_1H_WINDOW_MEAN = "precipitation_intensity_1h_window_mean"

# Snow Depth (cm)
SNOW_DEPTH_1H_WINDOW_MIN = "snow_depth_1h_window_min"
SNOW_DEPTH_1H_WINDOW_MAX = "snow_depth_1h_window_max"
SNOW_DEPTH_1H_WINDOW_MEAN = "snow_depth_1h_window_mean"

# Pressure MSL (hPa)
PRESSURE_1H_WINDOW_MIN = "pressure_msl_1h_window_min"
PRESSURE_1H_WINDOW_MAX = "pressure_msl_1h_window_max"
PRESSURE_1H_WINDOW_MEAN = "pressure_msl_1h_window_mean"

# Horizontal Visibility (m)
VISIBILITY_1H_WINDOW_MIN = "horizontal_visibility_1h_window_min"
VISIBILITY_1H_WINDOW_MAX = "horizontal_visibility_1h_window_max"
VISIBILITY_1H_WINDOW_MEAN = "horizontal_visibility_1h_window_mean"

# Cloud Amount (oktas, 0-8)
CLOUD_AMOUNT_1H_WINDOW_MIN = "cloud_amount_1h_window_min"
CLOUD_AMOUNT_1H_WINDOW_MAX = "cloud_amount_1h_window_max"
CLOUD_AMOUNT_1H_WINDOW_MEAN = "cloud_amount_1h_window_mean"

# -----------------------------------------------------------------------------
# Dictionary Mapping Feature Keys to Output Column Names
# Used for dynamic column generation in the processing method
# -----------------------------------------------------------------------------
WEATHER_1H_WINDOW_OUTPUT_COLS = {
    "air_temperature": {
        "min": AIR_TEMP_1H_WINDOW_MIN,
        "max": AIR_TEMP_1H_WINDOW_MAX,
        "mean": AIR_TEMP_1H_WINDOW_MEAN,
    },
    "wind_speed": {
        "min": WIND_SPEED_1H_WINDOW_MIN,
        "max": WIND_SPEED_1H_WINDOW_MAX,
        "mean": WIND_SPEED_1H_WINDOW_MEAN,
    },
    "relative_humidity": {
        "min": HUMIDITY_1H_WINDOW_MIN,
        "max": HUMIDITY_1H_WINDOW_MAX,
        "mean": HUMIDITY_1H_WINDOW_MEAN,
    },
    "precipitation_intensity": {
        "min": PRECIP_INTENSITY_1H_WINDOW_MIN,
        "max": PRECIP_INTENSITY_1H_WINDOW_MAX,
        "mean": PRECIP_INTENSITY_1H_WINDOW_MEAN,
    },
    "snow_depth": {
        "min": SNOW_DEPTH_1H_WINDOW_MIN,
        "max": SNOW_DEPTH_1H_WINDOW_MAX,
        "mean": SNOW_DEPTH_1H_WINDOW_MEAN,
    },
    "pressure_msl": {
        "min": PRESSURE_1H_WINDOW_MIN,
        "max": PRESSURE_1H_WINDOW_MAX,
        "mean": PRESSURE_1H_WINDOW_MEAN,
    },
    "horizontal_visibility": {
        "min": VISIBILITY_1H_WINDOW_MIN,
        "max": VISIBILITY_1H_WINDOW_MAX,
        "mean": VISIBILITY_1H_WINDOW_MEAN,
    },
    "cloud_amount": {
        "min": CLOUD_AMOUNT_1H_WINDOW_MIN,
        "max": CLOUD_AMOUNT_1H_WINDOW_MAX,
        "mean": CLOUD_AMOUNT_1H_WINDOW_MEAN,
    },
}

# -----------------------------------------------------------------------------
# List of All 1h Window Output Columns (for validation and reference)
# Total: 24 columns (8 features × 3 statistics)
# -----------------------------------------------------------------------------
ALL_1H_WINDOW_OUTPUT_COLUMNS = [
    # Air Temperature
    AIR_TEMP_1H_WINDOW_MIN,
    AIR_TEMP_1H_WINDOW_MAX,
    AIR_TEMP_1H_WINDOW_MEAN,
    # Wind Speed
    WIND_SPEED_1H_WINDOW_MIN,
    WIND_SPEED_1H_WINDOW_MAX,
    WIND_SPEED_1H_WINDOW_MEAN,
    # Relative Humidity
    HUMIDITY_1H_WINDOW_MIN,
    HUMIDITY_1H_WINDOW_MAX,
    HUMIDITY_1H_WINDOW_MEAN,
    # Precipitation Intensity
    PRECIP_INTENSITY_1H_WINDOW_MIN,
    PRECIP_INTENSITY_1H_WINDOW_MAX,
    PRECIP_INTENSITY_1H_WINDOW_MEAN,
    # Snow Depth
    SNOW_DEPTH_1H_WINDOW_MIN,
    SNOW_DEPTH_1H_WINDOW_MAX,
    SNOW_DEPTH_1H_WINDOW_MEAN,
    # Pressure MSL
    PRESSURE_1H_WINDOW_MIN,
    PRESSURE_1H_WINDOW_MAX,
    PRESSURE_1H_WINDOW_MEAN,
    # Horizontal Visibility
    VISIBILITY_1H_WINDOW_MIN,
    VISIBILITY_1H_WINDOW_MAX,
    VISIBILITY_1H_WINDOW_MEAN,
    # Cloud Amount
    CLOUD_AMOUNT_1H_WINDOW_MIN,
    CLOUD_AMOUNT_1H_WINDOW_MAX,
    CLOUD_AMOUNT_1H_WINDOW_MEAN,
]

# -----------------------------------------------------------------------------
# Feature Descriptions (for documentation/logging)
# -----------------------------------------------------------------------------
WEATHER_1H_WINDOW_FEATURE_DESCRIPTIONS = {
    "air_temperature": "Air temperature in degrees Celsius",
    "wind_speed": "Wind speed in meters per second",
    "relative_humidity": "Relative humidity percentage (0-100%)",
    "precipitation_intensity": "Precipitation intensity in mm/h",
    "snow_depth": "Snow depth in centimeters",
    "pressure_msl": "Atmospheric pressure at mean sea level in hPa",
    "horizontal_visibility": "Horizontal visibility in meters",
    "cloud_amount": "Cloud amount in oktas (0-8 scale)",
}