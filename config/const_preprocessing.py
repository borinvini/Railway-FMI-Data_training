# Pipeline State Machine Configuration
# Dictionary containing method names in execution order with enable/disable flags
PREPROCESSING_STATE_MACHINE = {
    "filter_by_target_station": True,
    "process_causes_column": True,
    "add_train_delayed_feature": True,
    "null_out_of_bounds_weather": True,
    "add_weather_scenarios_col": True,
    "add_rolling_weather_scenarios_col": True,
    "weather_scenario_one_hot_encoder": True,
    "add_wawa_group_col": True,
    "wawa_group_one_hot_encoder": True,
    "convert_scheduled_time_to_local": False,
    "process_actual_time_column": True,
    "filter_columns": True,
    "convert_boolean_to_numeric": True,
    "handle_missing_values": True,
    "save_month_df_to_parquet": True,
    "convert_hour_to_sincos": True,
    "convert_month_to_sincos": True,
    "convert_dayofweek_to_sincos": True,
    "convert_dayofmonth_to_sincos": True,
    "convert_winddirection_to_sincos": True,
    "drop_original_temporal_columns": False,
    "select_target": False,
    "remove_duplicates": True,
    "save_training_ready_parquet": True
}

# Value to be considered as a delay (in minutes)
# Long-distance trains: 5 min
# Short trains: 2-3 min
# The single source of this threshold: trainDelayed, trainDisrupted, the balance step, the regression
# metrics (const_training.DELAY_THRESHOLD_MINUTES) and the FCLL label all read it.
TRAIN_DELAY_MINUTES = 5

# Target feature to use for prediction
# 'trainDisrupted' = delayed by more than TRAIN_DELAY_MINUTES OR cancelled. A cancelled train has no
# delay value, so 'trainDelayed' alone can neither include nor label it. Switch to 'trainDelayed'
# to train on the pure delay again (cancelled trains stay in the data either way).
DEFAULT_TARGET_FEATURE = 'differenceInMinutes'
# Possible values: 'differenceInMinutes', 'differenceInMinutes_offset',
# 'differenceInMinutes_eachStation_offset', 'trainDelayed', 'cancelled', 'trainDisrupted'

# Target column to use for calculating trainDelayed feature
TRAIN_DELAYED_TARGET_COLUMN = 'differenceInMinutes'
# Possible values: 'differenceInMinutes', 'differenceInMinutes_offset', 'differenceInMinutes_eachStation_offset'

# Per-stage intermediate parquet files that are actually written. Every stage used
# to write one file per month; with filter_by_target_station off a month has about
# 507k rows and each stage folder costs 2.45-3.0 GB over 96 months (about 48 GB for
# all 17 folders, against about 0.5 GB at OL). Nothing downstream reads them: training
# reads only 101-preprocessed_training_ready. Keys are state-machine names.
# 100-preprocessed and 101-... keep their own switches (save_month_df_to_parquet,
# save_training_ready_parquet). Keeping "process_actual_time_column" gives a debug
# checkpoint with every engineered column next to the raw ones (~3.0 GB / 96 months).
SAVE_STAGE_OUTPUTS = {"process_actual_time_column"}

# Station identifier kept through filter_columns when filter_by_target_station is off,
# so rows from different stations stay distinguishable. stationUICCode is int64 with
# no nulls in all 96 months and maps 1:1 to stationShortCode, so it needs no encoder.
STATION_ID_COLUMN = "stationUICCode"

FOLDER_FILTER_BY_TARGET_STATION = "data/output/2-filter_by_target_station"
FOLDER_PROCESS_CAUSES_COLUMN = "data/output/3-process_causes_column"
FOLDER_ADD_TRAIN_DELAYED_FEATURE = "data/output/4-add_train_delayed_feature"
FOLDER_NULL_OUT_OF_BOUNDS_WEATHER = "data/output/4b-null_out_of_bounds_weather"
FOLDER_ADD_WEATHER_SCENARIOS_COL = "data/output/7-add_weather_scenarios_col"
FOLDER_ADD_ROLLING_WEATHER_SCENARIOS_COL = "data/output/7b-add_rolling_weather_scenarios_col"
FOLDER_WEATHER_SCENARIO_ONE_HOT_ENCODER = "data/output/8-weather_scenario_one_hot_encoder"

# Rolling window suffixes present in the input data (used by add_rolling_weather_scenarios_col
# and by weather_scenario_one_hot_encoder to encode the per-window scenario columns)
ROLLING_WINDOWS = ['12h', '24h', '72h']

FOLDER_CONVERT_SCHEDULED_TIME_TO_LOCAL = "data/output/8c-convert_scheduled_time_to_local"

# Column that month / hour / day_of_week / day_of_month are derived from. It must
# be the SCHEDULED time: actualTime is the scheduled time plus the delay, so the
# clock time of an actual arrival hands the model the label (a single hour_sin /
# hour_cos pair scored ROC-AUC 0.775 from actualTime against 0.706 from
# scheduledTime). Setting this to "actualTime" reproduces the old behaviour for an
# A/B comparison with earlier results.
TIME_FEATURE_SOURCE_COLUMN = "scheduledTime"

# Zone used by convert_scheduled_time_to_local (stage disabled by default). Every
# timestamp in the matched files is already UTC; the stage only changes what the
# clock feature means: a fixed timetable slot keeps one hour all year instead of
# shifting by one at each DST change. Measured effect on ROC-AUC of the clock
# feature alone is within +-0.008 and its sign depends on the encoding (+0.007 on
# raw minutes of day, -0.008 on hour_sin/hour_cos), so it is not a reliable gain;
# enable it for interpretability, not for score.
LOCAL_TIMEZONE = "Europe/Helsinki"

# The stage adds this suffix to TIME_FEATURE_SOURCE_COLUMN.
LOCAL_TIME_COLUMN_SUFFIX = "_local"

FOLDER_PROCESS_ACTUAL_TIME_COLUMN = "data/output/9-process_actual_time_column"
FOLDER_FILTER_COLUMNS = "data/output/10-filter_columns"
FOLDER_CONVERT_BOOLEAN_TO_NUMERIC = "data/output/11-convert_boolean_to_numeric"
FOLDER_HANDLE_MISSING_VALUES = "data/output/12-handle_missing_values"
FOLDER_CONVERT_HOUR_TO_SINCOS = "data/output/13-convert_hour_to_sincos"
FOLDER_CONVERT_MONTH_TO_SINCOS = "data/output/14-convert_month_to_sincos"
FOLDER_CONVERT_DAYOFWEEK_TO_SINCOS = "data/output/15-convert_dayofweek_to_sincos"
FOLDER_CONVERT_DAYOFMONTH_TO_SINCOS = "data/output/15b-convert_dayofmonth_to_sincos"
FOLDER_CONVERT_WINDDIRECTION_TO_SINCOS = "data/output/15c-convert_winddirection_to_sincos"
FOLDER_DROP_ORIGINAL_TEMPORAL_COLUMNS = "data/output/16-drop_original_temporal_columns"
FOLDER_SELECT_TARGET = "data/output/17-select_target"
FOLDER_REMOVE_DUPLICATES = "data/output/19-remove_duplicates"

DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
PREPROCESSED_OUTPUT_FOLDER = "data/output/100-preprocessed"
TRAINING_READY_OUTPUT_FOLDER = "data/output/101-preprocessed_training_ready"

# Weather column missing value threshold (drop columns with more missing values than this %).
# Applied ONCE over all months by PreprocessingPipeline.decide_weather_columns_to_drop
# (target-station rows, pooled counts); the run aborts if a month would decide
# differently, because merge_data_files keeps the column intersection and would
# drop the column from every month. At OL it drops only "Precipitation amount"
# (about 88% missing, in all 96 months); the worst month of any kept column is
# 0.6% missing. The pass reads only the weather columns: about 15 s, a few MB.
WEATHER_MISSING_THRESHOLD = 30.0

# Valid target features for selection
VALID_TARGET_FEATURES = [
    'differenceInMinutes', 
    'differenceInMinutes_offset', 
    'differenceInMinutes_eachStation_offset', 
    'trainDelayed', 
    'cancelled',
    'trainDisrupted'
]

CLASSIFICATION_PROBLEM = ['trainDelayed', 'cancelled', 'trainDisrupted']
REGRESSION_PROBLEM = ['differenceInMinutes', 'differenceInMinutes_offset', 'differenceInMinutes_eachStation_offset']

# Station short code to filter data for - only exact matches will be kept
TARGET_STATION_CODE = 'OL'  # Example: 'OL', 'HKI', 'ROI', etc.

# Column carried through preprocessing purely so split_dataset can group by
# operating day. It is NOT a feature: select_training_cols keeps it, the split
# groups on it, and the trainers drop it before fitting. Without it the split
# falls back to random rows, which leaks for any sequence model - a test row's
# preceding trains sit in the training set.
GROUP_KEY_COLUMN = "departureDate"

# Time feature selection configuration
# True = keep sin/cos features (drop original), False = keep original features (drop sin/cos)
USE_SIN_COS_APPROACH = True

TEMPORAL_SINCOS_FEATURES = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'day_week_sin', 'day_week_cos',
                            'day_month_sin', 'day_month_cos']

# Length of the day-of-month circle: day 31 sits next to day 1. Fixed like the other
# periods (12 months, 24 hours, 7 days); a per-month length would need the year.
DAY_OF_MONTH_CYCLE = 31

# Wind direction is circular (359 and 1 degrees are both north). Source column, the code FMI
# uses for calm (every row with direction 0 has wind speed 0.00 at OL; north is 360), and the
# columns the sin/cos stage adds. Calm gets (0, 0), the no-direction point, not "north".
WIND_DIRECTION_COLUMN = 'Wind direction'
WIND_CALM_DIRECTION = 0
WIND_SINCOS_FEATURES = ['wind_dir_sin', 'wind_dir_cos']

# Valid prediction features (non-target features used for training)
VALID_TRAIN_PREDICTION_FEATURES = ["trainStopping", "commercialStop","month","hour","day_of_week", "day_of_month", "causes_related_to_weather", "train_id", "weather_scenario"]

# Target features that are categoricals for classification problems
CATEGORIAL_TARGET_FEATURES = ['trainDelayed', 'cancelled', 'trainDisrupted']

# Columns that carry the answer. Only DEFAULT_TARGET_FEATURE may survive column selection (stage 502);
# the trainers use every non-target column as a feature, so any other label would leak the target.
LABEL_COLUMNS = ['trainDelayed', 'trainDisrupted', 'cancelled']

# Boolean features that need to be converted to numeric
BOOLEAN_FEATURES = ['trainStopping', 'commercialStop']

# Instant weather scenarios classify precipitation from "Precipitation intensity"
# only. "Precipitation amount" is null in 87.8% of OL rows (81-90% in every year);
# NaN compares False, so conditions on it silently stopped firing (Black Ice 1.4%
# of rows instead of about 10%, Dense Fog 0.04% instead of 0.36%). Intensity is
# complete (0.0% null). Calibrated on the 14,397 rows that have both: the old
# `amount <= 0.5` agrees with `intensity <= 0.5` on 97.7% of rows, and the old
# `amount <= 0.1` with `intensity <= 0.1` on 96.4%.
INSTANT_BLACK_ICE_MAX_INTENSITY = 0.5   # mm/h, "light or no active precipitation"
INSTANT_DENSE_FOG_MAX_INTENSITY = 0.1   # mm/h, "no or very light precipitation"

# Rolling scenarios scale with the window. The first version reused the instant
# cuts on worst-case aggregates in every window, so 10 mm was Heavy Rain whether
# it fell in 12 hours or 72, peak intensity counted too, and the +-2 C band had
# to hold for the whole window. At OL the 72h columns read Heavy Rain 17.3%,
# Blizzard 11.5%, Heavy Snow 9.2%, Black Ice 0.04% (instant: 0.18, 0.05, 0.02,
# 13.5%). Now the window state comes from the cumulative amount against a rate
# times the window length, and from the window means of temperature (phase and
# band) and humidity (RH max is above 80 in 99% of 72h windows). The old amount
# cuts (3 / 5 / 10 mm) are read as per-24h, giving 72h shares of Blizzard 1.8%,
# Heavy Snow 0.3%, Heavy Rain 1.3%, Black Ice 14.2% (12h: 0.7 / 1.7 / 3.3 / 13.5).
ROLLING_BLIZZARD_MIN_RATE = 3 / 24      # mm/h, window total > rate * hours
ROLLING_HEAVY_SNOW_MIN_RATE = 5 / 24    # mm/h
ROLLING_HEAVY_RAIN_MIN_RATE = 10 / 24   # mm/h


def rolling_window_hours(window):
    """Length in hours of a ROLLING_WINDOWS entry ('72h' -> 72)."""
    return int(window.rstrip("h"))

# Weather scenario categories, shared by the instant and rolling-window scenario stages
# and by weather_scenario_one_hot_encoder (order matters for consistent one-hot columns)
WEATHER_SCENARIO_CATEGORIES = [
    'Normal/Clear',
    'Blizzard',
    'Heavy Snow',
    'Extreme Cold',
    'Heavy Rain',
    'Freezing Rain',
    'Black Ice',
    'Dense Fog',
    'High Winds',
    'Extreme Heat'
]

def _weather_scenario_one_hot_names(prefix):
    """Build the one-hot column names for a given scenario column prefix
    (e.g. 'weather_scenario' or 'weather_scenario_24h')."""
    return [
        f'{prefix}_{category.replace("/", "_").replace(" ", "_")}'
        for category in WEATHER_SCENARIO_CATEGORIES
    ]

# Define weather scenario feature columns (instant + rolling windows)
VALID_WEATHER_SCENARIO_FEATURES = [
    *_weather_scenario_one_hot_names("weather_scenario"),
    *[name
      for window in ROLLING_WINDOWS
      for name in _weather_scenario_one_hot_names(f"weather_scenario_{window}")],
]

# ---------------------------------------------------------------------------
# wawa — WMO code table 4680 "present weather reported from an automatic
# weather station", the FMI column "Present weather (auto)".
#
# The codes are nominal, not ordinal: 71 ("snow, slight") is not ten more of
# anything than 61 ("rain, slight"). They are folded into meteorologically
# coherent groups and one-hot encoded. Groupings are transcribed from
# docs/wawa-group-proposal.txt — that file is the source of truth.
# ---------------------------------------------------------------------------

WAWA_SOURCE_COLUMN = 'Present weather (auto)'

# Missing, non-numeric, out-of-table, and non-emitted codes all fold to this.
WAWA_FALLBACK_GROUP = 'clear'

# The 14 groups that become one-hot columns. Order is fixed: every month file
# must emit the same columns in the same order or the downstream merge of the
# per-month parquets misaligns.
WAWA_EMITTED_GROUPS = [
    'clear',
    'haze',
    'mist',
    'fog',
    'precip_unspec',
    'drizzle',
    'rain',
    'freezing',
    'sleet',
    'snow',
    'ice_particles',
    'rain_shower',
    'snow_shower',
    'hail',
]

# Mapped so that no code falls through unassigned, but never emitted: these
# have zero occurrences in the 41.7M-row record, so a column for them would be
# always-zero. Rows landing here fold to WAWA_FALLBACK_GROUP, and
# add_wawa_group_col logs the count per group so a station that does report
# them announces itself instead of silently losing the rows.
WAWA_NON_EMITTED_GROUPS = [
    'thunder',
    'blowing_snow',
    'squall',
    'tornado',
]


def _wawa_codes(*spans):
    """Expand a mix of single codes and inclusive (start, end) ranges."""
    codes = []
    for span in spans:
        if isinstance(span, tuple):
            start, end = span
            codes.extend(range(start, end + 1))
        else:
            codes.append(span)
    return codes


# Codes 20-26 report the preceding hour rather than the observation instant.
# By the agreed design they merge into their corresponding "now" group
# (20 -> fog, 21 -> precip_unspec, 22 -> drizzle, 23 -> rain, 24 -> snow,
# 25 -> freezing) rather than forming a separate group.
WAWA_GROUP_CODES = {
    'clear':         _wawa_codes((0, 3), (6, 9), (13, 17), 19, (36, 39), 49, 59, 69, 79, 88, (97, 98)),
    'haze':          _wawa_codes(4, 5),
    'mist':          _wawa_codes(10),
    'fog':           _wawa_codes(20, (30, 35)),
    'precip_unspec': _wawa_codes(21, (40, 42)),
    'drizzle':       _wawa_codes(22, (50, 53), 57, 58),
    'rain':          _wawa_codes(23, 43, 44, (60, 63)),
    'freezing':      _wawa_codes(25, 47, 48, (54, 56), (64, 66)),
    'sleet':         _wawa_codes(67, 68),
    'snow':          _wawa_codes(24, 45, 46, (70, 73)),
    'ice_particles': _wawa_codes(11, (74, 78)),
    'rain_shower':   _wawa_codes((80, 84)),
    'snow_shower':   _wawa_codes((85, 87)),
    'hail':          _wawa_codes(89),
    # not emitted
    'thunder':       _wawa_codes(12, 26, (90, 96)),
    'blowing_snow':  _wawa_codes((27, 29)),
    'squall':        _wawa_codes(18),
    'tornado':       _wawa_codes(99),
}

WAWA_CODE_TO_GROUP = {
    code: group
    for group, codes in WAWA_GROUP_CODES.items()
    for code in codes
}


def _wawa_one_hot_names():
    """Build the one-hot column names for the emitted wawa groups."""
    return [f'wawa_group_{group}' for group in WAWA_EMITTED_GROUPS]


VALID_WAWA_FEATURES = _wawa_one_hot_names()

FOLDER_ADD_WAWA_GROUP_COL = "data/output/7c-add_wawa_group_col"
FOLDER_WAWA_GROUP_ONE_HOT_ENCODER = "data/output/8b-wawa_group_one_hot_encoder"

# Multi category features
CATEGORICAL_FEATURES = [
    "month",
    "hour",
    "day_of_week",
    "causes",
    "causes_related_to_weather",
    # Weather scenario one-hot encoded features (instant)
    *_weather_scenario_one_hot_names("weather_scenario"),
    # Weather scenario one-hot encoded features (rolling windows)
    *[name
      for window in ROLLING_WINDOWS
      for name in _weather_scenario_one_hot_names(f"weather_scenario_{window}")],
    # wawa (WMO 4680) group one-hot encoded features
    *_wawa_one_hot_names(),
]

# Set to True to drop trainStopping and commercialStop from training
DROP_TRAIN_FEATURES = True

# Set to True to drop trainStopping and commercialStop from training
delay = True

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

SKEWED_WEATHER_FEATURES = [
    'Precipitation amount',
    'Precipitation intensity',
    'Snow depth',
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

SELECTED_WEATHER_FEATURES = [
    'Air temperature', 
    'Wind speed', 
    'Gust speed', 
    'Snow depth',
]



# Physical limits per weather reading (low, high), applied to the base column and to its
# 12h/24h/72h aggregates by null_out_of_bounds_weather. A value outside, or not finite, is
# set to NaN (never dropped or clipped) and filled later by the train-only median imputation.
# The limits sit well outside the real data (air -38.8..34.0, pressure 947..1056, wind <= 26.6,
# gust <= 44.8, humidity 4..100, visibility <= 75000, cloud 0..9, snow depth <= 116, precipitation
# intensity <= 129.4 over 2018-2025), so only glitches trigger. Snow depth and the precipitation
# columns allow -1: that is the "not measured" sentinel, and window means that average it with
# real values land between -1 and 0; both are deliberately left alone here.
# Origin: Dew-point temperature = -84.8 at Jarvenpaa Sorto, 2019-03-23 05:29 UTC (air 1.5 C,
# humidity 63% imply about -4.8 C), present in the raw matched_data_flat_2019_03.
WEATHER_PHYSICAL_BOUNDS = {
    'Air temperature': (-60, 45),
    'Dew-point temperature': (-70, 35),
    'Relative humidity': (0, 100),
    'Wind speed': (0, 60),
    'Gust speed': (0, 90),
    'Wind direction': (0, 360),
    'Pressure (msl)': (900, 1100),
    'Horizontal visibility': (0, 200000),
    'Cloud amount': (0, 9),
    'Snow depth': (-1, 400),
    'Precipitation intensity': (-1, 400),
    'Precipitation amount': (-1, 1000),
}

# More than this share of one column out of bounds in one month is not a glitch but an
# upstream unit or schema change: the stage raises instead of nulling it silently.
WEATHER_BOUNDS_MAX_NULLED_SHARE = 0.001


def weather_bounds_for(column):
    """(low, high) for a weather column or one of its '<base> (<window> <agg>)' aggregates, else None."""
    for base in sorted(WEATHER_PHYSICAL_BOUNDS, key=len, reverse=True):
        if column == base or column.startswith(base + " ("):
            return WEATHER_PHYSICAL_BOUNDS[base]
    return None


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