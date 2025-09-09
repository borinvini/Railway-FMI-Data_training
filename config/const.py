# Pipeline execution control flag
EXECUTE_PREPROCESSING_DATA_PIPELINE = False
EXECUTE_TRAINING_PIPELINE = True

# Constants for file processing
FOLDER_NAME = "data"
INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"
DATA_FILE_PREFIX = "matched_data_"


# All columns available in the matched dataset before preprocessing (alphabetical order):
#
# actualTime                              - Actual arrival/departure time
# cancelled                               - Boolean indicating if train was cancelled
# causes                                  - Reasons for delays (nested data)
# causes_related_to_weather               - Weather-related delay indicator/score
# commercialStop                          - Boolean indicating if it's a commercial stop
# commercialTrack                         - Track number for commercial operations
# commuterLineID                          - ID for commuter train lines
# countryCode                             - Country code where station is located
# day_of_week                             - Day of week extracted from date (1-7)
# departureDate                           - Date when the train departed
# differenceInMinutes                     - Difference between scheduled and actual time
# differenceInMinutes_eachStation_offset  - Station-specific offset time difference
# differenceInMinutes_offset              - Offset-adjusted time difference
# hour                                    - Hour extracted from time
# month                                   - Month extracted from date
# operatorShortCode                       - Short code of the train operator
# operatorUICCode                         - UIC code of the train operator
# runningCurrently                        - Boolean indicating if train is currently running
# scheduledTime                           - Originally scheduled time
# stationName                             - Name of the station
# stationShortCode                        - Short code for the station
# stationUICCode                          - UIC code for the station
# timetableAcceptanceDate                 - Date when timetable was accepted
# timetableType                           - Type of timetable used
# trainCategory                           - Category classification of the train
# trainNumber                             - Unique identifier for the train
# trainReady                              - Boolean indicating if train is ready for departure
# trainStopping                           - Boolean indicating if train stops at station
# trainType                               - Type of train (e.g., passenger, freight)
# type                                    - Type of stop (arrival/departure)
# version                                 - Version of the data record
# =============================================================================