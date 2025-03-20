from scipy.stats import randint

# Constants for file processing
FOLDER_NAME = "data"
INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"
PREPROCESSED_OUTPUT_FOLDER = "data/output/preprocessed"
DECISION_TREE_OUTPUT_FOLDER = "data/output/decision_tree"
RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER = "data/output/randomized_search_cv"
DATA_FILE_PREFIX = "matched_data_"
DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/important_features_randomized_search"


# Model training parameters
IMPORTANCE_THRESHOLD = 0.10


# Parameter distributions for RandomizedSearchCV
RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS = {
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

# RandomizedSearchCV settings
RANDOM_SEARCH_ITERATIONS = 50
RANDOM_SEARCH_CV_FOLDS = 5