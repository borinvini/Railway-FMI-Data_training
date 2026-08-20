# The trainers, in array-index order. Sourced, never executed.
#
# One definition with three consumers, because they disagree destructively if
# they drift:
#
#   hpc/submit-scenarios.sh  sizes the array by ${#MODELS[@]}
#   hpc/train_scenarios.sh   picks its cell with MODELS[id % ${#MODELS[@]}]
#   hpc/train_array.sh       picks its model with MODELS[id]
#
# ORDER IS PART OF THE CONTRACT. The index math is positional, so reordering
# this list does not fail — it relabels every result directory, training
# lightgbm into a folder named for xgboost. Append new trainers at the end.
#
# The names are the keys of MODEL_TRAINERS in main.py:20; a name that is not a
# key fails in argparse, which is loud, and is the one drift mode that is safe.
MODELS=(xgboost lightgbm random_forest logistic_regression naive_bayes)
