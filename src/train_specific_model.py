# train_model.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from src.data_preprocessor import TrainingPipeline

def main():
    parser = argparse.ArgumentParser(description="Train a specific model for a given month_id.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=[
            "decision_tree", 
            "decision_tree_important", 
            "xgboost", 
            "xgboost_important", 
            "balanced_rf", 
            "regularized_regression"
        ],
        help="Which model to train"
    )
    parser.add_argument(
        "--month_id", 
        type=str, 
        required=True, 
        help="Month identifier (e.g., 2023-2024_12)"
    )
    args = parser.parse_args()

    pipeline = TrainingPipeline()

    if args.model == "decision_tree":
        result = pipeline.train_decision_tree_with_randomized_search_cv(args.month_id)
    elif args.model == "decision_tree_important":
        result = pipeline.train_decision_tree_rs_with_important_features(args.month_id)
    elif args.model == "xgboost":
        result = pipeline.train_xgboost_with_randomized_search_cv(args.month_id)
    elif args.model == "xgboost_important":
        result = pipeline.train_xgboost_rs_with_important_features(args.month_id)
    elif args.model == "balanced_rf":
        result = pipeline.train_balanced_random_forest(args.month_id)
    elif args.model == "regularized_regression":
        result = pipeline.train_regularized_regression(args.month_id)
    else:
        print(f"Unknown model: {args.model}")
        return

    print("\n=== Training Result ===")
    print(result)

if __name__ == "__main__":
    main()