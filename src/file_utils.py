from datetime import datetime
import json
import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, 
    average_precision_score, cohen_kappa_score, f1_score,
    classification_report, roc_curve
)

from config.const import (
    BORDERLINE_SMOTE_CONFIG,
    DATA_FILE_PREFIX, 
    DATA_FILE_PREFIX_FOR_TRAINING,
    DEFAULT_TARGET_FEATURE, 
    FOLDER_NAME, 
    INPUT_FOLDER,
    MERGED_TRAINING_READY_OUTPUT_FOLDER, 
    OUTPUT_FOLDER,
    THRESHOLD_OPTIMIZATION_CONFIG
)


def ensure_folder_structure():
    """
    Explicitly create the basic folder structure required by the application.
    
    This function guarantees that all necessary directories exist before
    the main processing begins, preventing any file operation errors.
    
    Returns:
    --------
    dict
        Summary of the folder creation operation with created/existing directories
    """
    print("Ensuring basic folder structure exists...")
    
    # List of all directories that need to be created
    required_directories = [
        FOLDER_NAME,                                    # data/
        INPUT_FOLDER,                                   # data/input/
        OUTPUT_FOLDER,                                  # data/output/
        os.path.join(FOLDER_NAME, "output", "log")      # data/output/log/
    ]
    
    created_dirs = []
    already_existed = []
    
    for directory in required_directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                created_dirs.append(directory)
                print(f"  ✓ Created directory: {directory}")
            except Exception as e:
                print(f"  ✗ Failed to create directory {directory}: {e}")
                raise
        else:
            already_existed.append(directory)
            print(f"  ✓ Directory already exists: {directory}")
    
    # Summary
    print(f"\nFolder structure verification complete:")
    print(f"  - Created: {len(created_dirs)} new directories")
    print(f"  - Existing: {len(already_existed)} directories")
    print(f"  - Total verified: {len(required_directories)} directories")
    
    if created_dirs:
        print(f"\nNewly created directories:")
        for dir_path in created_dirs:
            print(f"  - {dir_path}")
    
    return {
        "created": created_dirs,
        "existed": already_existed,
        "total_verified": len(required_directories)
    }

def check_csv_files():
    """
    Check if there are CSV files in the data/input folder and print their names.
    Returns a list of CSV file paths found.
    
    Note: The input folder is guaranteed to exist by the main initialization process.
    """
    # Using only Approach 1: Based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_folder = os.path.join(project_root, INPUT_FOLDER)
    
    print(f"Looking for CSV files in: {input_folder}")
    
    # Check for CSV files in the data/input folder
    csv_files = []
    if os.path.exists(input_folder):
        csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    # Report findings
    if csv_files:
        print(f"\n✓ Success! Found {len(csv_files)} CSV files in the input folder:")
        for file in csv_files:
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"  - {os.path.basename(file)} ({file_size:.1f} MB)")
    else:
        print(f"\n⚠️  No CSV files found in the input folder.")
        print(f"\nTo get started:")
        print(f"  1. Place your CSV files in: {input_folder}")
        print(f"  2. Expected file format: {DATA_FILE_PREFIX}YYYY_MM.csv")
        print(f"  3. Example: {DATA_FILE_PREFIX}2023_12.csv")
        print(f"  4. Then run: python main.py --target trainDelayed")
    
    return csv_files

def extract_date_range(csv_files):
    """
    Extract the date range (YYYY_MM) from the CSV filenames.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        tuple: (earliest_date, latest_date) where each date is a tuple of (year, month)
               Returns None if no valid dates found
    """
    if not csv_files:
        print("No files to extract date range from.")
        return None
    
    # Extract all YYYY_MM patterns from filenames
    dates = []
    pattern = f'{DATA_FILE_PREFIX}(\\d{{4}})_(\\d{{2}})\\.csv'  # Use the constant here
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        match = re.match(pattern, filename)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            dates.append((year, month))
    
    if not dates:
        print(f"No files match the expected naming pattern ({DATA_FILE_PREFIX}YYYY_MM.csv).")
        return None
    
    # Sort dates to find earliest and latest
    dates.sort()
    earliest = dates[0]
    latest = dates[-1]
    
    print(f"\nDate range found in files:")
    print(f"Earliest: {earliest[0]}_{earliest[1]:02d}")
    print(f"Latest: {latest[0]}_{latest[1]:02d}")
    
    return earliest, latest

def generate_output_path(input_file_path):
    """
    Generate an output file path based on the input file name.
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file.
        
    Returns:
    --------
    tuple
        (output_file_path, dataframe) - The generated output path and the loaded dataframe.
        
    Note: Basic folder structure is guaranteed to exist by the main initialization process,
          but we still ensure specific output directories exist for safety.
    """
    # Get just the filename without the path
    basename = os.path.basename(input_file_path)
    
    # Use regex to find year and month in the filename
    # This pattern looks for 4 digits (year) followed by optional characters and then 2 digits (month)
    match = re.search(r'(\d{4}).*?(\d{2})', basename)
    
    if match:
        year, month = match.groups()
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}{year}_{month}.csv"
    else:
        # Fallback if date pattern not found
        output_filename = f"{DATA_FILE_PREFIX_FOR_TRAINING}processed.csv"
        print(f"Warning: Could not extract date from filename. Using default: {output_filename}")
    
    # Ensure specific output directories exist (basic structure guaranteed by main initialization)
    # We still check these for safety in case this function is called independently
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Join the output directory path with the output filename
    output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)
    print(f"Will save to {OUTPUT_FOLDER} directory: {output_file_path}")
    
    # Load the CSV file
    print(f"Loading CSV file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    return output_file_path, df

def format_param_distributions_for_json(param_distributions):
    """
    Convert parameter distributions to a JSON-serializable format that's human readable.
    
    Args:
        param_distributions (dict): Dictionary containing parameter distributions
        
    Returns:
        dict: JSON-serializable version of the parameter distributions
    """
    formatted_params = {}
    
    for key, value in param_distributions.items():
        if hasattr(value, '__class__') and 'scipy.stats' in str(type(value)):
            # Handle scipy.stats distributions
            if hasattr(value, 'args') and hasattr(value, 'kwds'):
                if 'randint' in str(type(value)):
                    # For randint distributions, extract the range
                    low = value.args[0] if value.args else value.kwds.get('low', 'unknown')
                    high = value.args[1] if len(value.args) > 1 else value.kwds.get('high', 'unknown')
                    formatted_params[key] = f"randint({low}, {high})"
                elif 'uniform' in str(type(value)):
                    # For uniform distributions
                    loc = value.args[0] if value.args else value.kwds.get('loc', 'unknown')
                    scale = value.args[1] if len(value.args) > 1 else value.kwds.get('scale', 'unknown')
                    formatted_params[key] = f"uniform({loc}, {loc + scale})"
                else:
                    # For other scipy distributions, just show the type and args
                    type_name = str(type(value)).split('.')[-1].replace("'>", "")
                    formatted_params[key] = f"{type_name}{value.args}"
            else:
                formatted_params[key] = str(type(value))
        else:
            # For non-scipy objects (lists, strings, etc.), keep as-is
            formatted_params[key] = value
    
    return formatted_params

def plot_roc_curve_smote(fpr, tpr, roc_auc, optimal_threshold, file_identifier, output_dir):
    """Plot and save ROC curve for SMOTE models with optimal threshold marked."""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

    # Mark optimal threshold (simplified visualization)
    plt.plot([0, 1], [0, 1], 'ro', markersize=8, alpha=0.7,
    label=f'Optimal threshold = {optimal_threshold:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - SMOTE Model {file_identifier}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'smote_roc_curve_{file_identifier}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve_smote(precision, recall, pr_auc, optimal_threshold, file_identifier, output_dir):
    """Plot and save Precision-Recall curve for SMOTE models."""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - SMOTE Model {file_identifier}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'smote_precision_recall_curve_{file_identifier}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_analysis_smote(thresholds, fpr, tpr, precision, recall, file_identifier, output_dir):
    """Plot threshold analysis for SMOTE models showing how metrics change with threshold."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # TPR and FPR vs Threshold
    ax1.plot(thresholds, tpr, 'b-', label='True Positive Rate', linewidth=2)
    ax1.plot(thresholds, fpr, 'r-', label='False Positive Rate', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Rate')
    ax1.set_title('TPR and FPR vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision and Recall vs Threshold (use appropriate length)
    threshold_len = min(len(thresholds), len(precision), len(recall))
    ax2.plot(thresholds[:threshold_len], precision[:threshold_len], 'g-', label='Precision', linewidth=2)
    ax2.plot(thresholds[:threshold_len], recall[:threshold_len], 'orange', label='Recall', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision and Recall vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # F1 Score vs Threshold
    f1_scores = []
    for i in range(threshold_len):
        if precision[i] + recall[i] > 0:
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1 = 0
        f1_scores.append(f1)

    ax3.plot(thresholds[:threshold_len], f1_scores, 'purple', label='F1 Score', linewidth=2)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ROC Curve
    ax4.plot(fpr, tpr, color='darkorange', lw=2)
    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Threshold Analysis - SMOTE Model {file_identifier}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'smote_threshold_analysis_{file_identifier}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_summary_smote(all_optimal_thresholds, all_optimized_f1_scores, output_dir):
    """Create summary plot showing distribution of optimal thresholds for SMOTE models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram of optimal thresholds
    ax1.hist(all_optimal_thresholds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(all_optimal_thresholds), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_optimal_thresholds):.3f}')
    ax1.set_xlabel('Optimal Threshold')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Optimal Thresholds - SMOTE Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot of threshold vs F1 score
    ax2.scatter(all_optimal_thresholds, all_optimized_f1_scores, alpha=0.6, color='green', s=50)
    ax2.set_xlabel('Optimal Threshold')
    ax2.set_ylabel('Optimized F1 Score')
    ax2.set_title('Optimal Threshold vs F1 Score - SMOTE Models')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smote_threshold_optimization_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve_xgboost(y_true, y_proba, optimal_threshold, file_identifier, method_name, output_dir):
    """Plot and save ROC curve for XGBoost models with optimal threshold marked."""
    try:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
            
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
        # Mark optimal threshold point
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        optimal_fpr = np.mean(y_pred_optimal[y_true == 0])  # False Positive Rate
        optimal_tpr = np.mean(y_pred_optimal[y_true == 1])  # True Positive Rate (Recall)
            
        plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, 
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - XGBoost {method_name}\n{file_identifier}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
            
        # Save the plot
        plot_filename = f"roc_curve_xgboost_{file_identifier}_{method_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
            
    except Exception as e:
        print(f"      Warning: Could not create ROC plot for {file_identifier}_{method_name}: {str(e)}")

def plot_precision_recall_xgboost(y_true, y_proba, optimal_threshold, file_identifier, method_name, output_dir):
    """Plot and save Precision-Recall curve for XGBoost models with optimal threshold marked."""
    try:
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
            
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
            
        # Mark optimal threshold point
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        optimal_precision = precision_score(y_true, y_pred_optimal, zero_division=0)
        optimal_recall = recall_score(y_true, y_pred_optimal, zero_division=0)
            
        plt.plot(optimal_recall, optimal_precision, 'ro', markersize=10,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
           
        # Add baseline (random classifier)
        baseline = np.mean(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                label=f'Random Baseline (AP = {baseline:.3f})')
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - XGBoost {method_name}\n{file_identifier}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
            
        # Save the plot
        plot_filename = f"precision_recall_xgboost_{file_identifier}_{method_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
            
    except Exception as e:
        print(f"      Warning: Could not create Precision-Recall plot for {file_identifier}_{method_name}: {str(e)}")
    
def optimize_threshold_xgboost(y_true, y_proba, file_identifier, method_name, output_dir):
    """
    Find optimal threshold for XGBoost classification using ROC analysis.
        
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Probability predictions from XGBoost model
    file_identifier : str
        Identifier for the file being processed
    method_name : str
        Name of the XGBoost method being optimized
    output_dir : str
        Directory to save plots and results
            
    Returns:
    --------
    tuple
        (optimal_threshold, metrics_dict)
    """
    # Generate threshold range
    thresholds = np.arange(
        THRESHOLD_OPTIMIZATION_CONFIG["min_threshold"], 
        THRESHOLD_OPTIMIZATION_CONFIG["max_threshold"] + THRESHOLD_OPTIMIZATION_CONFIG["threshold_step"], 
        THRESHOLD_OPTIMIZATION_CONFIG["threshold_step"]
    )
        
    # Initialize tracking variables
    best_threshold = 0.5
    best_score = -1
    threshold_results = []
        
    optimization_metric = THRESHOLD_OPTIMIZATION_CONFIG["optimization_metric"]
        
    # Test each threshold
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
            
        # Select metric for optimization
        if optimization_metric == "f1":
            score = f1
        elif optimization_metric == "precision":
            score = precision
        elif optimization_metric == "recall":
            score = recall
        elif optimization_metric == "accuracy":
            score = accuracy
        else:
            score = f1  # Default to F1
            
        threshold_results.append({
            "threshold": threshold,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "optimization_score": score
        })
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
        
    # Get metrics for optimal threshold
    optimal_metrics = next(
        result for result in threshold_results 
        if result["threshold"] == best_threshold
    )
        
    # Create plots if enabled
    if THRESHOLD_OPTIMIZATION_CONFIG["plot_roc_curve"]:
        plot_roc_curve_xgboost(y_true, y_proba, best_threshold, file_identifier, method_name, output_dir)
        
    if THRESHOLD_OPTIMIZATION_CONFIG["plot_precision_recall"]:
        plot_precision_recall_xgboost(y_true, y_proba, best_threshold, file_identifier, method_name, output_dir)
        
    # Save threshold analysis results
    threshold_analysis_file = os.path.join(output_dir, f"threshold_analysis_{file_identifier}_{method_name}.json")
    with open(threshold_analysis_file, 'w') as f:
        json.dump({
            "file_identifier": file_identifier,
            "method_name": method_name,
            "optimization_metric": optimization_metric,
            "optimal_threshold": best_threshold,
            "optimal_metrics": optimal_metrics,
            "all_thresholds": threshold_results
        }, f, indent=2)
        
    return best_threshold, {
        "f1_score": optimal_metrics["f1_score"],
        "precision": optimal_metrics["precision"],
        "recall": optimal_metrics["recall"],
        "accuracy": optimal_metrics["accuracy"]
    }


def plot_smote_summary(smote_results, output_dir):
    """
    Create summary visualizations for BorderlineSMOTE data generation.

    Parameters:
    -----------
    smote_results : dict
        Results from BorderlineSMOTE generation
    output_dir : str
        Directory to save the plots
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        file_identifiers = []
        original_samples = []
        synthetic_samples = []
        augmentation_ratios = []
        
        for result in smote_results["file_results"]:
            if result.get("borderline_smote_successful", False):
                file_identifiers.append(result["file_identifier"])
                original_samples.append(result["original_samples"])
                synthetic_samples.append(result["synthetic_samples_added"])
                augmentation_ratios.append(result["augmentation_ratio"])
        
        # Plot 1: Original vs Synthetic Samples
        x_pos = np.arange(len(file_identifiers))
        width = 0.35
        
        ax1.bar(x_pos - width/2, original_samples, width, label='Original Samples', alpha=0.8, color='skyblue')
        ax1.bar(x_pos + width/2, synthetic_samples, width, label='Synthetic Samples Added', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('File Identifier')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Original vs Synthetic Samples by File')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(file_identifiers, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Augmentation Ratios
        bars = ax2.bar(file_identifiers, augmentation_ratios, alpha=0.8, color='lightgreen')
        ax2.set_xlabel('File Identifier')
        ax2.set_ylabel('Augmentation Ratio')
        ax2.set_title('Data Augmentation Ratio by File')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, augmentation_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}x', ha='center', va='bottom')
        
        # Plot 3: Total Sample Distribution
        total_original = sum(original_samples)
        total_synthetic = sum(synthetic_samples)
        
        ax3.pie([total_original, total_synthetic], 
                labels=['Original Samples', 'Synthetic Samples'], 
                autopct='%1.1f%%', 
                startangle=90,
                colors=['skyblue', 'lightcoral'])
        ax3.set_title('Overall Distribution: Original vs Synthetic Samples')
        
        # Plot 4: Summary Statistics
        ax4.axis('off')
        
        # Get overall class distribution info for display
        overall_before = smote_results["aggregate_metrics"]["overall_class_distributions"]["before_smote"]["distribution"]
        overall_after = smote_results["aggregate_metrics"]["overall_class_distributions"]["after_smote"]["distribution"]
        
        # Format class distribution strings
        before_dist_str = ", ".join([f"Class {k}: {v['count']:,} ({v['percentage']}%)" for k, v in overall_before.items()])
        after_dist_str = ", ".join([f"Class {k}: {v['count']:,} ({v['percentage']}%)" for k, v in overall_after.items()])
        
        summary_text = f"""
    BorderlineSMOTE Generation Summary

    Total Files Processed: {len(file_identifiers)}

    Sample Counts:
    Original Samples: {total_original:,}
    Synthetic Samples: {total_synthetic:,}
    Final Samples: {total_original + total_synthetic:,}

    Class Distribution Before SMOTE:
    {before_dist_str}

    Class Distribution After SMOTE:
    {after_dist_str}

    Augmentation Ratios:
    Average: {np.mean(augmentation_ratios):.2f}x
    Min: {min(augmentation_ratios):.2f}x
    Max: {max(augmentation_ratios):.2f}x

    BorderlineSMOTE Configuration:
    k_neighbors: {BORDERLINE_SMOTE_CONFIG['k_neighbors']}
    m_neighbors: {BORDERLINE_SMOTE_CONFIG['m_neighbors']}
    kind: {BORDERLINE_SMOTE_CONFIG['kind']}
    sampling_strategy: {BORDERLINE_SMOTE_CONFIG['sampling_strategy']}
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(output_dir, "borderline_smote_generation_summary.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    generate_borderline_smote_data: Summary plot saved to {plot_filename}")
        
    except Exception as e:
        print(f"    generate_borderline_smote_data: Error creating summary plot: {str(e)}")

def create_correlation_plot(correlations, filename, output_dir, title):
    """
    Create a Point-Biserial Correlation Analysis plot for a single file.
    
    Parameters:
    -----------
    correlations : dict
        Dictionary of feature names and their correlation values
    filename : str
        Name of the source file
    output_dir : str
        Directory to save the plot
    title : str
        Title for the plot
    """
    try:
        # Filter out NaN correlations
        valid_correlations = {k: v for k, v in correlations.items() if not pd.isna(v)}
        
        if not valid_correlations:
            print(f"        No valid correlations to plot for {filename}")
            return
        
        # Sort correlations by absolute value for better visualization
        sorted_correlations = sorted(valid_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [item[0] for item in sorted_correlations]
        corr_values = [item[1] for item in sorted_correlations]
        
        # Create figure with single plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Horizontal bar chart of Point-Biserial correlations
        colors = ['red' if val < 0 else 'steelblue' for val in corr_values]
        bars = ax.barh(range(len(features)), corr_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=11)
        ax.set_xlabel('Point-Biserial Correlation Coefficient (r_pb)', fontsize=12)
        ax.set_title(f'Point-Biserial Correlation Analysis\n{DEFAULT_TARGET_FEATURE} vs Weather Features', fontsize=14, pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(-1, 1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, corr_values)):
            label_x = value + (0.02 if value >= 0 else -0.02)
            ha = 'left' if value >= 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                    ha=ha, va='center', fontsize=10, fontweight='bold')
        
        # Add interpretation guide
        interpretation_text = (
            "Interpretation Guide:\n"
            "• |r_pb| ≥ 0.7: Very Strong relationship\n"
            "• 0.5 ≤ |r_pb| < 0.7: Strong relationship\n"
            "• 0.3 ≤ |r_pb| < 0.5: Moderate relationship\n"
            "• 0.1 ≤ |r_pb| < 0.3: Weak relationship\n"
            "• |r_pb| < 0.1: Very weak/no relationship\n\n"
            "Positive: Higher weather values → More delays\n"
            "Negative: Higher weather values → Fewer delays"
        )
        
        fig.text(0.02, 0.02, interpretation_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                verticalalignment='bottom')
        
        plt.suptitle(f'Point-Biserial Correlation Analysis: {filename}', fontsize=15, y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        # Save plot
        safe_filename = filename.replace('.csv', '').replace(' ', '_')
        plot_filename = f"point_biserial_correlation_{safe_filename}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved point-biserial correlation plot: {plot_filename}")
        
    except Exception as e:
        print(f"        Warning: Failed to create plot for {filename}: {str(e)}")

def create_distribution_plots(df, weather_features, filename, output_dir):
    """
    Create distribution plots for target feature and weather features.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing all features
    weather_features : list
        List of weather feature column names
    filename : str
        Name of the source file
    output_dir : str
        Directory to save the plots
    """
    try:
        print(f"        Creating distribution plots for {filename}...")
        
        # Check if target feature exists
        if DEFAULT_TARGET_FEATURE not in df.columns:
            print(f"        Warning: Target feature '{DEFAULT_TARGET_FEATURE}' not found for distribution plots")
            return
        
        # Filter weather features that exist in the dataframe
        valid_weather_features = [col for col in weather_features if col in df.columns]
        
        if not valid_weather_features:
            print(f"        Warning: No valid weather features found for distribution plots")
            return
        
        # Create comprehensive distribution analysis
        n_features = len(valid_weather_features)
        n_cols = 3
        n_rows = max(2, (n_features + n_cols - 1) // n_cols + 1)  # +1 for target distribution
        
        fig = plt.figure(figsize=(18, 6 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
        
        # 1. Target feature distribution (first row, span all columns)
        ax_target = fig.add_subplot(gs[0, :])
        target_counts = df[DEFAULT_TARGET_FEATURE].value_counts()
        target_proportions = df[DEFAULT_TARGET_FEATURE].value_counts(normalize=True)
        
        bars = ax_target.bar(
            ['Not Delayed (False)', 'Delayed (True)'], 
            [target_counts.get(False, 0), target_counts.get(True, 0)],
            color=['lightgreen', 'lightcoral'], 
            alpha=0.7,
            edgecolor='black',
            linewidth=1
        )
        
        ax_target.set_title(f'{DEFAULT_TARGET_FEATURE} Distribution\nTotal samples: {len(df):,}', 
                        fontsize=14, fontweight='bold')
        ax_target.set_ylabel('Count', fontsize=12)
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, [target_counts.get(False, 0), target_counts.get(True, 0)])):
            percentage = (count / len(df)) * 100
            ax_target.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(df)*0.01,
                        f'{count:,}\n({percentage:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax_target.grid(axis='y', alpha=0.3)
        
        # 2. Weather feature distributions and comparative box plots
        for i, feature in enumerate(valid_weather_features):
            row = (i // n_cols) + 1
            col = i % n_cols
            
            # Create subplot for this feature
            ax = fig.add_subplot(gs[row, col])
            
            # Get feature data, handling missing values
            feature_data = df[feature].dropna()
            if len(feature_data) == 0:
                ax.text(0.5, 0.5, f'No valid data\nfor {feature}', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title(feature, fontsize=12, fontweight='bold')
                continue
            
            # Create combined plot: histogram + box plot comparison
            # Top: Histogram of the feature
            ax_hist = ax
            ax_hist.hist(feature_data, bins=30, alpha=0.7, color='steelblue', 
                        edgecolor='black', linewidth=0.5)
            ax_hist.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
            ax_hist.set_xlabel(feature, fontsize=10)
            ax_hist.set_ylabel('Frequency', fontsize=10)
            ax_hist.grid(axis='y', alpha=0.3)
            
            # Add statistics text
            stats_text = (f'Mean: {feature_data.mean():.2f}\n'
                        f'Std: {feature_data.std():.2f}\n'
                        f'Missing: {df[feature].isna().sum():,}')
            ax_hist.text(0.02, 0.98, stats_text, transform=ax_hist.transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle(f'Feature Distributions Analysis: {filename}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save distribution plot
        safe_filename = filename.replace('.csv', '').replace(' ', '_')
        dist_plot_filename = f"distributions_{safe_filename}.png"
        dist_plot_path = os.path.join(output_dir, dist_plot_filename)
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate comparative box plots (delayed vs non-delayed)
        create_comparative_distributions(df, valid_weather_features, filename, output_dir)
        
        print(f"        Saved distribution plots: {dist_plot_filename}")
        
    except Exception as e:
        print(f"        Warning: Failed to create distribution plots for {filename}: {str(e)}")

def create_comparative_distributions(df, weather_features, filename, output_dir):
    """
    Create comparative distribution plots (box plots and violin plots) 
    comparing weather features between delayed and non-delayed cases.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing all features
    weather_features : list
        List of weather feature column names
    filename : str
        Name of the source file
    output_dir : str
        Directory to save the plots
    """
    try:
        print(f"        Creating comparative distribution plots for {filename}...")
        
        # Import required constant (assuming it's defined in your config)
        # DEFAULT_TARGET_FEATURE should be imported from your config
        from config.const import DEFAULT_TARGET_FEATURE
        
        # Check if target feature exists
        if DEFAULT_TARGET_FEATURE not in df.columns:
            print(f"        Warning: Target feature '{DEFAULT_TARGET_FEATURE}' not found for comparative plots")
            return
        
        # Filter weather features that exist in the dataframe
        valid_weather_features = [col for col in weather_features if col in df.columns]
        
        if not valid_weather_features:
            print(f"        Warning: No valid weather features found for comparative plots")
            return
        
        # Create comparative analysis
        n_features = len(valid_weather_features)
        n_cols = 2  # Two columns for better readability
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_features > 1 else [axes]
        elif n_features == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature in enumerate(valid_weather_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get feature data, handling missing values
            feature_data = df[[feature, DEFAULT_TARGET_FEATURE]].dropna()
            
            if len(feature_data) == 0:
                ax.text(0.5, 0.5, f'No valid data\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title(f'{feature} - No Data', fontsize=12, fontweight='bold')
                continue
            
            # Create box plot comparing delayed vs non-delayed
            delayed_data = feature_data[feature_data[DEFAULT_TARGET_FEATURE] == True][feature]
            not_delayed_data = feature_data[feature_data[DEFAULT_TARGET_FEATURE] == False][feature]
            
            # Create violin plot for better distribution visualization
            violin_data = [not_delayed_data.values, delayed_data.values]
            violin_labels = ['Not Delayed', 'Delayed']
            
            # Create violin plot
            parts = ax.violinplot(violin_data, positions=[1, 2], showmeans=True, showmedians=True)
            
            # Customize violin plot colors
            colors = ['lightgreen', 'lightcoral']
            for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Add box plot overlay for quartiles
            bp = ax.boxplot(violin_data, positions=[1, 2], widths=0.3, 
                          patch_artist=True, showfliers=False)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(violin_labels)
            ax.set_title(f'{feature}\nDelayed vs Not Delayed', fontsize=11, fontweight='bold')
            ax.set_ylabel(feature, fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            # Add statistics text
            delayed_count = len(delayed_data)
            not_delayed_count = len(not_delayed_data)
            delayed_mean = delayed_data.mean() if len(delayed_data) > 0 else 0
            not_delayed_mean = not_delayed_data.mean() if len(not_delayed_data) > 0 else 0
            
            stats_text = (f'Not Delayed: n={not_delayed_count:,}, μ={not_delayed_mean:.2f}\n'
                         f'Delayed: n={delayed_count:,}, μ={delayed_mean:.2f}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Hide empty subplots if we have an odd number of features
        if n_features % n_cols != 0:
            for i in range(n_features, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
        
        plt.suptitle(f'Comparative Weather Feature Distributions: {filename}', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save comparative distribution plot
        safe_filename = filename.replace('.csv', '').replace(' ', '_')
        comp_plot_filename = f"comparative_distributions_{safe_filename}.png"
        comp_plot_path = os.path.join(output_dir, comp_plot_filename)
        plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved comparative distribution plots: {comp_plot_filename}")
        
    except Exception as e:
        print(f"        Warning: Failed to create comparative distribution plots for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()  # For debugging purposes

def create_combined_correlation_analysis(correlation_results, output_dir):
    """
    Create combined Point-Biserial correlation analysis across all files.

    Parameters:
    -----------
    correlation_results : list
        List of correlation result dictionaries from each file
    output_dir : str
        Directory to save the combined analysis
    """
    try:
        print("    correlation_analysis: Creating combined point-biserial correlation analysis...")
        
        # Collect all unique weather features
        all_features = set()
        for result in correlation_results:
            all_features.update(result['correlations'].keys())
        all_features = sorted(list(all_features))
        
        # Calculate average correlations across files
        avg_correlations = {}
        correlation_ranges = {}
        
        for feature in all_features:
            feature_values = []
            for result in correlation_results:
                corr_value = result['correlations'].get(feature, np.nan)
                if not pd.isna(corr_value):
                    feature_values.append(corr_value)
            
            if feature_values:
                avg_correlations[feature] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values),
                    'count': len(feature_values)
                }
                correlation_ranges[feature] = (np.min(feature_values), np.max(feature_values))
        
        # Sort by absolute mean correlation
        sorted_features = sorted(avg_correlations.keys(), 
                            key=lambda x: abs(avg_correlations[x]['mean']), 
                            reverse=True)
        
        # Create comprehensive combined plot
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 2, 1])
        
        # Main plot: Average correlations with error bars
        ax1 = fig.add_subplot(gs[0, 0])
        
        y_positions = range(len(sorted_features))
        means = [avg_correlations[f]['mean'] for f in sorted_features]
        stds = [avg_correlations[f]['std'] for f in sorted_features]
        colors = ['red' if val < 0 else 'steelblue' for val in means]
        
        bars = ax1.barh(y_positions, means, xerr=stds, color=colors, alpha=0.7, 
                    capsize=5, edgecolor='black', linewidth=0.5)
        
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(sorted_features, fontsize=10)
        ax1.set_xlabel('Average Point-Biserial Correlation', fontsize=12)
        ax1.set_title('Average Point-Biserial Correlations\n(with Standard Deviation)', fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim(-1, 1)
        
        # Add value labels
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
            label_x = mean_val + (0.05 if mean_val >= 0 else -0.05)
            ha = 'left' if mean_val >= 0 else 'right'
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{mean_val:.3f}±{std_val:.3f}', 
                    ha=ha, va='center', fontsize=8, fontweight='bold')
        
        # Range analysis plot
        ax2 = fig.add_subplot(gs[0, 1])
        
        for i, feature in enumerate(sorted_features):
            min_val, max_val = correlation_ranges[feature]
            mean_val = avg_correlations[feature]['mean']
            
            # Plot range as line
            ax2.plot([min_val, max_val], [i, i], 'k-', linewidth=2, alpha=0.6)
            # Plot min and max as points
            ax2.plot(min_val, i, 'ro', markersize=6, alpha=0.8)
            ax2.plot(max_val, i, 'bo', markersize=6, alpha=0.8)
            # Plot mean as diamond
            ax2.plot(mean_val, i, 'gD', markersize=8, alpha=0.9)
        
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(sorted_features, fontsize=10)
        ax2.set_xlabel('Correlation Range Across Files', fontsize=12)
        ax2.set_title('Correlation Variability\n(Min-Max Range)', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim(-1, 1)
        ax2.legend(['Range', 'Minimum', 'Maximum', 'Mean'], loc='lower right', fontsize=9)
        
        # Strength classification plot
        ax3 = fig.add_subplot(gs[0, 2])
        
        strength_categories = {'Very Strong (≥0.7)': [], 'Strong (0.5-0.7)': [], 
                            'Moderate (0.3-0.5)': [], 'Weak (0.1-0.3)': [], 
                            'Very Weak (<0.1)': []}
        
        for feature in sorted_features:
            abs_mean = abs(avg_correlations[feature]['mean'])
            if abs_mean >= 0.7:
                strength_categories['Very Strong (≥0.7)'].append(feature)
            elif abs_mean >= 0.5:
                strength_categories['Strong (0.5-0.7)'].append(feature)
            elif abs_mean >= 0.3:
                strength_categories['Moderate (0.3-0.5)'].append(feature)
            elif abs_mean >= 0.1:
                strength_categories['Weak (0.1-0.3)'].append(feature)
            else:
                strength_categories['Very Weak (<0.1)'].append(feature)
        
        category_counts = [len(features) for features in strength_categories.values()]
        category_labels = list(strength_categories.keys())
        colors_pie = ['darkred', 'red', 'orange', 'yellow', 'lightgray']
        
        wedges, texts, autotexts = ax3.pie(category_counts, labels=category_labels, autopct='%1.0f%%', 
                                        colors=colors_pie, startangle=90)
        ax3.set_title('Correlation Strength\nDistribution', fontsize=12)
        
        # File-by-file heatmap
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Create matrix for heatmap
        heatmap_data = []
        file_labels = []
        
        for result in correlation_results:
            row_data = []
            for feature in sorted_features:
                corr_value = result['correlations'].get(feature, np.nan)
                row_data.append(corr_value)
            heatmap_data.append(row_data)
            file_labels.append(result['filename'].replace('.csv', ''))
        
        heatmap_matrix = np.array(heatmap_data)
        mask = np.isnan(heatmap_matrix)
        
        im = ax4.imshow(heatmap_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Point-Biserial Correlation Coefficient', fontsize=10)
        
        ax4.set_xticks(range(len(sorted_features)))
        ax4.set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=9)
        ax4.set_yticks(range(len(file_labels)))
        ax4.set_yticklabels(file_labels, fontsize=9)
        ax4.set_title('Point-Biserial Correlations Across All Files', fontsize=12)
        
        # Add text annotations for non-NaN values
        for i in range(len(file_labels)):
            for j in range(len(sorted_features)):
                if not mask[i, j]:
                    text_color = 'white' if abs(heatmap_matrix[i, j]) > 0.5 else 'black'
                    ax4.text(j, i, f'{heatmap_matrix[i, j]:.2f}', 
                            ha='center', va='center', color=text_color, fontsize=7)
        
        # Summary statistics text
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        summary_text = "Summary Statistics:\n\n"
        summary_text += f"Total Features: {len(sorted_features)}\n"
        summary_text += f"Files Analyzed: {len(correlation_results)}\n\n"
        
        if sorted_features:
            strongest_feature = sorted_features[0]
            strongest_corr = avg_correlations[strongest_feature]['mean']
            summary_text += f"Strongest Correlation:\n{strongest_feature}\n(r = {strongest_corr:.3f})\n\n"
        
        # Count by strength
        for category, features in strength_categories.items():
            if features:
                summary_text += f"{category}: {len(features)}\n"
        
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightblue", alpha=0.8))
        
        plt.suptitle(f'Comprehensive Point-Biserial Correlation Analysis\n{DEFAULT_TARGET_FEATURE} vs Weather Features', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save combined plot
        combined_plot_path = os.path.join(output_dir, "combined_point_biserial_analysis.png")
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed average correlations summary
        avg_summary_path = os.path.join(output_dir, "point_biserial_summary.txt")
        with open(avg_summary_path, 'w') as f:
            f.write("Point-Biserial Correlation Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Target Variable: {DEFAULT_TARGET_FEATURE} (Binary)\n")
            f.write(f"Weather Features: {len(all_features)} (Continuous)\n")
            f.write(f"Files Analyzed: {len(correlation_results)}\n\n")
            
            f.write("Correlation Strength Interpretation:\n")
            f.write("• |r_pb| ≥ 0.7: Very Strong relationship\n")
            f.write("• 0.5 ≤ |r_pb| < 0.7: Strong relationship\n")
            f.write("• 0.3 ≤ |r_pb| < 0.5: Moderate relationship\n")
            f.write("• 0.1 ≤ |r_pb| < 0.3: Weak relationship\n")
            f.write("• |r_pb| < 0.1: Very weak/no relationship\n\n")
            
            f.write("Weather Features (sorted by absolute correlation strength):\n")
            f.write("-" * 70 + "\n")
            for i, feature in enumerate(sorted_features, 1):
                stats = avg_correlations[feature]
                strength = ""
                abs_mean = abs(stats['mean'])
                if abs_mean >= 0.7:
                    strength = "Very Strong"
                elif abs_mean >= 0.5:
                    strength = "Strong"
                elif abs_mean >= 0.3:
                    strength = "Moderate"
                elif abs_mean >= 0.1:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                
                f.write(f"{i:2d}. {feature}:\n")
                f.write(f"    Mean correlation: {stats['mean']:+.4f} ({strength})\n")
                f.write(f"    Std deviation: {stats['std']:.4f}\n")
                f.write(f"    Range: [{stats['min']:+.4f}, {stats['max']:+.4f}]\n")
                f.write(f"    Files available: {stats['count']}/{len(correlation_results)}\n")
                
                # Interpretation
                if stats['mean'] > 0:
                    direction = "Higher values → More train delays"
                elif stats['mean'] < 0:
                    direction = "Higher values → Fewer train delays"
                else:
                    direction = "No clear relationship"
                f.write(f"    Interpretation: {direction}\n\n")
        
        print(f"        Saved comprehensive point-biserial correlation analysis")
        
    except Exception as e:
        print(f"        Warning: Failed to create combined analysis: {str(e)}")

def create_non_weather_correlation_plot(correlations, correlation_types, filename, output_dir, title):
    """
    Create a Non-Weather Features Correlation Analysis plot for a single file.

    This creates separate subplots for boolean and temporal features with their correlations.
    """
    try:
        # Separate correlations by type
        boolean_correlations = {}
        temporal_correlations = {}
        
        for feature, corr in correlations.items():
            if pd.isna(corr):
                continue
            corr_type = correlation_types.get(feature, "")
            if "Boolean" in corr_type:
                boolean_correlations[feature] = corr
            elif "Temporal" in corr_type:
                temporal_correlations[feature] = corr
        
        # Calculate subplot layout
        n_plots = 0
        if boolean_correlations:
            n_plots += 1
        if temporal_correlations:
            n_plots += 1
        
        if n_plots == 0:
            print(f"      No valid correlations to plot for {filename}")
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot boolean features
        if boolean_correlations:
            ax = axes[plot_idx]
            features = list(boolean_correlations.keys())
            values = list(boolean_correlations.values())
            colors = ['darkblue' if v >= 0 else 'darkred' for v in values]
            
            bars = ax.bar(features, values, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_title(f'Boolean Features - Phi Coefficient\n({len(features)} features)')
            ax.set_ylabel('Phi Coefficient (φ)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')
            
            # Rotate x-labels if needed
            if len(features) > 3:
                ax.tick_params(axis='x', rotation=45)
            
            plot_idx += 1
        
        # Plot temporal features
        if temporal_correlations:
            ax = axes[plot_idx]
            features = list(temporal_correlations.keys())
            values = list(temporal_correlations.values())
            colors = ['steelblue' if v >= 0 else 'crimson' for v in values]
            
            bars = ax.bar(features, values, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_title(f'Temporal Features (Sin-Cos) - Point-Biserial Correlation\n({len(features)} features)')
            ax.set_ylabel('Point-Biserial Correlation (r_pb)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')
            
            # Rotate x-labels if needed
            if len(features) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"non_weather_correlation_{filename.replace('.csv', '.png')}"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved non-weather correlation plot: {plot_filename}")
        
    except Exception as e:
        print(f"      Error creating non-weather correlation plot: {str(e)}")

def create_combined_non_weather_correlation_analysis(correlation_results, output_dir):
    """
    Create combined analysis across all non-weather correlation results.
    """
    try:
        print(f"    Creating combined non-weather correlation analysis...")
        
        # Aggregate correlations across all files
        all_features = set()
        for result in correlation_results:
            all_features.update(result['correlations'].keys())
        
        all_features = sorted(list(all_features))
        
        # Calculate average correlations
        avg_correlations = {}
        for feature in all_features:
            values = []
            for result in correlation_results:
                if feature in result['correlations'] and not pd.isna(result['correlations'][feature]):
                    values.append(result['correlations'][feature])
            
            if values:
                avg_correlations[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Create combined plot
        if avg_correlations:
            # Sort features by absolute correlation strength
            sorted_features = sorted(avg_correlations.keys(), 
                                key=lambda x: abs(avg_correlations[x]['mean']), 
                                reverse=True)
            
            # Separate by feature type for plotting
            boolean_features = [f for f in sorted_features if f in ['trainStopping', 'commercialStop']]
            temporal_features = [f for f in sorted_features if f not in boolean_features]
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot boolean features
            if boolean_features:
                ax = axes[0]
                values = [avg_correlations[f]['mean'] for f in boolean_features]
                errors = [avg_correlations[f]['std'] for f in boolean_features]
                colors = ['darkblue' if v >= 0 else 'darkred' for v in values]
                
                bars = ax.bar(boolean_features, values, yerr=errors, color=colors, 
                            alpha=0.7, edgecolor='black', capsize=5)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title(f'Boolean Features - Average Phi Coefficient Across All Files\n({len(boolean_features)} features)')
                ax.set_ylabel('Average Phi Coefficient (φ)')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value, error in zip(bars, values, errors):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (error + 0.01 if height >= 0 else -error - 0.01),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
            else:
                axes[0].text(0.5, 0.5, 'No Boolean Features Available', 
                        ha='center', va='center', transform=axes[0].transAxes, fontsize=14)
                axes[0].set_title('Boolean Features - No Data')
            
            # Plot temporal features
            if temporal_features:
                ax = axes[1]
                values = [avg_correlations[f]['mean'] for f in temporal_features]
                errors = [avg_correlations[f]['std'] for f in temporal_features]
                colors = ['steelblue' if v >= 0 else 'crimson' for v in values]
                
                bars = ax.bar(temporal_features, values, yerr=errors, color=colors, 
                            alpha=0.7, edgecolor='black', capsize=5)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title(f'Temporal Features - Average Point-Biserial Correlation Across All Files\n({len(temporal_features)} features)')
                ax.set_ylabel('Average Point-Biserial Correlation (r_pb)')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value, error in zip(bars, values, errors):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (error + 0.01 if height >= 0 else -error - 0.01),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
                
                # Rotate x-labels for temporal features
                ax.tick_params(axis='x', rotation=45)
            else:
                axes[1].text(0.5, 0.5, 'No Temporal Features Available', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
                axes[1].set_title('Temporal Features - No Data')
            
            plt.suptitle(f'Combined Non-Weather Features Correlation Analysis\n{DEFAULT_TARGET_FEATURE} vs Non-Weather Features', 
                        fontsize=16, y=0.98)
            plt.tight_layout()
            
            # Save combined plot
            combined_plot_path = os.path.join(output_dir, "combined_non_weather_correlation_analysis.png")
            plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save detailed average correlations summary
            avg_summary_path = os.path.join(output_dir, "non_weather_correlation_summary.txt")
            with open(avg_summary_path, 'w') as f:
                f.write("Combined Non-Weather Features Correlation Analysis\n")
                f.write("=" * 65 + "\n\n")
                f.write(f"Target Variable: {DEFAULT_TARGET_FEATURE} (Binary)\n")
                f.write(f"Non-Weather Features: {len(all_features)}\n")
                f.write(f"Files Analyzed: {len(correlation_results)}\n\n")
                
                f.write("Correlation Strength Interpretation:\n")
                f.write("• |correlation| ≥ 0.7: Very Strong relationship\n")
                f.write("• 0.5 ≤ |correlation| < 0.7: Strong relationship\n")
                f.write("• 0.3 ≤ |correlation| < 0.5: Moderate relationship\n")
                f.write("• 0.1 ≤ |correlation| < 0.3: Weak relationship\n")
                f.write("• |correlation| < 0.1: Very weak/no relationship\n\n")
                
                f.write("Non-Weather Features (sorted by absolute correlation strength):\n")
                f.write("-" * 80 + "\n")
                for i, feature in enumerate(sorted_features, 1):
                    stats = avg_correlations[feature]
                    strength = ""
                    abs_mean = abs(stats['mean'])
                    if abs_mean >= 0.7:
                        strength = "Very Strong"
                    elif abs_mean >= 0.5:
                        strength = "Strong"
                    elif abs_mean >= 0.3:
                        strength = "Moderate"
                    elif abs_mean >= 0.1:
                        strength = "Weak"
                    else:
                        strength = "Very Weak"
                    
                    feature_type = "Boolean" if feature in ['trainStopping', 'commercialStop'] else "Temporal"
                    
                    f.write(f"{i:2d}. {feature:20s} | {stats['mean']:+.4f} ± {stats['std']:.4f} | {strength:12s} | {feature_type}\n")
                    f.write(f"    Files: {stats['count']}/{len(correlation_results)} | Range: [{stats['min']:+.4f}, {stats['max']:+.4f}]\n\n")
            
            print(f"    Combined non-weather correlation analysis saved")
        
    except Exception as e:
        print(f"    Error creating combined non-weather correlation analysis: {str(e)}")

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive classification metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like  
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
        
    Returns:
    --------
    dict
        Dictionary containing various classification metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # Classification report metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    #metrics['weighted_avg_f1'] = report['weighted avg']['f1-score']
    #metrics['weighted_avg_precision'] = report['weighted avg']['precision']
    #metrics['weighted_avg_recall'] = report['weighted avg']['recall']

    # Probability-based metrics (if available)
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        try:
            # For binary classification, use positive class probabilities
            if y_pred_proba.ndim > 1:
                pos_proba = y_pred_proba[:, 1]
            else:
                pos_proba = y_pred_proba
                
            metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
            metrics['pr_auc'] = average_precision_score(y_true, pos_proba)
        except:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None

    # Class-specific metrics
    for class_label in report:
        if class_label not in ['weighted avg', 'macro avg', 'accuracy']:
            metrics[f'class_{class_label}_f1'] = report[class_label]['f1-score']
            metrics[f'class_{class_label}_precision'] = report[class_label]['precision']
            metrics[f'class_{class_label}_recall'] = report[class_label]['recall']
            metrics[f'class_{class_label}_support'] = report[class_label]['support']

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    for class_val, count in zip(unique, counts):
        metrics[f'class_{class_val}_test_count'] = count
        metrics[f'class_{class_val}_test_percentage'] = (count / len(y_true)) * 100

    return metrics

def save_confusion_matrix(conf_matrix, y_test, y_pred, file_identifier, output_dir):
    """
    Save confusion matrix as both CSV data and visualization plot.

    Parameters:
    -----------
    conf_matrix : array-like
        Confusion matrix from sklearn.metrics.confusion_matrix
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels  
    file_identifier : str
        Identifier for the file being processed
    output_dir : str
        Directory to save the confusion matrix files
        
    Returns:
    --------
    dict
        Dictionary with paths to saved confusion matrix files
    """
    try:
        # Get unique class labels
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        
        # Create confusion matrix DataFrame
        conf_df = pd.DataFrame(
            conf_matrix,
            index=[f'True_{label}' for label in unique_labels],
            columns=[f'Pred_{label}' for label in unique_labels]
        )
        
        # Save confusion matrix as CSV
        csv_filename = f"confusion_matrix_{file_identifier}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        conf_df.to_csv(csv_path)
        
        # Create and save confusion matrix visualization
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[f'Predicted {label}' for label in unique_labels],
            yticklabels=[f'Actual {label}' for label in unique_labels],
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'Confusion Matrix - {file_identifier}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('Actual Label', fontsize=12)
        
        # Add classification metrics as text
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary vs multiclass metrics
        if len(unique_labels) == 2:
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Add metrics text box
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"confusion_matrix_{file_identifier}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"    train_decision_tree: Saved confusion matrix to {csv_filename} and {plot_filename}")
        
        return {
            'csv_file': csv_filename,
            'plot_file': plot_filename,
            'csv_path': csv_path,
            'plot_path': plot_path
        }
        
    except Exception as e:
        print(f"    train_decision_tree: Warning - Failed to save confusion matrix: {str(e)}")
        return {
            'csv_file': None,
            'plot_file': None,
            'csv_path': None,
            'plot_path': None,
            'error': str(e)
        }

def find_optimal_thresholds(y_true, y_proba, fpr, tpr, thresholds, precision, recall, pr_thresholds):
    """
    Find optimal thresholds based on different metrics.

    Returns:
    --------
    dict
        Dictionary with optimal thresholds for different metrics
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    # Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    youden_optimal_idx = np.argmax(j_scores)
    youden_threshold = thresholds[youden_optimal_idx]

    # F1 score optimization
    f1_scores = []
    test_thresholds = np.arange(
        THRESHOLD_OPTIMIZATION_CONFIG["min_threshold"],
        THRESHOLD_OPTIMIZATION_CONFIG["max_threshold"],
        THRESHOLD_OPTIMIZATION_CONFIG["threshold_step"]
    )

    for threshold in test_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    f1_optimal_idx = np.argmax(f1_scores)
    f1_threshold = test_thresholds[f1_optimal_idx]

    # Precision optimization (highest precision with reasonable recall)
    precision_scores = []
    for threshold in test_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if np.sum(y_pred) > 0:  # Avoid division by zero
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            # Only consider thresholds with recall > 0.1
            if rec > 0.1:
                precision_scores.append(prec)
            else:
                precision_scores.append(0)
        else:
            precision_scores.append(0)

    precision_optimal_idx = np.argmax(precision_scores)
    precision_threshold = test_thresholds[precision_optimal_idx]

    # Recall optimization (highest recall with reasonable precision)
    recall_scores = []
    for threshold in test_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if np.sum(y_pred) > 0:
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            # Only consider thresholds with precision > 0.1
            if prec > 0.1:
                recall_scores.append(rec)
            else:
                recall_scores.append(0)
        else:
            recall_scores.append(0)

    recall_optimal_idx = np.argmax(recall_scores)
    recall_threshold = test_thresholds[recall_optimal_idx]

    # Accuracy optimization
    accuracy_scores = []
    for threshold in test_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        accuracy_scores.append(acc)

    accuracy_optimal_idx = np.argmax(accuracy_scores)
    accuracy_threshold = test_thresholds[accuracy_optimal_idx]

    return {
        "youden": float(youden_threshold),
        "f1": float(f1_threshold),
        "precision": float(precision_threshold),
        "recall": float(recall_threshold),
        "accuracy": float(accuracy_threshold)
    }

def plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold, file_identifier, output_dir):
    """Plot and save ROC curve with optimal threshold marked."""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

    # Mark optimal threshold
    from sklearn.metrics import roc_curve
    # Find closest threshold point
    optimal_idx = np.argmin(np.abs(np.array([0.5]) - optimal_threshold))  # Simplified for visualization
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
            label=f'Optimal threshold = {optimal_threshold:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {file_identifier}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_{file_identifier}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(precision, recall, pr_auc, optimal_threshold, file_identifier, output_dir):
    """Plot and save Precision-Recall curve."""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {file_identifier}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'precision_recall_curve_{file_identifier}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_analysis(thresholds, fpr, tpr, precision, recall, file_identifier, output_dir):
    """Plot threshold analysis showing how metrics change with threshold."""
    plt.figure(figsize=(15, 10))

    # Ensure all arrays have the same length
    min_len = min(len(thresholds), len(fpr), len(tpr), len(precision), len(recall))
    thresholds = thresholds[:min_len]
    fpr = fpr[:min_len]
    tpr = tpr[:min_len]
    precision = precision[:min_len]
    recall = recall[:min_len]

    plt.subplot(2, 2, 1)
    plt.plot(thresholds, tpr, 'b-', label='True Positive Rate')
    plt.plot(thresholds, fpr, 'r-', label='False Positive Rate')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('TPR and FPR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(thresholds, precision, 'g-', label='Precision')
    plt.plot(thresholds, recall, 'b-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    plt.plot(thresholds, f1_scores, 'purple', label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    # Calculate Youden's J statistic
    j_scores = tpr - fpr
    plt.plot(thresholds, j_scores, 'orange', label="Youden's J")
    plt.xlabel('Threshold')
    plt.ylabel("Youden's J")
    plt.title("Youden's J Statistic vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'Threshold Analysis - {file_identifier}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'threshold_analysis_{file_identifier}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_summary(optimal_thresholds, f1_scores, output_dir):

    """Plot summary of optimal thresholds across all models."""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.hist(optimal_thresholds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Optimal Threshold')
    plt.ylabel('Frequency')
    plt.title('Distribution of Optimal Thresholds Across All Models')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.scatter(optimal_thresholds, f1_scores, alpha=0.7, color='green')
    plt.xlabel('Optimal Threshold')
    plt.ylabel('Optimized F1 Score')
    plt.title('Optimal Threshold vs F1 Score Performance')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_optimization_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_nan_cleanup_summary(result, source_info, project_root):
    """
    Save a detailed summary of the NaN column cleanup operation.
    
    Parameters:
    -----------
    result : dict
        Results from the drop_nan_columns operation
    source_info : str
        Information about the source data (file path or description)
    project_root : str
        Root directory of the project
    """
    try:
        # Determine output directory
        if result.get("output_path"):
            summary_dir = os.path.dirname(result["output_path"])
        else:
            summary_dir = os.path.join(project_root, MERGED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(summary_dir, exist_ok=True)
        
        # Generate summary filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f"nan_cleanup_summary_{timestamp}.txt"
        summary_path = os.path.join(summary_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write("NaN Column Cleanup Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: {source_info}\n")
            f.write(f"Output file: {os.path.basename(result.get('output_path', 'N/A'))}\n\n")
            
            f.write("Dataset Shape Changes:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Original shape: {result['original_shape']}\n")
            f.write(f"Final shape: {result['final_shape']}\n")
            f.write(f"Columns dropped: {result['columns_dropped']}\n")
            f.write(f"Columns kept: {result['columns_kept']}\n\n")
            
            if result['columns_dropped'] > 0:
                # Separate different types of drops
                specific_drops = result.get('specific_drops', [])
                nan_analysis = result.get('nan_analysis', {})
                
                # Columns dropped due to NaN values
                nan_dropped_cols = [col for col in result['dropped_column_names'] if col not in specific_drops]
                
                if nan_dropped_cols:
                    f.write("Columns Dropped (contained NaN values):\n")
                    f.write("-" * 40 + "\n")
                    for col in nan_dropped_cols:
                        stats = nan_analysis.get(col, {})
                        f.write(f"• {col}: {stats.get('nan_count', 'N/A')} NaN values "
                            f"({stats.get('nan_percentage', 'N/A')}%)\n")
                    f.write("\n")
                
                # Columns dropped for optimization/training purposes
                if specific_drops:
                    f.write("Columns Dropped (optimization/training purposes):\n")
                    f.write("-" * 45 + "\n")
                    for col in specific_drops:
                        if col == 'causes_related_to_weather':
                            f.write(f"• {col}: Not needed for training (used only for filtering)\n")
                        else:
                            f.write(f"• {col}: Training optimization\n")
                    f.write("\n")
            
            f.write("Preserved Columns (clean and needed for training):\n")
            f.write("-" * 50 + "\n")
            for i, col in enumerate(result['kept_column_names'], 1):
                f.write(f"{i:3d}. {col}\n")
            
            f.write(f"\nOperation Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            if not result['success']:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            
            # Add optimization notes
            f.write(f"\nOptimization Notes:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"• Total columns processed: {len(result.get('nan_analysis', {}))}\n")
            f.write(f"• Columns with NaN values removed: {result['columns_dropped'] - len(result.get('specific_drops', []))}\n")
            f.write(f"• Columns specifically removed for training: {len(result.get('specific_drops', []))}\n")
            f.write(f"• Final dataset ready for model training: {'Yes' if result['success'] else 'No'}\n")
        
        print(f"    drop_nan_columns: Summary saved to {summary_path}")
        
    except Exception as e:
        print(f"    drop_nan_columns: Warning - Could not save summary: {str(e)}")