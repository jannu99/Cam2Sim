import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------- CONFIG --------------

# Metrics where *higher* is better.
HIGHER_IS_BETTER = {
    "PSNR", "SSIM", "SegScore",
    "Veh_Recall", "Veh_Precision", "Veh_AvgIoU",
    "IS_mean", "PRDC_Precision", "PRDC_Recall", 
    "PRDC_Density", "PRDC_Coverage",
}

# Metrics where *lower* is better.
LOWER_IS_BETTER = {
    "MSE", "FID", "KID_mean", "MMD_RBF", "CPL",
}

# Define which metrics belong to which category
METRIC_GROUPS = {
    "Score_Vehicle": {
        "Veh_Recall", "Veh_Precision", "Veh_AvgIoU"
    },
    "Score_Distribution": {
        "FID", "KID_mean", "IS_mean", "MMD_RBF", 
        "PRDC_Precision", "PRDC_Recall", "PRDC_Density", "PRDC_Coverage"
    },
    "Score_SingleImage": {
        "PSNR", "SSIM", "MSE", "SegScore", "CPL"
    }
}

def load_reports(folder):
    """Load all JSON report files from a folder."""
    systems = {}
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return {}

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".json"):
            continue
        
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not read {fpath}: {e}")
            continue
        
        # Use filename (without extension) as system name
        system_name = os.path.splitext(fname)[0]
        avg_metrics = data.get("average_metrics")
        
        if avg_metrics is None:
            print(f"Warning: no 'average_metrics' in {fpath}, skipping.")
            continue
        
        systems[system_name] = avg_metrics
    
    return systems

def build_dataframe(systems_dict):
    """Build a pandas DataFrame (systems x metrics)."""
    if not systems_dict:
        raise ValueError("No systems were loaded.")
    
    df = pd.DataFrame.from_dict(systems_dict, orient="index")
    df.index.name = "system"
    return df

def compute_scores(df):
    """
    Computes Z-scores for all metrics, flips signs for 'lower is better',
    and calculates grouped scores.
    """
    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=["number"]).copy()
    if df_numeric.empty:
        raise ValueError("No numeric metrics found.")
    
    # 1. Standardize columns (Z-score)
    # (x - mean) / std
    z_df = df_numeric.apply(lambda col: (col - col.mean()) / col.std(ddof=0) if col.std(ddof=0) != 0 else 0)
    
    # 2. Flip sign where lower is better
    for col in df_numeric.columns:
        if col in LOWER_IS_BETTER:
            z_df[col] = -z_df[col]

    # 3. Compute Category Scores
    results = df.copy()
    
    # Helper to calculate mean of available columns for a group
    for group_name, metric_set in METRIC_GROUPS.items():
        # Find which metrics from this group actually exist in the dataframe
        valid_cols = [c for c in z_df.columns if c in metric_set]
        
        if valid_cols:
            results[group_name] = z_df[valid_cols].mean(axis=1)
        else:
            results[group_name] = np.nan # Or 0 if you prefer

    # 4. Compute Overall Score
    # We take the mean of ALL numeric z-scores (not just the groups, to be inclusive)
    results["Score_Overall"] = z_df.mean(axis=1)
    
    return results

def main(folder, show_plot=True, save_csv=None):
    # 1. Load reports
    systems = load_reports(folder)
    if not systems:
        print("No valid systems found. Exiting.")
        return

    # 2. Build DataFrame
    df = build_dataframe(systems)

    # 3. Compute scores
    df_scored = compute_scores(df)

    # 4. Sort by Overall Score
    df_sorted = df_scored.sort_values("Score_Overall", ascending=False)

    # 5. Print overview
    print("\n=== Systems ranked by Score_Overall ===")
    print(df_sorted[["Score_Overall", "Score_SingleImage", "Score_Distribution", "Score_Vehicle"]])
    
    print(f"\nBest System: {df_sorted.index[0]}")

    # 6. Save to CSV
    if save_csv is not None:
        df_sorted.to_csv(save_csv)
        print(f"\nFull ranking saved to: {save_csv}")

    # 7. Plotting
    if show_plot:
        # We will plot the 3 categories stacked to show contribution
        # Note: Z-scores can be negative. Stacked bars with negatives can be visually tricky,
        # but it gives a good sense of relative strengths.
        
        plot_cols = ["Score_SingleImage", "Score_Distribution", "Score_Vehicle"]
        # Filter out columns that might be all NaN if data was missing
        plot_cols = [c for c in plot_cols if c in df_sorted.columns and not df_sorted[c].isna().all()]

        if not plot_cols:
            print("No category scores available to plot.")
            return

        # Dynamic figure size based on number of systems
        width = max(10, len(df_sorted) * 0.8)
        fig, ax = plt.subplots(figsize=(width, 7))
        
        df_sorted[plot_cols].plot(kind="bar", ax=ax, width=0.7)
        
        # --- FIX FOR READABILITY ---
        # Rotate labels 45 degrees and align them to the right
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        # Add a horizontal line at 0 for reference
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.title("Model Comparison: Normalized Scores (Z-Score)", fontsize=14)
        plt.ylabel("Standardized Score (Higher is better)", fontsize=12)
        plt.legend(title="Metric Category")
        
        # Adjust layout to make room for the rotated labels
        plt.subplots_adjust(bottom=0.25)
        
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model report JSON files.")
    parser.add_argument("folder", help="Folder containing JSON report files")
    parser.add_argument("--no-plot", action="store_true", help="Do not show a matplotlib plot")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save results as CSV")
    
    args = parser.parse_args()
    main(args.folder, show_plot=not args.no_plot, save_csv=args.csv)