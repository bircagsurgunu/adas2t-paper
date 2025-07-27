# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def create_plot(csv_path: str, metric: str, output_path: str):
    """
    Generates and saves a bar plot from the benchmark results CSV.

    Args:
        csv_path (str): Path to the input CSV file.
        metric (str): The metric to plot ('WER' or 'CER').
        output_path (str): Path to save the output plot image.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Combine dataset and split for a unique x-axis label
    df["dataset_label"] = df["dataset"] + " (" + df["split"] + ")"

    # Pivot the data for plotting
    try:
        pivot_df = df.pivot(index="dataset_label", columns="model", values=metric)
    except KeyError:
        print(f"Error: Metric '{metric}' not found in the CSV file. Available columns: {df.columns.tolist()}")
        return

    # Create the plot
    ax = pivot_df.plot(kind="bar", figsize=(15, 8), width=0.8)
    
    plt.title(f"ASR Model Comparison: {metric}", fontsize=16)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # Use argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Visualize ASR benchmark results.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="asr_benchmark_results.csv",
        help="Path to the results CSV file."
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default="WER",
        choices=["WER", "CER"],
        help="Metric to plot (WER or CER)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="asr_benchmark_plot.png",
        help="Path to save the output plot."
    )
    
    args = parser.parse_args()
    create_plot(args.input, args.metric, args.output)