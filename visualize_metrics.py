
import argparse
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize metrics from cv_results.csv")
    parser.add_argument("target_dir", type=str, help="Directory containing cv_results.csv")
    parser.add_argument("--output", type=str, default="metrics_analysis.png", help="Output filename for the plot")
    return parser.parse_args()

def calculate_mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

def main():
    args = parse_args()
    target_dir = Path(args.target_dir)
    csv_path = target_dir / "cv_results.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        sys.exit(1)

    print(f"Reading {csv_path}...", flush=True)

    # Data storage: metrics[metric_name] = [val_fold1, val_fold2, ...]
    metrics_data = defaultdict(list)
    
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if val and val.strip():
                    try:
                        metrics_data[key].append(float(val))
                    except ValueError:
                        pass
    
    if not metrics_data:
        print("No valid data found in CSV.")
        sys.exit(1)

    # Identify classes and metrics
    class_indices = set()
    for key in metrics_data.keys():
        match = re.match(r"class(\d+)_", key)
        if match:
            class_indices.add(int(match.group(1)))
    
    sorted_classes = sorted(list(class_indices))
    print(f"Found classes: {sorted_classes}", flush=True)

    # Calculate statistics
    stats = {}
    for key, values in metrics_data.items():
        mean, std = calculate_mean_std(values)
        stats[key] = (mean, std)
        print(f"{key}: {mean:.4f} ± {std:.4f}")

    # Prepare for plotting
    # We want grouped bars: Precision, Recall, F1 for each class
    metric_types = ["precision", "recall", "f1"]
    
    means = {m: [] for m in metric_types}
    stds = {m: [] for m in metric_types}
    labels = []

    for c in sorted_classes:
        labels.append(f"Class {c}")
        for m in metric_types:
            key = f"class{c}_{m}"
            if key in stats:
                means[m].append(stats[key][0])
                stds[m].append(stats[key][1])
            else:
                means[m].append(0.0)
                stds[m].append(0.0)

    # Plotting
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, means["precision"], width, yerr=stds["precision"], label='Precision', capsize=5)
    rects2 = ax.bar(x, means["recall"], width, yerr=stds["recall"], label='Recall', capsize=5)
    rects3 = ax.bar(x + width, means["f1"], width, yerr=stds["f1"], label='F1-Score', capsize=5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Metrics by Class (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)  # slightly above 1 for error bars
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Function to add labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    output_path = target_dir / args.output
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()
