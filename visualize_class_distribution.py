
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
import sys

# Add project root to path to allow imports from vas
sys.path.append(str(Path(__file__).parent))

from vas.config import Config
from vas.dataset import load_sessions, quantize_vas, VasGroup

def main():
    parser = argparse.ArgumentParser(description="Visualize class distribution for VAS dataset")
    parser.add_argument("--num-classes", type=int, default=Config().num_classes, help="Number of classes to divide the data into")
    parser.add_argument("--index-path", type=str, default=Config().index_path, help="Path to index.jsonl")
    parser.add_argument("--output", type=str, default="distribution.png", help="Path to save the output plot")
    
    args = parser.parse_args()

    print(f"Loading sessions from {args.index_path}...", flush=True)
    try:
        sessions = load_sessions(args.index_path)
    except FileNotFoundError:
        print(f"Error: Index file not found at {args.index_path}")
        return

    print(f"Calculating distribution for {args.num_classes} classes...")
    
    # Logic to count samples per class
    # We closely mimic how dataset.py constructs samples to get accurate counts
    # dataset.py:
    # 1. Iterate sessions
    # 2. Iterate vas_groups -> quantize_vas -> Label
    # 3. Add VasGroup with VAS=0 (vas0) -> Label 0
    # Note: dataset.py multiplies by samples_per_group. We probably just want to count the *groups* (files/segments) available,
    # or arguably the "potential samples". The prompt asks "how much training data is distributed".
    # Usually, counting the number of source groups (videos/segments) assigned to each class is the most informative
    # fundamental distribution. If we just multiply by a constant, the ratio is the same.
    # HOWEVER, dataset.py creates a "vas0" group virtually if normal_target_frames exist. We must count that.

    class_counts = Counter()
    
    # Config values relevant for "vas0" logic
    min_frames = Config().min_frames_per_window

    for session_id, session in sessions.items():
        # Check filters used in dataset.py
        if not session.anchor_frames:
            continue
        if not session.vas_groups and not session.normal_target_frames:
            continue
            
        # 1. Count existing VAS groups
        for group in session.vas_groups.values():
            label = quantize_vas(group.vas_value, args.num_classes)
            class_counts[label] += 1
            
        # 2. Count virtual VAS=0 group if applicable
        if len(session.normal_target_frames) >= min_frames:
             # Label 0 is always 0
             class_counts[0] += 1

    # Prepare data for plotting
    labels = sorted(class_counts.keys())
    counts = [class_counts[l] for l in labels]
    
    # Fill in missing classes with 0 if any
    all_classes = range(args.num_classes)
    final_counts = []
    for c in all_classes:
        final_counts.append(class_counts.get(c, 0))

    print("Counts per class:")
    for c, count in zip(all_classes, final_counts):
        print(f"  Class {c}: {count}")

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(all_classes, final_counts, color='skyblue', edgecolor='black')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Data Groups (Segments)')
    plt.title(f'Data Distribution across {args.num_classes} Classes')
    plt.xticks(all_classes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
