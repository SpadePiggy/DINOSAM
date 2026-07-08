#!/usr/bin/env python3
"""
Analyze layer preferences from trained model checkpoints.

Usage:
    python scripts/analyze_layer_preferences.py \
        --phase1_checkpoint outputs/phase1_best.pt \
        --phase2_checkpoint outputs/phase2_best.pt \
        --output_dir outputs/analysis
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_model_weights(checkpoint_path, device="cpu"):
    """Load model and extract fusion weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Extract fusion weights
    weights = {}
    for key, value in state_dict.items():
        if "mask_fusion.layer_weights" in key:
            weights["mask"] = value.cpu().numpy()
        elif "depth_fusion.layer_weights" in key:
            weights["depth"] = value.cpu().numpy()

    return weights


def plot_layer_preferences(mask_weights, depth_weights, output_dir):
    """Generate comparison plots of layer preferences."""
    os.makedirs(output_dir, exist_ok=True)

    n_layers = len(mask_weights)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot mask fusion weights
    ax1.bar(range(n_layers), mask_weights, alpha=0.7, label="Mask")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Weight")
    ax1.set_title("Mask Fusion Layer Weights")
    ax1.set_xticks(range(n_layers))
    ax1.grid(True, alpha=0.3)

    # Plot depth fusion weights
    ax2.bar(range(n_layers), depth_weights, alpha=0.7, label="Depth", color="orange")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Weight")
    ax2.set_title("Depth Fusion Layer Weights")
    ax2.set_xticks(range(n_layers))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_weights_comparison.png"), dpi=150)
    plt.close()

    # Create overlay plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_layers)
    width = 0.35

    ax.bar(x - width/2, mask_weights, width, label="Mask", alpha=0.7)
    ax.bar(x + width/2, depth_weights, width, label="Depth", alpha=0.7)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Weight")
    ax.set_title("Layer Weight Comparison: Mask vs Depth")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_weights_overlay.png"), dpi=150)
    plt.close()

    # Compute statistics
    mask_topk = np.argsort(mask_weights)[::-1][:4]
    depth_topk = np.argsort(depth_weights)[::-1][:4]

    overlap = len(set(mask_topk) & set(depth_topk))

    print(f"\n=== Layer Preference Analysis ===")
    print(f"Mask top-4 layers: {mask_topk}")
    print(f"Depth top-4 layers: {depth_topk}")
    print(f"Overlap: {overlap}/4 layers")
    print(f"Mask weights: {mask_weights}")
    print(f"Depth weights: {depth_weights}")

    # Save statistics to file
    with open(os.path.join(output_dir, "statistics.txt"), "w") as f:
        f.write("Layer Preference Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Mask top-4 layers: {mask_topk}\n")
        f.write(f"Depth top-4 layers: {depth_topk}\n")
        f.write(f"Overlap: {overlap}/4 layers\n\n")
        f.write(f"Mask weights: {mask_weights}\n")
        f.write(f"Depth weights: {depth_weights}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze layer preferences")
    parser.add_argument("--phase1_checkpoint", type=str, required=True,
                        help="Path to phase1 checkpoint")
    parser.add_argument("--phase2_checkpoint", type=str, required=True,
                        help="Path to phase2 checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis",
                        help="Output directory for analysis")
    args = parser.parse_args()

    print(f"Loading phase 1 checkpoint: {args.phase1_checkpoint}")
    phase1_weights = load_model_weights(args.phase1_checkpoint)

    print(f"Loading phase 2 checkpoint: {args.phase2_checkpoint}")
    phase2_weights = load_model_weights(args.phase2_checkpoint)

    # Use phase 1 for mask weights, phase 2 for depth weights
    mask_weights = phase1_weights.get("mask")
    depth_weights = phase2_weights.get("depth")

    if mask_weights is None:
        print("Warning: No mask fusion weights found in phase 1 checkpoint")
        return

    if depth_weights is None:
        print("Warning: No depth fusion weights found in phase 2 checkpoint")
        return

    plot_layer_preferences(mask_weights, depth_weights, args.output_dir)

    print(f"\nAnalysis complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
