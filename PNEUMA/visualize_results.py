"""
visualize_results.py
====================
Generates publication-quality plots from PNEUMA HGT training logs.
Requires: matplotlib, seaborn, pandas, numpy
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Set plot style for research papers
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

def plot_training_curves(history: list):
    """Plots training and validation loss/MAE over epochs."""
    epochs = [h['epoch'] for h in history]
    
    # Loss curves
    plt.figure(figsize=(10, 6))
    train_loss = [h['train']['loss'] for h in history]
    val_loss = [h['val']['loss'] for h in history]
    plt.plot(epochs, train_loss, label='Train Total Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Val Total Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=300)
    plt.close()

    # Vehicle MAE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h['train']['vehicle_mae'] for h in history], label='Train (Vehicle)', linewidth=2)
    plt.plot(epochs, [h['val']['vehicle_mae'] for h in history], label='Val (Vehicle)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Vehicle Kinematics Prediction Error (MAE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mae_vehicle.png", dpi=300)
    plt.close()

    # Segment MAE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h['train']['segment_mae'] for h in history], label='Train (Segment)', linewidth=2)
    plt.plot(epochs, [h['val']['segment_mae'] for h in history], label='Val (Segment)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Segment Speed Prediction Error (MAE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mae_segment.png", dpi=300)
    plt.close()

def plot_scatter_and_residuals(val_results: dict):
    """Plots Predicted vs Actual scatter and error distribution."""
    actual = np.array(val_results['actual'])
    predicted = np.array(val_results['predicted'])
    
    # Filter out zeros/extreme outliers for cleaner visualization if needed
    mask = (actual > 0.05) & (actual < 2.0)
    actual = actual[mask]
    predicted = predicted[mask]

    r2 = r2_score(actual, predicted)
    mae = np.mean(np.abs(actual - predicted))

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    # Hexbin for density as we have many snapshots
    plt.hexbin(actual, predicted, gridsize=50, cmap='YlGnBu', mincnt=1)
    plt.colorbar(label='Count')
    
    # 45-degree line
    lims = [0, max(actual.max(), predicted.max())]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=3, label='Perfect Prediction')
    
    plt.xlabel('Actual Normalised Speed')
    # plt.ylabel('Predicted Normalised Speed')
    plt.ylabel('Predicted Normalised Speed')
    plt.title(f'Segment Speed: Predicted vs Actual\n$R^2 = {r2:.3f}$, MAE = {mae:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_speed.png", dpi=300)
    plt.close()

    # Residual Distribution
    plt.figure(figsize=(10, 6))
    residuals = actual - predicted
    sns.histplot(residuals, kde=True, color='teal')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Speed Prediction Errors')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_distribution.png", dpi=300)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load history
    history_path = CHECKPOINT_DIR / "epoch_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print(f"Plotting training curves from {history_path}...")
        plot_training_curves(history)
    else:
        print(f"Warning: {history_path} not found.")

    # Load validation results
    results_path = CHECKPOINT_DIR / "val_results.json"
    if results_path.exists():
        with open(results_path) as f:
            val_results = json.load(f)
        print(f"Plotting evaluation charts from {results_path}...")
        plot_scatter_and_residuals(val_results)
    else:
        print(f"Warning: {results_path} not found.")

    print(f"All plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
