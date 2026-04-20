# PNEUMA HGT Training Pipeline

This directory contains a self-supervised training pipeline for the **PNEUMA** dataset using a **Heterogeneous Graph Transformer (HGT)**. The model learns to represent urban traffic dynamics by predicting future vehicle states and segment metrics from historical graph snapshots.

## Pipeline Overview

1.  **Static Graph Construction**: Road segments, junction controllers, and their topological relationships (e.g., `turns_into`, `merges_into`) are loaded from GeoPackage and JSON files.
2.  **Dynamic Snapshot Building**: Chronological snapshots of the traffic state are built. Each snapshot includes:
    *   **Vehicle Nodes**: Current kinematic features (speed, acceleration, etc.).
    *   **Segment Nodes**: Static road features and historical performance metrics.
    *   **Temporal Memory**: History of vehicle and segment states stored as separate "memory" nodes connected by temporal edges.
3.  **HGT Encoding**: A multi-layer HGT processes the heterogeneous graph using grouped message passing.
4.  **Self-Supervised Prediction**:
    *   **Vehicle Head**: Predicts the next-step kinematic features for each vehicle.
    *   **Segment Head**: Predicts the next-step traffic speed (EMA) for road segments.

## Model Architecture

The model uses a custom HGT implementation (`hgt_model.py`) that supports:
- **Grouped Sequential Message Passing**: Message passing is performed in logical groups:
    1.  **Temporal Injection**: Memory nodes update current vehicle/segment states.
    2.  **Interaction**: Vehicle and segment nodes interact (e.g., vehicles "on" segments).
    3.  **Topology Propagation**: Information flows across the road network topology.
    4.  **Control Influence**: Junction controllers influence their approach segments.
- **Pass-through Identity**: Node types not receiving messages in a specific group maintain their state for the next step, ensuring stable multi-layer representations.
- **Relative Temporal Encoding**: Edge time deltas (e.g., time since last segment update) are encoded as Fourier features and injected into the attention mechanism.

## Data Structure

The `PNEUMA/data/` folder should contain episode subfolders (e.g., `20181029_dX_0800_0830/`). Each subfolder must include:
- `*_processed.csv`: Vectorized vehicle trajectory data.
- `downsampled_aggregated_states.csv`: Aggregated road segment metrics.
- `osm_network.gpkg`: Road network geometry and static attributes.
- `controllers.json`: Junction controller topology.
- `segment_thresholds.json`: Free-flow speed thresholds.

`topological_adjacency.json` is shared and located in the `PNEUMA/` root.

## Optimization Features

- **Pre-vectorization**: Dataframes are vectorized once during initialization to avoid redundant processing during training epochs.
- **Gradient Accumulation**: Gradients are accumulated over multiple snapshots to simulate larger batch sizes and improve stability.
- **Warm-up Phase**: Temporal memory (deques) is populated for a set number of steps before loss calculation begins for a chunk.

## Execution Commands

Ensure you are using the `pyg_env` conda environment.

### 1. Run Smoke Test
Verify the pipeline with example files at the root:
```bash
conda run -n pyg_env python -m PNEUMA.pneuma
```
*(Note: Set `SMOKE_TEST = True` in `PNEUMA/pneuma.py`'s `Config` class before running)*

### 2. Full Training
Ensure your data is structured in `PNEUMA/data/` as described above:
```bash
conda run -n pyg_env python -m PNEUMA.pneuma
```

### 3. Visualization
After training is complete, generate plots for your research paper:
```bash
conda run -n pyg_env python PNEUMA/visualize_results.py
```
This will create a `PNEUMA/plots/` folder with:
*   `loss_curves.png`: Training vs Validation MSE.
*   `mae_curves.png`: Breakdown of MAE for vehicles and segments.
*   `scatter_speed.png`: Density scatter plot of predicted vs actual segment speeds (with $R^2$).
*   `error_distribution.png`: Histogram of prediction residuals.

### 4. Feature Statistics
If you add new data, the pipeline will automatically recompute normalization statistics or load them from `PNEUMA/data/feature_stats.json`.

## Refactoring Notes
- `hgt_conv.py` was identified as redundant and removed.
- `HGTConv` in `hgt_model.py` was updated to fix a pass-through bug where non-destination node types were being dropped during multi-group message passing.
- `tqdm` progress bars are disabled by default in the training scripts to prevent buffering issues in some CLI environments.
