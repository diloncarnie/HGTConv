# PNEUMA HGT: Heterogeneous Graph Transformer for Traffic Prediction

This repository implements a Heterogeneous Graph Transformer (HGT) model for traffic prediction based on the PNEUMA dataset. The pipeline models both static road topologies and dynamic vehicle trajectories as a complex heterogeneous graph, predicting future vehicle kinematics and road segment speeds.

All the valid scripts for the data pipeline, model definition, and training are contained within the `PNEUMA/` folder.

## Key Features

- **Heterogeneous Graph Formulation**: Models vehicles, road segments, controllers, and temporal memories as distinct node types.
- **Dynamic Snapshots**: Builds time-evolving graph snapshots from trajectory and aggregated state CSVs.
- **Self-Supervised Pre-Training**: Dual prediction heads for next-step vehicle kinematic features and segment temporal speeds.
- **Chunked Episode Training**: Handles large temporal datasets by chunking sequences to manage memory limits efficiently.

## Core Modules (`PNEUMA/`)

- **`pneuma.py`**: The main entry point for training. Runs self-supervised pre-training of the HGT model. Supports a `SMOKE_TEST` mode for quick verification on example CSVs, and a full training mode over multiple episode folders.
- **`hgt_model.py`**: Contains the core PyTorch Geometric implementation of the Heterogeneous Graph Transformer (HGTConv) operator.
- **`pneuma_model.py`**: Wraps the base HGT encoder with the self-supervised prediction heads (`vehicle_head` and `segment_head`) and defines the heterogeneous graph metadata constraints.
- **`graph_builder.py`**: Responsible for static and dynamic graph construction. Pre-processes road segments and controller features, and incrementally builds PyG `HeteroData` snapshots at each trajectory timestamp.
- **`pneuma_dataset.py`**: Implements data loading and chronologically iterates through episodes (processed CSV + aggregated states pairs) using a PyTorch `IterableDataset`.
- **`visualize_results.py`**: Utility script to parse training logs (`epoch_history.json` and `val_results.json`) and generate publication-quality evaluation plots (loss curves, MAE, error distributions).

## Data Structure

For full training, the pipeline expects episode folders inside `PNEUMA/data/`, where each folder contains:
- `osm_network.gpkg`
- `segment_thresholds.json`
- `controllers.json`
- `<foldername>_processed.csv`
- `downsampled_aggregated_states.csv`

A shared `topological_adjacency.json` is expected at `PNEUMA/topological_adjacency.json`.

## Usage

**Smoke Test (using example data):**
```bash
cd /path/to/repo
python -m PNEUMA.pneuma
```
*(Ensure `Config.SMOKE_TEST = True` is set in `pneuma.py`)*

**Full Training:**
1. Place your dataset episodes into `PNEUMA/data/`.
2. Set `Config.SMOKE_TEST = False` in `pneuma.py`.
3. Run the training pipeline:
```bash
python -m PNEUMA.pneuma
```

**Visualizations:**
After training, generate performance plots:
```bash
python -m PNEUMA.visualize_results
```
Plots will be saved to the `PNEUMA/plots/` directory.