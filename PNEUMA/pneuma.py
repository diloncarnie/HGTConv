"""
pneuma.py
=========
Self-supervised pre-training of the PNEUMA HGT model.

Usage (smoke test with example CSVs):
    cd C:/Users/dilon/Desktop/HGTConv
    conda run -n pyg_env python -m PNEUMA.pneuma

Usage (full training with episode folders under PNEUMA/data/):
    Set SMOKE_TEST = False and ensure PNEUMA/data/ contains the episode subfolders,
    each with its own osm_network.gpkg, segment_thresholds.json, controllers.json,
    <foldername>_processed.csv, and downsampled_aggregated_states.csv.
    Place topological_adjacency.json at PNEUMA/topological_adjacency.json.

Training strategy
-----------------
- Each episode folder is treated as an independent episode (memory resets between files).
- Within each episode, snapshots are iterated chronologically.
- Gradients are accumulated over GRAD_ACCUM_STEPS snapshots before each optimizer step.
- An epoch = one full pass through all training episodes (in shuffled order).
- Best model is saved by lowest validation loss.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
import torch
import torch_geometric

# Ensure PNEUMA package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PNEUMA.graph_builder import (
    PneumaStaticGraph,
    precompute_feature_stats,
    scan_segment_stats,
)
from PNEUMA.pneuma_dataset import PneumaDataModule, PneumaEpisode
from PNEUMA.pneuma_model import PneumaModel, PNEUMA_CUSTOM_ORDER

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE_TEST = False

ROOT_DIR = Path(__file__).resolve().parent.parent   # HGTConv repo root
PNEUMA_DIR = Path(__file__).resolve().parent        # PNEUMA/ package directory

# --- Full-training paths ---
# Each episode subfolder under DATA_DIR must contain:
#   <foldername>_processed.csv, downsampled_aggregated_states.csv,
#   osm_network.gpkg, segment_thresholds.json, controllers.json
DATA_DIR      = str(PNEUMA_DIR / "data")
TOPOLOGY_PATH = str(PNEUMA_DIR / "topological_adjacency.json")   # shared across episodes

# Feature-normalisation stats cache (written once, reused thereafter)
STATS_PATH    = str(PNEUMA_DIR / "data" / "feature_stats.json")

# --- Smoke-test paths (root-level example files) ---
# These are used when SMOKE_TEST = True and do not require the episode folder layout.
SMOKE_TOPOLOGY_PATH    = str(ROOT_DIR / "topological_adjacency.json")
SMOKE_GPKG_PATH        = str(ROOT_DIR / "osm_network.gpkg")
SMOKE_CONTROLLERS_PATH = str(ROOT_DIR / "controllers.json")
SMOKE_THRESHOLDS_PATH  = str(ROOT_DIR / "segment_thresholds.json")
SMOKE_PROC_PATH        = str(ROOT_DIR / "example_processed.csv")
SMOKE_AGG_PATH         = str(ROOT_DIR / "example_aggregated_states.csv")

CHECKPOINT_DIR = str(PNEUMA_DIR / "checkpoints")

# Model hyperparameters
HIDDEN_CHANNELS = 64
NUM_HEADS       = 2
NUM_LAYERS      = 1
DROPOUT         = 0.4
LAMBDA_V        = 1.0
LAMBDA_S        = 0.3

# Training hyperparameters
EPOCHS           = 50
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
GRAD_ACCUM_STEPS = 32   # accumulate gradients over this many snapshots
TRAIN_FRAC       = 0.8  # set < 1.0 (e.g. 0.8) when you have multiple episodes
SEED             = 42

# Chunk sampling hyperparameters
CHUNK_LENGTH     = 30  # Number of consecutive snapshots to predict in a row
CHUNKS_PER_EPOCH = 10   # Number of random sequences sampled per episode per epoch

# Sampling hyperparameters
USE_HGT_SAMPLING = False  # Set to True to enable budget-based HGT subgraph sampling
HGT_HOPS         = 2



# ---------------------------------------------------------------------------
# Train / val epoch functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: PneumaModel,
    episodes: List[PneumaEpisode],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
) -> Dict[str, float]:
    """
    Run one training pass over a list of PneumaEpisode objects.
    Gradients are accumulated over ``grad_accum_steps`` snapshots before each step.

    Returns averaged metrics across all snapshots.
    """
    model.train()

    total_loss   = 0.0
    total_v_loss = 0.0
    total_s_loss = 0.0
    n_snapshots  = 0
    accum_count  = 0

    optimizer.zero_grad()

    # Shuffle episode order within epoch
    ep_order = list(episodes)
    random.shuffle(ep_order)

    for episode in ep_order:
        pbar = tqdm(episode, desc="Training Snaps", leave=False)
        for snapshot, targets in pbar:
            snapshot = snapshot.to(device)

            preds = model(snapshot)
            loss, metrics = model.compute_loss(preds, targets)

            # Scale loss by accumulation factor so gradients average out.
            # Skip if no gradient targets exist this step (empty masks).
            if loss.requires_grad:
                (loss / grad_accum_steps).backward()
                accum_count += 1

            total_loss   += float(loss)
            total_v_loss += metrics['vehicle_loss']
            total_s_loss += metrics['segment_loss']
            n_snapshots  += 1

            if accum_count >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

    # Flush any remaining accumulated gradients
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    n = max(n_snapshots, 1)
    return {
        'loss':         total_loss  / n,
        'vehicle_loss': total_v_loss / n,
        'segment_loss': total_s_loss / n,
        'n_snapshots':  n_snapshots,
    }


@torch.no_grad()
def val_epoch(
    model: PneumaModel,
    episodes: List[PneumaEpisode],
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation pass. No gradient accumulation."""
    model.eval()

    total_loss   = 0.0
    total_v_loss = 0.0
    total_s_loss = 0.0
    n_snapshots  = 0

    for episode in episodes:
        pbar = tqdm(episode, desc="Validation Snaps", leave=False)
        for snapshot, targets in pbar:
            snapshot = snapshot.to(device)
            preds = model(snapshot)
            loss, metrics = model.compute_loss(preds, targets)

            total_loss   += float(loss)
            total_v_loss += metrics['vehicle_loss']
            total_s_loss += metrics['segment_loss']
            n_snapshots  += 1

    n = max(n_snapshots, 1)
    return {
        'loss':         total_loss  / n,
        'vehicle_loss': total_v_loss / n,
        'segment_loss': total_s_loss / n,
        'n_snapshots':  n_snapshots,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch_geometric.is_xpu_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Build episodes
    # -----------------------------------------------------------------------

    if SMOKE_TEST:
        # ---- Smoke test: use root-level example files ----
        print("SMOKE TEST mode: using root-level example files as a single episode.")
        print("Building static graph...")
        t0 = time.time()
        csv_stats = scan_segment_stats([SMOKE_PROC_PATH])
        static_graph = PneumaStaticGraph(
            SMOKE_TOPOLOGY_PATH,
            SMOKE_GPKG_PATH,
            SMOKE_CONTROLLERS_PATH,
            csv_stats,
        )
        print(f"  Built in {time.time() - t0:.1f}s | "
              f"{static_graph.num_segments} segs, {static_graph.num_controllers} controllers")

        # No feature normalisation in smoke test
        train_episodes = [PneumaEpisode(SMOKE_PROC_PATH, SMOKE_AGG_PATH, static_graph)]
        val_episodes   = [PneumaEpisode(SMOKE_PROC_PATH, SMOKE_AGG_PATH, static_graph)]

    else:
        # ---- Full training: per-episode folder layout ----
        print(f"Full training mode: scanning episode folders in {DATA_DIR}")

        # 1. Discover episodes and split into train/val
        data_module = PneumaDataModule(
            DATA_DIR, TOPOLOGY_PATH,
            train_frac=TRAIN_FRAC, seed=SEED,
            chunk_length=CHUNK_LENGTH,
            chunks_per_epoch=CHUNKS_PER_EPOCH,
            use_hgt_sampling=USE_HGT_SAMPLING,
            hgt_hops=HGT_HOPS,
        )

        # 2. Pre-scan all episodes for segment static metadata (csv_stats)
        print("Pre-scanning episode CSVs for segment stats...")
        t0 = time.time()
        all_proc = data_module.get_all_proc_paths()
        data_module.csv_stats = scan_segment_stats(all_proc)
        print(f"  Done in {time.time() - t0:.1f}s ({len(data_module.csv_stats)} segments)")

        # 3. Compute (or load cached) feature normalisation statistics
        stats_cache = Path(STATS_PATH)
        if stats_cache.exists():
            with open(stats_cache) as f:
                feature_stats = json.load(f)
            print(f"Loaded feature stats from {stats_cache}")
        else:
            train_proc, train_agg = data_module.get_train_proc_agg_paths()
            print(f"Computing feature stats from {len(train_proc)} training episodes...")
            t0 = time.time()
            feature_stats = precompute_feature_stats(train_proc, train_agg)
            stats_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_cache, 'w') as f:
                json.dump(feature_stats, f, indent=2)
            print(f"  Done in {time.time() - t0:.1f}s — saved to {stats_cache}")
        data_module.feature_stats = feature_stats

        # 4. Materialise episode objects (builds per-episode static graphs)
        print("Building per-episode static graphs and loading episodes...")
        t0 = time.time()
        train_episodes = list(data_module.iter_train())
        val_episodes   = list(data_module.iter_val())
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"Episodes: {len(train_episodes)} train, {len(val_episodes)} val")

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    print("Building model...")
    model = PneumaModel(
        hidden_channels=HIDDEN_CHANNELS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lambda_v=LAMBDA_V,
        lambda_s=LAMBDA_S,
    ).to(device)

    print(f"  Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    best_epoch    = -1

    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        t_epoch = time.time()

        train_metrics = train_epoch(model, train_episodes, optimizer, device)
        val_metrics   = val_epoch(model, val_episodes, device)

        elapsed = time.time() - t_epoch
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"(v={train_metrics['vehicle_loss']:.4f}, s={train_metrics['segment_loss']:.4f}) | "
            f"val loss={val_metrics['loss']:.4f} "
            f"(v={val_metrics['vehicle_loss']:.4f}, s={val_metrics['segment_loss']:.4f}) | "
            f"snaps={train_metrics['n_snapshots']} | "
            f"t={elapsed:.1f}s"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch    = epoch
            ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss,
            }, ckpt_path)
            print(f"  -> Saved best model to {ckpt_path}")

    print(f"\nTraining complete. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == '__main__':
    main()
