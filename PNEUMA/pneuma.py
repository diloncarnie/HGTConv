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
import logging
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

class Config:
    SMOKE_TEST = False

    ROOT_DIR = Path(__file__).resolve().parent.parent   # HGTConv repo root
    PNEUMA_DIR = Path(__file__).resolve().parent        # PNEUMA/ package directory

    # --- Full-training paths ---
    DATA_DIR      = PNEUMA_DIR / "data"
    TOPOLOGY_PATH = PNEUMA_DIR / "topological_adjacency.json"   # shared across episodes
    STATS_PATH    = PNEUMA_DIR / "data" / "feature_stats.json"

    # --- Smoke-test paths ---
    SMOKE_TOPOLOGY_PATH    = PNEUMA_DIR / "topological_adjacency.json"
    SMOKE_GPKG_PATH        = ROOT_DIR / "osm_network.gpkg"
    SMOKE_CONTROLLERS_PATH = ROOT_DIR / "controllers.json"
    SMOKE_THRESHOLDS_PATH  = ROOT_DIR / "segment_thresholds.json"
    SMOKE_PROC_PATH        = ROOT_DIR / "example_processed.csv"
    SMOKE_AGG_PATH         = ROOT_DIR / "example_aggregated_states.csv"

    CHECKPOINT_DIR = PNEUMA_DIR / "checkpoints"

    # Model hyperparameters
    HIDDEN_CHANNELS = 64
    NUM_HEADS       = 2
    NUM_LAYERS      = 2  # Increased default for HGT
    DROPOUT         = 0.4
    LAMBDA_V        = 1.0
    LAMBDA_S        = 0.3

    # Training hyperparameters
    EPOCHS           = 50
    LR               = 1e-3
    WEIGHT_DECAY     = 1e-4
    GRAD_ACCUM_STEPS = 32
    TRAIN_FRAC       = 0.8
    SEED             = 42

    # Chunk sampling hyperparameters
    CHUNK_LENGTH     = 15
    CHUNKS_PER_EPOCH = 3
    WARMUP_STEPS     = 0

    # Sampling hyperparameters
    USE_HGT_SAMPLING = False
    HGT_HOPS         = 2

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ---------------------------------------------------------------------------
# Train / val epoch functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: PneumaModel,
    episodes: List[PneumaEpisode],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = Config.GRAD_ACCUM_STEPS,
) -> Dict[str, float]:
    """
    Run one training pass. Returns averaged metrics across all snapshots.
    """
    model.train()

    totals = {
        'loss': 0.0, 'vehicle_loss': 0.0, 'vehicle_mae': 0.0, 'vehicle_rmse': 0.0,
        'segment_loss': 0.0, 'segment_mae': 0.0, 'segment_rmse': 0.0
    }
    n_snapshots  = 0
    accum_count  = 0

    optimizer.zero_grad()

    ep_order = list(episodes)
    random.shuffle(ep_order)

    for episode in ep_order:
        for snapshot, targets in episode:
            snapshot = snapshot.to(device)

            preds = model(snapshot)
            loss, metrics = model.compute_loss(preds, targets)

            if loss.requires_grad:
                (loss / grad_accum_steps).backward()
                accum_count += 1

            for k in totals:
                if k == 'loss':
                    totals[k] += float(loss)
                else:
                    totals[k] += metrics[k]
            n_snapshots  += 1

            if accum_count >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    n = max(n_snapshots, 1)
    res = {k: v / n for k, v in totals.items()}
    res['n_snapshots'] = n_snapshots
    return res


@torch.no_grad()
def val_epoch(
    model: PneumaModel,
    episodes: List[PneumaEpisode],
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation pass."""
    model.eval()

    totals = {
        'loss': 0.0, 'vehicle_loss': 0.0, 'vehicle_mae': 0.0, 'vehicle_rmse': 0.0,
        'segment_loss': 0.0, 'segment_mae': 0.0, 'segment_rmse': 0.0
    }
    n_snapshots  = 0

    for episode in episodes:
        for snapshot, targets in episode:
            snapshot = snapshot.to(device)
            preds = model(snapshot)
            loss, metrics = model.compute_loss(preds, targets)

            totals['loss'] += float(loss)
            for k in metrics:
                totals[k] += metrics[k]
            n_snapshots  += 1

    n = max(n_snapshots, 1)
    res = {k: v / n for k, v in totals.items()}
    res['n_snapshots'] = n_snapshots
    return res


@torch.no_grad()
def evaluate_model(
    model: PneumaModel,
    episodes: List[PneumaEpisode],
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Detailed evaluation to collect Actual vs Predicted values for segments.
    Used for scatter plots and error distribution.
    """
    model.eval()
    actuals = []
    preds_list = []

    for episode in episodes:
        for snapshot, targets in episode:
            snapshot = snapshot.to(device)
            preds = model(snapshot)
            
            if 'segment' in preds:
                seg_idx = targets['segment_mask_idx'].to(device)
                if seg_idx.shape[0] > 0:
                    s_pred = preds['segment'][seg_idx].view(-1).cpu().tolist()
                    s_tgt = targets['segment_feats'].to(device).view(-1).cpu().tolist()
                    actuals.extend(s_tgt)
                    preds_list.extend(s_pred)

    return {'actual': actuals, 'predicted': preds_list}

def main() -> None:
    # -----------------------------------------------------------------------
    # Build episodes
    # -----------------------------------------------------------------------

    if Config.SMOKE_TEST:
        # ---- Smoke test: use root-level example files ----
        logger.info("SMOKE TEST mode: using root-level example files as a single episode.")
        logger.info("Building static graph...")
        t0 = time.time()
        csv_stats = scan_segment_stats([str(Config.SMOKE_PROC_PATH)])
        static_graph = PneumaStaticGraph(
            str(Config.SMOKE_TOPOLOGY_PATH),
            str(Config.SMOKE_GPKG_PATH),
            str(Config.SMOKE_CONTROLLERS_PATH),
            csv_stats,
        )
        logger.info(f"  Built in {time.time() - t0:.1f}s | "
                    f"{static_graph.num_segments} segs, {static_graph.num_controllers} controllers")

        # No feature normalisation in smoke test
        train_episodes = [PneumaEpisode(str(Config.SMOKE_PROC_PATH), str(Config.SMOKE_AGG_PATH), static_graph, warmup_steps=Config.WARMUP_STEPS)]
        val_episodes   = [PneumaEpisode(str(Config.SMOKE_PROC_PATH), str(Config.SMOKE_AGG_PATH), static_graph, warmup_steps=Config.WARMUP_STEPS)]

    else:
        # ---- Full training: per-episode folder layout ----
        logger.info(f"Full training mode: scanning episode folders in {Config.DATA_DIR}")

        # 1. Discover episodes and split into train/val
        data_module = PneumaDataModule(
            str(Config.DATA_DIR), str(Config.TOPOLOGY_PATH),
            train_frac=Config.TRAIN_FRAC, seed=Config.SEED,
            chunk_length=Config.CHUNK_LENGTH,
            chunks_per_epoch=Config.CHUNKS_PER_EPOCH,
            use_hgt_sampling=Config.USE_HGT_SAMPLING,
            hgt_hops=Config.HGT_HOPS,
        )

        # 2. Pre-scan all episodes for segment static metadata (csv_stats)
        logger.info("Pre-scanning episode CSVs for segment stats...")
        t0 = time.time()
        all_proc = data_module.get_all_proc_paths()
        data_module.csv_stats = scan_segment_stats(all_proc)
        logger.info(f"  Done in {time.time() - t0:.1f}s ({len(data_module.csv_stats)} segments)")

        # 3. Compute (or load cached) feature normalisation statistics
        stats_cache = Config.STATS_PATH
        if stats_cache.exists():
            with open(stats_cache) as f:
                feature_stats = json.load(f)
            logger.info(f"Loaded feature stats from {stats_cache}")
        else:
            train_proc, train_agg = data_module.get_train_proc_agg_paths()
            logger.info(f"Computing feature stats from {len(train_proc)} training episodes...")
            t0 = time.time()
            feature_stats = precompute_feature_stats(train_proc, train_agg)
            stats_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_cache, 'w') as f:
                json.dump(feature_stats, f, indent=2)
            logger.info(f"  Done in {time.time() - t0:.1f}s — saved to {stats_cache}")
        data_module.feature_stats = feature_stats

        # 4. Materialise episode objects (builds per-episode static graphs)
        logger.info("Building per-episode static graphs and loading episodes...")
        t0 = time.time()
        train_episodes = list(data_module.iter_train())
        val_episodes   = list(data_module.iter_val())
        logger.info(f"  Done in {time.time() - t0:.1f}s")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch_geometric.is_xpu_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")

    logger.info(f"Episodes: {len(train_episodes)} train, {len(val_episodes)} val")

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    logger.info("Building model...")
    model = PneumaModel(
        hidden_channels=Config.HIDDEN_CHANNELS,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        lambda_v=Config.LAMBDA_V,
        lambda_s=Config.LAMBDA_S,
    ).to(device)

    logger.info(f"  Model parameters: "
                f"{sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    os.makedirs(str(Config.CHECKPOINT_DIR), exist_ok=True)
    best_val_loss = float('inf')
    best_epoch    = -1
    history       = []

    logger.info(f"\nStarting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        t_epoch = time.time()

        train_metrics = train_epoch(model, train_episodes, optimizer, device, grad_accum_steps=Config.GRAD_ACCUM_STEPS)
        val_metrics   = val_epoch(model, val_episodes, device)

        elapsed = time.time() - t_epoch
        logger.info(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"(v_mae={train_metrics['vehicle_mae']:.4f}, s_mae={train_metrics['segment_mae']:.4f}) | "
            f"val loss={val_metrics['loss']:.4f} "
            f"(v_mae={val_metrics['vehicle_mae']:.4f}, s_mae={val_metrics['segment_mae']:.4f}) | "
            f"t={elapsed:.1f}s"
        )

        epoch_stats = {
            'epoch': epoch,
            'train': train_metrics,
            'val':   val_metrics
        }
        history.append(epoch_stats)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch    = epoch
            ckpt_path = Config.CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss,
            }, ckpt_path)
            logger.info(f"  -> Saved best model to {ckpt_path}")

    # Save history
    history_path = Config.CHECKPOINT_DIR / "epoch_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Final evaluation on best model for visualization
    logger.info("Running final evaluation on best model...")
    ckpt = torch.load(Config.CHECKPOINT_DIR / "best_model.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    val_results = evaluate_model(model, val_episodes, device)
    
    val_results_path = Config.CHECKPOINT_DIR / "val_results.json"
    with open(val_results_path, 'w') as f:
        json.dump(val_results, f)
    logger.info(f"Saved validation results to {val_results_path}")

    logger.info(f"\nTraining complete. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == '__main__':
    main()
