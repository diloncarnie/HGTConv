"""
pneuma_dataset.py
=================
Data loading and episode iteration for the PNEUMA HGT training pipeline.

Classes
-------
PneumaEpisode
    IterableDataset for a single episode (one processed.csv + aggregated_states.csv pair).
    Yields (snapshot_t, targets_t) pairs in chronological order.

PneumaDataModule
    Manages the full set of episode subfolders under data_dir.
    Each subfolder must contain its own osm_network.gpkg, segment_thresholds.json,
    controllers.json, <foldername>_processed.csv, and downsampled_aggregated_states.csv.
    Shared topology file (topological_adjacency.json) is passed separately.
"""

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.loader import HGTLoader
from torch.utils.data import IterableDataset

from .graph_builder import PneumaStaticGraph, PneumaSnapshotBuilder


class PneumaEpisode(IterableDataset):
    """
    Iterates chronologically through one episode (a matched processed/aggregated CSV pair).

    Each iteration step yields ``(snapshot_t, targets_t)`` where:
      - ``snapshot_t`` is a PyG HeteroData object for timestamp ``t``
      - ``targets_t`` is a dict with vehicle and segment prediction targets for step t+1

    The last timestamp is skipped (no t+1 target available).

    Parameters
    ----------
    processed_csv :
        Path to the processed trajectory CSV file.
    aggregated_csv :
        Path to the matching aggregated segment-states CSV file.
    static_graph :
        Pre-built PneumaStaticGraph for this episode.
    filter_outliers :
        If True, rows with ``is_outlier`` or ``is_parked`` are excluded.
    feature_stats :
        Optional dict with keys 'vehicle_mean', 'vehicle_std', 'seg_metric_mean',
        'seg_metric_std' (from precompute_feature_stats). When provided, all dynamic
        node features and prediction targets are z-score normalised.
    """

    def __init__(
        self,
        processed_csv: str,
        aggregated_csv: str,
        static_graph: PneumaStaticGraph,
        filter_outliers: bool = True,
        feature_stats: Optional[Dict] = None,
        chunk_length: int = 200,
        chunks_per_epoch: int = 10,
        warmup_steps: int = 30,
        use_hgt_sampling: bool = False,
        hgt_samples: Optional[Dict[str, List[int]]] = None,
        hgt_hops: int = 2,
    ) -> None:
        self.processed_csv = processed_csv
        self.aggregated_csv = aggregated_csv
        self.static_graph = static_graph
        self.filter_outliers = filter_outliers
        self.feature_stats = feature_stats
        self.chunk_length = chunk_length
        self.chunks_per_epoch = chunks_per_epoch
        self.warmup_steps = warmup_steps
        self.use_hgt_sampling = use_hgt_sampling
        self.hgt_hops = hgt_hops
        
        # Default budget: 64 nodes per type per hop
        self.hgt_samples = hgt_samples or {
            ntype: [64] * hgt_hops for ntype in 
            ['vehicle', 'segment', 'controller', 'vehicle_memory', 'segment_metric']
        }

        # Load and sort by time
        self._proc_df = pd.read_csv(processed_csv, dtype={'segment_id': str})
        # Snap timestamps to the nearest 1 second (1Hz) to eliminate floating-point jitter duplication
        self._proc_df['time_bucket'] = self._proc_df['time'].round(0)
        self._proc_df = self._proc_df.drop_duplicates(subset=['time_bucket', 'track_id'])
        self._proc_df = self._proc_df.sort_values('time').reset_index(drop=True)
        
        self._agg_df = pd.read_csv(aggregated_csv, dtype={'segment_id': str}).sort_values('timestamp').reset_index(drop=True)

        self.timestamps: List[float] = sorted(self._proc_df['time_bucket'].unique().tolist())
        # We yield pairs (t, t_next) so the last timestamp has no target
        self._num_steps: int = max(0, len(self.timestamps) - 1)

    def __len__(self) -> int:
        if self._num_steps == 0:
            return 0
        actual_chunk_len = min(self.chunk_length, self._num_steps)
        return self.chunks_per_epoch * actual_chunk_len

    def __iter__(self):
        if self._num_steps == 0:
            return

        # Sample random chunk starting indices
        valid_starts = max(1, self._num_steps - self.chunk_length)
        start_indices = random.choices(range(valid_starts), k=self.chunks_per_epoch)

        # Initialize and vectorize dataframes exactly ONCE per episode
        builder = PneumaSnapshotBuilder(
            self.static_graph,
            self.filter_outliers,
            self.feature_stats,
        )
        builder.load_episode(self._proc_df, self._agg_df)

        for start_idx in start_indices:
            # Reset temporal states instantly for the new chunk
            builder.reset_state()

            # 1. Warm-up phase: Build historic context in the deques silently
            warmup_idx = max(0, start_idx - self.warmup_steps)
            for i in range(warmup_idx, start_idx):
                builder.get_snapshot(self.timestamps[i])  # t_next=None, no targets generated

            # 2. Training phase: Yield consecutive snapshots with targets
            end_idx = min(start_idx + self.chunk_length, self._num_steps)
            for i in range(start_idx, end_idx):
                t = self.timestamps[i]
                t_next = self.timestamps[i + 1]
                snapshot, targets = builder.get_snapshot(t, t_next)

                if self.use_hgt_sampling and snapshot['vehicle'].num_nodes > 0:
                    # 1. Tag original global IDs before sampling
                    for ntype in snapshot.node_types:
                        snapshot[ntype].n_id = torch.arange(snapshot[ntype].num_nodes)

                    # 2. Setup HGTLoader to sample around a random subset of vehicles
                    n_veh = snapshot['vehicle'].num_nodes
                    seed_size = min(n_veh, 128)
                    input_nodes = ('vehicle', torch.randperm(n_veh)[:seed_size])

                    loader = HGTLoader(
                        snapshot,
                        num_samples=self.hgt_samples,
                        num_hops=self.hgt_hops,
                        input_nodes=input_nodes,
                        batch_size=seed_size,
                        shuffle=False,
                    )
                    sampled_snapshot = next(iter(loader))

                    # 3. Re-map prediction targets to the newly sampled node indices
                    if targets:
                        if 'vehicle_feats' in targets:
                            v_n_id = sampled_snapshot['vehicle'].n_id
                            targets['vehicle_feats'] = targets['vehicle_feats'][v_n_id]
                            targets['vehicle_mask'] = targets['vehicle_mask'][v_n_id]
                        
                        if 'segment_mask_idx' in targets:
                            s_n_id = sampled_snapshot['segment'].n_id
                            global_to_local = {g.item(): l for l, g in enumerate(s_n_id)}
                            
                            new_seg_mask, new_seg_feats = [], []
                            for idx, g_idx in enumerate(targets['segment_mask_idx'].tolist()):
                                if g_idx in global_to_local:
                                    new_seg_mask.append(global_to_local[g_idx])
                                    new_seg_feats.append(targets['segment_feats'][idx])
                                    
                            targets['segment_mask_idx'] = torch.tensor(new_seg_mask, dtype=torch.long)
                            targets['segment_feats'] = torch.stack(new_seg_feats) if new_seg_feats else torch.zeros((0, 1), dtype=torch.float32)
                    snapshot = sampled_snapshot

                yield snapshot, targets


class PneumaDataModule:
    """
    Manages the collection of PNEUMA episode subfolders and their train/val split.

    Directory convention
    --------------------
    Each episode lives in its own subfolder under ``data_dir``
    (e.g., ``20181029_d1_0800_0830/``).  Inside every subfolder::

        <foldername>_processed.csv          — trajectory data
        downsampled_aggregated_states.csv   — segment metrics
        osm_network.gpkg                    — road network (per-episode)
        segment_thresholds.json             — free-flow speed thresholds (per-episode)
        controllers.json                    — junction controller topology (per-episode)

    ``topological_adjacency.json`` is shared across all episodes and must be
    supplied via ``topology_path``.

    Parameters
    ----------
    data_dir :
        Root directory containing the episode subfolders.
    topology_path :
        Path to the shared topological_adjacency.json file.
    train_frac :
        Fraction of episodes used for training (default 0.8 → 16 train, 4 val).
    seed :
        Random seed for the train/val split.
    filter_outliers :
        Passed through to each PneumaEpisode.
    """

    def __init__(
        self,
        data_dir: str,
        topology_path: str,
        train_frac: float = 0.8,
        seed: int = 42,
        filter_outliers: bool = True,
        chunk_length: int = 200,
        chunks_per_epoch: int = 10,
        use_hgt_sampling: bool = False,
        hgt_samples: Optional[Dict[str, List[int]]] = None,
        hgt_hops: int = 2,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.topology_path = topology_path
        self.filter_outliers = filter_outliers
        self.chunk_length = chunk_length
        self.chunks_per_epoch = chunks_per_epoch
        self.use_hgt_sampling = use_hgt_sampling
        self.hgt_samples = hgt_samples
        self.hgt_hops = hgt_hops

        # Set after construction (before iter_train / iter_val)
        self.csv_stats: Optional[Dict] = None
        self.feature_stats: Optional[Dict] = None

        all_episodes = self._discover_episodes()

        rng = random.Random(seed)
        shuffled = list(all_episodes)
        rng.shuffle(shuffled)

        n_train = max(1, round(len(shuffled) * train_frac))
        self.train_episodes: List[Dict] = shuffled[:n_train]
        self.val_episodes:   List[Dict] = shuffled[n_train:]

        print(
            f"PneumaDataModule: {len(all_episodes)} episodes discovered "
            f"({len(self.train_episodes)} train, {len(self.val_episodes)} val)"
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_train_proc_agg_paths(self) -> Tuple[List[str], List[str]]:
        """Return (processed_paths, aggregated_paths) for all training episodes."""
        proc = [str(ep['proc_path']) for ep in self.train_episodes]
        agg  = [str(ep['agg_path'])  for ep in self.train_episodes]
        return proc, agg

    def get_all_proc_paths(self) -> List[str]:
        """Return processed CSV paths for every discovered episode (train + val)."""
        return [str(ep['proc_path']) for ep in self.train_episodes + self.val_episodes]

    # ------------------------------------------------------------------
    # Iterators
    # ------------------------------------------------------------------

    def iter_train(self) -> Iterator[PneumaEpisode]:
        """Yield PneumaEpisode objects for each training episode."""
        for ep in self.train_episodes:
            yield self._make_episode(ep)

    def iter_val(self) -> Iterator[PneumaEpisode]:
        """Yield PneumaEpisode objects for each validation episode."""
        for ep in self.val_episodes:
            yield self._make_episode(ep)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_episode(self, ep: Dict) -> PneumaEpisode:
        """Build the per-episode static graph then wrap it in a PneumaEpisode."""
        static_graph = PneumaStaticGraph(
            self.topology_path,
            str(ep['gpkg_path']),
            str(ep['controllers_path']),
            self.csv_stats,
        )
        return PneumaEpisode(
            str(ep['proc_path']),
            str(ep['agg_path']),
            static_graph,
            self.filter_outliers,
            self.feature_stats,
            self.chunk_length,
            self.chunks_per_epoch,
            self.use_hgt_sampling,
            self.hgt_samples,
            self.hgt_hops,
        )

    def _discover_episodes(self) -> List[Dict]:
        """
        Scan ``data_dir`` for valid episode subfolders.

        A subfolder is valid when it contains all five required files:
          - ``<foldername>_processed.csv``
          - ``downsampled_aggregated_states.csv``
          - ``osm_network.gpkg``
          - ``segment_thresholds.json``
          - ``controllers.json``

        Returns a list of dicts with keys:
          folder, proc_path, agg_path, gpkg_path, thresholds_path, controllers_path
        """
        EXPECTED = 20
        episodes: List[Dict] = []

        for folder in sorted(self.data_dir.iterdir()):
            if not folder.is_dir():
                continue

            proc_csv    = folder / f"{folder.name}_processed.csv"
            agg_csv     = folder / "downsampled_aggregated_states.csv"
            gpkg        = folder / "osm_network.gpkg"
            thresholds  = folder / "segment_thresholds.json"
            controllers = folder / "controllers.json"

            required = {
                "processed csv":          proc_csv,
                "aggregated csv":         agg_csv,
                "osm_network.gpkg":       gpkg,
                "segment_thresholds.json": thresholds,
                "controllers.json":       controllers,
            }
            missing = [name for name, p in required.items() if not p.exists()]
            if missing:
                print(f"[PneumaDataModule] Skipping '{folder.name}': missing {missing}")
                continue

            episodes.append({
                'folder':           folder,
                'proc_path':        proc_csv,
                'agg_path':         agg_csv,
                'gpkg_path':        gpkg,
                'thresholds_path':  thresholds,
                'controllers_path': controllers,
            })

        if len(episodes) != EXPECTED:
            print(
                f"[PneumaDataModule] WARNING: expected {EXPECTED} episode folders, "
                f"found {len(episodes)}."
            )

        return episodes
