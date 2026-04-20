"""
graph_builder.py
================
Static and dynamic graph construction for the PNEUMA HGT training pipeline.

Classes
-------
PneumaStaticGraph
    Loads and pre-processes the fixed parts of the graph:
    road segment features, controller features, and all static edge indices.

PneumaSnapshotBuilder
    Incrementally builds PyG HeteroData snapshots at each trajectory timestamp,
    maintaining rotating memory deques for vehicle and segment history.
"""

import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGHWAY_TYPES = [
    'unknown', 'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'unclassified', 'residential',
]
HIGHWAY_TO_IDX: Dict[str, int] = {h: i for i, h in enumerate(HIGHWAY_TYPES)}

# Maps segment_type values found in processed CSVs to GeoPackage highway names
SEGMENT_TYPE_MAP: Dict[str, str] = {
    'primary': 'primary',
    'secondary': 'secondary',
    'tertiary': 'tertiary',
    'residential': 'residential',
}

# Topology edge types to include (u_turns_into and opposite_direction excluded)
TOPO_EDGE_TYPES: List[str] = [
    'to', 'from', 'turns_into', 'crosses', 'crossed_by',
    'merges_with', 'merged_by', 'merges_into', 'intersects_with',
]

# These relation types require bidirectional edges (A→B and B→A under same label)
BIDIRECTIONAL_TOPO: frozenset = frozenset({'turns_into', 'merges_into', 'intersects_with'})

# 12 vehicle feature columns (in order)
VEHICLE_FEATURE_COLS: List[str] = [
    'speed',
    'long_acc',
    'lat_acc',
    'proportionate_distance_travelled',
    'lane_norm',                        # derived: lane_index / num_lanes
    'relative_time_gap',
    'relative_kinematic_ratio',
    'relative_ego_speed',
    'relative_occupancy_proceeding',
    'relative_speed_proceeding',
    'relative_occupancy_following',
    'relative_speed_following',
]
NUM_VEHICLE_FEATURES: int = len(VEHICLE_FEATURE_COLS)   # 12
NUM_SEGMENT_FEATURES: int = 8
NUM_SEGMENT_METRIC_FEATURES: int = 4
NUM_CONTROLLER_FEATURES: int = 3

# Memory limits
MAX_VEHICLE_MEMORY: int = 30
MAX_SEGMENT_MEMORY: int = 10


# ---------------------------------------------------------------------------
# PneumaStaticGraph
# ---------------------------------------------------------------------------

class PneumaStaticGraph:
    """
    Loads and pre-processes the static parts of the PNEUMA graph.

    Segment features [N_seg, 8]:
        0  log_length          log(length + 1)
        1  num_lanes_norm      lanes / max_lanes_in_network
        2  highway_type_norm   label_index / (n_types - 1)
        3  speed_proxy_norm    maxspeed_norm (gpkg) or free_flow_speed_norm (csv)
        4  signal_at_end       0 / 1
        5  centroid_x_norm     (utm_x - mean) / std  [0 for non-gpkg segs]
        6  centroid_y_norm     (utm_y - mean) / std  [0 for non-gpkg segs]
        7  free_flow_speed_norm free_flow_speed / max_ffs  [from csv_stats]

    Controller features [N_ctrl, 3]:
        0  signal_count_norm   signal_count / max_signals
        1  junction_x_norm     (utm_x - mean) / std
        2  junction_y_norm     (utm_y - mean) / std

    Parameters
    ----------
    topology_path :
        Path to topological_adjacency.json
    gpkg_path :
        Path to osm_network.gpkg
    controllers_path :
        Path to controllers.json
    csv_stats :
        Optional dict mapping segment_id (str) to
        {'length': float, 'type': str, 'num_lanes': int, 'free_flow_speed': float}
        Pre-scanned from processed CSV files to fill features for non-GeoPackage segments.
    """

    def __init__(
        self,
        topology_path: str,
        gpkg_path: str,
        controllers_path: str,
        csv_stats: Optional[Dict[str, Dict]] = None,
    ) -> None:
        with open(topology_path) as f:
            self._topo: Dict = json.load(f)
        with open(controllers_path) as f:
            self._ctrl_raw: Dict = json.load(f)
        self._gdf: gpd.GeoDataFrame = gpd.read_file(gpkg_path)

        # Deterministic ordering: sorted segment IDs
        all_seg_ids = sorted(self._topo.keys())
        self.segment_id_to_idx: Dict[str, int] = {s: i for i, s in enumerate(all_seg_ids)}
        self.idx_to_segment_id: List[str] = all_seg_ids
        self.num_segments: int = len(all_seg_ids)

        ctrl_ids = sorted(self._ctrl_raw.keys())
        self.controller_id_to_idx: Dict[str, int] = {c: i for i, c in enumerate(ctrl_ids)}
        self.idx_to_controller_id: List[str] = ctrl_ids
        self.num_controllers: int = len(ctrl_ids)

        self.segment_features: Tensor = self._build_segment_features(csv_stats)
        self.controller_features: Tensor = self._build_controller_features()
        self.topology_edge_dict: Dict = self._build_topology_edges()
        self.controller_edge_dict: Dict = self._build_controller_edges()

    # ------------------------------------------------------------------
    # Segment feature construction
    # ------------------------------------------------------------------

    def _build_segment_features(self, csv_stats: Optional[Dict]) -> Tensor:
        gdf = self._gdf.copy()
        gdf_proj = gdf.to_crs(epsg=32634)
        gdf['centroid_x'] = gdf_proj.geometry.centroid.x
        gdf['centroid_y'] = gdf_proj.geometry.centroid.y
        gdf['seg_id_str'] = gdf['segment_id'].astype(str)

        # Normalisation statistics from GeoPackage
        cx_mean = float(gdf['centroid_x'].mean())
        cx_std = float(gdf['centroid_x'].std()) + 1e-8
        cy_mean = float(gdf['centroid_y'].mean())
        cy_std = float(gdf['centroid_y'].std()) + 1e-8
        lanes_max = float(gdf['lanes'].max()) if gdf['lanes'].notna().any() else 4.0

        maxspeed_num = pd.to_numeric(gdf['maxspeed'], errors='coerce')
        ms_median = float(maxspeed_num.median()) if not maxspeed_num.isna().all() else 50.0
        ms_max = float(maxspeed_num.max()) if not maxspeed_num.isna().all() else 50.0
        gdf['maxspeed_norm'] = maxspeed_num.fillna(ms_median) / max(ms_max, 1.0)

        # Free-flow speed max for normalisation
        ffs_vals: List[float] = []
        if csv_stats:
            ffs_vals = [
                float(v['free_flow_speed'])
                for v in csv_stats.values()
                if v.get('free_flow_speed') and not math.isnan(float(v['free_flow_speed']))
            ]
        ffs_max = max(ffs_vals) if ffs_vals else 1.0

        # Build per-segment lookup from GeoPackage
        gpkg_lookup: Dict[str, Dict] = {}
        for _, row in gdf.iterrows():
            sid = str(row['seg_id_str'])
            gpkg_lookup[sid] = {
                'log_length': math.log1p(float(row['length'])),
                'lanes_norm': float(row['lanes']) / lanes_max if not math.isnan(float(row['lanes'])) else 1.0 / lanes_max,
                'highway_idx': HIGHWAY_TO_IDX.get(str(row['highway']), 0),
                'maxspeed_norm': float(row['maxspeed_norm']),
                'signal_at_end': 1.0 if row['signal_at_end'] else 0.0,
                'cx_norm': (float(row['centroid_x']) - cx_mean) / cx_std,
                'cy_norm': (float(row['centroid_y']) - cy_mean) / cy_std,
            }

        n_highway_types = max(len(HIGHWAY_TYPES) - 1, 1)
        feats = np.zeros((self.num_segments, NUM_SEGMENT_FEATURES), dtype=np.float32)

        for i, seg_id in enumerate(self.idx_to_segment_id):
            row = feats[i]  # view into array
            in_gpkg = seg_id in gpkg_lookup
            in_csv = csv_stats is not None and seg_id in csv_stats

            if in_gpkg:
                g = gpkg_lookup[seg_id]
                row[0] = g['log_length']
                row[1] = g['lanes_norm']
                row[2] = float(g['highway_idx']) / n_highway_types
                row[3] = g['maxspeed_norm']
                row[4] = g['signal_at_end']
                row[5] = g['cx_norm']
                row[6] = g['cy_norm']
            elif in_csv:
                s = csv_stats[seg_id]
                row[0] = math.log1p(float(s.get('length', 0.0)))
                row[1] = float(s.get('num_lanes', 1)) / lanes_max
                htype = SEGMENT_TYPE_MAP.get(str(s.get('type', 'unknown')), 'unknown')
                row[2] = float(HIGHWAY_TO_IDX.get(htype, 0)) / n_highway_types
                # Features 3-6 left as 0 (no speed proxy, no centroid)

            # Feature 7: free_flow_speed from csv_stats (preferred over maxspeed)
            if in_csv:
                ffs = float(csv_stats[seg_id].get('free_flow_speed', 0.0))
                row[7] = ffs / ffs_max if ffs_max > 0.0 else 0.0
            elif in_gpkg:
                row[7] = gpkg_lookup[seg_id]['maxspeed_norm']

        return torch.tensor(feats, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Controller feature construction
    # ------------------------------------------------------------------

    def _build_controller_features(self) -> Tensor:
        gdf_proj = self._gdf.to_crs(epsg=32634)
        cx_mean = float(gdf_proj.geometry.centroid.x.mean())
        cx_std = float(gdf_proj.geometry.centroid.x.std()) + 1e-8
        cy_mean = float(gdf_proj.geometry.centroid.y.mean())
        cy_std = float(gdf_proj.geometry.centroid.y.std()) + 1e-8

        max_signals = max(v['signal_count'] for v in self._ctrl_raw.values())

        feats = np.zeros((self.num_controllers, NUM_CONTROLLER_FEATURES), dtype=np.float32)
        for ctrl_id, idx in self.controller_id_to_idx.items():
            c = self._ctrl_raw[ctrl_id]
            x = float(c['junction_centroid']['x_utm'])
            y = float(c['junction_centroid']['y_utm'])
            feats[idx, 0] = float(c['signal_count']) / max(max_signals, 1)
            feats[idx, 1] = (x - cx_mean) / cx_std
            feats[idx, 2] = (y - cy_mean) / cy_std

        return torch.tensor(feats, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Topology edge indices
    # ------------------------------------------------------------------

    def _build_topology_edges(self) -> Dict:
        # Accumulate (src, dst) pairs per edge type
        edge_lists: Dict[str, Tuple[List[int], List[int]]] = {
            et: ([], []) for et in TOPO_EDGE_TYPES
        }

        for src_id, data in self._topo.items():
            if src_id not in self.segment_id_to_idx:
                continue
            src_idx = self.segment_id_to_idx[src_id]
            for et in TOPO_EDGE_TYPES:
                targets = data.get(et, [])
                for tgt_id in targets:
                    tgt_str = str(tgt_id)
                    if tgt_str not in self.segment_id_to_idx:
                        continue
                    tgt_idx = self.segment_id_to_idx[tgt_str]
                    edge_lists[et][0].append(src_idx)
                    edge_lists[et][1].append(tgt_idx)
                    if et in BIDIRECTIONAL_TOPO:
                        edge_lists[et][0].append(tgt_idx)
                        edge_lists[et][1].append(src_idx)

        result: Dict = {}
        for et in TOPO_EDGE_TYPES:
            src_list, dst_list = edge_lists[et]
            if len(src_list) == 0:
                idx_tensor = torch.zeros((2, 0), dtype=torch.long)
            else:
                idx_tensor = torch.tensor([src_list, dst_list], dtype=torch.long)
                if et in BIDIRECTIONAL_TOPO:
                    idx_tensor = torch.unique(idx_tensor, dim=1)
            result[('segment', et, 'segment')] = idx_tensor

        return result

    # ------------------------------------------------------------------
    # Controller edge indices
    # ------------------------------------------------------------------

    def _build_controller_edges(self) -> Dict:
        ctrl_src, ctrl_dst = [], []
        seg_src, seg_dst = [], []

        for ctrl_id, data in self._ctrl_raw.items():
            if ctrl_id not in self.controller_id_to_idx:
                continue
            ctrl_idx = self.controller_id_to_idx[ctrl_id]
            for seg_id in data.get('approach_segments', []):
                seg_str = str(seg_id)
                if seg_str not in self.segment_id_to_idx:
                    continue
                seg_idx = self.segment_id_to_idx[seg_str]
                ctrl_src.append(ctrl_idx)
                ctrl_dst.append(seg_idx)
                seg_src.append(seg_idx)
                seg_dst.append(ctrl_idx)

        if ctrl_src:
            ctrl_edge = torch.tensor([ctrl_src, ctrl_dst], dtype=torch.long)
            seg_edge = torch.tensor([seg_src, seg_dst], dtype=torch.long)
        else:
            ctrl_edge = torch.zeros((2, 0), dtype=torch.long)
            seg_edge = torch.zeros((2, 0), dtype=torch.long)

        return {
            ('controller', 'controls', 'segment'): ctrl_edge,
            ('segment', 'controlled_by', 'controller'): seg_edge,
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_static_edge_index_dict(self) -> Dict:
        """Returns all static edge indices (topology + controller) as a single dict."""
        d: Dict = {}
        d.update(self.topology_edge_dict)
        d.update(self.controller_edge_dict)
        return d


# ---------------------------------------------------------------------------
# PneumaSnapshotBuilder
# ---------------------------------------------------------------------------

class PneumaSnapshotBuilder:
    """
    Incrementally builds HeteroData snapshots for one episode.

    Call ``load_episode`` at the start of each episode to reset state.
    Call ``get_snapshot(t, t_next)`` sequentially for each timestamp in the episode.

    Memory management
    -----------------
    vehicle_memory: deque per vehicle (maxlen=MAX_VEHICLE_MEMORY=30)
        Each entry = (timestamp: float, feature_vec: np.ndarray[12])
        At snapshot t, the deque holds states from t-1, t-2, …, t-30.

    segment_metric: deque per segment (maxlen=MAX_SEGMENT_MEMORY=10)
        Each entry = (timestamp: float, feature_vec: np.ndarray[4])
        Populated from aggregated_states rows with timestamp ≤ t.
    """

    def __init__(self, static_graph: PneumaStaticGraph, filter_outliers: bool = True, feature_stats: Optional[Dict] = None) -> None:
        self.static = static_graph
        self.filter_outliers = filter_outliers
        self.feature_stats = feature_stats

        if self.feature_stats:
            self._veh_mean = np.array(self.feature_stats['vehicle_mean'], dtype=np.float32)
            self._veh_std = np.array(self.feature_stats['vehicle_std'], dtype=np.float32)
            self._seg_mean = np.array(self.feature_stats['seg_metric_mean'], dtype=np.float32)
            self._seg_std = np.array(self.feature_stats['seg_metric_std'], dtype=np.float32)
        else:
            self._veh_mean = self._veh_std = self._seg_mean = self._seg_std = None

        # Episode-level state (reset by load_episode)
        self._time_groups: Optional[Dict[float, Dict]] = None
        self._agg_times: Optional[np.ndarray] = None
        self._agg_seg_idx: Optional[np.ndarray] = None
        self._agg_feats: Optional[np.ndarray] = None
        self._agg_pointer: int = 0  # index into sorted aggregated_df

        # Per-vehicle state
        self._vehicle_memory_deques: Dict[int, deque] = {}
        self._prev_vehicle_features: Dict[int, np.ndarray] = {}
        self._prev_vehicle_times: Dict[int, float] = {}
        self._prev_active_ids: set = set()
        self._last_t: float = -float('inf')

        # Per-segment state
        self._segment_metric_deques: Dict[int, deque] = {}

        # Pre-compute static edge indices and feature tensors for reuse
        self._static_edge_dict = static_graph.get_static_edge_index_dict()

    # ------------------------------------------------------------------
    # Episode loading
    # ------------------------------------------------------------------

    def load_episode(
        self,
        processed_df: pd.DataFrame,
        aggregated_df: pd.DataFrame,
    ) -> None:
        """Pre-process episode dataframes into highly optimised numpy arrays."""
        # 1. Vectorize processed trajectories
        df = processed_df.sort_values('time').reset_index(drop=True)
        
        if 'time_bucket' not in df.columns:
            df['time_bucket'] = df['time'].round(0)
        df = df.drop_duplicates(subset=['time_bucket', 'track_id']).reset_index(drop=True)
        if self.filter_outliers:
            df = df[~(df['is_outlier'].astype(bool) | df['is_parked'].astype(bool))]

        num_lanes = df['num_lanes'].clip(lower=1.0)
        lane_norm = df['lane_index'] / num_lanes
        rtg = df['relative_time_gap'].fillna(0.0).clip(upper=10.0) / 10.0
        rkr = df['relative_kinematic_ratio'].fillna(0.0)

        feats = np.column_stack([
            df['speed'].values,
            df['long_acc'].values,
            df['lat_acc'].values,
            df['proportionate_distance_travelled'].values,
            lane_norm.values,
            rtg.values,
            rkr.values,
            df['relative_ego_speed'].values,
            df['relative_occupancy_proceeding'].values,
            df['relative_speed_proceeding'].values,
            df['relative_occupancy_following'].values,
            df['relative_speed_following'].values,
        ]).astype(np.float32)

        if self._veh_mean is not None:
            feats = (feats - self._veh_mean) / self._veh_std

        seg_idx_map = self.static.segment_id_to_idx
        df['__seg_idx'] = df['segment_id'].astype(str).map(seg_idx_map).fillna(0).astype(int)
        df['__feat_idx'] = np.arange(len(df))

        self._time_groups = {}
        for t_bucket, group in df.groupby('time_bucket'):
            idx = group['__feat_idx'].values
            self._time_groups[t_bucket] = {
                'track_ids': group['track_id'].values.astype(int),
                'feats': feats[idx],
                'seg_indices': group['__seg_idx'].values.tolist(),
                'true_times': group['time'].values.astype(float),
            }

        # 2. Vectorize aggregated segment states
        agg = aggregated_df.sort_values('timestamp').reset_index(drop=True)
        agg['__seg_idx'] = agg['segment_id'].astype(str).map(seg_idx_map)
        agg = agg.dropna(subset=['__seg_idx']).copy()
        
        log_time = np.log1p(agg['time_since_last_update'].values)
        sm_feats = np.column_stack([
            agg['ema_temporal_speed'].values,
            agg['ema_spatial_speed'].values,
            agg['recalculated_rtsm'].values,
            log_time
        ]).astype(np.float32)

        if self._seg_mean is not None:
            sm_feats = (sm_feats - self._seg_mean) / self._seg_std

        self._agg_times = agg['timestamp'].values.astype(float)
        self._agg_seg_idx = agg['__seg_idx'].values.astype(int)
        self._agg_feats = sm_feats
        
        self.reset_state()

    def reset_state(self) -> None:
        """Resets the temporal memory tracking for a new sequential chunk."""
        self._agg_pointer = 0
        self._vehicle_memory_deques = {}
        self._prev_vehicle_features = {}
        self._prev_vehicle_times = {}
        self._prev_active_ids = set()
        self._last_t = -float('inf')
        self._segment_metric_deques = {}

    # ------------------------------------------------------------------
    # Snapshot construction
    # ------------------------------------------------------------------

    def get_snapshot(
        self,
        t: float,
        t_next: Optional[float] = None,
    ) -> Tuple[HeteroData, Dict]:
        """
        Build and return the graph snapshot at timestamp ``t``.

        Parameters
        ----------
        t :
            Current timestamp (seconds, within-episode).
        t_next :
            Optional next timestamp. When provided, prediction targets are
            computed and returned in the targets dict.

        Returns
        -------
        data : HeteroData
        targets : dict with optional keys 'vehicle' and 'segment'
        """
        assert self._time_groups is not None, "Call load_episode() before get_snapshot()."

        # 1. Advance aggregated pointer → update segment_metric deques up to t
        self._update_segment_metrics(t)

        # 2. Fetch vehicle rows at time t
        vehicle_feats, vehicle_local_idx, seg_indices, vehicle_times = self._extract_vehicle_data(t)
        active_ids = set(vehicle_local_idx.keys())

        # 3. Vehicles that disappeared since last step: clean up
        vanished = self._prev_active_ids - active_ids
        for vid in vanished:
            self._vehicle_memory_deques.pop(vid, None)
            self._prev_vehicle_features.pop(vid, None)
            self._prev_vehicle_times.pop(vid, None)

        # 4. New vehicles: initialise empty deque
        appeared = active_ids - self._prev_active_ids
        for vid in appeared:
            self._vehicle_memory_deques[vid] = deque(maxlen=MAX_VEHICLE_MEMORY)

        # 5. Push PREVIOUS features into deques (rotate memory)
        #    Only for vehicles that were active in the last step
        for vid in self._prev_active_ids.intersection(active_ids):
            if vid in self._prev_vehicle_features:
                exact_prev_t = self._prev_vehicle_times[vid]
                self._vehicle_memory_deques[vid].appendleft(
                    (exact_prev_t, self._prev_vehicle_features[vid])
                )

        # 7. Assemble memory data
        vm_feats, vm_src, vm_dst, vm_times = self._build_vehicle_memory_data(
            t, vehicle_local_idx, vehicle_times
        )
        sm_feats, sm_src, sm_dst, sm_times = self._build_segment_metric_data(t)

        # 8. Assemble HeteroData
        data = self._assemble_heterodata(
            vehicle_feats, seg_indices,
            vm_feats, vm_src, vm_dst, vm_times,
            sm_feats, sm_src, sm_dst, sm_times,
        )

        # 9. Build targets
        targets = {}
        if t_next is not None:
            targets = self._build_targets(active_ids, vehicle_local_idx, t, t_next)

        # 10. Update state
        id_list = list(vehicle_local_idx.keys())
        for vid, feat_row in zip(id_list, vehicle_feats):
            self._prev_vehicle_features[vid] = feat_row.copy()
            self._prev_vehicle_times[vid] = vehicle_times[vid]
        self._prev_active_ids = active_ids
        self._last_t = t

        return data, targets

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_vehicle_data(
        self,
        t: float,
    ) -> Tuple[np.ndarray, Dict[int, int], List[int], Dict[int, float]]:
        tg = self._time_groups.get(t)
        if tg is None:
            return np.zeros((0, NUM_VEHICLE_FEATURES), dtype=np.float32), {}, [], {}
        vehicle_local_idx = {vid: i for i, vid in enumerate(tg['track_ids'])}
        vehicle_times = {vid: tg['true_times'][i] for i, vid in enumerate(tg['track_ids'])}
        return tg['feats'], vehicle_local_idx, tg['seg_indices'], vehicle_times

    # ------------------------------------------------------------------
    # Memory data builders
    # ------------------------------------------------------------------

    def _build_vehicle_memory_data(
        self,
        t: float,
        vehicle_local_idx: Dict[int, int],
        vehicle_times: Dict[int, float],
    ) -> Tuple[np.ndarray, List[int], List[int], List[float]]:
        """
        Returns
        -------
        feats    : [N_mem, 12] float32
        src_list : [N_mem]  memory node local indices
        dst_list : [N_mem]  vehicle node local indices
        times    : [N_mem]  time delta (t - memory_timestamp)
        """
        feats_list: List[np.ndarray] = []
        src_list: List[int] = []
        dst_list: List[int] = []
        times: List[float] = []

        mem_node_idx = 0
        for vid, veh_local in vehicle_local_idx.items():
            dq = self._vehicle_memory_deques.get(vid)
            if dq is None:
                continue
            current_t = vehicle_times[vid]
            for mem_t, mem_feat in dq:
                feats_list.append(mem_feat)
                src_list.append(mem_node_idx)
                dst_list.append(veh_local)
                times.append(max(current_t - mem_t, 0.0))
                mem_node_idx += 1

        if feats_list:
            feats = np.stack(feats_list, axis=0).astype(np.float32)
        else:
            feats = np.zeros((0, NUM_VEHICLE_FEATURES), dtype=np.float32)

        return feats, src_list, dst_list, times

    def _update_segment_metrics(self, t: float) -> None:
        """Advance aggregated pointer to t, pushing new entries into deques."""
        while self._agg_pointer < len(self._agg_times):
            ts = self._agg_times[self._agg_pointer]
            if ts > t:
                break
            seg_idx = self._agg_seg_idx[self._agg_pointer]
            feat = self._agg_feats[self._agg_pointer]
            
            if seg_idx not in self._segment_metric_deques:
                self._segment_metric_deques[seg_idx] = deque(maxlen=MAX_SEGMENT_MEMORY)
            self._segment_metric_deques[seg_idx].appendleft((ts, feat))
            
            self._agg_pointer += 1

    def _build_segment_metric_data(
        self,
        t: float,
    ) -> Tuple[np.ndarray, List[int], List[int], List[float]]:
        """
        Returns
        -------
        feats    : [N_met, 4] float32
        src_list : [N_met]  metric node local indices
        dst_list : [N_met]  segment (global) indices
        times    : [N_met]  time delta (t - metric_timestamp)
        """
        feats_list: List[np.ndarray] = []
        src_list: List[int] = []
        dst_list: List[int] = []
        times: List[float] = []

        met_node_idx = 0
        for seg_idx, dq in self._segment_metric_deques.items():
            for met_t, met_feat in dq:
                feats_list.append(met_feat)
                src_list.append(met_node_idx)
                dst_list.append(seg_idx)
                times.append(max(t - met_t, 0.0))
                met_node_idx += 1

        if feats_list:
            feats = np.stack(feats_list, axis=0).astype(np.float32)
        else:
            feats = np.zeros((0, NUM_SEGMENT_METRIC_FEATURES), dtype=np.float32)

        return feats, src_list, dst_list, times

    # ------------------------------------------------------------------
    # HeteroData assembly
    # ------------------------------------------------------------------

    def _assemble_heterodata(
        self,
        vehicle_feats: np.ndarray,
        seg_indices: List[int],
        vm_feats: np.ndarray,
        vm_src: List[int],
        vm_dst: List[int],
        vm_times: List[float],
        sm_feats: np.ndarray,
        sm_src: List[int],
        sm_dst: List[int],
        sm_times: List[float],
    ) -> HeteroData:
        data = HeteroData()

        # --- Node features ---
        data['vehicle'].x = torch.tensor(vehicle_feats, dtype=torch.float32)
        data['vehicle_memory'].x = torch.tensor(vm_feats, dtype=torch.float32)
        data['segment'].x = self.static.segment_features          # shared reference
        data['segment_metric'].x = torch.tensor(sm_feats, dtype=torch.float32)
        data['controller'].x = self.static.controller_features    # shared reference

        # --- Dynamic edges ---
        N_veh = vehicle_feats.shape[0]
        N_seg = self.static.num_segments

        # vehicle_memory → vehicle
        if vm_src:
            vm_edge = torch.tensor([vm_src, vm_dst], dtype=torch.long)
            vm_time_tensor = torch.tensor(vm_times, dtype=torch.float32)
        else:
            vm_edge = torch.zeros((2, 0), dtype=torch.long)
            vm_time_tensor = torch.zeros(0, dtype=torch.float32)
        data['vehicle_memory', 'prev_state', 'vehicle'].edge_index = vm_edge
        data['vehicle_memory', 'prev_state', 'vehicle'].edge_time = vm_time_tensor

        # segment_metric → segment
        if sm_src:
            sm_edge = torch.tensor([sm_src, sm_dst], dtype=torch.long)
            sm_time_tensor = torch.tensor(sm_times, dtype=torch.float32)
        else:
            sm_edge = torch.zeros((2, 0), dtype=torch.long)
            sm_time_tensor = torch.zeros(0, dtype=torch.float32)
        data['segment_metric', 'prev_state', 'segment'].edge_index = sm_edge
        data['segment_metric', 'prev_state', 'segment'].edge_time = sm_time_tensor

        # vehicle → segment  (on) and reverse
        if N_veh > 0:
            veh_src = list(range(N_veh))
            on_edge = torch.tensor([veh_src, seg_indices], dtype=torch.long)
            rev_edge = torch.tensor([seg_indices, veh_src], dtype=torch.long)
        else:
            on_edge = torch.zeros((2, 0), dtype=torch.long)
            rev_edge = torch.zeros((2, 0), dtype=torch.long)
        data['vehicle', 'on', 'segment'].edge_index = on_edge
        data['segment', 'occupied_by', 'vehicle'].edge_index = rev_edge

        # --- Static edges ---
        for etype, edge_idx in self._static_edge_dict.items():
            src_type, rel, dst_type = etype
            data[src_type, rel, dst_type].edge_index = edge_idx

        return data

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _build_targets(
        self,
        active_ids: set,
        vehicle_local_idx: Dict[int, int],
        t: float,
        t_next: float,
    ) -> Dict:
        """
        Returns
        -------
        dict with keys:
          'vehicle_feats'    : Tensor [N_veh, 12]  (rows = vehicles present at both t and t+1)
          'vehicle_mask'     : Tensor [N_veh]  bool  (True = vehicle present at t+1)
          'segment_feats'    : Tensor [N_seg_update, 1]  ema_temporal_speed at t+1
          'segment_mask_idx' : Tensor [N_seg_update]  global segment indices with new entry
        """
        targets: Dict = {}

        # --- Vehicle targets ---
        tg_next = self._time_groups.get(t_next)
        N_veh = len(vehicle_local_idx)
        veh_target_feats = np.zeros((N_veh, NUM_VEHICLE_FEATURES), dtype=np.float32)
        veh_mask = np.zeros(N_veh, dtype=bool)

        if tg_next is not None:
            next_ids = tg_next['track_ids']
            next_feats = tg_next['feats']
            next_lookup = {vid: next_feats[i] for i, vid in enumerate(next_ids)}
            
            for vid, local_i in vehicle_local_idx.items():
                if vid in next_lookup:
                    veh_target_feats[local_i] = next_lookup[vid]
                    veh_mask[local_i] = True

        targets['vehicle_feats'] = torch.tensor(veh_target_feats, dtype=torch.float32)
        targets['vehicle_mask'] = torch.tensor(veh_mask, dtype=torch.bool)

        # --- Segment targets (new aggregated entries in window (t, t_next]) ---
        ptr = self._agg_pointer
        seg_idx_list: List[int] = []
        seg_speed_list: List[float] = []
        
        while ptr < len(self._agg_times):
            ts = self._agg_times[ptr]
            if ts > t_next:
                break
            seg_idx_list.append(self._agg_seg_idx[ptr])
            seg_speed_list.append(self._agg_feats[ptr][0])  # ema_temporal_speed already normalised
            ptr += 1

        if seg_idx_list:
            targets['segment_feats'] = torch.tensor(seg_speed_list, dtype=torch.float32).unsqueeze(1)
            targets['segment_mask_idx'] = torch.tensor(seg_idx_list, dtype=torch.long)
        else:
            targets['segment_feats'] = torch.zeros((0, 1), dtype=torch.float32)
            targets['segment_mask_idx'] = torch.zeros(0, dtype=torch.long)

        return targets


# ---------------------------------------------------------------------------
# Utility: pre-scan processed CSVs to collect per-segment static metadata
# ---------------------------------------------------------------------------

def precompute_feature_stats(proc_paths: List[str], agg_paths: List[str]) -> Dict[str, List[float]]:
    """
    Precompute normalisation statistics (mean and std) for dynamic features.
    """
    veh_feats = []
    seg_feats = []

    for path in proc_paths:
        df = pd.read_csv(path, usecols=[
            'speed', 'long_acc', 'lat_acc', 'proportionate_distance_travelled',
            'lane_index', 'num_lanes', 'relative_time_gap', 'relative_kinematic_ratio',
            'relative_ego_speed', 'relative_occupancy_proceeding',
            'relative_speed_proceeding', 'relative_occupancy_following',
            'relative_speed_following', 'is_outlier', 'is_parked'
        ])
        df = df[~(df['is_outlier'].astype(bool) | df['is_parked'].astype(bool))]
        num_lanes = df['num_lanes'].clip(lower=1.0)
        lane_norm = df['lane_index'] / num_lanes
        rtg = df['relative_time_gap'].fillna(0.0).clip(upper=10.0) / 10.0
        rkr = df['relative_kinematic_ratio'].fillna(0.0)
        
        feats = np.column_stack([
            df['speed'].values,
            df['long_acc'].values,
            df['lat_acc'].values,
            df['proportionate_distance_travelled'].values,
            lane_norm.values,
            rtg.values,
            rkr.values,
            df['relative_ego_speed'].values,
            df['relative_occupancy_proceeding'].values,
            df['relative_speed_proceeding'].values,
            df['relative_occupancy_following'].values,
            df['relative_speed_following'].values,
        ]).astype(np.float32)
        veh_feats.append(feats)

    for path in agg_paths:
        df = pd.read_csv(path, usecols=[
            'ema_temporal_speed', 'ema_spatial_speed', 
            'recalculated_rtsm', 'time_since_last_update'
        ])
        log_time = np.log1p(df['time_since_last_update'].values)
        feats = np.column_stack([
            df['ema_temporal_speed'].values,
            df['ema_spatial_speed'].values,
            df['recalculated_rtsm'].values,
            log_time
        ]).astype(np.float32)
        seg_feats.append(feats)

    def get_stats(f_list, dim):
        if not f_list:
            return [0.0] * dim, [1.0] * dim
        all_f = np.concatenate(f_list, axis=0)
        mean = np.nanmean(all_f, axis=0).tolist()
        std = np.nanstd(all_f, axis=0)
        std[std < 1e-5] = 1.0
        return mean, std.tolist()

    v_mean, v_std = get_stats(veh_feats, NUM_VEHICLE_FEATURES)
    s_mean, s_std = get_stats(seg_feats, NUM_SEGMENT_METRIC_FEATURES)

    return {
        'vehicle_mean': v_mean,
        'vehicle_std': v_std,
        'seg_metric_mean': s_mean,
        'seg_metric_std': s_std,
    }

def scan_segment_stats(csv_paths: List[str]) -> Dict[str, Dict]:
    """
    Pre-scan a list of processed CSV files to collect static segment metadata.
    Returns a dict: {segment_id_str -> {'length', 'type', 'num_lanes', 'free_flow_speed'}}
    Only the first seen value per segment is kept (they are static across files).
    """
    stats: Dict[str, Dict] = {}
    for path in csv_paths:
        df = pd.read_csv(path, usecols=[
            'segment_id', 'segment_length', 'segment_type',
            'num_lanes', 'segment_free_flow_speed',
        ], dtype={'segment_id': str})
        for _, row in df.drop_duplicates('segment_id').iterrows():
            sid = str(row['segment_id'])
            if sid not in stats:
                stats[sid] = {
                    'length': float(row['segment_length']),
                    'type': str(row['segment_type']),
                    'num_lanes': int(row['num_lanes']),
                    'free_flow_speed': float(row['segment_free_flow_speed']),
                }
    return stats
