"""
pneuma_model.py
===============
PneumaModel wraps the custom HGT encoder with two self-supervised prediction heads:
  - vehicle_head : predicts next-step vehicle kinematic features (12-d regression)
  - segment_head : predicts next-step segment ema_temporal_speed (1-d regression)

Also defines PNEUMA_METADATA and PNEUMA_CUSTOM_ORDER constants used throughout.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear

from .hgt_model import HGT
from .graph_builder import (
    NUM_VEHICLE_FEATURES,
    NUM_SEGMENT_FEATURES,
    NUM_SEGMENT_METRIC_FEATURES,
    NUM_CONTROLLER_FEATURES,
)

# ---------------------------------------------------------------------------
# Metadata constants
# ---------------------------------------------------------------------------

PNEUMA_NODE_TYPES: List[str] = [
    'vehicle',
    'vehicle_memory',
    'segment',
    'segment_metric',
    'controller',
]

PNEUMA_EDGE_TYPES: List[Tuple[str, str, str]] = [
    # Temporal memory edges
    ('vehicle_memory', 'prev_state',    'vehicle'),
    ('segment_metric', 'prev_state',    'segment'),
    # Dynamic vehicle ↔ segment
    ('vehicle',        'on',            'segment'),
    ('segment',        'occupied_by',   'vehicle'),
    # Static road topology (9 types)
    ('segment',        'to',            'segment'),
    ('segment',        'from',          'segment'),
    ('segment',        'turns_into',    'segment'),
    ('segment',        'crosses',       'segment'),
    ('segment',        'crossed_by',    'segment'),
    ('segment',        'merges_with',   'segment'),
    ('segment',        'merged_by',     'segment'),
    ('segment',        'merges_into',   'segment'),
    ('segment',        'intersects_with', 'segment'),
    # Static controller ↔ segment
    ('controller',     'controls',      'segment'),
    ('segment',        'controlled_by', 'controller'),
]

PNEUMA_METADATA: Tuple[List[str], List[Tuple]] = (PNEUMA_NODE_TYPES, PNEUMA_EDGE_TYPES)

# ---------------------------------------------------------------------------
# Custom message-passing order
# ---------------------------------------------------------------------------

PNEUMA_CUSTOM_ORDER: List[List[Tuple[str, str, str]]] = [
    # Group 1: Inject temporal history into current nodes (run in parallel)
    [
        ('vehicle_memory', 'prev_state',  'vehicle'),
        ('segment_metric', 'prev_state',  'segment'),
    ],
    # Group 2: Vehicle ↔ segment interaction using freshly updated representations
    [
        ('vehicle',  'on',          'segment'),
        ('segment',  'occupied_by', 'vehicle'),
    ],
    # Group 3: Road-network topology propagation (all 9 types in parallel)
    [
        ('segment', 'to',              'segment'),
        ('segment', 'from',            'segment'),
        ('segment', 'turns_into',      'segment'),
        ('segment', 'crosses',         'segment'),
        ('segment', 'crossed_by',      'segment'),
        ('segment', 'merges_with',     'segment'),
        ('segment', 'merged_by',       'segment'),
        ('segment', 'merges_into',     'segment'),
        ('segment', 'intersects_with', 'segment'),
    ],
    # Group 4: Controller influence on segments (parallel)
    [
        ('controller', 'controls',      'segment'),
        ('segment',    'controlled_by', 'controller'),
    ],
]


# ---------------------------------------------------------------------------
# PneumaModel
# ---------------------------------------------------------------------------

class PneumaModel(torch.nn.Module):
    """
    Self-supervised PNEUMA model for temporal graph representation learning.

    Consists of:
      - HGT encoder producing embeddings for all 5 node types
      - vehicle_head: Linear(hidden_channels → NUM_VEHICLE_FEATURES)
      - segment_head: Linear(hidden_channels → 1)

    The combined loss is:
        total_loss = lambda_v * vehicle_mse + lambda_s * segment_mse

    Parameters
    ----------
    hidden_channels :
        Embedding dimensionality. Must be divisible by num_heads.
    num_heads :
        Number of attention heads in each HGTConv layer.
    num_layers :
        Number of stacked HGTConv layers.
    dropout :
        Dropout probability applied after each HGTConv layer.
    lambda_v :
        Loss weight for the vehicle prediction head.
    lambda_s :
        Loss weight for the segment prediction head.
    prev_norm :
        Whether to apply LayerNorm after non-final HGT layers.
    last_norm :
        Whether to apply LayerNorm after the final HGT layer.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        lambda_v: float = 1.0,
        lambda_s: float = 0.5,
        prev_norm: bool = True,
        last_norm: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s

        in_channels_dict = {
            'vehicle': NUM_VEHICLE_FEATURES,
            'vehicle_memory': NUM_VEHICLE_FEATURES,
            'segment': NUM_SEGMENT_FEATURES,
            'segment_metric': NUM_SEGMENT_METRIC_FEATURES,
            'controller': NUM_CONTROLLER_FEATURES,
        }

        self.encoder = HGT(
            in_channels=in_channels_dict,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            metadata=PNEUMA_METADATA,
            dropout=dropout,
            prev_norm=prev_norm,
            last_norm=last_norm,
        )

        # Prediction heads (lazy: accept any hidden_channels via -1 not applicable here,
        # so we use the explicit hidden_channels)
        self.vehicle_head = Linear(hidden_channels, NUM_VEHICLE_FEATURES)
        self.segment_head = Linear(hidden_channels, 1)

    def forward(
        self,
        data: HeteroData,
        custom_order: Optional[List] = None,
    ) -> Dict[str, Tensor]:
        """
        Run the HGT encoder and produce predictions for vehicle and segment nodes.

        Parameters
        ----------
        data :
            HeteroData snapshot containing node features, edge indices, and edge_time
            attributes on the two temporal memory edge types.
        custom_order :
            Grouped sequential message-passing order. Defaults to PNEUMA_CUSTOM_ORDER.

        Returns
        -------
        dict with keys 'vehicle' [N_veh, 12] and 'segment' [N_seg, 1].
        """
        if custom_order is None:
            custom_order = PNEUMA_CUSTOM_ORDER

        # Build edge_time_dict from the two temporal edge types
        edge_time_dict: Dict = {}
        for etype in [
            ('vehicle_memory', 'prev_state', 'vehicle'),
            ('segment_metric', 'prev_state', 'segment'),
        ]:
            edge_store = data[etype[0], etype[1], etype[2]]
            if hasattr(edge_store, 'edge_time') and edge_store.edge_time is not None:
                edge_time_dict[etype] = edge_store.edge_time

        # Build x_dict and edge_index_dict, filtering absent node/edge types
        x_dict = {
            ntype: data[ntype].x
            for ntype in data.node_types
            if data[ntype].x is not None and data[ntype].x.shape[0] > 0
        }
        edge_index_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in data.edge_types
            if data[et[0], et[1], et[2]].edge_index.shape[1] > 0
        }

        # Filter custom_order to only include edge types present in edge_index_dict
        present_etypes = set(edge_index_dict.keys())
        filtered_order = [
            [et for et in group if et in present_etypes]
            for group in custom_order
        ]
        # Drop empty groups
        filtered_order = [g for g in filtered_order if g]

        # HGT encoder
        emb_dict = self.encoder(
            x_dict,
            edge_index_dict,
            edge_time_dict=edge_time_dict if edge_time_dict else None,
            custom_order=filtered_order if filtered_order else None,
        )

        preds: Dict[str, Tensor] = {}

        if 'vehicle' in emb_dict and emb_dict['vehicle'] is not None:
            preds['vehicle'] = self.vehicle_head(emb_dict['vehicle'])

        if 'segment' in emb_dict and emb_dict['segment'] is not None:
            preds['segment'] = self.segment_head(emb_dict['segment'])

        return preds

    def compute_loss(
        self,
        preds: Dict[str, Tensor],
        targets: Dict,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute the combined autoregressive loss and auxiliary metrics (MAE, RMSE).
        """
        metrics: Dict[str, float] = {}
        device = next(self.parameters()).device

        # --- Vehicle metrics ---
        vehicle_mse = torch.tensor(0.0, device=device)
        vehicle_mae = torch.tensor(0.0, device=device)
        if 'vehicle' in preds:
            v_mask = targets['vehicle_mask'].to(device)
            if v_mask.sum() > 0:
                v_pred = preds['vehicle'][v_mask]
                v_tgt = targets['vehicle_feats'].to(device)[v_mask]
                vehicle_mse = F.mse_loss(v_pred, v_tgt)
                vehicle_mae = F.l1_loss(v_pred, v_tgt)
        
        metrics['vehicle_loss'] = float(vehicle_mse)
        metrics['vehicle_mae']  = float(vehicle_mae)
        metrics['vehicle_rmse'] = float(torch.sqrt(vehicle_mse))

        # --- Segment metrics ---
        segment_mse = torch.tensor(0.0, device=device)
        segment_mae = torch.tensor(0.0, device=device)
        if 'segment' in preds:
            seg_idx = targets['segment_mask_idx'].to(device)
            if seg_idx.shape[0] > 0:
                s_pred = preds['segment'][seg_idx]           # [N_upd, 1]
                s_tgt = targets['segment_feats'].to(device)  # [N_upd, 1]
                segment_mse = F.mse_loss(s_pred, s_tgt)
                segment_mae = F.l1_loss(s_pred, s_tgt)
        
        metrics['segment_loss'] = float(segment_mse)
        metrics['segment_mae']  = float(segment_mae)
        metrics['segment_rmse'] = float(torch.sqrt(segment_mse))

        total_loss = self.lambda_v * vehicle_mse + self.lambda_s * segment_mse
        return total_loss, metrics
