import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import ModuleDict, ModuleList, LayerNorm, Dropout

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.basis_freq = Parameter(1 / 10 ** torch.linspace(0, 9, expand_dim))
        self.phase = Parameter(torch.zeros(expand_dim))

    def forward(self, ts):
        ts = ts.view(-1, 1)
        map_ts = ts * self.basis_freq + self.phase
        return torch.cos(map_ts)

class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        self.k_rel = ModuleDict()
        self.v_rel = ModuleDict()
        self.p_rel = ParameterDict()

        for edge_type in self.edge_types:
            edge_type_str = '__'.join(edge_type)
            self.k_rel[edge_type_str] = Linear(self.out_channels, self.out_channels, bias=False)
            self.v_rel[edge_type_str] = Linear(self.out_channels, self.out_channels, bias=False)
            self.p_rel[edge_type_str] = Parameter(torch.empty(1, heads))

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.time_encoder = TimeEncode(self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        for key in self.k_rel.keys():
            self.k_rel[key].reset_parameters()
            self.v_rel[key].reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        edge_time_dict: Optional[Dict[EdgeType, Tensor]] = None
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict = {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k
            q_dict[key] = q
            v_dict[key] = v

        out_dict = {node_type: [] for node_type in self.dst_node_types}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type_str = '__'.join(edge_type)
            edge_time = edge_time_dict.get(edge_type) if edge_time_dict is not None else None
            
            out = self.propagate(
                edge_index,
                q=q_dict[dst_type],
                k=k_dict[src_type],
                v=v_dict[src_type],
                edge_time=edge_time,
                etype_str=edge_type_str,
                size=(k_dict[src_type].size(0), q_dict[dst_type].size(0))
            )
            out_dict[dst_type].append(out)

        agg_out_dict = {}
        for node_type, outs in out_dict.items():
            if len(outs) > 0:
                agg_out_dict[node_type] = sum(outs)
            elif node_type in q_dict:
                agg_out_dict[node_type] = torch.zeros(q_dict[node_type].size(0), F, device=q_dict[node_type].device)

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k: torch.nn.functional.gelu(v)
            for k, v in agg_out_dict.items()
        })

        # Iterate over node types:
        res_dict = {}
        for node_type in x_dict.keys():
            if node_type in a_dict:
                out = a_dict[node_type]
                if out.size(-1) == x_dict[node_type].size(-1):
                    alpha = self.skip[node_type].sigmoid()
                    out = alpha * out + (1 - alpha) * x_dict[node_type]
                res_dict[node_type] = out
            else:
                res_dict[node_type] = x_dict[node_type]

        return res_dict

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, edge_time: Optional[Tensor],
                etype_str: str, index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        if edge_time is not None:
            t_emb = self.time_encoder(edge_time)
            k_j = k_j + t_emb
            v_j = v_j + t_emb

        k_j = self.k_rel[etype_str](k_j)
        v_j = self.v_rel[etype_str](v_j)

        H = self.heads
        D = self.out_channels // H

        q_i = q_i.view(-1, H, D)
        k_j = k_j.view(-1, H, D)
        v_j = v_j.view(-1, H, D)

        edge_attr = self.p_rel[etype_str]

        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, metadata,
                 dropout=0.2, prev_norm=True, last_norm=True, in_channels: Union[int, Dict[str, int]] = -1):
        super().__init__()
        self.node_types = metadata[0]
        self.num_layers = num_layers

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in self.node_types}

        # Initial Adaptation Layers
        self.lin_dict = ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(in_channels[node_type], hidden_channels)

        self.drop = Dropout(dropout)

        # Convolutional Layers
        self.convs = ModuleList()
        self.layer_norms = ModuleList()
        
        for i in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
            
            # Determine if this layer should use normalization
            is_last = (i == num_layers - 1)
            use_layer_norm = last_norm if is_last else prev_norm
            
            if use_layer_norm:
                norm_dict = ModuleDict({
                    node_type: LayerNorm(hidden_channels)
                    for node_type in self.node_types
                })
                self.layer_norms.append(norm_dict)
            else:
                self.layer_norms.append(None)

    def forward(self, x_dict, edge_index_dict, edge_time_dict=None, custom_order=None):
        # 1. Initial Projection & Activation
        x_dict = {
            node_type: self.drop(torch.tanh(self.lin_dict[node_type](x)))
            for node_type, x in x_dict.items()
        }

        # 2. Message Passing Layers
        for i, conv in enumerate(self.convs):
            if custom_order is not None:
                # Custom Grouped Sequential Message Passing
                for group in custom_order:
                    subset_edges = {etype: edge_index_dict[etype] for etype in group}
                    subset_times = {etype: edge_time_dict.get(etype) for etype in group} if edge_time_dict is not None else None
                    
                    # conv returns all nodes (pass-through), but we only want to update dst_types in this group
                    new_nodes = conv(x_dict, subset_edges, subset_times)
                    
                    dst_types = {etype[-1] for etype in group}
                    for node_type in dst_types:
                        if new_nodes.get(node_type) is not None:
                            x_dict[node_type] = new_nodes[node_type]
            else:
                # Default parallel message passing
                x_dict = conv(x_dict, edge_index_dict, edge_time_dict)

            # 3. Post-Conv Normalization & Dropout
            for node_type in x_dict:
                if x_dict[node_type] is not None:
                    x_dict[node_type] = self.drop(x_dict[node_type])
                    if self.layer_norms[i] is not None:
                        x_dict[node_type] = self.layer_norms[i][node_type](x_dict[node_type])

        return x_dict
