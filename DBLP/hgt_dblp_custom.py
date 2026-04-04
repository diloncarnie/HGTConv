import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ModuleDict, ModuleList, LayerNorm, Dropout

import torch_geometric
from torch_geometric.datasets import DBLP
from hgt_conv import HGTConv
from torch_geometric.nn import Linear

path = osp.join(osp.dirname(osp.realpath(__file__)), 'DBLP-dataset')
# We initialize conference node features with a single one-vector as feature:
dataset = DBLP(path)
data = dataset[0]
if ('x' not in data['conference']) or (data['conference'].x is None):
    data['conference'].x = torch.ones((data['conference'].num_nodes, 1))


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, 
                 dropout=0.2, prev_norm=True, last_norm=True):
        super().__init__()
        self.node_types = data.node_types
        self.num_layers = num_layers

        # Initial Adaptation Layers (matching pyHGT GNN adapt_ws)
        self.lin_dict = ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.drop = Dropout(dropout)

        # Convolutional Layers
        self.convs = ModuleList()
        self.layer_norms = ModuleList()
        
        for i in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)
            
            # Determine if this layer should use normalization
            is_last = (i == num_layers - 1)
            use_layer_norm = last_norm if is_last else prev_norm
            
            if use_layer_norm:
                # One LayerNorm per node type per layer (matching pyHGT HGTConv norms)
                norm_dict = ModuleDict({
                    node_type: LayerNorm(hidden_channels)
                    for node_type in self.node_types
                })
                self.layer_norms.append(norm_dict)
            else:
                self.layer_norms.append(None)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, custom_order=None):
        # 1. Initial Projection & Activation
        x_dict = {
            node_type: self.drop(torch.tanh(self.lin_dict[node_type](x)))
            for node_type, x in x_dict.items()
        }

        # 2. Message Passing Layers
        for i, conv in enumerate(self.convs):
            if custom_order is not None:
                # Custom Grouped Sequential Message Passing
                # custom_order is a List of Lists: [[R1, R2], [R3]]
                for group in custom_order:
                    # Filter only edges in this group
                    subset_edges = {etype: edge_index_dict[etype] for etype in group}
                    
                    # Compute messages for the whole group in one parallel pass
                    new_nodes = conv(x_dict, subset_edges)
                    
                    # Update ONLY the destination nodes involved in this group
                    dst_types = {etype[-1] for etype in group}
                    for node_type in dst_types:
                        if new_nodes.get(node_type) is not None:
                            x_dict[node_type] = new_nodes[node_type]
            else:
                # Default parallel message passing
                x_dict = conv(x_dict, edge_index_dict)

            # 3. Post-Conv Normalization & Dropout
            for node_type in x_dict:
                if x_dict[node_type] is not None:
                    x_dict[node_type] = self.drop(x_dict[node_type])
                    if self.layer_norms[i] is not None:
                        x_dict[node_type] = self.layer_norms[i][node_type](x_dict[node_type])

        return self.lin(x_dict['author'])


# Define a GROUPED specific order for DBLP relations
# Grouping independent relations maximizes GPU parallel utilization.
custom_order = [
    [('paper', 'to', 'conference')],
    # Group 1: Papers are updated by all their attributes at once
    [('conference', 'to', 'paper'), ('term', 'to', 'paper'), ('author', 'to', 'paper')],
    # Group 2: Authors are updated by the newly enriched papers
    [('paper', 'to', 'author')]
]
# Use default parallel message passing without grouping
# custom_order = None

epochs = 50
lr = 0.001
weight_decay = 0.001
model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=3, 
            dropout=0.4, prev_norm=True, last_norm=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')
data, model = data.to(device), model.to(device)


with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict, custom_order=custom_order)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict, custom_order=custom_order)
    mask = data['author'].train_mask
    loss = F.cross_entropy(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, custom_order=custom_order).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


best_val_acc = 0
best_test_acc = 0
best_epoch = 0
best_loss = 0

for epoch in range(0, epochs):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_epoch = epoch
        best_loss = loss

print(f'\nBest Performance at Epoch {best_epoch:03d}:')
print(f'Loss: {best_loss:.4f}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}')
