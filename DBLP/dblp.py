import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.datasets import DBLP
from hgt_model import HGT
from torch_geometric.nn import Linear

path = osp.join(osp.dirname(osp.realpath(__file__)), 'DBLP-dataset')
# We initialize conference node features with a single one-vector as feature:
dataset = DBLP(path)
data = dataset[0]
if ('x' not in data['conference']) or (data['conference'].x is None):
    data['conference'].x = torch.ones((data['conference'].num_nodes, 1))


class DBLPClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata,
                 dropout=0.4, prev_norm=True, last_norm=False):
        super().__init__()
        self.encoder = HGT(hidden_channels, num_heads, num_layers, metadata, 
                           dropout, prev_norm, last_norm)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, custom_order=None):
        out_dict = self.encoder(x_dict, edge_index_dict, custom_order=custom_order)
        return self.lin(out_dict['author'])


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
model = DBLPClassifier(hidden_channels=64, out_channels=4, num_heads=2, num_layers=3, 
                       metadata=data.metadata(), dropout=0.4, prev_norm=True, last_norm=True)

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
