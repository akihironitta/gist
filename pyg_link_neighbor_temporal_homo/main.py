import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

data = Data(
    x=torch.ones(10, 7),
    edge_index=torch.randint(0, 10, (2, 123)),
    edge_attr=torch.randn(123, 3),
    edge_time=torch.arange(123),
)
print(f"data\t{data}")
loader = LinkNeighborLoader(
    data,
    num_neighbors=[-1],
    edge_label=torch.ones(data.num_edges),
    time_attr="edge_time",
    edge_label_time=data.edge_time,
    batch_size=13,
    shuffle=False,
)
batch = next(iter(loader))
print(f"batch\t{batch}")
size = batch.edge_label_index.size()
assert size == (2, 13), f"{size} should be [2, batch_size]"
