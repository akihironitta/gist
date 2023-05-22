# https://pytorch-geometric.readthedocs.io/en/2.3.1/generated/torch_geometric.datasets.FakeDataset.html#torch_geometric.datasets.FakeDataset
# https://pytorch-geometric.readthedocs.io/en/2.3.1/modules/loader.html#torch_geometric.loader.NeighborLoader
# https://arxiv.org/abs/1706.02216
import torch
import torch_geometric


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)


def main():
    dataset = torch_geometric.datasets.FakeDataset(
        num_graphs=1,
        avg_num_nodes=10000,
        avg_degree=10,
        num_channels=64,
        edge_dim=0,
        num_classes=10,
        task="node",
        transform=None,
        pre_transform=None,
    )
    loader = torch_geometric.loader.NeighborLoader(
        dataset[0],
        # sample 4 neighbourhood nodes for each node for 2 iterations
        num_neighbors=[4] * 2,
        batch_size=128,
        # None means all nodes
        input_nodes=None,
    )
    model = GNN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
    )
    data = next(iter(loader))
    with torch.no_grad():  # lazy tensor initialisation
        model(data.x, data.edge_index)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )
    device = torch.device("cpu")
    model = model.to(device)
    for _ in range(10):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            optimizer.zero_grad()
            x_out = model(batch.x, batch.edge_index)
            # slicing `[:batch_size]` is necessary since we're only interested in calculating
            # the loss across the sampled `batch_size` nodes.
            loss = torch.nn.functional.nll_loss(
                x_out[:batch_size],
                batch.y[:batch_size],
            )
            loss.backward()
            optimizer.step()
            print(loss)


if __name__ == "__main__":
    main()
