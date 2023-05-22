# https://pytorch-geometric.readthedocs.io/en/2.3.1/tutorial/heterogeneous.html
# https://pytorch-geometric.readthedocs.io/en/2.3.1/generated/torch_geometric.datasets.FakeHeteroDataset.html#torch_geometric.datasets.FakeHeteroDataset
import torch
import torch_geometric


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels)
        self.conv2 = torch_geometric.nn.SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def main():
    dataset = torch_geometric.datasets.FakeHeteroDataset(
        num_graphs=1,
        num_node_types=2,
        num_edge_types=4,
        avg_num_nodes=1000,
        avg_degree=10,
        avg_num_channels=64,
        edge_dim=0,
        num_classes=10,
        task="auto",  # "node" | "graph"
        transform=None,
        pre_transform=None,
    )
    data = dataset[0]
    model = GNN(hidden_channels=64, out_channels=dataset._num_classes)
    model = torch_geometric.nn.to_hetero(model, data.metadata(), aggr="sum")
    with torch.no_grad():  # lazy tensor initialisation
        model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    data.to(device)
    model.to(device)
    for _ in range(10):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = torch.nn.functional.cross_entropy(out["v0"], data["v0"].y)
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == "__main__":
    main()
