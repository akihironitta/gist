# https://pytorch-geometric.readthedocs.io/en/2.3.1/get_started/introduction.html
# https://pytorch-geometric.readthedocs.io/en/2.3.1/generated/torch_geometric.datasets.FakeDataset.html#torch_geometric.datasets.FakeDataset
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
        avg_num_nodes=1000,
        avg_degree=10,
        num_channels=64,
        edge_dim=0,
        num_classes=10,
        task="auto",  # "node" | "graph"
        transform=None,
        pre_transform=None,
    )
    data = dataset[0]
    model = GNN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
    )
    with torch.no_grad():  # lazy tensor initialisation
        model(data.x, data.edge_index)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )
    device = torch.device("cpu")
    data.to(device)
    model.to(device)
    for _ in range(10):
        model.train()
        optimizer.zero_grad()
        x_out = model(data.x, data.edge_index)  # (num_nodes, num_classes)
        loss = torch.nn.functional.nll_loss(x_out, data.y)
        loss.backward()
        optimizer.step()
        print(loss)

    model.eval()
    print(torch.argmax(x_out, dim=1))  # (num_nodes,)


if __name__ == "__main__":
    main()
