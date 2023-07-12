"""
Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
https://arxiv.org/abs/1905.07953
"""
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.transforms import NormalizeFeatures


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=8):
        super().__init__()
        self.conv1 = GATConv(
            num_features, hidden_channels, heads=8, dropout=0.6
        )
        self.conv2 = GATConv(
            hidden_channels * 8, num_classes, heads=1, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    dataset = Planetoid(
        root="data/PubMed",
        name="PubMed",
        transform=NormalizeFeatures(),
    )
    data = dataset[0]  # on CPU
    cluster_data = ClusterData(data, num_parts=128)
    loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
    # model = GCN(
    #     dataset.num_features,
    #     dataset.num_classes,
    # )
    model = GAT(dataset.num_features, dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
    )

    model.train()
    for _ in range(50):
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            print(loss)

    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    print(test_acc)


if __name__ == "__main__":
    main()
