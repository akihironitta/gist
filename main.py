import torch
torch._dynamo.config.capture_dynamic_output_shape_ops = True

import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as conv
from torch_geometric.datasets import FakeDataset


class GNN(torch.nn.Module):
    def __init__(self, features, classes, hidden_width, layers):
        super().__init__()

        self.layers = torch.nn.ModuleList([conv(features, hidden_width, heads=1)])
        for i in range(layers-2):
            self.layers.append(conv(-1, hidden_width, heads=1))
        self.layers.append(conv(-1, classes))
        self.act = F.gelu

    def forward(self, x, edge_index):

        for l in self.layers[:-1]:
            x = l(x, edge_index)
            x = self.act(x)
        x = self.layers[-1](x, edge_index)

        return x


if __name__ == "__main__":

    num_channels = 2
    num_classes = 2

    model = GNN(num_channels, num_classes, 4, 4)
    model = torch.compile(model, dynamic=True, fullgraph=True)

    dataset = FakeDataset(num_channels=num_channels, num_classes=num_classes, task="node")

    for data in dataset:
        out = model(data.x, data.edge_index)
        print(out)