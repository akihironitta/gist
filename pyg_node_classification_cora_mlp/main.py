import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)
    model = MLP(
        dataset.num_features,
        dataset.num_classes,
        hidden_channels=16,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
    )

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = model(data.x)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(loss)

    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    print(test_acc)


if __name__ == "__main__":
    main()
