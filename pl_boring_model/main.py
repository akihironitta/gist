# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/bug_report/bug_report_model.py
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return self(batch).sum()

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.layer.parameters(), lr=torch.tensor(0.1))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        optimizer.step = torch.compile(optimizer.step)
        return [optimizer], [lr_scheduler]


def main():
    train_data = DataLoader(RandomDataset(32, 16), batch_size=2)
    model = BoringModel()
    trainer = Trainer(
        max_epochs=5,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_data)


if __name__ == "__main__":
    main()
