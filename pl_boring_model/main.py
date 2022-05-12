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
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def main():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model0 = BoringModel()
    trainer0 = Trainer(max_epochs=1)
    trainer0.fit(
        model0, train_dataloaders=train_data, val_dataloaders=val_data
    )
    checkpoint_path = "example.ckpt"
    trainer0.save_checkpoint(checkpoint_path)

    trainer1 = Trainer(max_epochs=1)
    model1 = BoringModel()
    trainer1.fit(
        model1,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
        ckpt_path=checkpoint_path,
    )
    old_state = trainer0.optimizers[0].state
    new_state = trainer1.optimizers[0].state
    print(old_state, new_state)


if __name__ == "__main__":
    main()
