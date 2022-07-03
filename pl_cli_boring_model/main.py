# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/bug_report/bug_report_model.py
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI
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

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)


def main():

    trainer_defaults = dict(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        benchmark=False,  # True by default in 1.6.{0-3}.
    )
    cli = LightningCLI(
        BoringModel,
        trainer_defaults=trainer_defaults,
        save_config_callback=None,
        run=False,
    )
    cli.trainer.fit(cli.model)


if __name__ == "__main__":
    main()
