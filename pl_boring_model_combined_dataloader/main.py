from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
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
        loss = self(batch["a"]).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch["a"]).sum()
        self.log("valid_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


class RandomDataModule(LightningDataModule):
    def __init__(
        self, mode: str = "min_size", len_a: int = 10, len_b: int = 6
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        # Training data has 10 batches in max_size_cycle model, and 6 in
        # min_size mode. Training should run oder 5 resp. 3 steps because we
        # have 2 GPUs. In fact, training runs over 10 resp. 6 steps, and each
        # GPU sees all the data.
        return CombinedLoader(
            {
                "a": DataLoader(
                    RandomDataset(32, self.hparams.len_a), batch_size=1
                ),
                "b": DataLoader(
                    RandomDataset(32, self.hparams.len_b), batch_size=1
                ),
            },
            mode=self.hparams.mode,
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                "a": DataLoader(
                    RandomDataset(32, self.hparams.len_a), batch_size=1
                ),
                "b": DataLoader(
                    RandomDataset(32, self.hparams.len_b), batch_size=1
                ),
            },
            mode=self.hparams.mode,
        )


def main():
    model = BoringModel()
    dm = RandomDataModule()
    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
