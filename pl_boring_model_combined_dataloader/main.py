import os

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
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


len_a = 10
len_b = 6
mode = "min_size"


class MyData(LightningDataModule):
    def train_dataloader(self):
        # Training data has 10 batches in max_size_cycle model, and 6 in min_size mode.
        # Training should run oder 5 resp. 3 steps because we have 2 GPUs.
        # In fact, training runs over 10 resp. 6 steps, and each GPU sees all the data.
        return CombinedLoader(
            {
                "a": DataLoader(RandomDataset(32, len_a), batch_size=1),
                "b": DataLoader(RandomDataset(32, len_b), batch_size=1),
            },
            mode=mode,
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                "a": DataLoader(RandomDataset(32, len_a), batch_size=1),
                "b": DataLoader(RandomDataset(32, len_b), batch_size=1),
            },
            mode=mode,
        )


def run():

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=MyData())


if __name__ == "__main__":
    run()
