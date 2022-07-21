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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, **kwargs):
        self.should_skip_lr_scheduler_step = False

        scaler = getattr(self.trainer.strategy.precision_plugin, "scaler", None)
        if scaler:
            scale_before_step = scaler.get_scale()

        optimizer.step(closure=optimizer_closure)

        if scaler:
            scale_after_step = scaler.get_scale()
            self.should_skip_lr_scheduler_step = scale_before_step > scale_after_step

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.should_skip_lr_scheduler_step:
            return

        scheduler.step()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
            "interval": "step",
            "frequency": 1,  # other small numbers may also cause this issue.
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        max_steps=5,
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision=16,
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    main()
