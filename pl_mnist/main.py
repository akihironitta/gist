# https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/mnist/main.py
# https://github.com/PyTorchLightning/pytorch-lightning/tree/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/basic_examples/mnist_examples
import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision.datasets import MNIST


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ImageClassifier(LightningModule):
    def __init__(self, model, lr=1.0, gamma=0.7):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model or Net()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=32, data_dir="./data"):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        MNIST(self.hparams.data_dir, download=True, train=True)
        MNIST(self.hparams.data_dir, download=True, train=False)

    def setup(self, stage):
        if stage in (None, "fit"):
            self.train_dataset, self.val_dataset = random_split(
                MNIST(
                    self.hparams.data_dir,
                    train=True,
                    download=True,
                    transform=self.transform,
                ),
                [55000, 5000],
            )

        if stage in (None, "test"):
            self.test_dataset = MNIST(
                self.hparams.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

        if stage in (None, "predict"):
            self.predict_dataset = MNIST(
                self.hparams.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size)


def main():
    model = ImageClassifier()
    dm = MNISTDataModule()
    trainer = Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)
    trainer.predict(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()
