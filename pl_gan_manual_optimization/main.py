# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/docs/source/common/optimizers.rst#use-multiple-optimizers-like-gans
# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/domain_templates/generative_adversarial_net.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torchvision.datasets import MNIST


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.img_shape = img_shape
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


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


class GAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()
        self.automatic_optimization = False

    def sample_z(self, n):
        sample = self._Z.sample((n,))
        return sample

    def sample_G(self, n):
        z = self.sample_z(n)
        return self.G(z)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        X, _ = batch
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.D(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.D(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)
        errD = errD_real + errD_fake

        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.D(g_X)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

    def configure_optimizers(self):
        g_opt = Adam(self.G.parameters(), lr=1e-5)
        d_opt = Adam(self.D.parameters(), lr=1e-5)
        return g_opt, d_opt


def main():
    model = GAN()
    dm = MNISTDataModule()
    trainer = Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
