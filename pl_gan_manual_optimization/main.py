# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/docs/source/common/optimizers.rst#use-multiple-optimizers-like-gans
# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/domain_templates/generative_adversarial_net.py
import numpy as np
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as T


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
    def __init__(self, batch_size=64, data_dir="./data"):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

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
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.hparams.batch_size
        )


class GAN(LightningModule):
    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        latent_dim: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)
        self.validation_z = torch.randn(8, latent_dim)
        self.example_input_array = torch.zeros(2, latent_dim)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        x, _ = batch
        batch_size = x.size(0)
        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)
        z = torch.randn(
            batch_size, self.hparams.latent_dim, device=self.device
        )
        g_x = self.generator(z)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.discriminator(x)
        d_loss_real = F.binary_cross_entropy_with_logits(d_x, real_label)

        d_z = self.discriminator(g_x.detach())
        d_loss_fake = F.binary_cross_entropy_with_logits(d_z, fake_label)

        d_loss = d_loss_real + d_loss_fake

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.discriminator(g_x)
        g_loss = F.binary_cross_entropy_with_logits(d_z, real_label)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("loss/generator", g_loss, prog_bar=True)
        self.log("loss/discriminator", d_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        opt_d = Adam(
            self.discriminator.parameters(), lr=0.002, betas=(0.5, 0.999)
        )
        return [opt_g, opt_d]

    def on_epoch_end(self):
        if self.logger:
            z = self.validation_z.type_as(self.generator.model[0].weight)
            sample_imgs = self(z)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(
                "generated_images", grid, self.current_epoch
            )


def main():
    model = GAN()
    dm = MNISTDataModule()
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        # enable_progress_bar=False,
        # enable_model_summary=False,
        # enable_checkpointing=False,
        # logger=False,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
