# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/domain_templates/generative_adversarial_net.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
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
    def __init__(
        self,
        batch_size: int = 32,
        data_dir: str = "./data",
        num_workers: int = 0,
    ):
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
                    transform=self.transform,
                ),
                [55000, 5000],
            )

        if stage in (None, "test"):
            self.test_dataset = MNIST(
                self.hparams.data_dir,
                train=False,
                transform=self.transform,
            )

        if stage in (None, "predict"):
            self.predict_dataset = MNIST(
                self.hparams.data_dir,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )


class GAN(LightningModule):
    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        learning_rate: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)
        self.validation_z = torch.randn(8, latent_dim)
        self.example_input_array = torch.zeros(2, latent_dim)

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        if optimizer_idx == 0:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            self.log_dict(tqdm_dict)
            return g_loss

        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake
            )
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            self.log_dict(tqdm_dict)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        if self.logger:
            self.logger.experiment.add_image(
                "generated_images", grid, self.current_epoch
            )


def main():
    model = GAN()
    dm = MNISTDataModule()
    trainer = Trainer(
        max_epochs=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
