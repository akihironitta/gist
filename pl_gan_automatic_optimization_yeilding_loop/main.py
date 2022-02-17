# https://github.com/PyTorchLightning/pytorch-lightning/blob/4dba492fb55e2862bb20f304e6bef43f1f151fc7/pl_examples/loop_examples/yielding_training_step.py
from functools import partial
import inspect

import numpy as np
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loops import OptimizerLoop
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
from pytorch_lightning.loops.utilities import _build_training_step_kwargs
from pytorch_lightning.utilities.exceptions import MisconfigurationException
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
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "./data",
        num_workers: int = 0,
    ):
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


class YieldLoop(OptimizerLoop):
    def __init__(self):
        super().__init__()
        self._generator = None

    def connect(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not connect any child loops."
        )

    def on_run_start(self, batch, optimizers, batch_idx):
        super().on_run_start(batch, optimizers, batch_idx)
        if not inspect.isgeneratorfunction(
            self.trainer.lightning_module.training_step
        ):
            raise MisconfigurationException(
                "The `LightningModule` does not yield anything in the"
                " `training_step`."
            )
        assert self.trainer.lightning_module.automatic_optimization

        # We request the generator once and save it for later
        # so we can call next() on it.
        self._generator = self._get_generator(batch, batch_idx, opt_idx=0)

    def _make_step_fn(self, split_batch, batch_idx, opt_idx):
        return partial(self._training_step, self._generator)

    def _get_generator(self, split_batch, batch_idx, opt_idx):
        step_kwargs = _build_training_step_kwargs(
            self.trainer.lightning_module,
            self.trainer.optimizers,
            split_batch,
            batch_idx,
            opt_idx,
            hiddens=None,
        )

        # Here we are basically calling `lightning_module.training_step()`
        # and this returns a generator! The `training_step` is handled by the
        # accelerator to enable distributed training.
        return self.trainer.strategy.training_step(*step_kwargs.values())

    def _training_step(self, generator):
        # required for logging
        self.trainer.lightning_module._current_fx_name = "training_step"

        # Here, instead of calling `lightning_module.training_step()`
        # we call next() on the generator!
        training_step_output = next(generator)
        self.trainer.strategy.post_training_step()

        model_output = self.trainer._call_lightning_module_hook(
            "training_step_end", training_step_output
        )
        strategy_output = self.trainer._call_strategy_hook(
            "training_step_end", training_step_output
        )
        training_step_output = (
            strategy_output if model_output is None else model_output
        )

        # The closure result takes care of properly detaching the loss for
        # logging and peforms some additional checks that the output format is
        # correct.
        result = ClosureResult.from_training_step_output(
            training_step_output, self.trainer.accumulate_grad_batches
        )
        return result


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

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, _ = batch
        batch_size = x.size(0)
        z = torch.randn(
            batch_size, self.hparams.latent_dim, device=self.device
        )
        g_x = self(z)

        #############
        # Generator #
        #############
        real_label = torch.ones(batch_size, 1, device=self.device)
        g_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(g_x), real_label
        )
        self.log("loss/generator", g_loss, prog_bar=True)
        yield g_loss

        #################
        # Discriminator #
        #################
        real_label = torch.ones(batch_size, 1, device=self.device)
        real_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(x), real_label
        )
        fake_label = torch.zeros(batch_size, 1, device=self.device)

        # We make use again of the generator_output
        fake_loss = self.adversarial_loss(
            self.discriminator(g_x.detach()), fake_label
        )
        d_loss = (real_loss + fake_loss) / 2
        self.log("loss/discriminator", d_loss, prog_bar=True)
        yield d_loss

    def configure_optimizers(self):
        opt_d = Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        opt_g = Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        return [opt_d, opt_g]

    def on_epoch_end(self):
        if self.logger:
            z = self.validation_z.type_as(self.generator.model[0].weight)
            sample_imgs = self.generator(z)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(
                "generated_images", grid, self.current_epoch
            )


def main():
    model = GAN()
    dm = MNISTDataModule()
    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=YieldLoop())
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
