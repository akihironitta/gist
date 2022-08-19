# https://github.com/Lightning-AI/lightning/blob/bbb406bce9493aeb104adb6aa40b9623201c2a29/examples/convert_from_pt_to_pl/image_classifier_2_lite.py
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy

from pytorch_lightning import seed_everything
from pytorch_lightning.demos.boring_classes import Net
from pytorch_lightning.demos.mnist_datamodule import MNIST
from pytorch_lightning.lite import LightningLite  # import LightningLite

DATASETS_PATH = "~/data/"


class Lite(LightningLite):
    def run(self, hparams):
        self.hparams = hparams
        seed_everything(hparams.seed)

        transform = T.Compose(
            [T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]
        )

        if self.is_global_zero:
            MNIST(DATASETS_PATH, download=True)
        self.barrier()
        train_dataset = MNIST(DATASETS_PATH, train=True, transform=transform)
        test_dataset = MNIST(DATASETS_PATH, train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=hparams.batch_size,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=hparams.batch_size
        )

        train_loader, test_loader = self.setup_dataloaders(
            train_loader, test_loader
        )

        model = Net()
        optimizer = optim.Adadelta(model.parameters(), lr=hparams.lr)
        model, optimizer = self.setup(model, optimizer)

        scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

        test_acc = Accuracy().to(self.device)

        for epoch in range(1, hparams.epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                self.backward(loss)  # instead of loss.backward()

                optimizer.step()
                if (batch_idx == 0) or (
                    (batch_idx + 1) % hparams.log_interval == 0
                ):
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
                    if hparams.dry_run:
                        break

            scheduler.step()

            # TESTING LOOP
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    # NOTE: no need to call `.to(device)` on the data, target
                    output = model(data)
                    test_loss += F.nll_loss(
                        output, target, reduction="sum"
                    ).item()

                    # WITHOUT TorchMetrics
                    # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    # correct += pred.eq(target.view_as(pred)).sum().item()

                    # WITH TorchMetrics
                    test_acc(output, target)

                    if hparams.dry_run:
                        break

            # all_gather is used to aggregated the value across processes
            test_loss = self.all_gather(test_loss).sum() / len(
                test_loader.dataset
            )

            print(
                f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({100 * test_acc.compute():.0f}%)\n"
            )
            test_acc.reset()

            if hparams.dry_run:
                break

        # When using distributed training, use `self.save`
        # to ensure the current process is allowed to save a checkpoint
        if hparams.save_model:
            self.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningLite MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    hparams = parser.parse_args()

    Lite(accelerator="auto", devices="auto").run(hparams)
