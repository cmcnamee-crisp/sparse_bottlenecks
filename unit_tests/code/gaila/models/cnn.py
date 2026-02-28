import pytorch_lightning as pl
import torch as th
import torch.nn as nn

# import pytorch_lightning.metrics.functional as metrics

from models.modules import Flatten


class CNN(pl.LightningModule):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            Flatten(),
            nn.Linear(1352, 512),
            nn.ReLU(),
        )
        self.ff_head = nn.Linear(512, num_classes)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        # acc = metrics.accuracy(logits, y)
        acc = (logits.squeeze().argmax(-1) == y).float().mean()
        return loss, {"loss": loss.item(), "acc": acc.item()}

    def forward(self, x):
        if isinstance(x, list):
            x, _ = x
        x = self.encoder(x)
        x = self.ff_head(x)
        return x  # logits

    def encode_image(self, x):
        return self.encoder(x)
