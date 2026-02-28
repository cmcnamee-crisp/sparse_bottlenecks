import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import numpy as np

# import pytorch_lightning.metrics.functional as metrics

from models.modules import Flatten


class Linear(pl.LightningModule):
    def __init__(self, num_features: int, num_output: int):
        super(Linear, self).__init__()
        self.save_hyperparameters()
        self.flatten = Flatten()
        self.linear = nn.Linear(num_features, num_output, bias=False)
        self.projected_linear = None

    def load_projection(self, projection):
        linear = nn.Linear(
            self.hparams.num_features, self.hparams.num_output, bias=False
        )
        with th.no_grad():
            linear.weight = nn.Parameter(self.linear.weight @ projection.float())
        self.projected_linear = linear

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-2)
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
        _, logs = self.step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return {f"test_{k}": v for k, v in logs.items()}

    def test_epoch_end(self, outputs):
        test_loss = sum([x["test_loss"] for x in outputs])
        test_acc = np.mean([x["test_acc"] for x in outputs])
        return {"test_loss": test_loss, "test_acc": test_acc}

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        acc = (logits.squeeze().argmax(-1) == y).float().mean()
        return loss, {"loss": loss.item(), "acc": acc.item()}

    def forward(self, x):
        if isinstance(x, list):
            x, _ = x
        if self.projected_linear is not None:
            return self.projected_linear(self.flatten(x.float()))
        return self.linear(self.flatten(x.float()))
