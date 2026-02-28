import pytorch_lightning as pl
import torch as th
import torch.nn as nn

# import pytorch_lightning.metrics.functional as metrics

from models.modules import Flatten

import torchvision


class ResNetWrapper(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNetWrapper, self).__init__()
        self.save_hyperparameters()

        resnet = torchvision.models.resnet18(pretrained=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool

        self.ff_head = nn.Linear(512, num_classes)

    def load_weights_from_vae(self, vae):
        self.encoder.load_state_dict(vae.encoder.encoder.state_dict(), strict=False)
        self.encoder_mu.load_state_dict(
            vae.encoder.encoder_mu.state_dict(), strict=False
        )

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
        logits = self.ff_head(self._encode_impl(x))
        return logits

    def _encode_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = th.flatten(x, 1)
        return x

    def encode_image(self, x):
        return self.encoder(x)
