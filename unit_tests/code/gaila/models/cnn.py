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

    def forward_from_layer(self, x, layer_name):
        """Passes a tensor `x` (already at `layer_name`) through the rest of the network.

        Handles two input formats:
        - (N, C*H*W): fully flattened (current _format_layer with flatten)
        - (N, H*W): channel-averaged (old _format_layer with mean(dim=1))

        For spatial layers, the input is unflattened to (N, C, H, W). If the flat
        dimension matches H*W instead of C*H*W (channel-averaged data), we expand
        the single averaged map across all C channels as an approximation.
        """
        # Layer shapes: (C, H, W) and their C*H*W and H*W sizes
        LAYER_SHAPES = {
            "conv_layer0": (64, 111, 111),   # C*H*W=788544, H*W=12321
            "layer1":      (32, 55, 55),     # C*H*W=96800,  H*W=3025
            "layer2":      (16, 27, 27),     # C*H*W=11664,  H*W=729
            "layer3":      (8,  13, 13),     # C*H*W=1352,   H*W=169
        }

        def _unflatten_spatial(x, C, H, W):
            """Unflatten x to (N, C, H, W), handling both full and channel-avg formats."""
            if x.ndim == 2:
                if x.shape[1] == C * H * W:
                    # Fully flattened — straightforward unflatten
                    x = x.unflatten(1, (C, H, W))
                elif x.shape[1] == H * W:
                    # Channel-averaged (N, H*W) — expand across C channels
                    x = x.unflatten(1, (H, W)).unsqueeze(1).expand(-1, C, -1, -1)
                    x = x.contiguous()
                else:
                    raise ValueError(
                        f"Unexpected flat dim {x.shape[1]} for layer shape ({C},{H},{W}): "
                        f"expected {C*H*W} (full) or {H*W} (channel-avg)"
                    )
            return x

        if layer_name == "conv_layer0":
            C, H, W = LAYER_SHAPES["conv_layer0"]
            x = _unflatten_spatial(x, C, H, W)
            x = self.encoder[2:](x)
            x = self.ff_head(x)
        elif layer_name == "layer1":
            C, H, W = LAYER_SHAPES["layer1"]
            x = _unflatten_spatial(x, C, H, W)
            x = self.encoder[5:](x)
            x = self.ff_head(x)
        elif layer_name == "layer2":
            C, H, W = LAYER_SHAPES["layer2"]
            x = _unflatten_spatial(x, C, H, W)
            x = self.encoder[8:](x)
            x = self.ff_head(x)
        elif layer_name == "layer3":
            C, H, W = LAYER_SHAPES["layer3"]
            x = _unflatten_spatial(x, C, H, W)
            x = self.encoder[11:](x)
            x = self.ff_head(x)
        elif layer_name == "fc":
            # Raw output of encoder[11:] is (N, 512) — already flat
            x = self.ff_head(x)
        else:
            raise ValueError(f"Unknown layer_name: {layer_name}")

        return x

    def forward_to_layer(self, x, layer_name):
        """Run the backbone up to (and including) `layer_name`.

        Returns the activation at that layer, still on the same device with
        gradients preserved.  For conv layers the output has shape
        (N, C, H, W); for 'fc' it is already flat (N, 512).
        """
        layers = self.encoder
        slices = {
            "conv_layer0": (0, 2),
            "layer1":      (0, 5),
            "layer2":      (0, 8),
            "layer3":      (0, 11),
            "fc":          (0, len(layers)),
        }
        if layer_name not in slices:
            raise ValueError(f"Unknown layer_name: {layer_name}")
        start, end = slices[layer_name]
        return layers[start:end](x)

    def encode_image(self, x):
        return self.encoder(x)
