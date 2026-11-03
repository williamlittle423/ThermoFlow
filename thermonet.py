"""Deterministic thermal-tomography baseline implemented as a compact 2D U-Net."""

import torch
from torch import nn
from torch.nn import functional as F

class _ConvBlock(nn.Sequential):
    """Extract spatial features without changing image resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class ThermoNet(nn.Module):
    """Map a surface thermal video to a volumetric inclusion reconstruction.

    Physical time is encoded in the input channels; depth is encoded in the
    output channels. Convolutions act over the surface coordinates (y, x).

    Args:
        in_channels: Number of recorded thermal frames (default: 20).
        out_channels: Number of reconstructed depth slices (default: 12).
        base_channels: Feature width at the highest resolution (default: 32).

    Input:
        Floating-point tensor of shape (batch, in_channels, height, width).
        Height and width must each be at least 4. Preserve frame ordering and
        apply the same training-derived or fixed normalization to every video.

    Output:
        Inclusion logits of shape (batch, out_channels, height, width), with
        depth increasing away from the observed surface. Use these logits with
        BCEWithLogitsLoss; use predict_proba() for sigmoid probabilities or a
        soft-Dice term. For the default materials, k_hat = 0.20 + 0.30 * p.

    Example:
        model = ThermoNet()
        logits = model(video)                 # (B, 12, 48, 48)
        probabilities = torch.sigmoid(logits)
    """

    def __init__(self, in_channels: int = 20, out_channels: int = 12,
                 base_channels: int = 32):
        super().__init__()
        for name, value in (("in_channels", in_channels),
                            ("out_channels", out_channels),
                            ("base_channels", base_channels)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        c = base_channels

        # Encoder: 48×48 -> 24×24 -> 12×12 for the default slab.
        self.encoder1 = _ConvBlock(in_channels, c)
        self.encoder2 = _ConvBlock(c, 2*c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _ConvBlock(2*c, 4*c)

        # Each decoder combines coarse context with its matching skip features.
        self.decoder2 = _ConvBlock(4*c + 2*c, 2*c)
        self.decoder1 = _ConvBlock(2*c + c, c)
        self.output = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Return raw inclusion logits, retaining the input spatial resolution."""
        if video.ndim != 4:
            raise ValueError("Expected video shape (batch, frames, height, width)")
        if video.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} frames, got {video.shape[1]}")
        if min(video.shape[-2:]) < 4:
            raise ValueError("Video height and width must each be at least 4")

        skip1 = self.encoder1(video)
        skip2 = self.encoder2(self.pool(skip1))
        features = self.bottleneck(self.pool(skip2))

        # Match skip sizes explicitly, including odd-sized surface images.
        features = F.interpolate(features, size=skip2.shape[-2:],
                                 mode="bilinear", align_corners=False)
        features = self.decoder2(torch.cat((features, skip2), dim=1))
        features = F.interpolate(features, size=skip1.shape[-2:],
                                 mode="bilinear", align_corners=False)
        features = self.decoder1(torch.cat((features, skip1), dim=1))
        return self.output(features)

    def predict_proba(self, video: torch.Tensor) -> torch.Tensor:
        """Return inclusion probabilities; gradients and train/eval mode are unchanged.

        For inference, call model.eval() and use torch.inference_mode().
        """
        return torch.sigmoid(self(video))
