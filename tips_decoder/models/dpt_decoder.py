"""DPT-style decoder blocks for TIPS patch tokens."""

import torch
import torch.nn.functional as F
from torch import nn


class ReassembleLayer(nn.Module):
    """Project and upsample patch tokens into multi-scale feature maps."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.resample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(self.project(x))


class FusionBlock(nn.Module):
    """Fuse multi-scale features with a residual conv block and upsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.residual_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.residual_conv(x + skip)
        return self.upsample(x)


class DPTDepthDecoder(nn.Module):
    """DPT-style depth decoder for frozen TIPS patch tokens."""

    def __init__(
        self,
        embed_dim: int = 384,
        channels: int = 256,
        output_scale: int = 2,
    ):
        super().__init__()
        self.reassemble_layers = nn.ModuleList(
            [
                ReassembleLayer(embed_dim, channels, scale_factor=4),
                ReassembleLayer(embed_dim, channels, scale_factor=2),
                ReassembleLayer(embed_dim, channels, scale_factor=1),
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                FusionBlock(channels),
                FusionBlock(channels),
            ]
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )
        self.output_scale = output_scale

    def forward(self, spatial_feats: torch.Tensor) -> torch.Tensor:
        feats = [layer(spatial_feats) for layer in self.reassemble_layers]
        x = self.fusion_blocks[0](feats[-1], feats[-2])
        x = self.fusion_blocks[1](x, feats[-3])
        depth = self.output_head(x)
        if self.output_scale != 1:
            depth = F.interpolate(
                depth, scale_factor=self.output_scale, mode="bilinear", align_corners=False
            )
        return depth
