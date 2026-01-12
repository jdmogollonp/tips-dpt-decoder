"""NYUv2 dataset helpers for frozen TIPS feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NYUv2TIPSDataset(Dataset):
    """Yield frozen TIPS patch tokens and 256x256 depth maps."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int,
        tips_encoder: torch.nn.Module,
        device: torch.device,
        use_cls_token: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.device = device
        self.encoder = tips_encoder.to(self.device).eval()
        self.use_cls_token = use_cls_token
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )
        self.depth_resize = transforms.Resize((256, 256))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        rgb_path, depth_path = self.df.iloc[idx]
        image = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)

        image = self.transform(image).to(self.device)
        depth = self.depth_resize(depth)
        depth = transforms.ToTensor()(depth)

        with torch.no_grad():
            output = self.encoder(image.unsqueeze(0))
            spatial_tokens = output[2]  # [1, HW, D]
            batch, hw, dim = spatial_tokens.shape
            grid = int(hw**0.5)
            spatial = spatial_tokens.view(batch, grid, grid, dim).permute(0, 3, 1, 2).squeeze(0)

            if self.use_cls_token:
                cls_token = output[0][:, 0, :]  # [1, D]
                cls_token = cls_token.view(dim, 1, 1)
                spatial = spatial + cls_token

        return spatial, depth


def load_nyuv2_dataframe(csv_path: Path, base_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the NYUv2 CSV and optionally expand relative paths."""
    df = pd.read_csv(csv_path, header=None)
    if base_path is not None:
        df[0] = df[0].map(lambda x: base_path / x)
        df[1] = df[1].map(lambda x: base_path / x)
    return df
