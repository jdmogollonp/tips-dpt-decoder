"""Depth-map postprocessing helpers for saving and visualization."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to [0, 1] for visualization."""
    depth = depth.astype("float32")
    min_val = depth.min()
    max_val = depth.max()
    return (depth - min_val) / (max_val - min_val + 1e-8)


def depth_to_grayscale(depth: np.ndarray) -> Image.Image:
    """Convert a depth map to an 8-bit grayscale PIL image."""
    norm_depth = normalize_depth(depth)
    img = (norm_depth * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(img)


def depth_to_colormap(
    depth: np.ndarray, cmap: str = "inferno", invert: bool = False
) -> Image.Image:
    """Convert a depth map to a colormapped PIL image."""
    norm_depth = normalize_depth(depth)
    if invert:
        norm_depth = 1.0 - norm_depth
    colorized = plt.get_cmap(cmap)(norm_depth)[..., :3]
    img = (colorized * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(img)


def resize_depth(depth: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a depth map using bilinear resampling."""
    image = Image.fromarray(depth.astype("float32"))
    return np.array(image.resize(size, resample=Image.BILINEAR))
