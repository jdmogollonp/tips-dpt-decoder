"""Inference pipeline for TIPS-based depth decoding."""

import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tips.pytorch import image_encoder
from tips_decoder.models.dpt_decoder import DPTDepthDecoder
from tips_decoder.utils.depth import depth_to_colormap, depth_to_grayscale

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TipsDepthInferencePipeline:
    """Load the frozen TIPS encoder and a trained DPT decoder for inference."""

    def __init__(
        self,
        image_ckpt: str,
        decoder_ckpt: str,
        image_size: int = 448,
        patch_size: int = 14,
        embed_dim: int = 384,
        channels: int = 256,
        output_scale: int = 2,
        device: torch.device | None = None,
    ):
        self.image_ckpt = image_ckpt
        self.decoder_ckpt = decoder_ckpt
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.output_scale = output_scale
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = self._load_image_encoder()
        self.decoder = self._load_decoder()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )

    def _load_image_encoder(self) -> torch.nn.Module:
        logging.info("Loading TIPS image encoder weights...")
        weights = dict(np.load(self.image_ckpt, allow_pickle=False))
        weights = {k: torch.tensor(v) for k, v in weights.items()}
        model = image_encoder.vit_small(
            img_size=self.image_size,
            patch_size=self.patch_size,
            ffn_layer="mlp",
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
        model.load_state_dict(weights)
        return model.to(self.device).eval()

    def _load_decoder(self) -> torch.nn.Module:
        logging.info("Loading DPT decoder weights...")
        decoder = DPTDepthDecoder(
            embed_dim=self.embed_dim, channels=self.channels, output_scale=self.output_scale
        ).to(self.device)
        decoder.load_state_dict(torch.load(self.decoder_ckpt, map_location=self.device))
        return decoder.eval()

    def _preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        return self.transform(image).unsqueeze(0).to(self.device), original_size

    @torch.no_grad()
    def infer_depth(self, image_tensor: torch.Tensor) -> np.ndarray:
        output = self.encoder(image_tensor)
        cls_token = output[0][:, 0:1, :]
        spatial_tokens = output[2]
        all_tokens = torch.cat([spatial_tokens, cls_token], dim=1)

        batch, n_tokens, dim = all_tokens.shape
        grid = int((n_tokens - 1) ** 0.5)
        spatial = all_tokens[:, :-1, :].view(batch, grid, grid, dim).permute(0, 3, 1, 2)
        cls_token_injected = all_tokens[:, -1, :].view(batch, dim, 1, 1)
        spatial = spatial + cls_token_injected

        depth = self.decoder(spatial)
        return depth.squeeze(0).squeeze(0).cpu().numpy()

    def infer_depth_from_path(self, image_path: str, resize_to_original: bool = True) -> np.ndarray:
        image_tensor, original_size = self._preprocess_image(image_path)
        depth = self.infer_depth(image_tensor)
        if resize_to_original:
            depth_img = Image.fromarray(depth.astype("float32"))
            depth = np.array(depth_img.resize(original_size, resample=Image.BILINEAR))
        return depth

    def save_depth_map(self, depth: np.ndarray, out_path: str) -> None:
        depth_to_grayscale(depth).save(out_path)
        logging.info("Saved depth map to %s", out_path)

    def save_depth_colormap(self, depth: np.ndarray, out_path: str, cmap: str = "inferno") -> None:
        depth_to_colormap(depth, cmap=cmap, invert=True).save(out_path)
        logging.info("Saved depth colormap to %s", out_path)
