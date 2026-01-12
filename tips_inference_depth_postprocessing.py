import os
import glob
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tips.pytorch import image_encoder
from tips.scenic.utils import feature_viz
import torch.nn as nn
import torch.nn.functional as F

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMAGE_SIZE = 448
PATCH_SIZE = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- DPT Decoder ----------------------
class ReassembleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.resample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.resample(self.project(x))


class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.residual_conv(x + skip)
        return self.upsample(x)


class DPTDepthDecoder(nn.Module):
    def __init__(self, embed_dim=384, channels=256):
        super().__init__()
        self.reassemble_layers = nn.ModuleList([
            ReassembleLayer(embed_dim, channels, scale_factor=4),
            ReassembleLayer(embed_dim, channels, scale_factor=2),
            ReassembleLayer(embed_dim, channels, scale_factor=1),
        ])
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(channels),
            FusionBlock(channels),
        ])
        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1)
        )

    def forward(self, spatial_feats):
        feats = [layer(spatial_feats) for layer in self.reassemble_layers]
        x = self.fusion_blocks[0](feats[-1], feats[-2])
        x = self.fusion_blocks[1](x, feats[-3])
        x = self.output_head(x)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


# ---------------------- Pipeline ----------------------
class TipsDepthInferencePipeline:
    def __init__(self, image_ckpt, decoder_ckpt):
        logging.info("🚀 Initializing TipsDepthInferencePipeline")
        self.encoder = self._load_image_encoder(image_ckpt)
        self.decoder = self._load_decoder(decoder_ckpt)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1)),
        ])

    def _load_image_encoder(self, ckpt_path):
        weights = {k: torch.tensor(v) for k, v in np.load(ckpt_path, allow_pickle=False).items()}
        model = image_encoder.vit_small(
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            ffn_layer='mlp',
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
        model.load_state_dict(weights)
        return model.to(DEVICE).eval()

    def _load_decoder(self, ckpt_path):
        decoder = DPTDepthDecoder(embed_dim=384, channels=256).to(DEVICE)
        decoder.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        return decoder.eval()

    def _preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(DEVICE)

    def infer_depth(self, image_tensor):
        with torch.no_grad():
            output = self.encoder(image_tensor)
            cls_token = output[0][:, 0:1, :]         # [B, 1, D]
            spatial_tokens = output[2]              # [B, HW, D]
            all_tokens = torch.cat([spatial_tokens, cls_token], dim=1)

            B, N, D = all_tokens.shape
            H = W = int((N - 1) ** 0.5)
            spatial = all_tokens[:, :-1, :].view(B, H, W, D).permute(0, 3, 1, 2)

            # ➕ Add CLS token globally
            cls_token_injected = all_tokens[:, -1, :].view(B, D, 1, 1)
            spatial = spatial + cls_token_injected

            depth = self.decoder(spatial).squeeze().cpu().numpy()
            return depth

    def save_colormapped_depth(self, depth_map, out_path, cmap='plasma', invert=True):
        norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        if invert:
            norm_depth = 1.0 - norm_depth
        colorized = plt.get_cmap(cmap)(norm_depth)[..., :3]  # drop alpha
        img = Image.fromarray((colorized * 255).astype(np.uint8))
        img.save(out_path)
        logging.info(f"📸 Saved depth colormap: {out_path}")

    def process_folder(self, input_folder, output_folder):
        logging.info(f"📂 Scanning folder: {input_folder}")
        os.makedirs(output_folder, exist_ok=True)
        image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

        if not image_paths:
            logging.warning("⚠️ No images found.")
            return

        for idx, image_path in enumerate(image_paths):
            try:
                logging.info(f"[{idx+1}/{len(image_paths)}] 🔍 Processing {os.path.basename(image_path)}")
                image_tensor = self._preprocess_image(image_path)
                depth_map = self.infer_depth(image_tensor)

                filename = os.path.splitext(os.path.basename(image_path))[0]
                out_path = os.path.join(output_folder, f"{filename}_depth.png")
                self.save_colormapped_depth(depth_map, out_path)

            except Exception as e:
                logging.error(f"❌ Error processing {image_path}: {e}")


# ---------------------- Run ----------------------
if __name__ == "__main__":
    IMAGE_CKPT = "/usr/mvl2/jdmcnw/Projects/2025/VOTS/TIPS/tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_vision.npz"
    DECODER_CKPT = "/usr/mvl2/jdmcnw/Projects/2025/VOTS/TIPS/checkpoints/dpt_decoder_epoch_99.pth"

    INPUT_FOLDER = "/usr/mvl2/jdmcnw/Projects/2025/VOTS/TIPS/images/inputs/bicycle/color"
    OUTPUT_FOLDER = "/usr/mvl2/jdmcnw/Projects/2025/VOTS/TIPS/images/outputs/bicycle/depth"

    pipeline = TipsDepthInferencePipeline(IMAGE_CKPT, DECODER_CKPT)
    pipeline.process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
