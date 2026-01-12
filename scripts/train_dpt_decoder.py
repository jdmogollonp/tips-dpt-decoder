"""Train a DPT-style decoder on frozen TIPS features."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from tips.pytorch import image_encoder
from tips_decoder.data.nyuv2 import NYUv2TIPSDataset, load_nyuv2_dataframe
from tips_decoder.models.dpt_decoder import DPTDepthDecoder


class SyntheticDepthDataset(Dataset):
    """Synthetic dataset to validate the training loop quickly."""

    def __init__(self, num_samples: int, embed_dim: int = 384, grid: int = 32):
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.grid = grid

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        features = torch.randn(self.embed_dim, self.grid, self.grid)
        depth = torch.randn(1, 256, 256)
        return features, depth


def load_tips_encoder(
    ckpt_path: Path, image_size: int, patch_size: int, device: torch.device
):
    """Load the frozen TIPS vision encoder from an .npz checkpoint."""
    weights = dict(np.load(ckpt_path, allow_pickle=False))
    weights = {k: torch.tensor(v) for k, v in weights.items()}
    model = image_encoder.vit_small(
        img_size=image_size,
        patch_size=patch_size,
        ffn_layer="mlp",
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    model.load_state_dict(weights)
    return model.to(device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DPT-style decoder on TIPS features."
    )
    parser.add_argument(
        "--train-csv", type=Path, default=None, help="Path to NYUv2 train CSV."
    )
    parser.add_argument(
        "--base-path", type=Path, default=None, help="Base path for CSV-relative files."
    )
    parser.add_argument(
        "--image-ckpt",
        type=Path,
        default=None,
        help="TIPS image encoder .npz checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Where to save checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--synthetic", action="store_true", help="Run a synthetic smoke training loop."
    )
    parser.add_argument(
        "--max-steps", type=int, default=0, help="Limit steps per epoch for quick runs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if args.synthetic:
        train_dataset = SyntheticDepthDataset(num_samples=32)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        if args.train_csv is None or args.image_ckpt is None:
            raise SystemExit("Provide --train-csv and --image-ckpt for real training.")

        df = load_nyuv2_dataframe(args.train_csv, base_path=args.base_path)
        train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True)
        train_df = train_df.reset_index(drop=True)[: args.max_train_samples]
        val_df = val_df.reset_index(drop=True)[: args.max_val_samples]

        encoder = load_tips_encoder(
            args.image_ckpt, image_size=448, patch_size=14, device=device
        )
        train_dataset = NYUv2TIPSDataset(
            train_df, image_size=448, tips_encoder=encoder, device=device
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )

    decoder = DPTDepthDecoder(embed_dim=384, channels=256, output_scale=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        decoder.train()
        total_loss = 0.0
        start_time = time.time()

        for step, (features, gt_depth) in enumerate(train_loader):
            features = features.to(device)
            gt_depth = gt_depth.to(device)

            if epoch == 0 and step == 0:
                print(f"Features: {features.shape}, Depth GT: {gt_depth.shape}")

            optimizer.zero_grad()
            pred_depth = decoder(features)
            loss = criterion(pred_depth, gt_depth)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if args.max_steps and step + 1 >= args.max_steps:
                break

        avg_loss = total_loss / max(1, len(train_loader))
        duration = time.time() - start_time
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Time: {duration:.2f}s")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = args.output_dir / f"dpt_decoder_epoch_{epoch}.pth"
        torch.save(decoder.state_dict(), save_path)
        print(f"Saved decoder weights to {save_path}")


if __name__ == "__main__":
    main()
