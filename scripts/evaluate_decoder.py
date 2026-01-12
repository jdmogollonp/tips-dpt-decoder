"""Evaluate a trained DPT decoder against NYUv2 depth targets."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError

from tips.pytorch import image_encoder
from tips_decoder.data.nyuv2 import NYUv2TIPSDataset, load_nyuv2_dataframe
from tips_decoder.models.dpt_decoder import DPTDepthDecoder


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
    parser = argparse.ArgumentParser(description="Evaluate a trained DPT decoder.")
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Path to NYUv2 validation CSV. Use --train-csv to split like notebooks.",
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
        required=True,
        help="TIPS image encoder .npz checkpoint.",
    )
    parser.add_argument("--decoder-ckpt", type=Path, required=True, help="Decoder checkpoint.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches", type=int, default=0, help="Limit batches for quick eval."
    )
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Limit samples for quick eval."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.val_csv is None and args.train_csv is None:
        raise SystemExit("Provide --val-csv or --train-csv for evaluation.")

    if args.val_csv is None:
        df = load_nyuv2_dataframe(args.train_csv, base_path=args.base_path)
        train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True)
        val_df, _ = train_test_split(val_df, test_size=0.1, shuffle=True)
        df = val_df.reset_index(drop=True)
    else:
        df = load_nyuv2_dataframe(args.val_csv, base_path=args.base_path)
    if args.max_samples:
        df = df.iloc[: args.max_samples].reset_index(drop=True)
    encoder = load_tips_encoder(
        args.image_ckpt, image_size=448, patch_size=14, device=device
    )
    dataset = NYUv2TIPSDataset(df, image_size=448, tips_encoder=encoder, device=device)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    decoder = DPTDepthDecoder(embed_dim=384, channels=256, output_scale=2).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
    decoder.eval()

    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for idx, (features, gt_depth) in enumerate(loader):
            features = features.to(device)
            gt_depth = gt_depth.to(device)
            preds = decoder(features)
            mse.update(preds, gt_depth)
            ssim.update(preds, gt_depth)
            if args.max_batches and idx + 1 >= args.max_batches:
                break

    print(f"MSE: {mse.compute().item():.6f}")
    print(f"SSIM: {ssim.compute().item():.6f}")


if __name__ == "__main__":
    main()
