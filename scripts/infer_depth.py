"""Run inference with a trained DPT depth decoder."""

import argparse
from pathlib import Path

from tips_decoder.pipeline import TipsDepthInferencePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TIPS DPT depth inference.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument(
        "--image-ckpt",
        type=Path,
        required=True,
        help="TIPS image encoder .npz checkpoint.",
    )
    parser.add_argument(
        "--decoder-ckpt", type=Path, required=True, help="Trained decoder checkpoint."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output depth image path."
    )
    parser.add_argument("--colormap", action="store_true", help="Save colormapped depth.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TipsDepthInferencePipeline(
        image_ckpt=str(args.image_ckpt),
        decoder_ckpt=str(args.decoder_ckpt),
    )
    depth = pipeline.infer_depth_from_path(str(args.image))
    if args.colormap:
        pipeline.save_depth_colormap(depth, str(args.output))
    else:
        pipeline.save_depth_map(depth, str(args.output))


if __name__ == "__main__":
    main()
