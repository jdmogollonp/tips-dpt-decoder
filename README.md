# TIPS DPT Decoder (Unofficial)

This repository packages a public-facing version of the TIPS DPT-style depth
decoder and training pipeline. It is built on top of the official TIPS release
(`tips/`) and provides clean, scriptable entry points for training and
inference.

## Repository Layout
- `tips_decoder/`: Python package for the decoder, datasets, and inference.
- `scripts/`: CLI scripts for training, inference, and evaluation.
- `config/`: Example YAML configs for common runs.
- `notebooks/`: Research notebooks (kept as historical notes).
- `tips/`: Official TIPS implementation (kept intact for reference).
- `checkpoints/`: Example decoder checkpoints.

## Setup
```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

For editable installs and CLI entry points:
```bash
pip install -e .
```

## Dataset (NYUv2 via KaggleHub)
- Kaggle dataset: https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2
- KaggleHub: https://github.com/Kaggle/kagglehub

Example download snippet:
```bash
python - <<'PY'
import kagglehub
path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")
print(path)
PY
```

## Inference (Depth From Image)
```bash
IMAGE_PATH=/path/to/input.jpg
IMAGE_CKPT=tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_vision.npz
DECODER_CKPT=checkpoints/dpt_decoder_epoch_99.pth
OUTPUT_PATH=/path/to/output_depth.png

python scripts/infer_depth.py \
  --image "$IMAGE_PATH" \
  --image-ckpt "$IMAGE_CKPT" \
  --decoder-ckpt "$DECODER_CKPT" \
  --output "$OUTPUT_PATH" \
  --colormap
```

CLI entry point (after `pip install -e .`):
```bash
tips-infer \
  --image "$IMAGE_PATH" \
  --image-ckpt "$IMAGE_CKPT" \
  --decoder-ckpt "$DECODER_CKPT" \
  --output "$OUTPUT_PATH" \
  --colormap
```

## Training (NYUv2)
```bash
TRAIN_CSV=/path/to/nyu2_train.csv
BASE_PATH=/path/to/nyu_data
IMAGE_CKPT=tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_vision.npz
OUTPUT_DIR=checkpoints

python scripts/train_dpt_decoder.py \
  --train-csv "$TRAIN_CSV" \
  --base-path "$BASE_PATH" \
  --image-ckpt "$IMAGE_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-4
```

CLI entry point:
```bash
tips-train \
  --train-csv "$TRAIN_CSV" \
  --base-path "$BASE_PATH" \
  --image-ckpt "$IMAGE_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-4
```

## Evaluation
Evaluation follows the same DPT training step shown in the notebooks, using a
frozen TIPS encoder and an MSE loss on 256x256 depth maps.
```bash
VAL_CSV=/path/to/nyu2_val.csv
BASE_PATH=/path/to/nyu_data
IMAGE_CKPT=tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_vision.npz
DECODER_CKPT=checkpoints/dpt_decoder_epoch_19.pth

python scripts/evaluate_decoder.py \
  --val-csv "$VAL_CSV" \
  --base-path "$BASE_PATH" \
  --image-ckpt "$IMAGE_CKPT" \
  --decoder-ckpt "$DECODER_CKPT" \
  --batch-size 4
```

CLI entry point:
```bash
tips-eval \
  --val-csv "$VAL_CSV" \
  --base-path "$BASE_PATH" \
  --image-ckpt "$IMAGE_CKPT" \
  --decoder-ckpt "$DECODER_CKPT" \
  --batch-size 4
```

For a smoke test that executes the DPT training step without dataset setup:
```bash
python scripts/train_dpt_decoder.py --synthetic --epochs 1 --max-steps 1
```

## Lint and Test
```bash
ruff check tips_decoder scripts tests
pytest
```

## Notes
- The decoder architecture and training loop are aligned to the reference
  notebook in `notebooks/TIP_DECODER DPT IMAGE+TEXT.ipynb`.
- TIPS checkpoints are required for feature extraction; use the provided
  checkpoints in `tips/pytorch/checkpoints`.

## Credits
If you use this repository, please cite the original TIPS paper:
```bibtex
@InProceedings{tips_paper,
    Title={{TIPS: Text-Image Pretraining with Spatial Awareness}},
    Author={Maninis, Kevis-Kokitsi and Chen, Kaifeng and Ghosh, Soham and Karpur, Arjun and Chen, Koert and Xia, Ye and Cao, Bingyi and Salz, Daniel and Han, Guangxing and Dlabal, Jan and Gnanapragasam, Dan and Seyedhosseini, Mojtaba and Zhou, Howard and Araujo, Andr\'e},
    Booktitle={ICLR},
    year={2025},
}
```
