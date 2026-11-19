# Pretrained YOLOX weights

These are upstream COCO-pretrained PyTorch checkpoints (80 classes), not
fork-trained models. The fork's downloader fetches the six non-legacy variants
from the official YOLOX `0.1.1rc0` GitHub release; Darknet53 is excluded.

## Download

From the repository root, with Bash and curl installed:

```sh copy Copy Download All Weights
bash tools/download_weights.sh
```

The [downloader](../tools/download_weights.sh) always writes to this `weights/`
directory, even when invoked by its path from another working directory. It
skips existing nonempty files, retries curl transfers up to three times, and
stops on failure. Downloads use temporary files that are renamed on success
and removed on failure. All six checkpoints together need about 1.55 GB of disk
space (decimal units).

To download just one variant, use its **Download** link below or run, for example,
from the repository root:

```sh copy Copy Download Nano Weights
mkdir -p weights
curl --fail --location --retry 3 \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth \
  --output weights/yolox_nano.pth
```

The single-file curl command overwrites its destination and does not provide
the script's skip/temporary-file protection. Neither method verifies hashes;
an existing nonempty file is not necessarily complete or trustworthy. See
[model asset provenance](../docs/model_assets.md) for the recorded Nano SHA-256
and verification instructions (use `weights/yolox_nano.pth` as the path when
running from the repository root). Load only trusted checkpoints.

## Variants, smallest to largest

Sorted by parameter count; checkpoint sizes and FLOPs also increase in this
order. Checkpoint sizes are rounded decimal MB (1 MB = 1,000,000 bytes), measured
from the locally downloaded release files. They are not inference RAM/VRAM
requirements or exported ONNX sizes.

| Variant / CLI model name | Checkpoint | Size (MB) | Parameters (M) | Input (pixels) | FLOPs (G) | COCO val AP (%) | Download |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| Nano / `yolox-nano` | `yolox_nano.pth` | 7.7 | 0.91 | 416 × 416 | 1.08 | 25.8 | [Download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |
| Tiny / `yolox-tiny` | `yolox_tiny.pth` | 40.8 | 5.06 | 416 × 416 | 6.45 | 32.8 | [Download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |
| S / `yolox-s` | `yolox_s.pth` | 72.1 | 9.0 | 640 × 640 | 26.8 | 40.5 | [Download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
| M / `yolox-m` | `yolox_m.pth` | 203.1 | 25.3 | 640 × 640 | 73.8 | 46.9 | [Download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
| L / `yolox-l` | `yolox_l.pth` | 434.4 | 54.2 | 640 × 640 | 155.6 | 49.7 | [Download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| X / `yolox-x` | `yolox_x.pth` | 793.5 | 99.1 | 640 × 640 | 281.9 | 51.1 | [Download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |

Parameter counts, FLOPs and AP are historical upstream figures from the retained
[benchmark tables](../README.md#benchmark), not fresh measurements of this fork.
AP uses IoU 0.50:0.95 on COCO validation; input sizes are the corresponding
benchmark/default evaluation resolutions. Accuracy on custom classes and latency
on CPU, Raspberry Pi CM5 or other hardware must not be inferred from these
figures. In particular, Nano/Tiny and S–X are benchmarked at different resolutions.

- **Nano:** smallest footprint and compute budget; uses depthwise convolutions
  in its backbone and head, unlike the other five variants.
- **Tiny:** a larger lightweight option with higher reported COCO AP than Nano.
- **S:** smallest standard 640 × 640 variant.
- **M, L, X:** progressively larger models with higher reported COCO AP and
  higher compute/storage costs; X is the largest and most accurate in this table.

Experiment definitions live in [exps/default](../exps/default). Start with Nano
when resource constraints dominate, then compare larger variants on your data
and runtime rather than treating model size as a latency guarantee.

## Use a checkpoint

Match the architecture to the checkpoint. For example, run CPU image inference
in the repository's synced CPU/export environment:

```sh copy Copy Nano Inference Command
uv run --no-sync python tools/demo.py image -n yolox-nano \
  -c weights/yolox_nano.pth --path assets/dog.jpg --device cpu --save_result
```

For CUDA fine-tuning, pass the matching `-n` and `-c` to `tools/train.py`
**without `--resume`**. Resume is for continuing an existing training run, not
initializing one with COCO weights. See the main README's
[training instructions](../README.md#training-resume-and-distributed-evaluation)
and [portable ONNX deployment guide](../docs/cm5_deployment.md) for export and
headless inference.
