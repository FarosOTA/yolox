# Apple Silicon training

This fork supports single-device FP32 training and evaluation using PyTorch's
MPS backend on Apple Silicon. Use the locked environment; neither CoreML nor
the Apple Neural Engine is involved in this training path.

## Install and train

From the repository root:

```sh
uv sync --locked --extra cpu --extra export
uv run --no-sync python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Despite the `cpu` extra's name, its macOS ARM64 wheels include MPS. The availability
check must print `True`. On the tested M5 the lock installs torch 2.14.1 and
torchvision 0.29.1. No additional MPS package or global CPU-fallback environment
variable is needed for the tested paths.

Prepare a local COCO/VOC dataset and experiment using the existing
[custom-data guide](train_custom_data.md), then run, replacing the example paths:

```sh
uv run --no-sync python tools/train.py -f /path/to/experiment.py --device mps -d 1 -b 4
uv run --no-sync python tools/train.py -f /path/to/experiment.py --device mps -b 4 --resume
uv run --no-sync python tools/eval.py -f /path/to/experiment.py --device mps -b 4 -c /path/to/latest_ckpt.pth
```

For fine-tuning, add `-c weights/yolox_s.pth` to training (choose the checkpoint
matching the experiment); omit `--resume`. Only load trusted checkpoints.
The Linux [CVAT/NFS supervisor](cvat_training_data.md) is unchanged and does not
run on macOS. MPS support does not change the public model factory or demo CLI
device selectors. Python callers can use `exp.get_model().to("mps")`.

- `--device auto` (the default) selects CUDA, then MPS, then CPU. Explicit
  `--device mps` never silently switches the entire model to CPU.
- MPS/CPU require one device, one machine and machine rank zero. CUDA keeps its
  existing multi-GPU launch path. `-d` is a device **count**.
- MPS/CPU use FP32. `--fp16` and `--occupy` fail clearly outside CUDA;
  TensorRT evaluation is also CUDA-only. MPS AMP is not a supported mode here.
- Start with Nano/Tiny/S and batch 2–4. Shared system memory is not dedicated
  VRAM; leave room for macOS, data workers, model/optimizer states and EMA.
  Reduce batch size or image size on out-of-memory errors. Do not disable
  PyTorch's MPS memory safety limits. Automatic label-assignment OOM recovery
  remains CUDA-specific.
- The default loaders pin memory only when CUDA is available. For local
  troubleshooting, append `data_num_workers 0` after all CLI options. Custom
  experiments must also avoid CUDA-only overrides of loaders/resizing.
- Multiscale training can grow beyond the initial image size. For a fixed
  640×640 run of the standard S experiment, append `input_size '(640,640)'
  test_size '(640,640)' multiscale_range 0`. Nano/Tiny define `random_size`
  separately; also override it to `'(20,20)'` to hold those models at 640.
- Checkpoint weights and optimizer tensors load onto the selected device.
  When resuming an FP32 checkpoint with CUDA AMP, an empty saved scaler state
  leaves the new CUDA scaler initialized normally. Changing backend or precision
  is not a promise of bitwise-identical continuation.

## Compare M5 and RTX 3060 training steps

Run the same revision, lockfile, model, input size, batch size, seed, warmup and
measurement length on both machines. The benchmark needs no downloaded weights
or dataset. It generates the same CPU-seeded images and three boxes per image,
copies them to the selected device once, and repeatedly trains on that batch.
Its JSON includes a data digest, hardware/runtime versions, precision, losses
and synchronized throughput. Nonfinite losses or final gradients fail the run.

Mac FP32:

```sh
uv run --no-sync python tools/benchmark_train.py --device mps -n yolox-s -b 4 --size 640 --warmup 10 --steps 30 > m5.json
```

RTX 3060, after installing the locked CUDA profile on that machine:

```sh
uv sync --locked --extra cuda --extra export
uv run --no-sync python tools/benchmark_train.py --device cuda -n yolox-s -b 4 --size 640 --warmup 10 --steps 30 > rtx3060-fp32.json
uv run --no-sync python tools/benchmark_train.py --device cuda -n yolox-s -b 4 --size 640 --warmup 10 --steps 30 --fp16 > rtx3060-amp.json
```

Compare FP32 to FP32 first, then MPS FP32 to CUDA AMP for the practical supported
training configurations. Divide RTX images/second by M5 images/second for the
measured ratio. If either machine runs out of memory, lower the batch size on
**both** machines. Match power mode and record whether the laptop is plugged in.

This measures forward/backward, SimOTA assignment, SGD and EMA, with GPU
synchronization after each step. It excludes loading, augmentation, transfers,
multiscale changes, validation and checkpoint I/O. It is neither full-epoch
throughput nor an accuracy/convergence benchmark. Longer runs (e.g. `--steps
300`) help expose thermal effects; a short run does not establish sustained
fanless-laptop throughput.

## Measured M5 evidence

The initial short run on a 16 GB Apple M5, macOS 27.2.1 arm64, Python 3.12.13,
torch 2.14.1 and torchvision 0.29.1 measured **14.74 images/second** (median
**270.36 ms per batch**) for YOLOX-s FP32, 640×640, batch 4. After 10 warmup
steps, 30 measured steps took 8.14 seconds. The repeated synthetic batch's loss
fell from 39.26 to 11.52 across all 40 steps; losses and final gradients were
finite. `PYTORCH_ENABLE_MPS_FALLBACK` was unset. Input digest:
`11edd1e0289a65aa233f0ecc86381302c2a62c55f2e26f1963a3888819cdbbbf`.

Real CPU and MPS CLI regression cases also exercise a small local COCO fixture:
training, EMA, checkpoint save, resumed optimizer updates, validation and
standalone evaluation. Model tests include empty/positive images and L1 loss.
A separate 36-iteration MPS smoke run exercised two spawned workers, 64/96-pixel
multiscale resizing and the mosaic-to-L1 transition, then resumed its checkpoint
on CPU for 12 more iterations. CPU/MPS synthetic Nano loss and decoded inference
agreed at `atol=rtol=0.001`; analytical box-IoU checks also passed. A synthetic
perfect-prediction VOC fixture produced AP 1.0 on both backends, including NMS
and an empty image. These checks used no global MPS CPU fallback.

The local Python 3.12 suite passed 22 tests; 118 Linux-only CVAT/LAN tests were
skipped on macOS. Ruff, lockfile validation and distribution builds passed.
Sphinx built with 17 existing warnings outside the new guide. CUDA hardware,
CUDA AMP and distributed execution were not revalidated in this Mac run.
These are smoke/regression checks, not full-dataset accuracy or long-run
convergence certification. The RTX 3060 has not run this benchmark yet, so no
measured M5-to-3060 ratio is available.
