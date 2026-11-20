#!/usr/bin/env python3
"""Repeatable training-step benchmark, excluding data loading and validation."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch
import torchvision

from yolox.exp import get_exp
from yolox.utils import ModelEMA, resolve_device, time_synchronized


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-n", "--name", default="yolox-s")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="CUDA AMP only")
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 1:
        parser.error("batch size, steps and warmup must be positive")
    if args.size < 64 or args.size % 32:
        parser.error("size must be a multiple of 32 and at least 64")
    device = resolve_device(args.device)
    if args.fp16 and device.type != "cuda":
        parser.error("--fp16 requires CUDA; MPS/CPU use FP32")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    exp = get_exp(exp_name=args.name)
    model = exp.get_model().to(device).train()
    exp.warmup_epochs = 0
    optimizer = exp.get_optimizer(args.batch_size)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    ema = ModelEMA(model, 0.9998)

    # Generate once on CPU so MPS and CUDA see exactly the same data.
    images = torch.rand(args.batch_size, 3, args.size, args.size) * 255
    targets = torch.tensor([
        [1, 0.23, 0.31, 0.14, 0.22],
        [7, 0.67, 0.58, 0.28, 0.19],
        [19, 0.46, 0.81, 0.11, 0.13],
    ]).repeat(args.batch_size, 1, 1)
    targets[:, :, 1:] *= args.size
    digest = hashlib.sha256(images.numpy().tobytes() + targets.numpy().tobytes()).hexdigest()
    dtype = torch.float16 if args.fp16 else torch.float32
    images, targets = images.to(device, dtype), targets.to(device, dtype)

    def step():
        with torch.amp.autocast(device.type, enabled=args.fp16):
            loss = model(images, targets)["total_loss"]
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        return loss.detach()

    print(f"Warming up {args.name} on {device} ({args.warmup} steps)...", file=sys.stderr)
    losses = [step() for _ in range(args.warmup)]
    time_synchronized(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Timing {args.steps} training steps...", file=sys.stderr)
    durations = []
    for _ in range(args.steps):
        start = time.perf_counter()
        losses.append(step())
        time_synchronized(device)
        durations.append(time.perf_counter() - start)

    # Check outside the timed region, but fail instead of reporting bogus throughput.
    losses = torch.stack(losses).cpu()
    if not torch.isfinite(losses).all() or not all(
        torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None
    ):
        raise RuntimeError("Nonfinite loss or gradients in benchmark")
    hardware = platform.processor()
    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    elif sys.platform == "darwin":
        hardware = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    result = {
        "scope": "synthetic fixed-batch forward/backward/SGD/EMA; no loader or evaluation",
        "hardware": hardware,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "precision": "cuda-amp-fp16" if args.fp16 else "fp32",
        "model": args.name,
        "batch_size": args.batch_size,
        "size": args.size,
        "objects_per_image": 3,
        "seed": args.seed,
        "data_sha256": digest,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "seconds": sum(durations),
        "images_per_second": args.batch_size * args.steps / sum(durations),
        "median_step_ms": statistics.median(durations) * 1000,
        "first_loss": losses[0].item(),
        "last_loss": losses[-1].item(),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    elif device.type == "mps":
        result["current_allocated_mib"] = torch.mps.current_allocated_memory() / 1024**2
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
