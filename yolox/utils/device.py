"""Device selection shared by training, evaluation and benchmarking."""

import torch


def resolve_device(device="auto", local_rank=0):
    """Resolve auto to CUDA, MPS, then CPU; explicit requests never fall back."""
    # Parent-side discovery must not initialize CUDA before RAM-cache workers fork.
    # device_count uses NVML when available; workers check runtime availability.
    if device == "auto":
        device = "cuda" if torch.cuda.device_count() > 0 else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device == "cuda":
        if torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda", local_rank)
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}")
