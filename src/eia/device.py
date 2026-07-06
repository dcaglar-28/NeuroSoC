"""Device selection for Apple Silicon (MPS), CUDA, or CPU."""

from __future__ import annotations


def get_device(prefer: str = "auto"):
    """Return the best available torch device.

    Args:
        prefer: "auto" (default), "mps", "cuda", or "cpu". On an M-series Mac,
            "auto" resolves to MPS. Some spiking ops lack MPS kernels and fall
            back to CPU automatically; set prefer="cpu" to force CPU if you hit
            an unsupported-op error.
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
