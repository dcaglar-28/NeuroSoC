"""EIA — event-driven multimodal edge diagnostics (Phase-0 prototype).

Public API:
    from eia import encoding, datasets, energy
    from eia.device import get_device
"""

__version__ = "0.1.0"

from . import encoding, energy, report  # noqa: F401  (numpy-only, always importable)

__all__ = ["encoding", "energy", "report", "__version__"]
