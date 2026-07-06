"""Event encoders: turn a continuous waveform into sparse spike events.

This is the "neuromorphic front end". Instead of feeding every sample to the
network, we emit an event only when the signal changes meaningfully. Downstream
compute then scales with the number of *events*, not the sample rate — which is
where the energy advantage of an event-driven pipeline comes from.

All functions here are pure NumPy (no torch), so they import and run anywhere.
"""

from __future__ import annotations

import numpy as np


def delta_encode(signal: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Delta / level-crossing modulation.

    Emits +1 when the signal rises by `threshold` since the last event, -1 when
    it falls by `threshold`, and 0 otherwise. This is the classic asynchronous
    delta-modulation scheme used by event cameras and neuromorphic ADCs.

    Args:
        signal: 1-D array, shape (T,). Should be roughly unit-scaled (see
            `normalize`) so a single threshold works across channels.
        threshold: change in signal amplitude required to emit an event.

    Returns:
        int8 array of {-1, 0, +1}, shape (T,).
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    spikes = np.zeros_like(signal, dtype=np.int8)
    reference = signal[0]
    for i in range(1, signal.size):
        diff = signal[i] - reference
        if diff >= threshold:
            spikes[i] = 1
            reference = signal[i]
        elif diff <= -threshold:
            spikes[i] = -1
            reference = signal[i]
    return spikes


def delta_encode_2ch(signal: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Same as `delta_encode` but split into two non-negative channels (ON/OFF).

    Returns:
        uint8 array, shape (2, T): row 0 = ON spikes, row 1 = OFF spikes.
        This is the form most spiking networks expect (non-negative inputs).
    """
    d = delta_encode(signal, threshold)
    on = (d > 0).astype(np.uint8)
    off = (d < 0).astype(np.uint8)
    return np.stack([on, off], axis=0)


def threshold_crossing(signal: np.ndarray, levels: int = 16) -> np.ndarray:
    """Quantize into `levels` bands and spike on every band crossing.

    A denser encoding than delta modulation — useful when you want to preserve
    more amplitude detail at the cost of more events.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-12:
        return np.zeros_like(signal, dtype=np.int8)
    bands = np.floor((signal - lo) / (hi - lo) * (levels - 1)).astype(int)
    spikes = np.zeros_like(signal, dtype=np.int8)
    spikes[1:] = np.sign(np.diff(bands))
    return spikes.astype(np.int8)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalization. Robust to constant signals."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    std = signal.std()
    if std < 1e-12:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


def event_count(spikes: np.ndarray) -> int:
    """Number of non-zero events in a spike array (any shape)."""
    return int(np.count_nonzero(spikes))


def event_rate(spikes: np.ndarray) -> float:
    """Fraction of timesteps that carry an event — the input sparsity."""
    spikes = np.asarray(spikes)
    return float(np.count_nonzero(spikes) / spikes.size) if spikes.size else 0.0
