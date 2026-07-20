"""Modality-agnostic windowed-signal feature front-end.

Originally built for EEG seizure detection (docs/eeg_frontend_results.md,
retired 2026-07-13 — see docs/eeg_frontend_results.md), generalized here so
the extraction machinery survives for the next modality that needs it
(heart-sound murmur detection, docs/heart_sounds_results.md: raw delta-encoding
came back ~chance, and a murmur is a spectral/turbulent-flow signature, not
an edge, so a band-power front-end is the documented escalation). Domain-
specific choices (which bands, which features) live with the caller
(`eia.datasets`), not here — this module takes `bands`/`feature_names`
explicitly and makes no assumption about what kind of signal it's given.

- **Line length** — sum of |x[t] - x[t-1]|. Amplitude+frequency complexity in
  one number (Esteller et al. 2001).
- **Relative band power** — Welch PSD, power per band as a FRACTION of total
  power (amplitude-invariant).
- **Spectral entropy** — the rhythmicity/turbulence measure (see
  `spectral_entropy` for why entropy, not peak-to-mean PSD ratio).

Pure NumPy + SciPy — no torch, offline-testable, matches the rest of this
repo's numpy-only encoding/report/case_level modules.
"""

from __future__ import annotations

import numpy as np


def line_length(x: np.ndarray) -> np.ndarray:
    """Sum of |x[t] - x[t-1]| along the last axis. x: (..., T) -> (...)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


def _welch_psd(x: np.ndarray, fs: float):
    from scipy.signal import welch
    x = np.asarray(x, dtype=np.float64)
    nperseg = min(x.shape[-1], 128)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def relative_band_power(x: np.ndarray, fs: float, bands: dict) -> dict:
    """Welch PSD -> power per band as a fraction of total power over all
    computed frequencies (0..fs/2). x: (..., T) -> {band_name: (...) array}.

    Sums PSD bins directly rather than trapezoidal-integrating over
    frequency: with short sub-windows (fine frequency resolution is not
    guaranteed), trapz's edge weighting double-counts or drops area near a
    band boundary that lands close to a bin. Band edges are half-open
    [lo, hi) so a bin sitting exactly on a shared boundary is counted in
    exactly one band, not two.
    """
    freqs, psd = _welch_psd(x, fs)
    total = psd.sum(axis=-1)
    total = np.where(total > 0, total, 1.0)  # avoid /0 on a flat/silent segment
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            out[name] = np.zeros(psd.shape[:-1])
            continue
        out[name] = psd[..., mask].sum(axis=-1) / total
    return out


def spectral_entropy(x: np.ndarray, fs: float) -> np.ndarray:
    """Normalized Shannon entropy of the PSD distribution, in [0, 1]: near 0
    = power concentrated in a narrow band (rhythmic/tonal); near 1 = power
    spread flat across frequencies (noise/turbulence-like). Chosen over a
    peak-to-mean PSD ratio because entropy pools the WHOLE distribution
    rather than a single bin, so a lone noisy PSD spike can't dominate it —
    more numerically stable for a compact feature set feeding a tiny SNN.
    x: (..., T) -> (...) array in [0, 1].
    """
    freqs, psd = _welch_psd(x, fs)
    eps = 1e-12
    psd_sum = psd.sum(axis=-1, keepdims=True)
    psd_norm = psd / np.where(psd_sum > 0, psd_sum, 1.0)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + eps), axis=-1)
    n_bins = psd.shape[-1]
    max_entropy = np.log2(n_bins) if n_bins > 1 else 1.0
    return entropy / max_entropy


def extract_window_features(sig: np.ndarray, fs: float, n_subwindows: int,
                             feature_names: tuple, bands: dict) -> np.ndarray:
    """Split a (n_channels, window_samples) signal into `n_subwindows` equal
    sub-windows — these become the SNN's timesteps — and compute
    `feature_names` for each channel x sub-window.

    Returns (len(feature_names) * n_channels, n_subwindows), feature-major
    (all channels for feature 0, then all channels for feature 1, ...) —
    features and channels flattened into one axis so the existing
    multi-channel delta encoder (scripts/xylo_verify.py's `_encode_batch`)
    can spike-encode the feature envelopes exactly like a raw front-end's
    per-channel signal, with zero new encoding code.
    """
    n_ch, n_samples = sig.shape
    sub_len = n_samples // n_subwindows
    if sub_len < 4:
        raise ValueError(
            f"sub-window too short ({sub_len} samples) for {n_subwindows} "
            f"sub-windows over {n_samples} samples -- use fewer sub-windows "
            f"or a longer capture window.")

    out = np.zeros((len(feature_names) * n_ch, n_subwindows), dtype=np.float32)
    for c in range(n_ch):
        for w in range(n_subwindows):
            seg = sig[c, w * sub_len:(w + 1) * sub_len]
            bp = None
            for fi, fname in enumerate(feature_names):
                row = fi * n_ch + c
                if fname == "line_length":
                    out[row, w] = line_length(seg)
                elif fname == "spectral_entropy":
                    out[row, w] = spectral_entropy(seg, fs)
                else:
                    if bp is None:
                        bp = relative_band_power(seg, fs, bands)
                    if fname not in bp:
                        raise ValueError(f"unknown feature name {fname!r}")
                    out[row, w] = bp[fname]
    return out


def normalize_features_train_only(Xtr: np.ndarray, Xval: np.ndarray, Xte: np.ndarray,
                                   eps: float = 1e-8):
    """Z-score each feature-channel using stats fit on `Xtr` ONLY (mean/std
    pooled over the sample and timestep axes), applied identically to
    `Xval`/`Xte`. Guards the classic normalization leak — fitting stats on
    the whole dataset before splitting silently lets val/test statistics
    influence what the model sees as "normal" during training.

    Xtr/Xval/Xte: (n, C, T) each -> same shapes, z-scored with Xtr's stats.
    """
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True)
    std = np.where(std > eps, std, 1.0)
    return (Xtr - mean) / std, (Xval - mean) / std, (Xte - mean) / std
