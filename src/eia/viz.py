"""Visualization helpers — what the data looks like and how it's encoded.

Every function returns a matplotlib Figure so it renders inline in a notebook,
or can be saved with `fig.savefig(...)`. matplotlib is imported lazily so the
numpy-only parts of `eia` still import without it.
"""

from __future__ import annotations

import numpy as np

from . import encoding


def plot_waveforms(data, n_per_class: int = 3, seed: int = 0):
    """Overlay a few normalized example windows for each class.

    Shows what the model actually sees per class (e.g. normal vs PVC ECG beats,
    or normovolemic vs hypovolemic PPG pulses).
    """
    import matplotlib.pyplot as plt

    X, y = np.asarray(data.X), np.asarray(data.y).astype(int)
    fs = float(getattr(data, "fs", 0.0) or 0.0)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    t = (np.arange(X.shape[1]) / fs) if fs else np.arange(X.shape[1])
    xlabel = "time (s)" if fs else "sample"

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 3.2),
                             squeeze=False)
    labels = {0: "class 0", 1: "class 1"}
    for ax, c in zip(axes[0], classes):
        idx = np.where(y == c)[0]
        pick = rng.choice(idx, size=min(n_per_class, idx.size), replace=False)
        for i in pick:
            ax.plot(t, encoding.normalize(X[i]), alpha=0.8, lw=1)
        ax.set_title(f"{labels.get(int(c), f'class {c}')}  (n={idx.size})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("normalized amplitude")
    fig.suptitle(f"{type(data).__name__} — example windows per class "
                 f"[{getattr(data, 'source', '')}]")
    fig.tight_layout()
    return fig


def plot_encoding(signal, threshold: float = 0.2, fs: float | None = None):
    """Illustrate the delta / level-crossing encoding on one window.

    Three panels: (1) normalized signal with the +/- threshold bands that trigger
    events, (2) the ON/OFF delta events, (3) the 2-channel input raster the SNN
    actually receives.
    """
    import matplotlib.pyplot as plt

    sig = encoding.normalize(np.asarray(signal))
    d = encoding.delta_encode(sig, threshold)          # {-1,0,+1}
    ch = encoding.delta_encode_2ch(sig, threshold)     # (2, window)
    n = sig.size
    t = (np.arange(n) / fs) if fs else np.arange(n)
    xlabel = "time (s)" if fs else "sample"
    rate = encoding.event_rate(d)

    fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)

    ax[0].plot(t, sig, color="black", lw=1.2, label="normalized signal")
    ax[0].set_title(f"1. Signal + level-crossing threshold (±{threshold})")
    ax[0].set_ylabel("amplitude")

    on = d > 0
    off = d < 0
    ax[1].vlines(t[on], 0, 1, color="tab:green", lw=1, label="ON (+)")
    ax[1].vlines(t[off], -1, 0, color="tab:red", lw=1, label="OFF (−)")
    ax[1].set_ylim(-1.3, 1.3)
    ax[1].set_title(f"2. Delta events  (event rate = {rate:.2f} — sparser is cheaper)")
    ax[1].set_ylabel("event")
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].imshow(ch, aspect="auto", cmap="Greys", interpolation="nearest",
                 extent=[t[0], t[-1], 1.5, -0.5])
    ax[2].set_yticks([0, 1])
    ax[2].set_yticklabels(["ON", "OFF"])
    ax[2].set_title("3. Input raster fed to the SNN  (2 channels × window timesteps)")
    ax[2].set_xlabel(xlabel)

    fig.tight_layout()
    return fig


def plot_class_balance(data):
    """Bar chart of class counts — a quick imbalance check."""
    import matplotlib.pyplot as plt

    y = np.asarray(data.y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([str(int(c)) for c in classes], counts, color=["tab:blue", "tab:orange"])
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_title(f"class balance [{getattr(data, 'source', '')}]")
    fig.tight_layout()
    return fig
