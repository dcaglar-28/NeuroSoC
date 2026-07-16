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


def plot_crm_lead_effect(windows_per_subject: int = 24, hr_baseline: float = 70.0,
                          window_sec: float = 3.0, fs: float = 100.0, seed: int = 0):
    """Demonstrate the CRM "lead effect" (docs/synthetic_crm_task.md) on one
    synthetic subject's trajectory: pulse MORPHOLOGY degrades from the very
    start of the decline, while heart rate stays flat at baseline until
    reserve drops well below the "compromised" label threshold — i.e. a
    classifier tracking morphology alone can flag the compromise before a
    pulse check (HR) would show anything unusual.

    Top row: three example windows (early/mid/late on the trajectory) —
    amplitude drop and dicrotic-notch blunting are visible by eye. Bottom
    panel: reserve `r(t)` and heart rate `HR(t)` over the trajectory, with
    the "occult" band (`_HR_RISE_R < r <= CRI_THRESHOLD` — labelled
    compromised, HR still at baseline) shaded.

    Returns a matplotlib Figure (same convention as this module's other
    plots — `fig.savefig(...)` to export, e.g. for docs/synthetic_crm_results.md).
    """
    import matplotlib.pyplot as plt

    from . import datasets

    rng = np.random.default_rng(seed)
    r_traj = datasets._reserve_trajectory(windows_per_subject, rng)
    hr_traj = datasets._hr_from_r(r_traj, hr_baseline)
    windows = [
        datasets._crm_window(float(r), float(hr), window_sec, fs, rng, 1.0, 1.0, noise=0.02)
        for r, hr in zip(r_traj, hr_traj)
    ]

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])
    t_wave = np.arange(windows[0].size) / fs

    snapshot_idx = [0, windows_per_subject // 2, windows_per_subject - 1]
    for col, i in enumerate(snapshot_idx):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_wave, windows[i], lw=1, color="tab:blue")
        ax.set_title(f"window {i}  (r={r_traj[i]:.2f}, HR={hr_traj[i]:.0f} bpm)")
        ax.set_xlabel("time (s)")
        if col == 0:
            ax.set_ylabel("PPG (a.u.)")

    ax2 = fig.add_subplot(gs[1, :])
    step = np.arange(windows_per_subject)
    occult = (r_traj > datasets._HR_RISE_R) & (r_traj <= datasets.CRI_THRESHOLD)
    if occult.any():
        first, last = np.where(occult)[0][[0, -1]]
        ax2.axvspan(first, last, color="tab:red", alpha=0.12,
                     label="occult band: labelled compromised, HR still baseline")

    ax2r = ax2.twinx()
    ax2.plot(step, r_traj, color="tab:green", lw=2, label="reserve r(t)")
    ax2.axhline(datasets.CRI_THRESHOLD, color="tab:green", ls="--", lw=1, alpha=0.6,
                 label=f"CRI threshold ({datasets.CRI_THRESHOLD})")
    ax2r.plot(step, hr_traj, color="tab:purple", lw=2, label="heart rate (bpm)")
    ax2.set_xlabel("window index (time along trajectory)")
    ax2.set_ylabel("reserve r", color="tab:green")
    ax2r.set_ylabel("heart rate (bpm)", color="tab:purple")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax2.set_title("Morphology (reserve-driven) declines well before heart rate moves")

    fig.suptitle("Synthetic CRM trajectory — the morphology-leads-HR effect")
    fig.tight_layout()
    return fig
