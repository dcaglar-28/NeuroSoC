"""ECG and PPG dataset loading.

Each modality has two paths:
  * a real loader streamed from PhysioNet via the `wfdb` package (needs
    network + `pip install wfdb`, or the `eia[data]` extra).
  * a self-contained synthetic generator so the whole pipeline runs
    end-to-end offline, with zero downloads, the moment you clone the repo.

`load_ecg()` / `load_ppg()` try the real dataset and fall back to synthetic,
so notebooks and tests always work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EcgData:
    X: np.ndarray  # (n_samples, window) float32
    y: np.ndarray  # (n_samples,) int64  — 0 = normal, 1 = abnormal
    fs: float      # sampling rate (Hz)
    source: str    # "mitbih" or "synthetic"
    # True iff the caller asked for real data (prefer_real=True), regardless
    # of whether it actually got real data. Combined with `source`, this is
    # what distinguishes "synthetic because that's what was asked for" from
    # "synthetic because the real load silently failed" — see
    # `report.data_card`'s provenance line and `load_ecg`'s [warn] print.
    requested_real: bool = False


@dataclass
class PpgData:
    X: np.ndarray  # (n_samples, window) float32
    y: np.ndarray  # (n_samples,) int64  — 0 = normal, 1 = abnormal
    fs: float      # sampling rate (Hz)
    source: str    # "bidmc" or "synthetic"
    requested_real: bool = False


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #
def _gaussian(t, center, width, amp):
    return amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def _normal_beat(n: int, rng: np.random.Generator) -> np.ndarray:
    """A stylised normal PQRST complex over `n` samples in [0, 1)."""
    t = np.linspace(0, 1, n)
    jitter = rng.normal(0, 0.01)
    beat = (
        _gaussian(t, 0.20 + jitter, 0.025, 0.10)   # P wave
        - _gaussian(t, 0.38, 0.012, 0.15)          # Q
        + _gaussian(t, 0.42, 0.010, 1.00)          # R (tall, narrow)
        - _gaussian(t, 0.46, 0.012, 0.25)          # S
        + _gaussian(t, 0.68, 0.040, 0.30)          # T wave
    )
    return beat


def _abnormal_beat(n: int, rng: np.random.Generator) -> np.ndarray:
    """PVC-like beat: wide QRS, no clear P wave, discordant T."""
    t = np.linspace(0, 1, n)
    beat = (
        _gaussian(t, 0.42, 0.045, 1.10)            # wide, bizarre QRS
        - _gaussian(t, 0.52, 0.050, 0.55)
        - _gaussian(t, 0.72, 0.060, 0.40)          # discordant (inverted) T
    )
    return beat


def make_synthetic_ecg(
    n_samples: int = 2000,
    window: int = 187,
    fs: float = 125.0,
    abnormal_frac: float = 0.4,
    noise: float = 0.04,
    seed: int = 0,
) -> EcgData:
    """Generate a balanced-ish synthetic ECG beat-classification set.

    `window=187` and `fs=125` mirror the popular MIT-BIH preprocessing so a model
    trained here has the same input shape as one trained on the real data.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, window), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        if rng.random() < abnormal_frac:
            beat = _abnormal_beat(window, rng)
            y[i] = 1
        else:
            beat = _normal_beat(window, rng)
            y[i] = 0
        beat = beat + rng.normal(0, noise, size=window)
        X[i] = beat.astype(np.float32)
    return EcgData(X=X, y=y, fs=fs, source="synthetic")


# --------------------------------------------------------------------------- #
# Real MIT-BIH via wfdb (optional)
# --------------------------------------------------------------------------- #
def load_mitbih(
    records=("100", "101", "103", "105", "111", "118", "200", "201", "210", "214"),
    window: int = 187,
    resample_to: int | None = None,
) -> EcgData:
    """Stream a subset of MIT-BIH Arrhythmia beats from PhysioNet.

    Normal (N) vs. non-normal AAMI classes -> binary label. Requires `wfdb` and
    network access. Raises ImportError/RuntimeError if unavailable; callers that
    want a guaranteed result should use `load_ecg()` instead.

    Args:
        window: native samples extracted per beat, at MIT-BIH's native 360 Hz
            (i.e. a *physiological capture duration*, not a Xylo timestep
            budget — see `resample_to`).
        resample_to: if given, FFT-resample each extracted beat from `window`
            native samples down (or up) to this many samples, and rescale
            `fs` to match. This decouples "how much signal do we capture"
            from "how many timesteps the Xylo net spends processing it" —
            the fix for the bug documented in
            `docs/ecg_quant_diagnosis.md`/`rockpool_models.build_xylo_snn`:
            naively setting `window=90` on real data captured only 90/360 =
            250ms, not the 90/125 = 720ms the same number meant for the
            synthetic generator. Capture the physiological duration you want
            via `window` (at native fs), then use `resample_to` to pick the
            Xylo timestep count independently.
    """
    try:
        import wfdb
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the data extra: pip install 'eia[data]'") from e

    normal_syms = {"N", "L", "R", "e", "j"}
    half = window // 2
    X_list, y_list, fs = [], [], 360.0
    for rec in records:
        sig, fields = wfdb.rdsamp(rec, pn_dir="mitdb", channels=[0])
        ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")
        fs = float(fields["fs"])
        x = sig[:, 0]
        for idx, sym in zip(ann.sample, ann.symbol):
            if idx - half < 0 or idx + half + 1 > x.size:
                continue
            if sym not in normal_syms and sym not in {"V", "A", "F", "!", "E"}:
                continue
            beat = x[idx - half: idx + half + 1][:window]
            if beat.size < window:
                continue
            X_list.append(beat.astype(np.float32))
            y_list.append(0 if sym in normal_syms else 1)
    if not X_list:
        raise RuntimeError("No beats extracted from MIT-BIH.")
    X = np.stack(X_list)
    if resample_to is not None and resample_to != window:
        from scipy.signal import resample as _fft_resample
        X = _fft_resample(X, resample_to, axis=1)
        fs = fs * (resample_to / window)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return EcgData(X=X.astype(np.float32), y=np.array(y_list, dtype=np.int64),
                   fs=fs, source="mitbih")


def load_ecg(prefer_real: bool = True, require_real: bool = False,
             **synth_kwargs) -> EcgData:
    """Load ECG data, preferring real MIT-BIH but falling back to synthetic.

    Args:
        prefer_real: try `load_mitbih()` first.
        require_real: if real data was requested (`prefer_real=True`) and
            fails to load, raise instead of silently falling back to
            synthetic. Every fallback that IS allowed still prints a `[warn]`
            line naming the reason, so provenance is never silent even when
            this is left False.
        resample_to (via **synth_kwargs): real-data only (see
            `load_mitbih`) — resample each captured beat down to this many
            Xylo timesteps, independent of the capture window. Ignored for
            synthetic, which already generates directly at the target size.
    """
    if require_real and not prefer_real:
        raise ValueError("require_real=True has no effect with prefer_real=False "
                          "(nothing was asked to be real).")
    # `window`, if given, applies to whichever loader actually runs — real or
    # synthetic — so a caller probing the window/XyloSim-agreement trade-off
    # (see rockpool_models.py) gets a like-for-like comparison either way.
    window = synth_kwargs.pop("window", None)
    resample_to = synth_kwargs.pop("resample_to", None)
    if prefer_real:
        try:
            mitbih_kwargs = {}
            if window is not None:
                mitbih_kwargs["window"] = window
            if resample_to is not None:
                mitbih_kwargs["resample_to"] = resample_to
            data = load_mitbih(**mitbih_kwargs)
            data.requested_real = True
            return data
        except Exception as e:  # noqa: BLE001  — any failure -> synthetic (or raise)
            if require_real:
                raise RuntimeError(
                    f"--require-real: real MIT-BIH failed to load ({e}); "
                    "refusing to silently substitute synthetic ECG."
                ) from e
            print(f"[warn] real MIT-BIH unavailable ({e}); falling back to synthetic ECG.")
    if window is not None:
        synth_kwargs["window"] = window
    data = make_synthetic_ecg(**synth_kwargs)
    data.requested_real = prefer_real
    return data


# --------------------------------------------------------------------------- #
# PPG — synthetic generator
# --------------------------------------------------------------------------- #
def _normal_pulse(n: int, rng: np.random.Generator) -> np.ndarray:
    """A stylised normovolemic PPG pulse: sharp systolic peak + clear
    dicrotic notch (the reflected-wave bump), full pulse amplitude.
    """
    t = np.linspace(0, 1, n)
    jitter = rng.normal(0, 0.01)
    pulse = (
        _gaussian(t, 0.28 + jitter, 0.055, 1.00)    # systolic peak
        + _gaussian(t, 0.55, 0.030, 0.15)           # dicrotic notch (small dip)
        + _gaussian(t, 0.62, 0.070, 0.35)           # diastolic (reflected) wave
    )
    return pulse


def _hypovolemic_pulse(n: int, rng: np.random.Generator) -> np.ndarray:
    """A blood-loss-like PPG pulse: reduced overall amplitude, blunted /
    absent dicrotic notch, and a narrower pulse — the qualitative waveform
    changes that underlie Compensatory Reserve Index estimation as central
    volume drops before vital signs change.
    """
    t = np.linspace(0, 1, n)
    amp = rng.uniform(0.35, 0.60)  # markedly reduced pulse amplitude
    pulse = (
        _gaussian(t, 0.30, 0.040, amp)              # narrower, smaller peak
        + _gaussian(t, 0.50, 0.060, amp * 0.25)      # blunted runoff, no notch
    )
    return pulse


def make_synthetic_ppg(
    n_samples: int = 2000,
    window: int = 125,
    fs: float = 125.0,
    abnormal_frac: float = 0.4,
    noise: float = 0.03,
    seed: int = 0,
) -> PpgData:
    """Generate a balanced-ish synthetic PPG pulse-classification set.

    `window=125` at `fs=125` Hz is one second — roughly one pulse cycle —
    mirroring the per-beat framing used for ECG.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, window), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        if rng.random() < abnormal_frac:
            pulse = _hypovolemic_pulse(window, rng)
            y[i] = 1
        else:
            pulse = _normal_pulse(window, rng)
            y[i] = 0
        pulse = pulse + rng.normal(0, noise, size=window)
        X[i] = pulse.astype(np.float32)
    return PpgData(X=X, y=y, fs=fs, source="synthetic")


# --------------------------------------------------------------------------- #
# PPG — real BIDMC PPG & Respiration dataset via wfdb (optional)
# --------------------------------------------------------------------------- #
def load_bidmc_ppg(
    records=None,
    window_sec: float = 4.0,
    spo2_threshold: float = 95.0,
) -> PpgData:
    """Stream PLETH (PPG) windows from the BIDMC PPG & Respiration dataset.

    BIDMC has no hemorrhage/blood-loss labels, so this uses SpO2 desaturation
    (mean SpO2 over the window < `spo2_threshold`) as an accessible real-data
    proxy for physiological compromise — the same binary-window framing the
    hemorrhage task will eventually use, not a clinical hemorrhage claim.
    Requires `wfdb` and network access.
    """
    try:
        import wfdb
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the data extra: pip install 'eia[data]'") from e

    if records is None:
        records = [f"bidmc{i:02d}" for i in range(1, 11)]

    fs = 125.0
    window = int(round(window_sec * fs))
    X_list, y_list = [], []
    for rec_name in records:
        sig_rec = wfdb.rdrecord(rec_name, pn_dir="bidmc")
        num_rec = wfdb.rdrecord(rec_name + "n", pn_dir="bidmc")

        pleth_idx = next(
            i for i, name in enumerate(sig_rec.sig_name) if "PLETH" in name
        )
        spo2_idx = next(
            i for i, name in enumerate(num_rec.sig_name) if "SpO2" in name
        )
        pleth = sig_rec.p_signal[:, pleth_idx]
        spo2 = num_rec.p_signal[:, spo2_idx]  # 1 Hz

        n_windows = min(pleth.size // window, spo2.size // int(window_sec))
        for w in range(n_windows):
            seg = pleth[w * window: (w + 1) * window]
            spo2_seg = spo2[w * int(window_sec): (w + 1) * int(window_sec)]
            if np.isnan(seg).any() or np.isnan(spo2_seg).any():
                continue
            X_list.append(seg.astype(np.float32))
            y_list.append(1 if spo2_seg.mean() < spo2_threshold else 0)

    if not X_list:
        raise RuntimeError("No PPG windows extracted from BIDMC.")
    X = np.stack(X_list)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return PpgData(X=X.astype(np.float32), y=np.array(y_list, dtype=np.int64),
                    fs=fs, source="bidmc")


def load_ppg(prefer_real: bool = True, require_real: bool = False,
             **synth_kwargs) -> PpgData:
    """Load PPG data, preferring real BIDMC but falling back to synthetic.

    Args:
        prefer_real: try `load_bidmc_ppg()` first.
        require_real: if real data was requested (`prefer_real=True`) and
            fails to load, raise instead of silently falling back to
            synthetic. Every fallback that IS allowed still prints a `[warn]`
            line naming the reason, so provenance is never silent even when
            this is left False.
    """
    if require_real and not prefer_real:
        raise ValueError("require_real=True has no effect with prefer_real=False "
                          "(nothing was asked to be real).")
    if prefer_real:
        try:
            data = load_bidmc_ppg()
            data.requested_real = True
            return data
        except Exception as e:  # noqa: BLE001  — any failure -> synthetic (or raise)
            if require_real:
                raise RuntimeError(
                    f"--require-real: real BIDMC PPG failed to load ({e}); "
                    "refusing to silently substitute synthetic PPG."
                ) from e
            print(f"[warn] real BIDMC PPG unavailable ({e}); falling back to synthetic PPG.")
    data = make_synthetic_ppg(**synth_kwargs)
    data.requested_real = prefer_real
    return data
