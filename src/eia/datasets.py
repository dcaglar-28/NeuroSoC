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
    source: str    # "bidmc", "vitaldb", or "synthetic"
    requested_real: bool = False
    # Case/subject id per sample, for datasets where multiple windows come
    # from the same underlying case (e.g. VitalDB: many windows per surgical
    # case). None for datasets with no such grouping (bidmc, synthetic) —
    # callers must split by group instead of by window when this is set, to
    # avoid case-level leakage between train/val/test.
    groups: np.ndarray | None = None


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


# --------------------------------------------------------------------------- #
# PPG — real VitalDB (intraoperative), with a genuine blood-loss label
# --------------------------------------------------------------------------- #
def vitaldb_ebl_labels(ebl: np.ndarray, ebl_threshold: float = 500.0) -> np.ndarray:
    """Binary blood-loss label from case-level estimated blood loss (mL):
    1 = `ebl >= ebl_threshold` (significant), 0 = below it. Pulled out as a
    pure function (no network) so the exact thresholding logic used by
    `load_vitaldb_ppg` is unit-testable on a plain array.
    """
    ebl = np.asarray(ebl, dtype=float)
    return (ebl >= ebl_threshold).astype(np.int64)


def load_vitaldb_ppg(
    max_cases: int | None = 30,
    window_sec: float = 4.0,
    resample_to: int | None = 125,
    max_windows_per_case: int = 40,
    ebl_threshold: float = 500.0,
    cache_dir: str = "data/vitaldb",
    seed: int = 0,
) -> PpgData:
    """Stream fingertip PPG (SNUADC/PLETH) windows from VitalDB, labelled by
    case-level estimated blood loss (`intraop_ebl`) — the first *open* PPG
    dataset in this repo with a real (if coarse) hemorrhage-relevant label,
    vs. BIDMC's SpO2-desaturation proxy. Requires `vitaldb` (`pip install
    'eia[data]'`) and network access.

    Field names verified against the installed `vitaldb` (1.5.8) library and
    live API before writing this loader (see docs/vitaldb_ppg_hemorrhage_task.md
    Part 0): clinical table via `load_clinical_data(caseids=list(range(1,6389)))`
    — note `caseids=[]` returns an EMPTY frame despite the docstring claiming
    "all cases", so every caseid must be listed explicitly — has an
    `intraop_ebl` column (mL, non-null for 3987/6388 cases); the raw waveform
    track is `SNUADC/PLETH` (confirmed 500 Hz via `load_case(..., interval=1/500)`,
    present in 6157/6388 cases); 3781 cases have both.

    Label (v1, binary, case-level, documented as coarse): 1 = "significant"
    blood loss (`intraop_ebl >= ebl_threshold` mL, default 500 mL — a common
    surgical-transfusion-trigger threshold; ~14% of qualifying cases at this
    cutoff), 0 = below it. Every window from a case inherits that case's
    single label — this is NOT a moment-of-bleeding label (see caveats below).

    Args:
        max_cases: subset of qualifying cases to actually download and use
            (a full VitalDB pull is large; None = use all ~3781 qualifying
            cases). A seeded random subset, not just the first N by caseid,
            so a small run isn't biased toward low case-ids.
        window_sec: physiological capture duration per segment, at VitalDB's
            native 500 Hz (mirrors `load_mitbih`'s `window`/`resample_to`
            split: this is NOT the Xylo timestep count).
        resample_to: FFT-resample each captured window down to this many
            Xylo timesteps (default 125, matching the existing PPG framing);
            None keeps the native `window_sec * 500` samples.
        max_windows_per_case: cap segments taken from any one case, so a
            multi-hour case doesn't dominate the dataset relative to a short
            one.
        ebl_threshold: mL cutoff for the binary label (see above).
        cache_dir: where per-case raw waveforms are cached as .npy (gitignored
            `data/`), so repeat runs (e.g. a multi-seed sweep) don't re-pull
            the same case from VitalDB every time.
        seed: controls the random subset of cases when `max_cases` is set.

    Returns:
        PpgData with `source="vitaldb"` and `groups` set to the case id of
        each window, for case-level (not window-level) train/val/test splits
        — required because windows from the same case are highly correlated
        and share one label by construction.
    """
    try:
        import vitaldb
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the data extra: pip install 'eia[data]'") from e
    import os

    native_fs = 500.0
    clinical = vitaldb.load_clinical_data(caseids=list(range(1, 6389)))
    ebl_by_case = clinical.set_index("caseid")["intraop_ebl"]
    ebl_cases = set(ebl_by_case[ebl_by_case.notna()].index.tolist())
    pleth_cases = set(vitaldb.find_cases("SNUADC/PLETH"))
    qualifying = sorted(ebl_cases & pleth_cases)
    if not qualifying:
        raise RuntimeError("No VitalDB cases with both SNUADC/PLETH and intraop_ebl.")

    if max_cases is not None and max_cases < len(qualifying):
        rng = np.random.default_rng(seed)
        qualifying = sorted(rng.choice(qualifying, size=max_cases, replace=False).tolist())

    os.makedirs(cache_dir, exist_ok=True)
    window = int(round(window_sec * native_fs))
    X_list, y_list, group_list = [], [], []
    for caseid in qualifying:
        cache_path = os.path.join(cache_dir, f"case{caseid}_pleth.npy")
        if os.path.exists(cache_path):
            wave = np.load(cache_path)
        else:
            wave = vitaldb.load_case(caseid, ["SNUADC/PLETH"], interval=1 / native_fs).ravel()
            np.save(cache_path, wave)

        n_windows = wave.size // window
        good_segs = []
        for w in range(n_windows):
            seg = wave[w * window:(w + 1) * window]
            if np.isnan(seg).any() or np.std(seg) < 1e-6:
                continue
            good_segs.append(seg)
        if not good_segs:
            continue
        if len(good_segs) > max_windows_per_case:
            idx = np.linspace(0, len(good_segs) - 1, max_windows_per_case).round().astype(int)
            good_segs = [good_segs[i] for i in idx]

        label = int(vitaldb_ebl_labels(np.array([ebl_by_case.loc[caseid]]), ebl_threshold)[0])
        for seg in good_segs:
            X_list.append(seg.astype(np.float32))
            y_list.append(label)
            group_list.append(caseid)

    if not X_list:
        raise RuntimeError("No PPG windows extracted from VitalDB.")
    X = np.stack(X_list)
    fs = native_fs
    if resample_to is not None and resample_to != window:
        from scipy.signal import resample as _fft_resample
        X = _fft_resample(X, resample_to, axis=1)
        fs = native_fs * (resample_to / window)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return PpgData(X=X.astype(np.float32), y=np.array(y_list, dtype=np.int64),
                    fs=fs, source="vitaldb", groups=np.array(group_list, dtype=np.int64))


def load_ppg_vitaldb(prefer_real: bool = True, require_real: bool = False,
                      **kwargs) -> PpgData:
    """Load VitalDB PPG, falling back to synthetic PPG on failure — mirrors
    `load_ppg`'s provenance contract exactly, but for VitalDB specifically
    (kept separate from `load_ppg`/BIDMC per docs/vitaldb_ppg_hemorrhage_task.md:
    "keep BIDMC as a secondary dataset", not folded into the same fallback
    chain).

    Args:
        prefer_real: try `load_vitaldb_ppg()` first.
        require_real: if real data was requested and fails to load, raise
            instead of silently substituting synthetic. Every allowed
            fallback still prints a `[warn]` line naming the reason.
        **kwargs: forwarded to `load_vitaldb_ppg` (max_cases, window_sec,
            resample_to, max_windows_per_case, ebl_threshold, cache_dir,
            seed) when real; ignored when falling back to synthetic (the
            synthetic generator has its own unrelated kwargs).
    """
    if require_real and not prefer_real:
        raise ValueError("require_real=True has no effect with prefer_real=False "
                          "(nothing was asked to be real).")
    if prefer_real:
        try:
            data = load_vitaldb_ppg(**kwargs)
            data.requested_real = True
            return data
        except Exception as e:  # noqa: BLE001  — any failure -> synthetic (or raise)
            if require_real:
                raise RuntimeError(
                    f"--require-real: real VitalDB PPG failed to load ({e}); "
                    "refusing to silently substitute synthetic PPG."
                ) from e
            print(f"[warn] real VitalDB PPG unavailable ({e}); falling back to synthetic PPG.")
    data = make_synthetic_ppg()
    data.requested_real = prefer_real
    return data
