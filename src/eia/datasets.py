"""ECG, PPG, and EEG dataset loading.

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


@dataclass
class EegData:
    X: np.ndarray  # (n_samples, n_channels, window) float32 — multi-channel,
                    # unlike EcgData/PpgData's (n_samples, window): Xylo's
                    # input framing needs one axis per montage channel, each
                    # delta-encoded into its own ON/OFF pair (see
                    # scripts/xylo_verify.py's `_encode_batch`).
    y: np.ndarray   # (n_samples,) int64 — 0 = non-seizure, 1 = seizure
    fs: float       # effective sampling rate (Hz) of the LAST axis (window),
                    # i.e. after any resample_to — matches `report.data_card`'s
                    # `duration_s = window / fs` for both 2-D and 3-D X.
    source: str     # "chbmit" or "synthetic"
    requested_real: bool = False
    # Canonical patient id per window (chb21 folded into chb01 — see
    # `eeg_canonical_patient`). Required: CHB-MIT's headline metric is
    # subject-independent, so every consumer must split by patient, never by
    # window (the VitalDB case-leakage lesson, same mechanism as PpgData.groups).
    groups: np.ndarray | None = None
    # Source record id per window (e.g. "chb01_03"). Finer-grained than
    # `groups`: used by `eeg_patient_specific_split` to hold out whole
    # RECORDS (not random windows) within one patient for the patient-
    # specific diagnostic (docs/eeg_seizure_task.md) — a within-patient
    # window-random split would leak the same seizure event's neighbouring
    # windows across train/test.
    record_ids: np.ndarray | None = None


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


# --------------------------------------------------------------------------- #
# EEG — real CHB-MIT seizure detection (MARCH "H", Phase 1)
# --------------------------------------------------------------------------- #
# Fixed bipolar montage (10-20 "double banana"), applied identically to every
# subject — NOT patient-specific channel selection, since a field device has
# never seen this patient (docs/eeg_seizure_task.md Flag 2). 6 channels ->
# 12 spike channels after ON/OFF delta encoding, leaving margin under Xylo's
# 16-input-channel ceiling (8ch would sit exactly at the limit with zero
# headroom). Left/right frontotemporal pairs (common seizure-focus region)
# plus the midline central-parietal pair, verified present in every
# successfully-read CHB-MIT header checked (chb01, chb02, chb03, chb05,
# chb08, chb24 — see docs/eeg_seizure_task.md Part 0).
EEG_MONTAGE = ["FP1-F7", "F7-T7", "FP2-F8", "F8-T8", "FZ-CZ", "CZ-PZ"]

EEG_NATIVE_FS = 256.0  # CHB-MIT's fixed sampling rate, confirmed via wfdb

# chb21 is the SAME PATIENT as chb01 (a second recording session) — CHB-MIT's
# own documentation states this. Canonicalizing prevents one patient's
# windows from landing in both train and test under two different ids.
_EEG_PATIENT_ALIASES = {"chb21": "chb01"}

# Subjects with confirmed-readable headers (verified via a header_only probe
# against every subject's first record before writing this default list).
# chb12, chb13, chb14, chb16, chb17, chb18, chb19, chb21 all raise a "math
# domain error" from wfdb's EDF parser on their very first record — a real,
# reproducible limitation of `wfdb.io.convert.edf.read_edf` on those specific
# files, not guessed and not per-record flakiness. `load_chbmit` also skips
# any subject/record it can't read at load time and prints why, so passing
# an unreadable subject here is harmless (0 windows from it).
# chb20, chb22, chb23, chb24 are ALSO confirmed-readable but excluded from
# this default purely for download time (each subject costs ~2-3 records x
# ~40MB streamed through wfdb's EDF reader, observed at several minutes per
# record) — pass them explicitly via `subjects=` if you want a larger pool.
DEFAULT_EEG_SUBJECTS = ["chb01", "chb02", "chb03", "chb05", "chb06", "chb07",
                         "chb08", "chb09", "chb10"]


def eeg_canonical_patient(subject: str) -> str:
    """Map a CHB-MIT subject id to its canonical patient id (chb21 -> chb01,
    the one documented same-patient alias in this dataset). Pure function so
    the group-merging logic is unit-testable without any network access."""
    return _EEG_PATIENT_ALIASES.get(subject, subject)


def select_montage(sig: np.ndarray, sig_name: list, montage: list) -> np.ndarray:
    """Select and reorder fixed montage channels from a (n_samples, n_sig)
    array using the record's own channel-name list. Raises (does not
    silently substitute a different channel) if any montage channel is
    missing — the caller should catch this and skip the record, mirroring
    VitalDB's "skip cases missing a required track" pattern.
    """
    missing = [ch for ch in montage if ch not in sig_name]
    if missing:
        raise ValueError(f"missing montage channel(s): {missing}")
    idx = [sig_name.index(ch) for ch in montage]
    return sig[:, idx]


def bandpass_eeg(sig: np.ndarray, fs: float, band: tuple = (0.5, 25.0)) -> np.ndarray:
    """4th-order Butterworth band-pass, zero-phase (`filtfilt`), along axis 0
    (time), applied to every channel. Restricts to the seizure-relevant band
    BEFORE resampling to a short Xylo timestep budget — the point being that
    256 Hz full-bandwidth resolution isn't needed once the signal is band-
    limited to ~25 Hz, so a much shorter timestep count still holds the
    seizure signature (docs/eeg_seizure_task.md's timestep-count lesson,
    same mechanism as `docs/ecg_quant_diagnosis.md`'s root cause: on-chip
    fidelity degrades with timestep count, not weight bits).
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(4, band, btype="band", fs=fs)
    return filtfilt(b, a, sig, axis=0)


def resample_windows(X: np.ndarray, resample_to: int) -> np.ndarray:
    """FFT-resample the last axis (time) of a 2-D (n, window) or 3-D
    (n, channels, window) array to `resample_to` samples — the
    `load_mitbih`-style decoupling of "how much signal is captured" from
    "how many Xylo timesteps it costs." Generic over ndim so it serves both
    the single-channel ECG/PPG path and EEG's multi-channel one.
    """
    from scipy.signal import resample as _fft_resample
    return _fft_resample(X, resample_to, axis=-1)


def parse_chbmit_summary(text: str) -> dict:
    """Parse a CHB-MIT `chbXX-summary.txt` into {record_name: [(start, end), ...]}
    (seizure intervals in seconds since the record started; empty list = no
    seizures in that record). Handles both the singular format used when a
    record has exactly one seizure (`Seizure Start Time:`) and the numbered
    format used for 2+ (`Seizure 2 Start Time:`) — confirmed against live
    files for both cases before writing this (docs/eeg_seizure_task.md Part 0).
    Pure function, no network — unit-testable on a small literal string.
    """
    import re
    records: dict = {}
    for block in text.split("File Name: ")[1:]:
        lines = block.splitlines()
        fname = lines[0].strip()
        rec = fname[:-4] if fname.endswith(".edf") else fname
        starts = [int(m) for m in re.findall(
            r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", block)]
        ends = [int(m) for m in re.findall(
            r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", block)]
        records[rec] = list(zip(starts, ends))
    return records


def _label_seizure_window(t0: float, t1: float, seizures: list) -> int:
    """1 if window [t0, t1) (seconds) overlaps any seizure interval."""
    return int(any(t0 < e and t1 > s for s, e in seizures))


def _fetch_chbmit_summary(subject: str, cache_dir: str) -> dict:
    import os
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{subject}-summary.txt")
    if not os.path.exists(path):
        url = f"https://physionet.org/files/chbmit/1.0.0/{subject}/{subject}-summary.txt"
        urllib.request.urlretrieve(url, path)
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    return parse_chbmit_summary(text)


def _load_chbmit_record_montage(subject: str, record: str, montage: list,
                                 cache_dir: str):
    """Stream one CHB-MIT record's EDF via wfdb, select the fixed montage,
    and cache the (much smaller) montage-only array to disk — repeat runs
    (e.g. a --resample-to sweep) don't need to re-download the full
    23-channel, ~42 MB-per-hour EDF every time.
    """
    import os
    path = os.path.join(cache_dir, f"{subject}_{record}_montage.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["sig"], float(d["fs"])
    from wfdb.io.convert.edf import read_edf
    os.makedirs(cache_dir, exist_ok=True)
    h = read_edf(record + ".edf", pn_dir=f"chbmit/{subject}")
    sig = select_montage(h.p_signal.astype(np.float32), h.sig_name, montage)
    fs = float(h.fs)
    np.savez(path, sig=sig, fs=fs)
    return sig, fs


def load_chbmit(
    subjects: list | None = None,
    seizure_records_per_subject: int = 2,
    nonseizure_records_per_subject: int = 1,
    window_sec: float = 4.0,
    resample_to: int | None = 128,
    montage: list | None = None,
    band_pass: tuple = (0.5, 25.0),
    neg_per_record: int = 60,
    cache_dir: str = "data/chbmit",
    seed: int = 0,
) -> EegData:
    """Stream CHB-MIT scalp EEG, windowed and labelled for seizure detection.

    Hardware-first pipeline (docs/eeg_seizure_task.md): 23 native channels ->
    fixed 6-channel bipolar montage (`EEG_MONTAGE`) -> band-pass to the
    seizure band -> per-window z-score -> FFT-resample to `resample_to`
    Xylo timesteps. Requires `wfdb` (`pip install 'eia[data]'`) and network.

    Args:
        subjects: CHB-MIT subject ids to pull from (default
            `DEFAULT_EEG_SUBJECTS`). chb21 is automatically folded into
            chb01's patient group if included (`eeg_canonical_patient`).
        seizure_records_per_subject: up to this many of the subject's
            seizure-containing records are used (ALL seizure windows from a
            chosen record are kept; see `neg_per_record` for negatives).
        nonseizure_records_per_subject: up to this many records with zero
            seizures, for non-ictal background diversity.
        window_sec: physiological capture duration per window, at native
            256 Hz (a duration, not a Xylo timestep budget — see
            `resample_to`, same decoupling as `datasets.load_mitbih`).
        resample_to: FFT-resample each window from `window_sec * 256` native
            samples down to this many Xylo timesteps; None keeps native.
        montage: override the fixed channel list (default `EEG_MONTAGE`).
        band_pass: seizure-band Butterworth cutoffs in Hz.
        neg_per_record: cap on non-seizure windows kept per record (a 1-hour
            record has ~900 windows at window_sec=4; keeping all of them
            would swamp the rare seizure windows and balloon dataset size).
            ALL seizure windows in a chosen record are always kept.
        cache_dir: per-record montage cache (gitignored `data/`).
        seed: controls which negative windows are subsampled per record.

    Returns:
        EegData with `source="chbmit"`, `groups` = canonical patient id per
        window (subject-independent split key).
    """
    try:
        import wfdb  # noqa: F401  (import check; actual reads use wfdb.io.convert.edf)
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the data extra: pip install 'eia[data]'") from e

    subjects = subjects or DEFAULT_EEG_SUBJECTS
    montage = montage or EEG_MONTAGE
    rng = np.random.default_rng(seed)
    window_native = int(round(window_sec * EEG_NATIVE_FS))

    X_list, y_list, group_list, record_list = [], [], [], []
    for subject in subjects:
        try:
            summary = _fetch_chbmit_summary(subject, cache_dir)
        except Exception as e:
            print(f"[warn] chbmit {subject}: failed to fetch summary ({e}); skipping subject.")
            continue

        seizure_recs = sorted(r for r, sz in summary.items() if sz)[:seizure_records_per_subject]
        nonseizure_recs = sorted(
            r for r, sz in summary.items() if not sz)[:nonseizure_records_per_subject]

        for record in seizure_recs + nonseizure_recs:
            try:
                sig, fs = _load_chbmit_record_montage(subject, record, montage, cache_dir)
            except Exception as e:
                print(f"[warn] chbmit {subject}/{record}: failed to load ({e}); skipping record.")
                continue

            filtered = bandpass_eeg(sig, fs, band_pass)
            n_windows = filtered.shape[0] // window_native
            seizures = summary.get(record, [])

            pos_w, neg_w = [], []
            for w in range(n_windows):
                t0, t1 = w * window_native / fs, (w + 1) * window_native / fs
                (pos_w if _label_seizure_window(t0, t1, seizures) else neg_w).append(w)
            if len(neg_w) > neg_per_record:
                neg_w = list(rng.choice(neg_w, size=neg_per_record, replace=False))

            patient = eeg_canonical_patient(subject)
            for w in pos_w + neg_w:
                seg = filtered[w * window_native:(w + 1) * window_native, :].T  # (n_ch, window)
                X_list.append(seg.astype(np.float32))
                y_list.append(1 if w in pos_w else 0)
                group_list.append(patient)
                record_list.append(record)

    if not X_list:
        raise RuntimeError("No EEG windows extracted from CHB-MIT.")

    X = np.stack(X_list)  # (n, n_ch, window_native)
    fs_eff = EEG_NATIVE_FS
    if resample_to is not None and resample_to != window_native:
        X = resample_windows(X, resample_to)
        fs_eff = EEG_NATIVE_FS * (resample_to / window_native)
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)

    return EegData(X=X.astype(np.float32), y=np.array(y_list, dtype=np.int64),
                    fs=fs_eff, source="chbmit",
                    groups=np.array(group_list, dtype="<U8"),
                    record_ids=np.array(record_list, dtype="<U16"))


def eeg_patient_specific_split(data: "EegData", patient: str, seed: int,
                                test_frac: float = 0.3, val_frac: float = 0.2):
    """Within-patient train/val/test split for the patient-specific EEG
    diagnostic (docs/eeg_seizure_task.md follow-up: disambiguating "too
    little cross-patient data" from "the front-end destroyed the seizure
    signal"). Holds out whole RECORDS for test — never individual windows —
    so a seizure event's neighbouring windows can't straddle train/test (the
    within-patient analogue of the case/patient-level leakage guard used
    everywhere else in this repo). `val` is carved window-level from the
    training records only, for checkpoint selection — that's fine because it
    is never the reported metric, only the record-disjoint test set is.

    Returns the same 9-tuple shape as `case_level.split_data`
    (Xtr, Xval, Xte, ytr, yval, yte, groups_tr, groups_val, groups_te), or
    `None` if this patient can't support a class-balanced record split:
    fewer than 2 distinct records, or no permutation of records (tried up to
    10 times) puts both classes in both the train and test halves.
    """
    if data.record_ids is None or data.groups is None:
        raise ValueError("eeg_patient_specific_split needs data.record_ids and data.groups")

    mask = data.groups == patient
    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        return None
    rec = data.record_ids[idx_all]
    y = data.y[idx_all]
    unique_records = np.unique(rec)
    if unique_records.size < 2:
        return None

    rng = np.random.default_rng(seed)
    te_mask = tr_mask = None
    for _attempt in range(10):
        perm = rng.permutation(unique_records)
        n_test = max(1, round(len(perm) * test_frac))
        n_test = min(n_test, len(perm) - 1)  # always leave >=1 record for train
        test_records = set(perm[:n_test].tolist())
        train_records = set(perm[n_test:].tolist())
        cand_te_mask = np.isin(rec, list(test_records))
        cand_tr_mask = np.isin(rec, list(train_records))
        if np.unique(y[cand_te_mask]).size > 1 and np.unique(y[cand_tr_mask]).size > 1:
            te_mask, tr_mask = cand_te_mask, cand_tr_mask
            break
    if te_mask is None:
        return None  # no record permutation gave both classes on both sides

    te_idx = idx_all[te_mask]
    tr_idx_full = idx_all[tr_mask]

    # Window-level split of the TRAINING records only, for checkpoint
    # selection — record-disjointness doesn't matter here since val is
    # never part of the reported test metric.
    y_tr_full = data.y[tr_idx_full]
    if np.unique(y_tr_full).size > 1 and tr_idx_full.size >= 4:
        from sklearn.model_selection import train_test_split
        tr_idx, val_idx = train_test_split(
            tr_idx_full, test_size=val_frac, random_state=seed, stratify=y_tr_full)
    else:
        tr_idx, val_idx = tr_idx_full, tr_idx_full

    # The last 3 slots match `case_level.split_data`'s shape (train_modality
    # only needs *a* grouping value there, unused for training itself) but
    # carry RECORD ids here, not patient ids — within one patient the patient
    # id is constant/uninformative, whereas the record id is what this split
    # actually partitioned on, and is what a leakage check should verify.
    return (data.X[tr_idx], data.X[val_idx], data.X[te_idx],
            data.y[tr_idx], data.y[val_idx], data.y[te_idx],
            data.record_ids[tr_idx], data.record_ids[val_idx], data.record_ids[te_idx])


def make_synthetic_eeg(
    n_samples: int = 800,
    n_channels: int = len(EEG_MONTAGE),
    window: int = 128,
    fs: float = 128.0,
    abnormal_frac: float = 0.15,
    noise: float = 0.3,
    seed: int = 0,
) -> EegData:
    """Generate a synthetic seizure/non-seizure EEG-like set: non-seizure
    windows are band-limited noise per channel; seizure windows add a
    higher-amplitude rhythmic (spike-wave-like) oscillation shared with
    correlated timing across channels, roughly mimicking hypersynchronous
    ictal activity vs. background — a stylised proxy, not a clinical claim
    (same spirit as the other `make_synthetic_*` generators).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, window / fs, window)
    X = np.empty((n_samples, n_channels, window), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        seizure = rng.random() < abnormal_frac
        y[i] = 1 if seizure else 0
        freq = rng.uniform(2.5, 4.0)  # spike-wave-like frequency if seizure
        for c in range(n_channels):
            sig = rng.normal(0, noise, size=window)
            if seizure:
                phase = rng.normal(0, 0.1)  # near-shared phase across channels
                sig = sig + 1.5 * np.sin(2 * np.pi * freq * t + phase)
            X[i, c] = sig.astype(np.float32)
    return EegData(X=X, y=y, fs=fs, source="synthetic")


def load_eeg(prefer_real: bool = True, require_real: bool = False,
             **kwargs) -> EegData:
    """Load EEG data, preferring real CHB-MIT but falling back to synthetic.

    Args:
        prefer_real: try `load_chbmit()` first.
        require_real: if real data was requested and fails to load, raise
            instead of silently substituting synthetic. Every allowed
            fallback still prints a `[warn]` line naming the reason.
        **kwargs: forwarded to `load_chbmit` (subjects,
            seizure_records_per_subject, nonseizure_records_per_subject,
            window_sec, resample_to, montage, band_pass, neg_per_record,
            cache_dir, seed) when real; ignored when falling back to
            synthetic.
    """
    if require_real and not prefer_real:
        raise ValueError("require_real=True has no effect with prefer_real=False "
                          "(nothing was asked to be real).")
    if prefer_real:
        try:
            data = load_chbmit(**kwargs)
            data.requested_real = True
            return data
        except Exception as e:  # noqa: BLE001  — any failure -> synthetic (or raise)
            if require_real:
                raise RuntimeError(
                    f"--require-real: real CHB-MIT EEG failed to load ({e}); "
                    "refusing to silently substitute synthetic EEG."
                ) from e
            print(f"[warn] real CHB-MIT EEG unavailable ({e}); falling back to synthetic EEG.")
    data = make_synthetic_eeg()
    data.requested_real = prefer_real
    return data
