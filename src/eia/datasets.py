"""ECG, PPG, and heart-sound dataset loading.

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
class HeartData:
    X: np.ndarray  # (n_samples, window) float32 — single-channel PCG, same
                    # 2-D convention as EcgData (heart sounds are an
                    # audio-class 1-D signal, same shape as ECG).
    y: np.ndarray   # (n_samples,) int64 — 0 = normal, 1 = abnormal
    fs: float       # sampling rate (Hz)
    source: str     # "cinc2016" or "synthetic"
    requested_real: bool = False
    # Subject id per window, when recoverable. CinC 2016's distributed files
    # (RECORDS/REFERENCE.csv/.hea headers — verified, not guessed; see
    # docs/heart_sounds_task.md Part 0) do NOT expose a subject id, even
    # though the dataset's own documentation confirms one subject may
    # contribute 1-6 recordings — so this is always None for "cinc2016" and
    # the split is by RECORDING, not by subject (documented caveat, not a
    # silent gap). Kept for the same reason PpgData carries it: so a future
    # subject-mapping (if one becomes available) slots in without an API
    # change.
    groups: np.ndarray | None = None
    # "raw" (default, delta-encoded waveform, per-window self-normalized —
    # X is (n_samples, window)) or "features" (docs/heart_sounds_task.md's
    # escalation path: raw delta came back ~chance, murmurs are spectral —
    # line length / relative band power / spectral entropy per sub-window
    # via `eia.signal_features`, X becomes (n_samples, n_features,
    # n_subwindows), NOT yet normalized — callers must run
    # `signal_features.normalize_features_train_only` AFTER splitting, never
    # before (train-only fit)). Real cinc2016 only; synthetic stays raw.
    frontend: str = "raw"


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
# Heart sounds (PCG) — synthetic generator
# --------------------------------------------------------------------------- #
def _s1s2_beat(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stylised normal heart-sound cycle: S1 (louder, lower-pitch "lub") then
    S2 (softer, higher-pitch "dub") with a systolic gap between them and a
    longer diastolic gap after — the two-beat cadence a listener recognizes
    as a normal heartbeat.
    """
    t = np.linspace(0, 1, n)
    jitter = rng.normal(0, 0.01)
    s1 = _gaussian(t, 0.15 + jitter, 0.02, 1.0) * np.sin(2 * np.pi * 12 * t)
    s2 = _gaussian(t, 0.45 + jitter, 0.015, 0.6) * np.sin(2 * np.pi * 18 * t)
    return s1 + s2


def _murmur_beat(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stylised abnormal cycle: the same S1/S2 structure, plus a systolic
    murmur — smoothed (crudely band-limited) noise filling the S1-S2 gap,
    mimicking turbulent flow through a diseased valve (mitral regurgitation /
    aortic stenosis — the causes CinC 2016's own documentation names as
    typical for its abnormal recordings).
    """
    beat = _s1s2_beat(n, rng)
    t = np.linspace(0, 1, n)
    systolic = ((t > 0.18) & (t < 0.42)).astype(np.float64)
    raw_noise = rng.normal(0, 1.0, size=n)
    # Cheap band-limiting: a short moving-average smooths out the highest
    # frequencies without importing scipy just for synthetic data (matches
    # the other make_synthetic_* generators' pure-NumPy convention).
    kernel = np.ones(5) / 5
    murmur = np.convolve(raw_noise, kernel, mode="same") * systolic * 0.4
    return beat + murmur


def make_synthetic_heart(
    n_samples: int = 2000,
    window: int = 128,
    fs: float = 128.0,
    abnormal_frac: float = 0.3,
    noise: float = 0.03,
    seed: int = 0,
) -> HeartData:
    """Generate a balanced-ish synthetic heart-sound (PCG) set: normal
    S1-S2 "lub-dub" cycles vs. abnormal cycles with an added systolic
    murmur — a stylised proxy for valve regurgitation/stenosis, not a
    clinical claim (same spirit as `make_synthetic_ecg`/`make_synthetic_ppg`).
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, window), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        if rng.random() < abnormal_frac:
            beat = _murmur_beat(window, rng)
            y[i] = 1
        else:
            beat = _s1s2_beat(window, rng)
            y[i] = 0
        beat = beat + rng.normal(0, noise, size=window)
        X[i] = beat.astype(np.float32)
    return HeartData(X=X, y=y, fs=fs, source="synthetic")


# --------------------------------------------------------------------------- #
# Heart sounds (PCG) — real PhysioNet/CinC Challenge 2016 via wfdb (optional)
# --------------------------------------------------------------------------- #
CINC2016_SETS = ("a", "b", "c", "d", "e", "f")
CINC2016_NATIVE_FS = 2000.0

# Bands for the "features" front-end (docs/heart_sounds_task.md's escalation
# path: raw delta-encoding measured ~chance, and a murmur is a spectral/
# turbulent-flow signature within the 20-400 Hz PCG band, not an edge — see
# `eia.signal_features`). "low" ~ S1/S2 fundamental energy; "high" ~ where
# turbulent-flow murmur energy concentrates (mitral regurgitation/aortic
# stenosis) — the same "two extremes of the band" pattern the EEG front-end
# used (delta+beta) before it was retired.
PCG_BANDS = {"low": (20.0, 100.0), "mid": (100.0, 200.0), "high": (200.0, 400.0)}
PCG_FEATURE_NAMES = ("line_length", "low", "high", "spectral_entropy")


def _bandpass_filter(sig: np.ndarray, fs: float, band: tuple) -> np.ndarray:
    """4th-order Butterworth band-pass, zero-phase (`filtfilt`), along the
    signal's time axis. Generic (not modality-specific) — heart sounds use
    ~20-400 Hz (S1/S2 energy is low, murmurs higher).
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(4, band, btype="band", fs=fs)
    return filtfilt(b, a, sig, axis=0)


def _fetch_cinc2016_index(training_set: str, cache_dir: str,
                           timeout_sec: float = 30.0) -> list:
    """Fetch `RECORDS`/`REFERENCE.csv`/`REFERENCE-SQI.csv` for one CinC 2016
    training set (small text files, cached to disk) and return
    `[(record_name, label, quality), ...]`.

    Confirmed live against the installed data before writing this loader
    (docs/heart_sounds_task.md Part 0): `REFERENCE.csv` is
    `record,label` with **label 1 = abnormal, -1 = normal** (cross-checked
    against the record's own `.hea` comment line, e.g. `# Abnormal`);
    `REFERENCE-SQI.csv` is `record,label,quality` with quality 0/1 (a
    signal-quality index — ~4% of training-a is quality=0, skewed heavily
    toward the abnormal class). `quality` is `None` if a set's
    `REFERENCE-SQI.csv` can't be fetched (kept, not excluded, since quality
    is unknown rather than confirmed poor).
    """
    import os
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    base = f"https://physionet.org/files/challenge-2016/1.0.0/training-{training_set}"

    def _fetch_text(name):
        path = os.path.join(cache_dir, f"training-{training_set}-{name}")
        if not os.path.exists(path):
            with urllib.request.urlopen(f"{base}/{name}", timeout=timeout_sec) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
        with open(path, "r", errors="ignore") as f:
            return f.read()

    labels = {}
    for line in _fetch_text("REFERENCE.csv").strip().splitlines():
        rec, lab = line.split(",")
        labels[rec] = int(lab)

    quality = {}
    try:
        for line in _fetch_text("REFERENCE-SQI.csv").strip().splitlines():
            parts = line.split(",")
            quality[parts[0]] = int(parts[2])
    except Exception:  # noqa: BLE001 — some sets could lack this file; unknown, not excluded
        pass

    return [(rec, lab, quality.get(rec)) for rec, lab in labels.items()]


def _load_cinc2016_record(training_set: str, record: str, cache_dir: str):
    """Stream one CinC 2016 record's PCG channel via wfdb and cache it.

    Selects the channel named `"PCG"` explicitly rather than assuming a
    fixed index — training-a's records also carry a simultaneous `"ECG"`
    reference channel (confirmed live), so index 0 is not reliably the PCG
    channel across every training set (all others carry PCG only).
    """
    import os
    path = os.path.join(cache_dir, f"training-{training_set}_{record}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["sig"], float(d["fs"])
    import wfdb
    os.makedirs(cache_dir, exist_ok=True)
    rec = wfdb.rdrecord(record, pn_dir=f"challenge-2016/1.0.0/training-{training_set}")
    pcg_idx = rec.sig_name.index("PCG")
    sig = rec.p_signal[:, pcg_idx].astype(np.float32)
    fs = float(rec.fs)
    np.savez(path, sig=sig, fs=fs)
    return sig, fs


def load_cinc2016(
    training_sets: tuple | None = None,
    max_records: int | None = 300,
    window_sec: float = 3.0,
    resample_to: int | None = 128,
    band_pass: tuple = (20.0, 400.0),
    min_quality: int = 1,
    max_windows_per_record: int = 20,
    cache_dir: str = "data/cinc2016",
    seed: int = 0,
    verbose: bool = True,
    record_timeout_sec: float = 60.0,
    retries: int = 1,
    heart_frontend: str = "features",
    n_subwindows: int = 24,
    feature_names: tuple | None = None,
) -> HeartData:
    """Stream PhysioNet/CinC Challenge 2016 heart-sound (PCG) recordings,
    windowed and labelled normal/abnormal.

    Part 0 findings (docs/heart_sounds_task.md — verified against the live
    dataset before writing this loader, not assumed):
    - WFDB-native (`.hea`/`.dat`, also `.wav`), streams directly via
      `wfdb.rdrecord(record, pn_dir="challenge-2016/1.0.0/training-<set>")`
      — no format conversion needed (unlike CHB-MIT's EDF).
    - Native fs = 2000 Hz, confirmed live (`CINC2016_NATIVE_FS`).
    - 6 training sets (a-f), 3240 recordings total, 665 abnormal / 2575
      normal (~20.5% abnormal) — imbalanced, but far milder than CHB-MIT's
      seizure prevalence; the existing class-weighted-CE + balanced-accuracy
      machinery is applied from the start regardless.
    - **Subject ids are NOT recoverable.** CinC 2016's own documentation
      states a subject may contribute 1-6 recordings, but no subject-id
      field exists in any distributed file (`RECORDS`, `REFERENCE.csv`,
      `REFERENCE-SQI.csv`, or the `.hea` header) — confirmed by direct
      inspection, not assumed. `HeartData.groups` is therefore always
      `None` for this loader; the split is by RECORDING, documented here
      and in the data card, matching common practice in published CinC
      2016 work (patient ids are withheld from the public release).
    - Recordings vary from ~5s to ~120s (documented range) — segmented
      into fixed `window_sec` windows here, capped per recording via
      `max_windows_per_record` so long recordings don't dominate.

    Args:
        training_sets: which of `CINC2016_SETS` ("a".."f") to pull from
            (default all six).
        max_records: cap on how many recordings (across all requested sets)
            to actually pull, a seeded random subset of the qualifying pool
            — keeps a default run fast; increase for a fuller result.
        window_sec: physiological capture duration per window, at native
            2000 Hz (a duration, not a Xylo timestep budget — see
            `resample_to`, same decoupling as `load_mitbih`).
        resample_to: `heart_frontend="raw"` only. FFT-resample each window
            from `window_sec * 2000` native samples down to this many Xylo
            timesteps. **Measured to break the signal, not just cost
            fidelity**: at the defaults (window_sec=3.0, resample_to=128)
            the effective rate is 2000*(128/6000) = ~42.7 Hz, a Nyquist of
            ~21 Hz — below where heart sounds live (S1/S2 ~20-150 Hz,
            murmurs up to ~400+ Hz), so the raw-waveform path band-limits
            away the diagnostic content before the SNN ever sees it (float
            balanced acc measured ~0.50, flat chance — see
            docs/heart_sounds_results.md). This is *why* `heart_frontend`
            defaults to `"features"` instead: `"features"` computes
            band-power at/near the NATIVE 2000 Hz per sub-window, so the
            aggressive rate reduction happens AFTER the spectral content is
            captured, not before.
        band_pass: PCG-band Butterworth cutoffs in Hz (~20-400 Hz: S1/S2
            energy is low, murmurs extend higher).
        min_quality: exclude recordings with a `REFERENCE-SQI.csv` quality
            flag below this (default 1 = exclude the ~4% flagged poor-
            quality; a documented Part-0 choice, not a silent default).
        max_windows_per_record: cap on windows kept per recording (a 120s
            recording at window_sec=3 gives 40 windows; capping keeps one
            long recording from dominating the dataset).
        cache_dir: per-recording cache (gitignored `data/`).
        seed: controls the random `max_records` subsample and any
            downstream window subsampling.
        verbose: print the pull plan and per-recording progress.
        record_timeout_sec: wall-clock cap on each index fetch / record
            read (`_call_with_timeout`, a daemon-thread timeout — same
            resumable/watchable pattern used for CHB-MIT). A timed-out or
            failing recording is skipped, never a hang.
        retries: extra attempts (beyond the first) before giving up.
        heart_frontend: `"features"` (default) or `"raw"` — see the
            `resample_to` note above for why raw-waveform delta-encoding at
            a downsampled rate measured ~chance. `"features"` extracts line
            length / relative band power (`PCG_BANDS`) / spectral entropy
            per sub-window at/near native 2000 Hz (`eia.signal_features`),
            ignores `resample_to` (`n_subwindows` sets the timestep count
            instead), and returns X NOT yet normalized — callers must run
            `signal_features.normalize_features_train_only` after splitting.
            `"raw"` is kept selectable for A/B (`--heart-frontend raw` on
            `scripts/xylo_verify.py`), not removed.
        n_subwindows: only used when `heart_frontend="features"` — number of
            equal sub-windows per capture window (each becomes one Xylo
            timestep).
        feature_names: only used when `heart_frontend="features"` — default
            `PCG_FEATURE_NAMES` if None.

    Returns:
        HeartData with `source="cinc2016"`, `groups=None` (see above).
    """
    try:
        import wfdb  # noqa: F401  (import check; actual reads use wfdb.rdrecord)
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the data extra: pip install 'eia[data]'") from e

    import time

    if heart_frontend not in ("raw", "features"):
        raise ValueError(f"heart_frontend must be 'raw' or 'features', got {heart_frontend!r}")
    if heart_frontend == "features" and feature_names is None:
        feature_names = PCG_FEATURE_NAMES

    training_sets = training_sets or CINC2016_SETS
    rng = np.random.default_rng(seed)
    window_native = int(round(window_sec * CINC2016_NATIVE_FS))

    # Phase 1: fetch each set's index (small text files) and build the full
    # pull plan up front, filtered by quality.
    pool = []  # [(training_set, record, label01), ...]
    for ts in training_sets:
        entries, err = _call_with_timeout(
            _fetch_cinc2016_index, record_timeout_sec, retries, ts, cache_dir)
        if err is not None:
            print(f"[warn] cinc2016 training-{ts}: failed to fetch index ({err}); skipping set.")
            continue
        for rec, lab, qual in entries:
            if qual is not None and qual < min_quality:
                continue
            pool.append((ts, rec, 1 if lab == 1 else 0))

    if not pool:
        raise RuntimeError("No CinC 2016 records found (index fetch failed for every training set).")

    if max_records is not None and max_records < len(pool):
        chosen = rng.choice(len(pool), size=max_records, replace=False)
        pool = [pool[i] for i in sorted(chosen.tolist())]

    n_abn = sum(1 for _, _, lab in pool if lab == 1)
    if verbose:
        print(f"[cinc2016] plan: {len(pool)} recordings "
              f"({n_abn} abnormal, {len(pool) - n_abn} normal), cache_dir={cache_dir!r}")

    # Phase 2: load + window each planned recording.
    X_list, y_list = [], []
    for i, (ts, record, label) in enumerate(pool, 1):
        if verbose:
            print(f"[cinc2016] training-{ts}/{record} ({i}/{len(pool)})")
        t0 = time.time()
        loaded, err = _call_with_timeout(
            _load_cinc2016_record, record_timeout_sec, retries, ts, record, cache_dir)
        elapsed = time.time() - t0
        if err is not None:
            print(f"[warn] cinc2016 training-{ts}/{record}: failed to load "
                  f"({err}) after {elapsed:.0f}s; skipping record.")
            continue
        sig, fs = loaded
        if not np.isfinite(sig).all():
            # A real, confirmed data-quality issue (not guessed): a handful
            # of CinC 2016 recordings (e.g. a0018, a0204) have NaN samples in
            # their raw signal that REFERENCE-SQI.csv's quality flag does
            # NOT catch -- filtfilt propagates a single NaN sample across the
            # WHOLE filtered signal, silently poisoning every window from
            # that recording. Skip the recording rather than train on it.
            if verbose:
                print(f"[cinc2016]   non-finite raw signal; skipping record.")
            continue

        filtered = _bandpass_filter(sig, fs, band_pass)
        n_windows = filtered.shape[0] // window_native
        if n_windows == 0:
            if verbose:
                print(f"[cinc2016]   shorter than one {window_sec}s window; skipping.")
            continue
        win_idx = list(range(n_windows))
        if len(win_idx) > max_windows_per_record:
            win_idx = sorted(rng.choice(win_idx, size=max_windows_per_record, replace=False).tolist())
        for w in win_idx:
            seg = filtered[w * window_native:(w + 1) * window_native]
            if heart_frontend == "features":
                from eia import signal_features
                feat = signal_features.extract_window_features(
                    seg[None, :], fs, n_subwindows, feature_names, PCG_BANDS)
                X_list.append(feat.astype(np.float32))  # (n_features, n_subwindows)
            else:
                X_list.append(seg.astype(np.float32))
            y_list.append(label)

        if verbose:
            print(f"[cinc2016]   loaded in {elapsed:.1f}s -- "
                  f"{len(win_idx)} windows this recording, {len(X_list)} cumulative")

    if not X_list:
        raise RuntimeError("No PCG windows extracted from CinC 2016.")

    X = np.stack(X_list)  # (n, window_native) raw, or (n, n_features, n_subwindows) features
    if heart_frontend == "features":
        # fs of the TIMESTEP axis (sub-windows), not the native PCG rate --
        # matches report.data_card's generic `duration_s = window / fs`.
        fs_eff = n_subwindows / window_sec
        # Deliberately NOT normalized here -- z-score stats must be fit on
        # the TRAIN split only (signal_features.normalize_features_train_only),
        # and this function doesn't know the split.
    else:
        fs_eff = CINC2016_NATIVE_FS
        if resample_to is not None and resample_to != window_native:
            X = resample_windows(X, resample_to)
            fs_eff = CINC2016_NATIVE_FS * (resample_to / window_native)
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    return HeartData(X=X.astype(np.float32), y=np.array(y_list, dtype=np.int64),
                      fs=fs_eff, source="cinc2016", groups=None, frontend=heart_frontend)


def load_heart(prefer_real: bool = True, require_real: bool = False,
               **kwargs) -> HeartData:
    """Load heart-sound data, preferring real CinC 2016 but falling back to
    synthetic. Mirrors `load_ecg`'s provenance contract exactly.

    Args:
        prefer_real: try `load_cinc2016()` first.
        require_real: if real data was requested and fails to load, raise
            instead of silently substituting synthetic. Every allowed
            fallback still prints a `[warn]` line naming the reason.
        **kwargs: forwarded to `load_cinc2016` (training_sets, max_records,
            window_sec, resample_to, band_pass, min_quality,
            max_windows_per_record, cache_dir, seed) when real; ignored
            when falling back to synthetic.
    """
    if require_real and not prefer_real:
        raise ValueError("require_real=True has no effect with prefer_real=False "
                          "(nothing was asked to be real).")
    if prefer_real:
        try:
            data = load_cinc2016(**kwargs)
            data.requested_real = True
            return data
        except Exception as e:  # noqa: BLE001  — any failure -> synthetic (or raise)
            if require_real:
                raise RuntimeError(
                    f"--require-real: real CinC 2016 heart sounds failed to load ({e}); "
                    "refusing to silently substitute synthetic heart sounds."
                ) from e
            print(f"[warn] real CinC 2016 unavailable ({e}); falling back to synthetic heart sounds.")
    data = make_synthetic_heart()
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
# Generic loader helpers (resampling + network timeout/retry) — originally
# added for EEG's CHB-MIT loader, kept because the heart-sound CinC 2016
# loader (`load_cinc2016`) also depends on both. Not modality-specific.
# --------------------------------------------------------------------------- #
def resample_windows(X: np.ndarray, resample_to: int) -> np.ndarray:
    """FFT-resample the last axis (time) of a 2-D (n, window) or 3-D
    (n, channels, window) array to `resample_to` samples — the
    `load_mitbih`-style decoupling of "how much signal is captured" from
    "how many Xylo timesteps it costs." Generic over ndim so it serves any
    single-channel (ECG/PPG/heart) or multi-channel modality alike.
    """
    from scipy.signal import resample as _fft_resample
    return _fft_resample(X, resample_to, axis=-1)


def _call_with_timeout(fn, timeout_sec: float, retries: int = 0, *args, **kwargs):
    """Run `fn(*args, **kwargs)` in a daemon thread with a wall-clock
    timeout, retrying up to `retries` times on timeout or exception. Returns
    `(result, None)` on success, `(None, error)` if every attempt failed —
    never raises, so callers can skip-and-continue uniformly (mirrors the
    "skip records that fail to load" behavior in `load_cinc2016`).

    A thread, not `signal.alarm`: works the same on Colab (Linux) and
    locally (macOS/Windows), and lets tests exercise it directly with a
    deliberately slow dummy callable — no real network or platform-specific
    signal delivery needed. The thread is a daemon and is NOT forcibly
    killed on timeout (Python has no safe way to kill a thread); a
    genuinely stuck network call keeps running in the background after we
    give up on it, but that's harmless here since each attempt only ever
    touches its own local variables / cache file, never shared state.
    """
    import threading

    last_err: Exception | None = None
    for _attempt in range(retries + 1):
        outcome: dict = {}

        def _target():
            try:
                outcome["value"] = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — reported to the caller, not raised here
                outcome["error"] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout_sec)
        if t.is_alive():
            last_err = TimeoutError(f"timed out after {timeout_sec:.0f}s")
            continue
        if "error" in outcome:
            last_err = outcome["error"]
            continue
        return outcome.get("value"), None
    return None, last_err

