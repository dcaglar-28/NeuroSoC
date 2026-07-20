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
class MiData:
    """Myocardial-infarction detection from 12-lead ECG (PTB-XL) — deepens
    the ECG modality from arrhythmia (`EcgData`/MIT-BIH, single-lead,
    per-beat) to MI (12-lead, per-recording). Its OWN dataclass, not a reuse
    of `EcgData`, because the shape genuinely differs: MI needs spatial
    lead information (`(leads, time)`, 2-D per sample), not a single-lead
    beat window — see docs/ptbxl_mi_results.md."""
    X: np.ndarray  # (n_samples, 12, 1000) float32 — 12-lead PTB-XL recording,
                    # 100 Hz, 10 s, RAW mV (not yet normalized — per-lead
                    # z-score is fit on the TRAIN split only, post-split, via
                    # `signal_features.normalize_features_train_only`, the
                    # exact function heart sounds already uses for the same
                    # reason: per-channel scales differ and must not leak
                    # val/test statistics into the fit).
    y: np.ndarray   # (n_samples,) int64 — 0 = NORM (confidently normal), 1 = MI
                     # (confidently myocardial infarction) — see the label
                     # rule in `load_ptbxl`'s docstring.
    fs: float       # sampling rate (Hz) — 100.0 (PTB-XL's `records100/`)
    source: str     # "ptbxl" or "synthetic"
    requested_real: bool = False
    # Patient id per window — ALWAYS set (PTB-XL is patient-respecting by
    # design: some patients contribute multiple recordings). Callers MUST
    # split by group (`case_level.split_data`), never by record — a
    # `strat_fold`-equivalent guarantee via `GroupShuffleSplit` on this
    # field, confirmed against PTB-XL's own shipped `strat_fold` column to
    # give the same patient-disjointness property (see docs/ptbxl_mi_results.md
    # Part 0).
    groups: np.ndarray | None = None


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
    # docs/heart_sounds_results.md Part 0) do NOT expose a subject id, even
    # though the dataset's own documentation confirms one subject may
    # contribute 1-6 recordings — so this is always None for "cinc2016" and
    # the split is by RECORDING, not by subject (documented caveat, not a
    # silent gap). Kept for the same reason PpgData carries it: so a future
    # subject-mapping (if one becomes available) slots in without an API
    # change.
    groups: np.ndarray | None = None
    # "raw" (default, delta-encoded waveform, per-window self-normalized —
    # X is (n_samples, window)) or "features" (docs/heart_sounds_results.md's
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


@dataclass
class ShockableData:
    """Shockable-rhythm (VF / rapid-VT) vs. non-shockable binary
    classification from ECG — the defibrillate-or-not (AED) decision (see
    docs/shockable_rhythm_task.md). Deepens ECG under Circulation:
    arrhythmia (MIT-BIH) -> MI (PTB-XL) -> shockable rhythm (VFDB/CUDB).
    This is a morphology/rhythm signal (VF = disorganized, no clear QRS; VT =
    wide, fast, regular QRS), so it uses the RAW-WAVEFORM front-end, same 2-D
    `(n_samples, window)` convention as `EcgData`/`CrmData` — NOT heart
    sounds' spectral filterbank map."""
    X: np.ndarray  # (n_samples, window) float32 — single-lead ECG window,
                    # per-window z-scored (same convention as `load_mitbih`).
    y: np.ndarray   # (n_samples,) int64 — 0 = non-shockable, 1 = shockable
                     # (VF/VFIB/VFL, or VT at/above `vt_rate_threshold` bpm —
                     # see `rhythm_code_to_shockable`).
    fs: float       # sampling rate (Hz) of the TIMESTEP axis (post
                     # `resample_to`, same decoupling as `load_mitbih`).
    source: str     # "vfdb_cudb" or "synthetic"
    requested_real: bool = False
    # Record id per window (e.g. "vfdb:418", "cudb:cu01"), ALWAYS set for
    # real data — VFDB/CUDB records are different PATIENTS/recordings, so
    # callers MUST split by record (`case_level.split_data`'s
    # GroupShuffleSplit path), never by window — same discipline as
    # `MiData`/`CrmData`'s `groups`. `None` for synthetic (no natural
    # grouping, matching every other synthetic generator here).
    groups: np.ndarray | None = None


@dataclass
class CrmData:
    """A synthetic, TIME-RESOLVED Compensatory Reserve trajectory dataset —
    see docs/synthetic_crm_results.md. Same 2-D (n_samples, window) PPG-shaped
    convention as `PpgData` (deliberately its own dataclass, not a reuse of
    `PpgData`, only because it needs one extra field, `cri`, that no other
    PPG source has: the continuous ground-truth reserve value, not just the
    thresholded binary label — see `cri` below)."""
    X: np.ndarray  # (n_samples, window) float32 — multi-pulse PPG windows
    y: np.ndarray   # (n_samples,) int64 — 0 = reserve intact (cri >= threshold),
                     # 1 = compromised (cri < threshold); see `cri`.
    fs: float       # sampling rate (Hz)
    source: str     # "synthetic" ONLY — no real loader exists (real LBNP/
                     # CRM-induction data is gated; see
                     # docs/synthetic_crm_results.md). Never claim this is a
                     # clinical accuracy number — see `report._CARDS`.
    requested_real: bool = False
    # Synthetic SUBJECT id per window — ALWAYS set (unlike other synthetic
    # generators, which have no natural grouping): many highly-correlated
    # windows come from one subject's hypovolemia TRAJECTORY, so splitting
    # must be subject-grouped (`case_level.split_data`), never by window.
    groups: np.ndarray | None = None
    # (n_samples,) float32 in [0, 1] — the CONTINUOUS reserve fraction r(t)
    # AT THIS WINDOW'S EXACT POINT on its trajectory (1.0 = full reserve,
    # 0.0 = decompensated). `y` is `cri < threshold` thresholded from this.
    # This is the field that makes the label genuinely TIME-ALIGNED, unlike
    # VitalDB's one-whole-case-EBL-number-stamped-on-every-window flaw this
    # generator exists to fix — kept for unit-testing the alignment
    # property directly and as the target for a future CRI-regression
    # variant (not built here, see docs/synthetic_crm_results.md).
    cri: np.ndarray | None = None


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
# Myocardial infarction (12-lead ECG, PTB-XL) — deepens ECG from arrhythmia
# to MI/ischemia (docs/ptbxl_mi_results.md). Real loader + synthetic fallback.
# --------------------------------------------------------------------------- #
def _mi_beat_leads(n: int, rng: np.random.Generator, is_mi: bool,
                    n_leads: int = 12) -> np.ndarray:
    """One stylized `n_leads`-lead beat cycle, `n` samples: reuses
    `_normal_beat`'s single-lead PQRST shape per lead (with per-lead
    amplitude variation, a crude stand-in for real inter-lead morphology
    differences), and for `is_mi=True` adds a stylized ST-segment elevation
    -- a slow positive deflection between the QRS and T wave, the single
    most teachable ECG sign of myocardial infarction -- to a random SUBSET
    of leads (mimicking a regional infarct, which doesn't show on every
    lead). A synthetic PIPELINE fallback only; never preferred over real
    PTB-XL (see `load_mi`) and not a diagnostic-accuracy claim.
    """
    base = _normal_beat(n, rng)
    t = np.linspace(0, 1, n)
    affected = (set(rng.choice(n_leads, size=int(rng.integers(2, 5)), replace=False).tolist())
                if is_mi else set())
    leads = np.empty((n_leads, n), dtype=np.float64)
    for lead in range(n_leads):
        sig = base * rng.uniform(0.7, 1.3)
        if lead in affected:
            sig = sig + _gaussian(t, 0.55, 0.09, rng.uniform(0.15, 0.35))
        leads[lead] = sig
    return leads


def make_synthetic_mi(
    n_samples: int = 800,
    n_leads: int = 12,
    window: int = 1000,
    fs: float = 100.0,
    abnormal_frac: float = 0.3,
    beats_per_window: int = 10,
    noise: float = 0.03,
    seed: int = 0,
) -> MiData:
    """Generate a synthetic 12-lead MI-vs-NORM set: `window=1000` at
    `fs=100` Hz mirrors PTB-XL's `records100/` shape (10 s), with
    `beats_per_window` stylized beats tiled across the window (mirroring
    the CRM generator's multi-pulse tiling), each independently drawn via
    `_mi_beat_leads`. `groups=None` (no natural patient grouping for
    synthetic samples, matching every other synthetic generator here).
    """
    rng = np.random.default_rng(seed)
    beat_len = max(4, window // beats_per_window)
    X = np.empty((n_samples, n_leads, window), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        is_mi = rng.random() < abnormal_frac
        y[i] = 1 if is_mi else 0
        sig = np.zeros((n_leads, window), dtype=np.float64)
        onset = 0
        while onset < window:
            blen = min(beat_len, window - onset)
            sig[:, onset:onset + blen] += _mi_beat_leads(beat_len, rng, is_mi, n_leads)[:, :blen]
            onset += beat_len
        sig = sig + rng.normal(0, noise, size=sig.shape)
        X[i] = sig.astype(np.float32)
    return MiData(X=X, y=y, fs=fs, source="synthetic")


PTBXL_VERSION = "1.0.3"
PTBXL_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
PTBXL_NATIVE_FS = 100.0  # records100/ -- confirmed live, docs/ptbxl_mi_results.md Part 0


def _fetch_ptbxl_index(cache_dir: str, timeout_sec: float = 60.0) -> "pd.DataFrame":
    """Fetch/cache `ptbxl_database.csv` + `scp_statements.csv` and return a
    DataFrame of every record CONFIDENTLY labeled MI-only or NORM-only for
    the binary task, with columns `filename_lr`, `patient_id`, `strat_fold`,
    `label` (1=MI, 0=NORM).

    **Label rule (verified against PTB-XL's own shipped `example_physionet.py`,
    not guessed — docs/ptbxl_mi_results.md Part 0):** `scp_codes` is a
    stringified `{SCP_code: likelihood}` dict; likelihood `0.0` means
    "present but not confidence-scored" — NOT "absent" — so, matching the
    dataset's own official aggregation example, we do NOT threshold by
    likelihood. Superclass membership: filter `scp_statements.csv` to rows
    with `diagnostic==1` (44 of 71 codes; confirmed this is exactly the set
    with `diagnostic_class` non-null), then a record's superclass set is
    every `diagnostic_class` among its `scp_codes` keys that's in that
    diagnostic-eligible set (a record can have >1 superclass — the
    multi-label case this binary task deliberately excludes). **A record
    is "confidently MI"** iff that set is EXACTLY `{"MI"}` (not merely
    containing MI alongside e.g. STTC or CD); **"confidently NORM"** iff
    EXACTLY `{"NORM"}`. Confirmed live counts (full dataset, all 21,799
    records / 18,869 patients): 2,532 MI-only + 9,069 NORM-only = 11,601
    eligible (~21.8% MI) -- imbalanced, class-weighted loss is applied
    downstream. `validated_by_human` (True for ~73.7% of records) is
    reported for information but NOT used to filter, matching the official
    example and the overwhelming majority of published PTB-XL baselines.

    `strat_fold` (1-10) is PTB-XL's own shipped, patient-respecting
    stratified fold assignment — confirmed live that no patient appears in
    more than one fold, and that folds 1-8 (train) / 9 (val) / 10 (test)
    each keep the ~20-22% MI ratio. `load_ptbxl` doesn't use `strat_fold`
    directly, though: it sets `MiData.groups = patient_id` and leaves
    splitting to the existing `case_level.split_data` (`GroupShuffleSplit`
    on patient id) — equally patient-safe, and reuses the exact split
    machinery every other modality here already uses, rather than adding a
    parallel fold-based splitter for one modality.
    """
    import os
    import urllib.request

    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    base = f"https://physionet.org/files/ptb-xl/{PTBXL_VERSION}"

    def _fetch(name):
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            with urllib.request.urlopen(f"{base}/{name}", timeout=timeout_sec) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
        return path

    db = pd.read_csv(_fetch("ptbxl_database.csv"), index_col="ecg_id")
    scp = pd.read_csv(_fetch("scp_statements.csv"), index_col=0)
    diagnostic = scp[scp["diagnostic"] == 1]
    scp_to_class = diagnostic["diagnostic_class"].to_dict()

    import ast
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    db["superclasses"] = db["scp_codes"].apply(
        lambda d: scp_codes_to_superclasses(d, scp_to_class))
    db["label"] = db["superclasses"].apply(mi_norm_label)
    sel = db[db["label"].notna()].copy()
    sel["label"] = sel["label"].astype("int64")
    return sel[["filename_lr", "patient_id", "strat_fold", "label"]]


def scp_codes_to_superclasses(scp_codes: dict, scp_to_class: dict) -> set:
    """Pure mapping step of `_fetch_ptbxl_index`'s label rule, pulled out
    for direct unit-testing on a literal dict (no pandas/network needed):
    every diagnostic SUPERCLASS among `scp_codes`' keys that's present in
    `scp_to_class` (an SCP-code -> diagnostic_class lookup, i.e. already
    filtered to `scp_statements.csv`'s `diagnostic==1` rows). Deliberately
    ignores `scp_codes`' likelihood VALUES — see `_fetch_ptbxl_index`'s
    docstring for why (0.0 means "present, unscored," not "absent," per
    PTB-XL's own official `example_physionet.py`).
    """
    return set(scp_to_class[k] for k in scp_codes.keys() if k in scp_to_class)


def mi_norm_label(superclasses: set) -> int | None:
    """The binary MI-vs-NORM label rule, pulled out for direct testing:
    1 if `superclasses` is EXACTLY `{"MI"}`, 0 if EXACTLY `{"NORM"}`,
    `None` (excluded from the binary task) for anything else — empty,
    single-other-superclass (STTC/CD/HYP alone), or ANY multi-superclass
    combination (including one that contains MI alongside something else,
    e.g. `{"MI", "STTC"}` — that record is NOT "confidently MI" for this
    binary task's purposes, it's excluded, not folded into either class).
    """
    if superclasses == {"MI"}:
        return 1
    if superclasses == {"NORM"}:
        return 0
    return None


def _load_ptbxl_record(filename_lr: str, cache_dir: str):
    """Stream one PTB-XL 100 Hz 12-lead record via wfdb and cache it.
    Selects/reorders leads by NAME (`PTBXL_LEADS`), not position — confirmed
    live that every checked record's `sig_name` already matches this exact
    order, but selecting by name is the same "don't assume, verify by name"
    discipline `_load_cinc2016_record` already applies for its bonus-ECG-
    channel gotcha, cheap insurance against a record that doesn't.
    """
    import os
    safe_name = filename_lr.replace("/", "_")
    path = os.path.join(cache_dir, f"{safe_name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["sig"], float(d["fs"])
    import wfdb
    os.makedirs(cache_dir, exist_ok=True)
    record_dir, record_name = filename_lr.rsplit("/", 1)
    rec = wfdb.rdrecord(record_name, pn_dir=f"ptb-xl/{PTBXL_VERSION}/{record_dir}")
    lead_idx = [rec.sig_name.index(lead) for lead in PTBXL_LEADS]
    sig = rec.p_signal[:, lead_idx].T.astype(np.float32)  # (12, 1000)
    fs = float(rec.fs)
    np.savez(path, sig=sig, fs=fs)
    return sig, fs


def load_ptbxl(
    max_records: int | None = 2000,
    cache_dir: str = "data/ptbxl",
    seed: int = 0,
    verbose: bool = True,
    record_timeout_sec: float = 60.0,
    retries: int = 1,
) -> MiData:
    """Stream PhysioNet PTB-XL 12-lead ECG recordings, labelled MI-vs-NORM
    (see `_fetch_ptbxl_index`'s docstring for the full label-rule
    derivation and Part-0 findings). Requires `wfdb` and network access.

    Args:
        max_records: cap on how many eligible recordings to actually pull —
            a seeded random subset of the ~11,601-record eligible pool
            (2,532 MI + 9,069 NORM), not just the first N, so a capped run
            isn't biased toward low ecg_ids. `None` pulls the full pool.
        cache_dir: per-record cache (gitignored `data/`), plus the two
            small index CSVs.
        seed: controls the random `max_records` subsample.
        verbose: print the pull plan and per-record progress.
        record_timeout_sec/retries: `_call_with_timeout` wall-clock cap and
            retry count per record (skip-and-continue, never hang).

    Returns:
        MiData with `source="ptbxl"`, `groups` = patient id per window (see
        `MiData`'s docstring — split via `case_level.split_data`, never by
        record).
    """
    try:
        import wfdb  # noqa: F401  (import check; actual reads use wfdb.rdrecord)
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the data extra: pip install 'eia[data]'") from e

    import time

    rng = np.random.default_rng(seed)
    index, err = _call_with_timeout(_fetch_ptbxl_index, record_timeout_sec, retries, cache_dir)
    if err is not None:
        raise RuntimeError(f"failed to fetch/parse the PTB-XL index: {err}")

    if max_records is not None and max_records < len(index):
        chosen = rng.choice(len(index), size=max_records, replace=False)
        index = index.iloc[sorted(chosen.tolist())]

    n_mi = int((index["label"] == 1).sum())
    if verbose:
        print(f"[ptbxl] plan: {len(index)} recordings "
              f"({n_mi} MI, {len(index) - n_mi} NORM), cache_dir={cache_dir!r}")

    X_list, y_list, group_list = [], [], []
    for i, (_ecg_id, row) in enumerate(index.iterrows(), 1):
        if verbose and (i % 100 == 0 or i == len(index)):
            print(f"[ptbxl] {i}/{len(index)} recordings loaded so far ({len(X_list)} kept)")
        t0 = time.time()
        loaded, err = _call_with_timeout(
            _load_ptbxl_record, record_timeout_sec, retries, row["filename_lr"], cache_dir)
        elapsed = time.time() - t0
        if err is not None:
            print(f"[warn] ptbxl {row['filename_lr']}: failed to load ({err}) "
                  f"after {elapsed:.0f}s; skipping record.")
            continue
        sig, _fs = loaded
        if not np.isfinite(sig).all():
            # Same real, confirmed data-quality guard as CinC 2016's loader
            # (docs/heart_sounds_results.md) -- skip rather than train on it.
            print(f"[warn] ptbxl {row['filename_lr']}: non-finite raw signal; skipping record.")
            continue
        X_list.append(sig)
        y_list.append(int(row["label"]))
        group_list.append(int(row["patient_id"]))

    if not X_list:
        raise RuntimeError("No PTB-XL records loaded.")

    return MiData(X=np.stack(X_list), y=np.array(y_list, dtype=np.int64),
                   fs=PTBXL_NATIVE_FS, source="ptbxl",
                   groups=np.array(group_list, dtype=np.int64))


def load_mi(prefer_real: bool = True, require_real: bool = False,
            **kwargs) -> MiData:
    """Load MI-vs-NORM data, preferring real PTB-XL but falling back to
    synthetic. Mirrors `load_ecg`'s provenance contract exactly.

    Args:
        prefer_real: try `load_ptbxl()` first.
        require_real: if real data was requested and fails to load, raise
            instead of silently substituting synthetic. Every allowed
            fallback still prints a `[warn]` line naming the reason.
        **kwargs: forwarded to `load_ptbxl` (max_records, cache_dir, seed)
            when real; ignored when falling back to synthetic.
    """
    if require_real and not prefer_real:
        raise ValueError("require_real=True has no effect with prefer_real=False "
                          "(nothing was asked to be real).")
    if prefer_real:
        try:
            data = load_ptbxl(**kwargs)
            data.requested_real = True
            return data
        except Exception as e:  # noqa: BLE001  — any failure -> synthetic (or raise)
            if require_real:
                raise RuntimeError(
                    f"--require-real: real PTB-XL failed to load ({e}); "
                    "refusing to silently substitute synthetic MI data."
                ) from e
            print(f"[warn] real PTB-XL unavailable ({e}); falling back to synthetic MI data.")
    data = make_synthetic_mi()
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

# Bands for the "features" front-end (docs/heart_sounds_results.md's escalation
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
    (docs/heart_sounds_results.md Part 0): `REFERENCE.csv` is
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

    Part 0 findings (docs/heart_sounds_results.md — verified against the live
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
    live API before writing this loader (see docs/vitaldb_ppg_results.md
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
    (kept separate from `load_ppg`/BIDMC per docs/vitaldb_ppg_results.md:
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
# CRM — synthetic time-resolved Compensatory Reserve / occult-hemorrhage
# generator (docs/synthetic_crm_results.md)
# --------------------------------------------------------------------------- #
# Why this exists: VitalDB settled ~chance because one whole-CASE
# `intraop_ebl` number was stamped on every window regardless of when in the
# case it was captured (see docs/vitaldb_case_level_results.md) — the label
# never matched the window's actual physiological state. This generator
# fixes that BY CONSTRUCTION: each window is labelled with the reserve
# fraction at its own exact point on a simulated hypovolemia trajectory, so
# it is a genuine (if synthetic) test of whether the pipeline can track
# time-varying occult volume loss, not just classify a fixed waveform shape.
#
# Physiological grounding (see docs/synthetic_crm_results.md and the
# explainable-CRM paper, MDPI Bioengineering 2023, "An Explainable Machine
# Learning Model for the Assessment of Compensatory Reserve"): as central
# reserve `r` falls from 1.0 (normovolemic) toward 0.0 (decompensated),
# PPG pulse amplitude drops, the dicrotic notch (the paper's most
# discriminative CRM feature) blunts and then disappears, the systolic
# upstroke narrows, and the diastolic/reflected-wave component flattens —
# all BEFORE heart rate rises (compensatory tachycardia only engages once
# reserve is significantly depleted). That morphology-leads-vitals ordering
# is the entire clinical value proposition CRM demonstrates and is deliberately
# reproduced here: see `_HR_RISE_R` < `CRI_THRESHOLD` below.
CRI_THRESHOLD = 0.5   # default binary label boundary: cri < this = "compromised"
_HR_RISE_R = 0.30      # HR starts rising only BELOW this r -- deliberately
                        # LOWER than CRI_THRESHOLD, so the r in (0.30, 0.50]
                        # band is labelled "compromised" (y=1) while HR is
                        # STILL AT BASELINE — the occult/pre-vitals-change
                        # detection window the whole generator exists to test.
_HR_COLLAPSE_R = 0.10  # below this, HR falls back toward a low terminal
                        # value (decompensation), rather than continuing to rise.


def _reserve_trajectory(n_windows: int, rng: np.random.Generator) -> np.ndarray:
    """Reserve fraction r(t) in [0, 1] for one synthetic subject's
    trajectory: 1.0 (full reserve) declining to 0.0 (decompensated) over
    `n_windows` steps. Monotonically NON-INCREASING by construction (no
    noise added to `r` itself — all randomness lives in the pulse signal
    and the per-subject baseline parameters, not the trajectory), so this
    is exactly and cheaply unit-testable. `shape` (per-subject, randomized)
    varies whether the decline front-loads or back-loads, so the pipeline
    can't memorize one fixed trajectory curve.
    """
    shape = rng.uniform(0.7, 1.8)
    t = np.linspace(0.0, 1.0, n_windows)
    return np.clip(1.0 - t ** shape, 0.0, 1.0)


def _hr_from_r(r, hr_baseline: float) -> np.ndarray:
    """Heart rate (bpm) as a function of reserve `r` — flat at
    `hr_baseline` while `r > _HR_RISE_R` (this flat zone is what makes the
    morphology-leads-HR demonstration possible), rising toward a
    compensatory-tachycardia peak as `r` falls through
    (`_HR_COLLAPSE_R`, `_HR_RISE_R`], then falling back toward a low
    terminal value below `_HR_COLLAPSE_R` (decompensation/collapse, not
    sustained tachycardia at r=0)."""
    r = np.asarray(r, dtype=np.float64)
    tachy_peak = hr_baseline * 1.6
    terminal = hr_baseline * 0.6
    rising = hr_baseline + (tachy_peak - hr_baseline) * (
        (_HR_RISE_R - r) / (_HR_RISE_R - _HR_COLLAPSE_R))
    collapsing = terminal + (tachy_peak - terminal) * (r / _HR_COLLAPSE_R)
    return np.where(r > _HR_RISE_R, hr_baseline, np.where(r > _HR_COLLAPSE_R, rising, collapsing))


def _crm_pulse(r: float, n: int, rng: np.random.Generator,
                amp_scale: float = 1.0, width_scale: float = 1.0) -> np.ndarray:
    """One PPG pulse cycle at reserve state `r` in [0, 1], `n` samples long.
    Continuously interpolates the same qualitative endpoints
    `_normal_pulse` (r=1)/`_hypovolemic_pulse` (r=0) already encode, as a
    function of `r` rather than a hard binary switch — amplitude and the
    diastolic component scale down smoothly, the pulse narrows, and the
    dicrotic notch (`notch_amp`) is the STRONGEST cue: it has the widest
    proportional range, -> ~0 as r -> 0, matching the CRM literature's
    finding that notch morphology is the most discriminative single feature.
    """
    t = np.linspace(0, 1, n)
    jitter = rng.normal(0, 0.01)
    amp = amp_scale * (0.35 + 0.65 * r)
    width = 0.055 * width_scale * (0.65 + 0.35 * r)
    notch_amp = 0.15 * r
    diastolic_amp = 0.35 * (0.20 + 0.80 * r)
    return (
        _gaussian(t, 0.28 + jitter, width, amp)
        + _gaussian(t, 0.55, 0.030, notch_amp)
        + _gaussian(t, 0.62, 0.070, diastolic_amp)
    )


def _crm_window(r: float, hr_bpm: float, window_sec: float, fs: float,
                 rng: np.random.Generator, amp_scale: float, width_scale: float,
                 noise: float) -> np.ndarray:
    """Build one fixed-duration window at reserve state `r` / heart rate
    `hr_bpm`: tile individual `_crm_pulse` cycles back-to-back, spaced by
    the CURRENT inter-beat interval (60/hr_bpm seconds) — so both pulse
    MORPHOLOGY (shape, via `r`) and pulse-to-pulse SPACING (apparent HR,
    via `hr_bpm`) are visible within one window, exactly like a real
    fixed-duration PPG capture. This is what lets a window from the
    r in (_HR_RISE_R, CRI_THRESHOLD] "occult" band show degraded morphology
    at a still-baseline pulse spacing — the lead effect.
    """
    n_total = int(round(window_sec * fs))
    ibi_samples = max(4, int(round(fs * 60.0 / hr_bpm)))
    sig = np.zeros(n_total, dtype=np.float64)
    onset = 0
    while onset < n_total:
        pulse_len = min(ibi_samples, n_total - onset)
        if pulse_len < 4:
            break
        sig[onset:onset + pulse_len] += _crm_pulse(
            r, ibi_samples, rng, amp_scale, width_scale)[:pulse_len]
        onset += ibi_samples
    sig = sig + rng.normal(0, noise, size=n_total)
    return sig.astype(np.float32)


def make_synthetic_crm(
    n_subjects: int = 50,
    windows_per_subject: int = 24,
    window_sec: float = 3.0,
    fs: float = 100.0,
    cri_threshold: float = CRI_THRESHOLD,
    hr_baseline_range: tuple = (60.0, 80.0),
    amp_scale_range: tuple = (0.8, 1.2),
    width_scale_range: tuple = (0.85, 1.15),
    noise: float = 0.03,
    seed: int = 0,
) -> CrmData:
    """Generate the synthetic time-resolved CRM / occult-hemorrhage
    dataset: `n_subjects` independent hypovolemia trajectories, each
    contributing `windows_per_subject` time-aligned windows.

    Each subject gets its own randomized baseline heart rate, pulse
    amplitude/width scale, and trajectory decline shape (`_reserve_trajectory`)
    — so the model can't memorize one fixed waveform or one fixed
    trajectory curve; it has to track the r-dependent morphology/HR
    relationships themselves.

    Args:
        n_subjects: number of independent synthetic trajectories.
        windows_per_subject: windows sampled along each trajectory
            (evenly spaced from r=1 to r=0 — see `_reserve_trajectory`).
        window_sec/fs: capture duration and sample rate per window
            (window = `round(window_sec * fs)` samples).
        cri_threshold: binary label boundary — `y = 1` iff `cri < threshold`.
            Deliberately ABOVE `_HR_RISE_R` (0.30), so some windows in every
            trajectory are labelled positive while HR is still at baseline —
            the "detect before vitals move" property this generator exists
            to demonstrate (see module-level comment above).
        hr_baseline_range/amp_scale_range/width_scale_range: per-subject
            randomization ranges for baseline HR and pulse morphology scale.
        noise: additive Gaussian noise std on the final signal.
        seed: RNG seed (subject cohort + per-subject/per-window randomness).

    Returns:
        CrmData with `source="synthetic"`, `groups` = subject id per window
        (ALWAYS set — split by subject, never by window), and `cri` = the
        continuous reserve value each window's `y` was thresholded from.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list, group_list, cri_list = [], [], [], []
    for subj in range(n_subjects):
        subj_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        hr_baseline = subj_rng.uniform(*hr_baseline_range)
        amp_scale = subj_rng.uniform(*amp_scale_range)
        width_scale = subj_rng.uniform(*width_scale_range)
        r_traj = _reserve_trajectory(windows_per_subject, subj_rng)
        for w in range(windows_per_subject):
            r = float(r_traj[w])
            hr = float(_hr_from_r(r, hr_baseline))
            sig = _crm_window(r, hr, window_sec, fs, subj_rng, amp_scale, width_scale, noise)
            X_list.append(sig)
            y_list.append(1 if r < cri_threshold else 0)
            group_list.append(subj)
            cri_list.append(r)

    return CrmData(
        X=np.stack(X_list).astype(np.float32),
        y=np.array(y_list, dtype=np.int64),
        fs=fs, source="synthetic",
        groups=np.array(group_list, dtype=np.int64),
        cri=np.array(cri_list, dtype=np.float32),
    )


def load_crm(prefer_real: bool = False, require_real: bool = False,
             **kwargs) -> CrmData:
    """Load the synthetic time-resolved CRM dataset. ALWAYS synthetic —
    there is no real loader: real LBNP/CRM-induction data is gated (request
    access separately; see docs/synthetic_crm_results.md). Mirrors the other
    loaders' provenance-contract SHAPE for API consistency (`requested_real`
    recorded honestly, `require_real` raises rather than silently
    substituting), even though there is no real branch to fall back FROM.

    Args:
        prefer_real: accepted for API symmetry; has no effect except being
            recorded in `requested_real` (so a caller that asked for real
            data can see, from the data card, that it silently got
            synthetic instead — this loader has nothing else to warn about).
        require_real: if True, raises immediately (real data genuinely does
            not exist in this repo) rather than pretending synthetic is real.
        **kwargs: forwarded to `make_synthetic_crm`.
    """
    if require_real:
        raise RuntimeError(
            "--require-real: no real CRM/LBNP loader exists in this repo — "
            "real hypovolemia-induction data is gated (see "
            "docs/synthetic_crm_results.md); refusing to pretend synthetic "
            "data is real.")
    data = make_synthetic_crm(**kwargs)
    data.requested_real = prefer_real
    return data


# --------------------------------------------------------------------------- #
# Shockable-rhythm (VF/VT) detection — the defibrillate-or-not (AED) decision
# (docs/shockable_rhythm_task.md). Real loader (VFDB + CUDB via wfdb) +
# synthetic fallback.
#
# PART 0 FINDINGS (verified live against PhysioNet before writing this
# loader, not assumed — see docs/shockable_rhythm_task.md):
#   - VFDB (MIT-BIH Malignant Ventricular Ectopy Database, `pn_dir="vfdb"`):
#     22 records, confirmed live via `wfdb.get_record_list('vfdb')`
#     (`VFDB_RECORDS`), 250 Hz, 2 channels (both named "ECG" — only channel 0
#     used here, for shape consistency with CUDB's single channel; see
#     "single-lead, not 2-D" below).
#   - CUDB (Creighton University Ventricular Tachyarrhythmia Database,
#     `pn_dir="cudb"`): 35 records (`CUDB_RECORDS`), 250 Hz, 1 channel.
#   - **Rhythm annotations are episode markers, not per-beat labels — the
#     load-bearing detail.** In both datasets' `.atr` files, rhythm changes
#     are recorded as annotations with `symbol == '+'` and the actual rhythm
#     code in `aux_note` (parenthesis-prefixed, NUL-padded, e.g. `'(VF\x00'`)
#     — confirmed live by dumping every annotation's `(symbol, aux_note)`
#     pair. CUDB's `.atr` ALSO carries ~950 per-beat `symbol='N'` annotations
#     per record with EMPTY `aux_note` (a generic beat marker — CUDB's own
#     documentation states "all beats are labelled normal (although many are
#     ectopic)", i.e. NOT diagnostically informative per-beat) — these are
#     filtered out (`_rhythm_intervals_from_annotations` keeps only
#     `symbol=='+'`) before building the rhythm-episode timeline; conflating
#     them with rhythm annotations was an early bug in this loader's Part-0
#     exploration (inflated episode counts to ~20,000 with sub-second
#     "episode" durations) — caught by checking `ann.symbol`, not just
#     `aux_note`, before trusting any duration statistic.
#   - **Rhythm codes actually present** (confirmed live, cross-checked
#     against PhysioNet's own VFDB documentation, which spells out N/NSR,
#     VF/VFIB, VFL, VT, NOISE, ASYS, NOD, SBR, HGEA, B, BI, PM, VER, AFIB —
#     every VFDB code found live is in that list): VFDB has AFIB, ASYS, B,
#     BI, HGEA, N, NOD, NOISE, NSR, PM, SBR, SVTA, VER, VF, VFIB, VFL, VT.
#     CUDB has the smaller set AF, N, VF, VT (AF read as the standard
#     atrial-fibrillation abbreviation, consistent with VFDB's AFIB — CUDB's
#     own PhysioNet page does not itself spell out code meanings, only
#     confirms 250 Hz / 35 records / single-channel and that annotations
#     mark rhythm-of-interest onsets).
#   - **Episode-duration distribution is why window LABELING can't require
#     an episode to be longer than the window** (measured live): VFDB's VFL
#     episodes have median duration 2.8s (73% under 5s), VT episodes median
#     4.2s (54% under 5s) — SHORTER than the "5-8s" AED-window range itself
#     for a majority of cases. A full-window-containment labeling rule
#     therefore does NOT require picking a window shorter than these
#     episodes (that's not achievable for the majority of them regardless of
#     window choice within 4-8s); it just means many individual short
#     episodes contribute zero windows (a fine-grained rhythm alternation is
#     dropped as ambiguous, not force-labeled) — see `min_coverage` below.
#
# WINDOW DURATION + LABELING RULE (`SHOCKABLE_WINDOW_SEC=5.0`,
# `SHOCKABLE_MIN_COVERAGE=1.0`) — chosen empirically in Part 0 by sweeping
# window_sec in {4,5,6,8} x min_coverage in {0.5,0.6,0.8,1.0} against the
# cached real annotation timelines and comparing total-window / class-
# balance yield: 5s at FULL containment (min_coverage=1.0 — a window is
# labeled only if ONE rhythm code covers it end-to-end; anything straddling
# a rhythm change is dropped as ambiguous/transition, per
# docs/shockable_rhythm_task.md's task framing) still yields ~8,000 windows
# combined (~7,600 VFDB + ~450 CUDB) with a healthy shockable-class count —
# full containment costs comparatively little yield vs. the loosest
# (min_coverage=0.5) config, so the strictest, least label-ambiguous rule
# was kept rather than trading label quality for a modest yield increase.
# 5.0s sits at the low end of the task's specified 5-8s AED-window range —
# the empirical sweep showed longer windows (6s, 8s) cost yield roughly
# proportionally with no offsetting quality gain, so 5.0s was preferred.
#
# SHOCKABLE DEFINITION (adopted, cited as in docs/shockable_rhythm_task.md —
# the de Bruin et al. / AHA-aligned rule): SHOCKABLE = ventricular
# fibrillation (VF/VFIB in VFDB's vocabulary, or VF/VFL — ventricular
# flutter, clinically treated as VF-equivalent and shockable) OR rapid
# ventricular tachycardia (VT at or above `VT_RATE_THRESHOLD_BPM`).
# NON-SHOCKABLE = everything else: normal sinus (N/NSR), sinus bradycardia
# (SBR), nodal/junctional (NOD), supraventricular tachyarrhythmia (SVTA),
# atrial fibrillation (AFIB/AF), bigeminy/1st-degree-block-adjacent codes
# (B/BI), ventricular escape rhythm (VER), paced rhythm (PM), high-grade
# ventricular ectopic activity short of sustained VT/VF (HGEA), asystole
# (ASYS — clinically a NON-shockable rhythm: AEDs correctly advise no shock
# for a flat line, matching the parenthetical example list in
# docs/shockable_rhythm_task.md's Part-0 rule statement), and slow VT (below
# the rate threshold). NOISE is DROPPED, not folded into non-shockable — per
# docs/shockable_rhythm_task.md's task-framing section ("drop... windows
# dominated by annotated noise"): noise is a signal-quality state, not a
# rhythm, and training a classifier to call noise "non-shockable" would
# reward learning nothing about the actual decision.
#
# VT RATE HANDLING: VFDB carries NO per-beat annotations (confirmed —
# `ann.symbol` is `{'+'}` only across every VFDB record), so a VT episode's
# rate can't be read off annotation timing directly; `_estimate_rate_bpm`
# computes it per WINDOW from the raw waveform instead (5-25 Hz band-pass +
# peak-picking on the squared signal, 200ms minimum peak spacing = a 300bpm
# physiological ceiling) — validated in Part 0 against known-VT windows
# (measured 173-275bpm across VFDB record 421's 50 VT episodes and across a
# long 885s VT episode in record 427, all comfortably above threshold; a
# sanity check against normal-rhythm 'N' windows in the same record measured
# 103-144bpm, correctly well below). `VT_RATE_THRESHOLD_BPM=150.0` is the LOW
# end of the commonly-cited 150-180bpm range (per docs/shockable_rhythm_task.md
# Part 0) — chosen for sensitivity, matching the AHA's own priority for this
# decision (missing a shockable rhythm is worse than an unnecessary shock
# advisory). A VT window whose estimated rate can't be computed (fewer than 2
# detected peaks — can happen on a short/noisy segment) is DROPPED, not
# guessed into either class.
# --------------------------------------------------------------------------- #
SHOCKABLE_NATIVE_FS = 250.0  # VFDB + CUDB, confirmed live (see Part 0 above)
SHOCKABLE_WINDOW_SEC = 5.0
SHOCKABLE_MIN_COVERAGE = 1.0
VT_RATE_THRESHOLD_BPM = 150.0

SHOCKABLE_RHYTHM_CODES = frozenset({"VF", "VFIB", "VFL"})
NONSHOCKABLE_RHYTHM_CODES = frozenset({
    "N", "NSR", "SBR", "NOD", "SVTA", "AFIB", "AF", "B", "BI", "VER", "PM",
    "ASYS", "HGEA",
})
DROP_RHYTHM_CODES = frozenset({"NOISE"})
VT_RHYTHM_CODE = "VT"

VFDB_RECORDS = ("418", "419", "420", "421", "422", "423", "424", "425", "426",
                 "427", "428", "429", "430", "602", "605", "607", "609", "610",
                 "611", "612", "614", "615")
CUDB_RECORDS = tuple(f"cu{i:02d}" for i in range(1, 36))


def rhythm_code_to_shockable(code: str, vt_rate_bpm: float | None = None,
                              vt_rate_threshold: float = VT_RATE_THRESHOLD_BPM
                              ) -> int | None:
    """Pure mapping, directly unit-testable on a literal code list (no
    network/wfdb needed): a rhythm code (already stripped of the leading '('
    and NUL padding, e.g. `'VF'`, `'N'`, `'VT'`) -> 1 (shockable), 0
    (non-shockable), or `None` (drop — noise or an unrecognized code). See
    the module-level "SHOCKABLE DEFINITION" comment above for the full rule
    and its citation.

    Args:
        code: rhythm code string.
        vt_rate_bpm: required to classify `VT_RHYTHM_CODE` — if `None`, a VT
            code is dropped (`None` returned) rather than guessed into
            either class.
        vt_rate_threshold: bpm cutoff for "rapid" VT (default
            `VT_RATE_THRESHOLD_BPM`).
    """
    if code in SHOCKABLE_RHYTHM_CODES:
        return 1
    if code in NONSHOCKABLE_RHYTHM_CODES:
        return 0
    if code == VT_RHYTHM_CODE:
        if vt_rate_bpm is None:
            return None
        return 1 if vt_rate_bpm >= vt_rate_threshold else 0
    return None  # DROP_RHYTHM_CODES (NOISE) and any unrecognized code


def _rhythm_intervals_from_annotations(symbol: list, sample: list, aux_note: list,
                                        sig_len: int) -> list:
    """Pure function: raw wfdb annotation fields -> `[(start, end, code), ...]`
    rhythm-episode intervals covering `[0, sig_len)`. Only `symbol == '+'`
    entries carry rhythm `aux_note`s (see the module Part-0 comment above for
    why this filter matters — CUDB's per-beat annotations use other symbols
    and must be excluded first). Each episode runs from its own annotation's
    sample to the NEXT rhythm annotation's sample (or `sig_len` for the last
    one) — PhysioNet's VFDB documentation confirms "rhythm change annotations
    are placed at the beginning of the episode of the indicated rhythm."
    """
    rhythm_idx = [i for i, s in enumerate(symbol) if s == "+"]
    rsamples = [sample[i] for i in rhythm_idx]
    rnotes = [aux_note[i].rstrip("\x00").lstrip("(") for i in rhythm_idx]
    intervals = []
    for i, code in enumerate(rnotes):
        end = rsamples[i + 1] if i + 1 < len(rsamples) else sig_len
        intervals.append((int(rsamples[i]), int(end), code))
    return intervals


def _dominant_rhythm(intervals: list, w_start: int, w_end: int,
                      min_coverage: float = SHOCKABLE_MIN_COVERAGE) -> str | None:
    """Pure function: the rhythm code covering the largest fraction of
    `[w_start, w_end)`, or `None` if no interval overlaps the window at all,
    OR the largest single code's coverage fraction is below `min_coverage`
    (an ambiguous/transition window straddling a rhythm change — dropped,
    per docs/shockable_rhythm_task.md's task framing). At the default
    `min_coverage=1.0`, this means "no interval fully contains the window."
    """
    cov: dict = {}
    total = w_end - w_start
    if total <= 0:
        return None
    for (s, e, code) in intervals:
        overlap = max(0, min(e, w_end) - max(s, w_start))
        if overlap > 0:
            cov[code] = cov.get(code, 0) + overlap
    if not cov:
        return None
    code, covered = max(cov.items(), key=lambda kv: kv[1])
    return code if (covered / total) >= min_coverage else None


def _estimate_rate_bpm(seg: np.ndarray, fs: float) -> float:
    """Lightweight per-window heart-rate estimate via band-pass + peak-
    picking on the squared signal — NOT a diagnostic-grade QRS detector, only
    accurate enough to discriminate "rapid" from "slow" VT (see the module
    Part-0 comment above for validation against known-VT and known-normal
    windows). Returns 0.0 if fewer than 2 peaks are found (segment too
    short/flat to estimate a rate from).
    """
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, [5.0, 25.0], btype="band", fs=fs)
    filt = filtfilt(b, a, seg)
    energy = filt ** 2
    min_dist = max(1, int(round(0.2 * fs)))  # 200ms refractory -> <=300bpm ceiling
    thresh = np.percentile(energy, 75)
    peaks, _ = find_peaks(energy, distance=min_dist, height=thresh)
    if len(peaks) < 2:
        return 0.0
    dur = (peaks[-1] - peaks[0]) / fs
    return (len(peaks) - 1) / dur * 60.0 if dur > 0 else 0.0


def _load_shockable_record(db: str, record: str, cache_dir: str):
    """Stream (or load from cache) one VFDB/CUDB record's channel-0 signal +
    rhythm-episode intervals. Caches to a `.npz` (gitignored `data/`) so
    repeated runs don't re-hit the network.
    """
    import os
    path = os.path.join(cache_dir, f"{db}_{record}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return (d["sig"], float(d["fs"]),
                list(zip(d["starts"].tolist(), d["ends"].tolist(), d["codes"].tolist())))
    import wfdb
    os.makedirs(cache_dir, exist_ok=True)
    sig, fields = wfdb.rdsamp(record, pn_dir=db, channels=[0])
    ann = wfdb.rdann(record, "atr", pn_dir=db)
    fs = float(fields["fs"])
    intervals = _rhythm_intervals_from_annotations(
        ann.symbol, ann.sample.tolist(), ann.aux_note, sig.shape[0])
    x = sig[:, 0].astype(np.float32)
    starts = np.array([iv[0] for iv in intervals], dtype=np.int64)
    ends = np.array([iv[1] for iv in intervals], dtype=np.int64)
    codes = np.array([iv[2] for iv in intervals], dtype="<U16")
    np.savez(path, sig=x, fs=fs, starts=starts, ends=ends, codes=codes)
    return x, fs, intervals


def load_vfdb_cudb(
    vfdb_records: tuple = VFDB_RECORDS,
    cudb_records: tuple = CUDB_RECORDS,
    window_sec: float = SHOCKABLE_WINDOW_SEC,
    min_coverage: float = SHOCKABLE_MIN_COVERAGE,
    vt_rate_threshold: float = VT_RATE_THRESHOLD_BPM,
    resample_to: int | None = 500,
    max_windows_per_record: int | None = 40,
    cache_dir_vfdb: str = "data/vfdb",
    cache_dir_cudb: str = "data/cudb",
    seed: int = 0,
    verbose: bool = True,
    record_timeout_sec: float = 60.0,
    retries: int = 1,
) -> ShockableData:
    """Stream VFDB + CUDB, window, and label shockable-vs-non-shockable —
    see the module-level Part-0 comment block above for the full derivation
    (dataset access, rhythm-code vocabulary, window duration/coverage
    choice, the shockable rule + citation, and VT-rate handling).

    Args:
        vfdb_records/cudb_records: which records to pull (default: the full
            confirmed 22 + 35).
        window_sec: analysis-window duration at native 250 Hz (a capture
            duration, not a Xylo/Akida timestep budget — see `resample_to`,
            same decoupling as `load_mitbih`).
        min_coverage: minimum fraction of a window one rhythm code must
            cover to be labeled (default 1.0 = full containment — anything
            straddling a rhythm change is dropped as ambiguous).
        vt_rate_threshold: bpm cutoff for "rapid" (shockable) VT.
        resample_to: FFT-resample each `window_sec*250` native-sample window
            down to this many timesteps (default 500 = 5s at an effective
            100 Hz — comfortably above VF's ~3-9 Hz dominant-frequency band
            and a common downsampling target in published rhythm-
            classification work; `None` keeps the native 1250 samples/window).
        max_windows_per_record: cap on windows kept per record (a seeded
            random subset) — keeps a few "busy" records from dominating the
            dataset, same rationale as `load_cinc2016`'s
            `max_windows_per_record`.
        cache_dir_vfdb/cache_dir_cudb: per-record caches (gitignored `data/`).
        seed: controls the random per-record window subsample.
        verbose: print the pull plan and per-record progress.
        record_timeout_sec/retries: `_call_with_timeout` wall-clock cap and
            retry count per record (skip-and-continue, never hang).

    Returns:
        ShockableData with `source="vfdb_cudb"`, `groups` = record id per
        window (ALWAYS set — split via `case_level.split_data`, never by
        window).
    """
    try:
        import wfdb  # noqa: F401  (import check; actual reads use wfdb.rdsamp/rdann)
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the data extra: pip install 'eia[data]'") from e

    import time

    rng = np.random.default_rng(seed)
    win_native = int(round(window_sec * SHOCKABLE_NATIVE_FS))

    plan = ([("vfdb", r, cache_dir_vfdb) for r in vfdb_records]
            + [("cudb", r, cache_dir_cudb) for r in cudb_records])
    if verbose:
        print(f"[shockable] plan: {len(plan)} records "
              f"({len(vfdb_records)} vfdb + {len(cudb_records)} cudb), "
              f"window_sec={window_sec}, min_coverage={min_coverage}")

    X_list, y_list, group_list = [], [], []
    n_drop_ambig = n_drop_noise = n_drop_vt = 0
    for i, (db, record, cache_dir) in enumerate(plan, 1):
        t0 = time.time()
        loaded, err = _call_with_timeout(
            _load_shockable_record, record_timeout_sec, retries, db, record, cache_dir)
        elapsed = time.time() - t0
        if err is not None:
            print(f"[warn] shockable {db}/{record}: failed to load ({err}) "
                  f"after {elapsed:.0f}s; skipping record.")
            continue
        sig, fs, intervals = loaded
        if not np.isfinite(sig).all():
            print(f"[warn] shockable {db}/{record}: non-finite raw signal; skipping record.")
            continue

        n_windows = sig.shape[0] // win_native
        win_idx = list(range(n_windows))
        if max_windows_per_record is not None and len(win_idx) > max_windows_per_record:
            win_idx = sorted(rng.choice(win_idx, size=max_windows_per_record,
                                         replace=False).tolist())

        kept = 0
        for w in win_idx:
            w_start, w_end = w * win_native, (w + 1) * win_native
            code = _dominant_rhythm(intervals, w_start, w_end, min_coverage)
            if code is None:
                n_drop_ambig += 1
                continue
            if code in DROP_RHYTHM_CODES:
                n_drop_noise += 1
                continue
            vt_rate = None
            if code == VT_RHYTHM_CODE:
                estimated = _estimate_rate_bpm(sig[w_start:w_end], fs)
                # `_estimate_rate_bpm` returns 0.0, not None, when it can't
                # find >=2 peaks -- treat that as "no estimate" here (drop),
                # not as a genuinely slow rate (which would silently mislabel
                # an unclassifiable window as non-shockable).
                vt_rate = estimated if estimated > 0.0 else None
            label = rhythm_code_to_shockable(code, vt_rate_bpm=vt_rate,
                                              vt_rate_threshold=vt_rate_threshold)
            if label is None:
                n_drop_vt += 1
                continue
            X_list.append(sig[w_start:w_end])
            y_list.append(label)
            group_list.append(f"{db}:{record}")
            kept += 1
        if verbose:
            print(f"[shockable] {db}/{record} ({i}/{len(plan)}): {kept} windows kept "
                  f"in {elapsed:.1f}s, {len(X_list)} cumulative")

    if not X_list:
        raise RuntimeError("No shockable-rhythm windows extracted from VFDB/CUDB.")

    X = np.stack(X_list).astype(np.float32)
    if resample_to is not None and resample_to != win_native:
        X = resample_windows(X, resample_to)
        fs_eff = SHOCKABLE_NATIVE_FS * (resample_to / win_native)
    else:
        fs_eff = SHOCKABLE_NATIVE_FS
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    y = np.array(y_list, dtype=np.int64)
    n_shock = int(y.sum())
    if verbose:
        print(f"[shockable] done: {len(X_list)} windows "
              f"({n_shock} shockable / {len(X_list) - n_shock} non-shockable, "
              f"{n_shock / len(X_list):.1%} shockable); dropped "
              f"{n_drop_ambig} ambiguous/transition, {n_drop_noise} noise, "
              f"{n_drop_vt} unclassifiable-VT")

    return ShockableData(X=X.astype(np.float32), y=y, fs=fs_eff, source="vfdb_cudb",
                          groups=np.array(group_list, dtype=object))


def _vf_window(n: int, rng: np.random.Generator, fs: float) -> np.ndarray:
    """Stylised VF: chaotic, no organized QRS — a dominant-frequency
    sinusoid in VF's typical 3-9 Hz band (see the module Part-0 comment
    above) with phase jitter and additive noise, deliberately NOT a clinical
    VF-morphology claim (same spirit as every other `make_synthetic_*`
    generator here — a pipeline check, not real physiology).
    """
    t = np.arange(n) / fs
    dom_freq = rng.uniform(3.0, 9.0)
    phase_jitter = rng.uniform(0, 2 * np.pi, size=n) * 0.3
    chaos = np.sin(2 * np.pi * dom_freq * t + phase_jitter)
    return (chaos + rng.normal(0, 0.3, size=n)) * rng.uniform(0.5, 1.2)


def _vt_window(n: int, rng: np.random.Generator, fs: float, hr_bpm: float) -> np.ndarray:
    """Stylised VT: `_abnormal_beat`'s wide/bizarre QRS shape (already used
    as ECG-arrhythmia's PVC-like beat) tiled at a RAPID, REGULAR rate —
    unlike a single PVC amid normal rhythm, VT is sustained, organized, and
    fast.
    """
    beat_len = max(4, int(round(60.0 / hr_bpm * fs)))
    sig = np.zeros(n, dtype=np.float64)
    onset = 0
    while onset < n:
        blen = min(beat_len, n - onset)
        sig[onset:onset + blen] += _abnormal_beat(beat_len, rng)[:blen]
        onset += beat_len
    return sig


def _organized_rhythm_window(n: int, rng: np.random.Generator, fs: float,
                              hr_bpm: float) -> np.ndarray:
    """Non-shockable stand-in: `_normal_beat`'s normal PQRST shape tiled at
    `hr_bpm`. Real non-shockable rhythms cover many distinct patterns (see
    the module Part-0 comment's `NONSHOCKABLE_RHYTHM_CODES`); this generator
    is a pipeline check, not a claim of rhythm diversity.
    """
    beat_len = max(4, int(round(60.0 / hr_bpm * fs)))
    sig = np.zeros(n, dtype=np.float64)
    onset = 0
    while onset < n:
        blen = min(beat_len, n - onset)
        sig[onset:onset + blen] += _normal_beat(beat_len, rng)[:blen]
        onset += beat_len
    return sig


def make_synthetic_shockable(
    n_samples: int = 2000,
    window_sec: float = SHOCKABLE_WINDOW_SEC,
    fs: float = 100.0,
    shockable_frac: float = 0.3,
    vt_frac_of_shockable: float = 0.5,
    hr_normal_range: tuple = (60.0, 100.0),
    hr_vt_range: tuple = (150.0, 220.0),
    noise: float = 0.03,
    seed: int = 0,
) -> ShockableData:
    """Generate a balanced-ish synthetic shockable-rhythm set: non-shockable
    windows are an `_organized_rhythm_window` at a normal rate; shockable
    windows are a mix of `_vf_window` (chaotic) and `_vt_window` (fast, wide,
    regular QRS, rate drawn from `hr_vt_range` — always at/above
    `VT_RATE_THRESHOLD_BPM` by construction, so every synthetic VT-like
    window is genuinely shockable, no rate-threshold ambiguity to encode
    here). `fs=100` (vs. VFDB/CUDB's native 250) matches this loader's own
    default `resample_to` effective rate, so a model trained here has the
    same input shape/scale as one trained on the real (resampled) data.
    """
    rng = np.random.default_rng(seed)
    window = int(round(window_sec * fs))
    X = np.empty((n_samples, window), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        if rng.random() < shockable_frac:
            y[i] = 1
            if rng.random() < vt_frac_of_shockable:
                sig = _vt_window(window, rng, fs, rng.uniform(*hr_vt_range))
            else:
                sig = _vf_window(window, rng, fs)
        else:
            y[i] = 0
            sig = _organized_rhythm_window(window, rng, fs, rng.uniform(*hr_normal_range))
        sig = sig + rng.normal(0, noise, size=window)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        X[i] = sig.astype(np.float32)
    return ShockableData(X=X, y=y, fs=fs, source="synthetic")


def load_shockable(prefer_real: bool = True, require_real: bool = False,
                    **kwargs) -> ShockableData:
    """Load shockable-rhythm data, preferring real VFDB+CUDB but falling
    back to synthetic. Mirrors `load_ecg`/`load_mi`'s provenance contract
    exactly.

    Args:
        prefer_real: try `load_vfdb_cudb()` first.
        require_real: if real data was requested and fails to load, raise
            instead of silently substituting synthetic. Every allowed
            fallback still prints a `[warn]` line naming the reason.
        **kwargs: forwarded to `load_vfdb_cudb` when real; ignored when
            falling back to synthetic.
    """
    if require_real and not prefer_real:
        raise ValueError("require_real=True has no effect with prefer_real=False "
                          "(nothing was asked to be real).")
    if prefer_real:
        try:
            data = load_vfdb_cudb(**kwargs)
            data.requested_real = True
            return data
        except Exception as e:  # noqa: BLE001  — any failure -> synthetic (or raise)
            if require_real:
                raise RuntimeError(
                    f"--require-real: real VFDB/CUDB failed to load ({e}); "
                    "refusing to silently substitute synthetic shockable-rhythm data."
                ) from e
            print(f"[warn] real VFDB/CUDB unavailable ({e}); falling back to "
                  "synthetic shockable-rhythm data.")
    data = make_synthetic_shockable()
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

