"""Per-dataset data cards + red-flag warnings.

Design principle: each dataset is its own experiment. `data_card()` prints a
short, structured summary of exactly what is being trained on — source, label
definition, class balance, and known limitations — and raises explicit warnings
for the traps that are easy to miss (class imbalance, tiny N, a model that only
matches the majority-class base rate, NaNs). It never pools or compares across
datasets; call it once per dataset.

Pure NumPy — imports and runs without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Per-(modality, source) label definition + known limitations. Keep honest:
# say plainly when a label is a proxy.
_CARDS = {
    ("ecg", "synthetic"): (
        "0 = normal PQRST complex, 1 = PVC-like beat (wide QRS, no P, discordant T).",
        "Separable by construction — expect ~1.0 accuracy. Proves the pipeline, "
        "NOT a real-accuracy claim.",
    ),
    ("ecg", "mitbih"): (
        "0 = normal AAMI beat (N/L/R/e/j), 1 = abnormal (V/A/F/!/E).",
        "Real clinical ECG (MIT-BIH). Single-lead, subset of records.",
    ),
    ("mi", "synthetic"): (
        "0 = normal 12-lead beat, 1 = stylized ST-segment elevation on a "
        "random SUBSET of leads (mimicking a regional infarct) — a crude "
        "proxy for the single most teachable ECG-MI sign.",
        "Separable by construction — expect ~1.0 accuracy. Proves the "
        "pipeline, NOT a real-accuracy claim.",
    ),
    ("mi", "ptbxl"): (
        "0 = confidently NORM (scp_codes' diagnostic superclass set is "
        "EXACTLY {NORM}), 1 = confidently MI (set is EXACTLY {MI}) — "
        "records with any other/mixed superclass (STTC, CD, HYP, or "
        "MI+something-else) are EXCLUDED from this binary task, not "
        "folded into either class. Label rule verified against PTB-XL's "
        "own shipped `example_physionet.py` (not guessed): scp_codes "
        "likelihood is NOT thresholded (0.0 means 'present, unscored', not "
        "'absent') — matches the dataset's own official aggregation.",
        "REAL, cardiologist-labeled 12-lead ECG (PTB-XL) — this deepens "
        "the ECG modality from arrhythmia (MIT-BIH, single-lead) to "
        "myocardial infarction (12-lead, benchmarked ~0.9+ AUROC in the "
        "literature). ~21.8% MI at the full-pool level (imbalanced — "
        "class-weighted loss + balanced-accuracy selection apply). Split "
        "by PATIENT (`groups`) — PTB-XL is patient-respecting by design, "
        "some patients contribute multiple recordings.",
    ),
    ("shockable", "synthetic"): (
        "0 = organized normal-rate rhythm (stylised normal PQRST tiled at "
        "60-100 bpm), 1 = shockable: chaotic VF-like (dominant 3-9 Hz "
        "sinusoid + noise, no organized QRS) or fast/wide/regular VT-like "
        "(wide-QRS beat tiled at 150-220 bpm, always at/above the rate "
        "threshold by construction).",
        "Separable by construction — expect ~1.0 accuracy. Proves the "
        "pipeline, NOT a real-accuracy claim.",
    ),
    ("shockable", "vfdb_cudb"): (
        "0 = non-shockable rhythm (normal sinus, sinus brady, nodal, SVTA, "
        "AFIB/AF, bigeminy-adjacent, ventricular escape, paced, high-grade "
        "ectopy short of VT/VF, asystole, or VT below the rate threshold); "
        "1 = shockable (VF/VFIB/VFL, or VT at/above 150 bpm — de Bruin et "
        "al. / AHA-aligned rule, see docs/shockable_rhythm_task.md). "
        "5-second AED-style windows, labeled by the rhythm annotation "
        "covering the window END-TO-END (min_coverage=1.0) — windows "
        "straddling a rhythm change, or dominated by annotated noise, are "
        "dropped, not force-labeled. VT rate is estimated PER WINDOW from "
        "the raw waveform (band-pass + peak-picking; neither dataset "
        "carries usable per-beat rate annotations for VFDB).",
        "REAL clinical ECG (MIT-BIH Malignant Ventricular Ectopy Database + "
        "Creighton University Ventricular Tachyarrhythmia Database, "
        "single-lead, 250 Hz native). Shockable is the MINORITY class. "
        "Split by RECORD (`groups`) — VFDB/CUDB records are different "
        "patients. VT-rate estimation is a lightweight peak-picker, not a "
        "diagnostic-grade QRS detector (validated against known-VT/known-"
        "normal windows in Part 0, not a clinical-accuracy guarantee).",
    ),
    ("heart", "synthetic"): (
        "0 = normal S1-S2 'lub-dub' cycle, 1 = added systolic murmur "
        "(band-limited noise filling the S1-S2 gap) — a stylised proxy for "
        "valve regurgitation/stenosis.",
        "Separable by construction — expect ~1.0 accuracy. Proves the "
        "pipeline, NOT a real-accuracy claim.",
    ),
    ("heart", "cinc2016"): (
        "0 = normal, 1 = abnormal (PhysioNet/CinC Challenge 2016 REFERENCE.csv, "
        "label 1 = abnormal / -1 = normal). Abnormal = a confirmed cardiac "
        "diagnosis (typically valve defects or coronary artery disease); the "
        "challenge does not provide a finer-grained diagnosis.",
        "Real clinical + non-clinical PCG recordings, ~20.5% abnormal. "
        "Recordings with signal-quality flag 0 (~4% of training-a) are "
        "excluded by default (min_quality). SUBJECT IDS ARE NOT RECOVERABLE "
        "from the distributed files (confirmed, not assumed) despite one "
        "subject possibly contributing 1-6 recordings — split is by "
        "RECORDING, not subject; some cross-recording leakage risk is "
        "possible and not eliminable from this public release.",
    ),
    ("ppg", "synthetic"): (
        "0 = normovolemic pulse, 1 = hypovolemic (reduced amplitude, blunted "
        "dicrotic notch) — a Compensatory-Reserve-style waveform change.",
        "Synthetic. Closest generator to the real hemorrhage target, but not "
        "real physiology.",
    ),
    ("ppg", "bidmc"): (
        "0 = SpO2 >= 95%, 1 = SpO2 < 95% over the window.",
        "SpO2-desaturation PROXY for physiological compromise — NOT a hemorrhage "
        "label. High sim-agreement here can mean 'not learning', not 'good'.",
    ),
    ("ppg", "vitaldb"): (
        "0 = intraop_ebl < 500 mL (case-level), 1 = intraop_ebl >= 500 mL — a "
        "real, case-level estimated-blood-loss label (VitalDB).",
        "REAL blood-loss label, but coarse: (1) intraoperative, ANESTHETIZED "
        "patients — anesthesia/vasopressors/surgical context confound PPG vs. "
        "conscious field trauma; (2) EBL is an ESTIMATE and a WHOLE-CASE "
        "total, not time-aligned to the moment of bleeding — every window "
        "from a case shares one label; (3) split by case, not window "
        "(`groups`) — windows within a case are highly correlated.",
    ),
    ("ppg", "lbnp"): (
        "Binary compromise from lower-body-negative-pressure stage / blood-volume "
        "decrement (see loader for exact threshold).",
        "Real induced central hypovolemia — the intended hemorrhage signal. "
        "Usually few subjects: split by subject, watch N.",
    ),
    ("crm", "synthetic"): (
        "0 = reserve intact (cri >= threshold, default 0.5), 1 = compromised "
        "(cri < threshold) — cri = the reserve fraction r(t) AT THIS WINDOW'S "
        "OWN POINT on a simulated hypovolemia trajectory (1.0 = normovolemic, "
        "0.0 = decompensated), TIME-ALIGNED per window, not a whole-trajectory "
        "label. The positive class deliberately includes windows where heart "
        "rate is STILL AT BASELINE (r in (0.30, 0.50]) — detecting these is "
        "the entire point: occult reserve loss before vitals move.",
        "SYNTHETIC — separable by construction; a high number here proves the "
        "PIPELINE and that the time-aligned label is learnable, NOT clinical "
        "hemorrhage-detection accuracy. Physiologically grounded (pulse "
        "amplitude/dicrotic-notch/width/diastolic-component relationships from "
        "the explainable-CRM literature, MDPI Bioeng. 2023) but real validation "
        "needs gated LBNP/CRM-induction data — see docs/synthetic_crm_task.md "
        "and docs/synthetic_crm_results.md. Split by SUBJECT (`groups`, always "
        "set) — windows within one trajectory are highly correlated.",
    ),
}


@dataclass
class DataCard:
    modality: str
    source: str
    n_samples: int
    window: int
    fs: float
    duration_s: float
    class_counts: dict
    class_fracs: dict
    majority_base_rate: float
    label_definition: str
    limitations: str
    provenance: str = "unknown"  # "REAL", "SYNTHETIC", or "SYNTHETIC (FALLBACK)"
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 66,
            f" DATA CARD — {self.modality.upper()} / {self.source}",
            "=" * 66,
            f" provenance     : {self.provenance}",
            f" samples        : {self.n_samples}",
            f" window         : {self.window} samples  (~{self.duration_s:.2f}s @ {self.fs:g} Hz)",
            f" class balance  : {self.class_fracs}  (counts {self.class_counts})",
            f" majority base  : {self.majority_base_rate:.3f}  "
            f"(a model at this accuracy has learned nothing)",
            f" label          : {self.label_definition}",
            f" limitations    : {self.limitations}",
        ]
        if self.warnings:
            lines.append(" " + "-" * 64)
            for w in self.warnings:
                lines.append(f" [warn] {self.source}: {w}")
        lines.append("=" * 66)
        return "\n".join(lines)


def _modality_of(data) -> str:
    name = type(data).__name__.lower()
    if name.startswith("ecg"):
        return "ecg"
    if name.startswith("heart"):
        return "heart"
    if name.startswith("ppg"):
        return "ppg"
    if name.startswith("crm"):
        return "crm"
    if name.startswith("mi"):
        return "mi"
    if name.startswith("shockable"):
        return "shockable"
    return getattr(data, "modality", "unknown")


def data_card(data, model_acc: float | None = None, verbose: bool = True,
              imbalance_frac: float = 0.15, small_n: int = 200,
              base_rate_hi: float = 0.65, not_learning_margin: float = 0.03) -> DataCard:
    """Build (and optionally print) a data card for one dataset.

    Args:
        data: an EcgData / PpgData / HeartData (has .X, .y, .fs, .source).
            `X` is (n_samples, window); `window` here always means the LAST
            axis (time), so it and `duration_s` stay correct even for a
            future multi-channel modality.
        model_acc: optional test accuracy of a trained model on this dataset;
            if within `not_learning_margin` of the base rate, raises a warning.
        verbose: print the card.
    """
    X = np.asarray(data.X)
    y = np.asarray(data.y).astype(int)
    modality = _modality_of(data)
    source = getattr(data, "source", "unknown")
    n = int(y.size)
    window = int(X.shape[-1]) if X.ndim >= 2 else 0
    fs = float(getattr(data, "fs", 0.0) or 0.0)
    duration_s = (window / fs) if fs else 0.0

    classes, counts = np.unique(y, return_counts=True)
    class_counts = {int(c): int(k) for c, k in zip(classes, counts)}
    class_fracs = {int(c): round(float(k) / n, 3) for c, k in zip(classes, counts)} if n else {}
    base_rate = float(counts.max() / n) if n else 0.0

    label_def, limitations = _CARDS.get(
        (modality, source), ("(undocumented label)", "(no limitations recorded)"))

    requested_real = bool(getattr(data, "requested_real", False))
    if source == "synthetic":
        provenance = "SYNTHETIC (FALLBACK — real data request failed)" if requested_real \
            else "SYNTHETIC (requested)"
    else:
        provenance = f"REAL ({source})"

    warnings = []
    if n and len(classes) > 1:
        minority = counts.min() / n
        if minority < imbalance_frac:
            warnings.append(
                f"class imbalance — minority class is {minority:.1%} of samples.")
    if len(classes) < 2:
        warnings.append("only one class present — classification is degenerate.")
    if n < small_n:
        warnings.append(f"small dataset (N={n} < {small_n}); results are noisy.")
    if base_rate > base_rate_hi:
        warnings.append(
            f"high majority base rate ({base_rate:.1%}); treat near-base-rate "
            f"accuracy as 'not learning', not success.")
    if model_acc is not None and (model_acc - base_rate) < not_learning_margin:
        warnings.append(
            f"model accuracy {model_acc:.3f} is within {not_learning_margin:.0%} "
            f"of base rate {base_rate:.3f} — model is NOT learning the signal.")
    if X.size and not np.isfinite(X).all():
        warnings.append("non-finite values present in X (NaN/inf).")

    card = DataCard(
        modality=modality, source=source, n_samples=n, window=window, fs=fs,
        duration_s=duration_s, class_counts=class_counts, class_fracs=class_fracs,
        majority_base_rate=base_rate, label_definition=label_def,
        limitations=limitations, provenance=provenance, warnings=warnings,
    )
    if verbose:
        print(card)
    return card


def assert_provenance(card: DataCard, data, expected_modality: str) -> None:
    """Guard against training/verifying on different data than the card
    describes — the exact bug class this repo hit once already: a notebook
    cell requested PPG (data card + plots all showed PPG) but the training
    call silently defaulted to ECG because `modality=` wasn't threaded
    through. Call this right after `data_card(data)`, before any training.

    Raises ValueError (not a bare `assert`, which `-O` can strip) if the
    card's modality doesn't match what the caller is about to train/verify
    on, or if the card's source doesn't match the actual data object's.
    """
    data_source = getattr(data, "source", None)
    if card.modality != expected_modality:
        raise ValueError(
            f"provenance mismatch: about to train/verify modality="
            f"{expected_modality!r} but the data card reports modality="
            f"{card.modality!r} — wrong data object for this run.")
    if card.source != data_source:
        raise ValueError(
            f"provenance mismatch: data card source={card.source!r} but the "
            f"data object's source={data_source!r} — card was built from a "
            f"different (or since-mutated) data object.")
