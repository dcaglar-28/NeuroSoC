# EEG front-end redesign — measured results

Implements `docs/eeg_frontend_task.md`, the follow-up to the settled diagnosis
in `docs/eeg_seizure_results.md`: subject-independent AND patient-specific
splits were both ~chance, and critically the **float** model (no chip, no
quantization) was at chance too — so the seizure signal was being destroyed
by the raw-waveform delta-encode **front-end**, not by data volume or
hardware. This task replaces that front-end with the feature representation
clinical seizure detectors actually use (line length, relative band power,
spectral entropy) and re-measures, on the exact same cached data, before
touching anything else.

## Outcome, stated up front

**STILL ~CHANCE.** The feature front-end does not clear chance on the float
model, on either split, on the same 4-patient pool the raw front-end was
also ~chance on. Balanced accuracy and AUROC for the feature front-end are
statistically indistinguishable from — and nominally slightly *below* — the
raw-delta baseline on both splits:

| Split | Front-end | Float balanced acc | Float AUROC |
|---|---|---|---|
| Subject-independent | raw (baseline) | 0.551 ± 0.042 | 0.562 ± 0.044 |
| Subject-independent | **features** | 0.527 ± 0.024 | 0.540 ± 0.036 |
| Patient-specific | raw (baseline) | 0.521 ± 0.071 | 0.528 ± 0.105 |
| Patient-specific | **features** | 0.488 ± 0.073 | 0.475 ± 0.074 |

Per the task's own decision rule: **the front-end was not the whole story.**
This is reported plainly, per the task's explicit instruction not to keep
tuning features or expand the feature set to force a result. **Next
suspects: montage/channel choice (only 2 of 6 available channels used here,
to fit the Xylo input budget) and true data scale (4 patients, 868 windows,
is still very small for either front-end to show its hand).** XyloSim
on-chip fidelity was not chased further — consistent with the diagnosis's
own lesson not to look at quantization fidelity until the float model
learns, which it still hasn't.

## The redesign

```
montage-selected, band-passed EEG channel (per FEATURE_MONTAGE)
  -> split each 4s window into 8 sub-windows (0.5s / 128 samples each)
  -> per sub-window, per channel: line length, relative delta power,
     relative beta power, spectral entropy
  -> (4 features x 2 channels = 8 feature-channels) x 8 sub-window timesteps
  -> per-feature z-score, TRAIN-ONLY stats (eeg_features.normalize_features_train_only)
  -> delta ON/OFF encode each feature-channel (existing encoder, unchanged)
     -> 16 spike channels
  -> build_xylo_snn(n_in=16, n_hidden=63, n_out=2)
```

Implemented in `src/eia/eeg_features.py` (pure NumPy/SciPy, offline-testable)
and wired into `datasets.load_chbmit` behind `eeg_frontend={"raw","features"}`
— `"raw"` is completely unchanged (the A/B baseline), `"features"` is new.
`scripts/xylo_verify.py` and `train.py` expose it via `--eeg-frontend`.

### Feature set and why

- **Line length** — sum of `|x[t] - x[t-1]|` over the sub-window. The single
  best-evidenced scalp-EEG seizure feature (Esteller et al. 2001); used in
  essentially every clinical detector since.
- **Relative band power** — Welch PSD, power per band as a fraction of total
  power (amplitude-invariant). All four canonical bands (delta/theta/alpha/
  beta) are implemented in `eeg_features.EEG_BANDS`, with beta pinned to
  **12.5–25 Hz** (the literature's best-performing range per the task doc,
  not the more common 13–30 Hz). Only **delta** and **beta** are wired into
  the compact 4-feature set actually used — the two band-power extremes,
  chosen as the most complementary (least redundant) pair of the four bands,
  capturing both slow/spike-wave and fast rhythmic seizure activity.
- **Spectral entropy** — the rhythmicity measure. Normalized Shannon entropy
  of the PSD distribution: near 0 = power concentrated in a narrow band
  (rhythmic, seizure-like), near 1 = power spread flat (noise-like). Chosen
  over a peak-to-mean PSD ratio because entropy pools the *whole*
  distribution rather than one bin, so a single noisy PSD spike can't
  dominate it — more numerically stable for a compact 4-feature set where a
  fragile rhythmicity measure would be an outsized fraction of what the
  network sees.

### Input budget — the binding constraint

The task doc's own worked example (`~4 channels x 4 features = 16`) doesn't
account for the ON/OFF delta-doubling every other modality in this repo
already uses (`scripts/xylo_verify.py`'s `_encode_batch`) — reusing that
exact same encoder (rather than inventing a new one) means the real
constraint is `(features x channels) x 2 <= 16`, i.e. `features x channels
<= 8`. Chosen layout: **`FEATURE_MONTAGE` = 2 channels (`F7-T7`, `F8-T8` —
left/right temporal, the same seizure-focus reasoning behind the original
6-channel `EEG_MONTAGE`) x 4 features (line length, delta, beta, spectral
entropy) = 8 feature-channels x 2 (ON/OFF) = 16 spike channels**, confirmed
in the mapped footprint every run: `input 16/16` — exactly at the ceiling,
zero headroom. `theta` and `alpha` remain implemented in `EEG_BANDS` and
available if the budget allows expanding later.

**The committed Akida decision (CLAUDE.md's "Hardware target" section)
loosens this ceiling** — Xylo's 16-input constraint is what forced the
2-channel trim here; Akida's larger input budget would let the full
6-channel montage and all four bands be used without this compression, a
concrete point in favor of that single-vendor decision.

## Cache reuse — no new downloads

Confirmed zero new PhysioNet access: `_load_chbmit_record_montage` now
tags cache filenames by montage (`_montage_cache_tag`) so the 2-channel
`FEATURE_MONTAGE` and 6-channel `EEG_MONTAGE` cached arrays can't collide —
and since `FEATURE_MONTAGE` is a strict subset of `EEG_MONTAGE`, the
2-channel array is derived directly from the already-cached 6-channel one
(`data/chbmit/*_montage.npz`) with zero network access, confirmed by every
run completing in ~30–40 seconds (pure local compute, no download wait).

## Run configuration

Same 4-patient / 12-record pool for all four runs (`chb01, chb02, chb03,
chb05`, `seizure_records_per_subject=2, nonseizure_records_per_subject=1`,
868 windows, class balance 82.9%/17.1%), 5 seeds x 5 restarts x 20 epochs.

```bash
# subject-independent, features vs. raw
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --eeg-frontend features --eeg-subjects chb01,chb02,chb03,chb05 \
  --split subject-independent --epochs 20 --n-restarts 5 --n-seeds 5 \
  --max-verify 300 --no-combined

python scripts/xylo_verify.py --modality eeg --real --require-real \
  --eeg-frontend raw --eeg-subjects chb01,chb02,chb03,chb05 \
  --split subject-independent --epochs 20 --n-restarts 5 --n-seeds 5 \
  --max-verify 300 --no-combined

# patient-specific, features vs. raw
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --eeg-frontend features --eeg-subjects chb01,chb02,chb03,chb05 \
  --split patient-specific --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300

python scripts/xylo_verify.py --modality eeg --real --require-real \
  --eeg-frontend raw --eeg-subjects chb01,chb02,chb03,chb05 \
  --split patient-specific --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300
```

## Subject-independent — full comparison (mean ± std over 5 seeds)

| Metric | raw (baseline) | features |
|---|---|---|
| Float balanced accuracy | 0.551 ± 0.042 | 0.527 ± 0.024 |
| Float per-class recall | [0.526±0.131, 0.576±0.199] | [0.838±0.165, 0.217±0.205] |
| Float sensitivity / specificity | 0.576±0.199 / 0.526±0.131 | 0.217±0.205 / 0.838±0.165 |
| Float AUROC | 0.562 ± 0.044 | 0.540 ± 0.036 |
| Float AUPRC | 0.200 ± 0.074 | 0.214 ± 0.082 |
| Float false alarms/hour | 427 ± 118 | 146 ± 149 |
| XyloSim balanced accuracy | 0.505 ± 0.052 | 0.533 ± 0.025 |
| Float vs. XyloSim agreement | 0.699 ± 0.127 | 0.868 ± 0.072 |

The feature front-end's much lower false-alarm rate (146 vs. 427/hour) is
**not** a sign it learned better — its recall pattern ([0.838, 0.217], i.e.
mostly predicting the majority/non-seizure class) shows a more conservative,
majority-leaning classifier, which trivially lowers false alarms at the cost
of sensitivity. This is the same shape of result the raw front-end showed in
earlier runs on other pools (`docs/eeg_seizure_results.md`) — a model that
hasn't found real signal defaults to one of two degenerate patterns (near-
all-negative or near-all-positive), and which one comes out varies by seed.

(Float vs. XyloSim agreement is higher for features (0.868) than raw (0.699)
— plausibly because the feature front-end's near-majority-collapsed float
decisions are *more* confident/less marginal than raw's, and confident
decisions are known to be more quantization-robust, per
`docs/ecg_quant_diagnosis.md`'s margin finding. Not chased further here, per
the instruction not to touch on-chip fidelity until the float model learns.)

## Patient-specific — full comparison

**Per-patient (float, mean ± std over 5 seeds), features front-end:**

| Patient | Balanced acc | AUROC | AUPRC | Sensitivity | Specificity | FA/hour |
|---|---|---|---|---|---|---|
| chb01 | 0.510 ± 0.094 | 0.490 ± 0.096 | 0.144 ± 0.028 | 0.350 ± 0.215 | 0.670 ± 0.155 | 297 ± 140 |
| chb02 | 0.467 ± 0.072 | 0.465 ± 0.079 | 0.267 ± 0.041 | 0.238 ± 0.186 | 0.697 ± 0.233 | 273 ± 210 |
| chb03 | 0.500 ± 0.074 | 0.479 ± 0.065 | 0.213 ± 0.043 | 0.489 ± 0.220 | 0.510 ± 0.208 | 441 ± 187 |
| chb05 | 0.476 ± 0.019 | 0.466 ± 0.043 | 0.322 ± 0.040 | 0.279 ± 0.250 | 0.673 ± 0.275 | 294 ± 248 |

**Per-patient (float, mean ± std over 5 seeds), raw baseline (for reference):**

| Patient | Balanced acc | AUROC | AUPRC | Sensitivity | Specificity | FA/hour |
|---|---|---|---|---|---|---|
| chb01 | 0.473 ± 0.100 | 0.437 ± 0.147 | 0.120 ± 0.037 | 0.300 ± 0.257 | 0.647 ± 0.177 | 318 ± 159 |
| chb02 | 0.565 ± 0.032 | 0.588 ± 0.022 | 0.312 ± 0.025 | 0.590 ± 0.260 | 0.540 ± 0.228 | 414 ± 205 |
| chb03 | 0.560 ± 0.042 | 0.596 ± 0.049 | 0.273 ± 0.061 | 0.381 ± 0.174 | 0.740 ± 0.215 | 234 ± 194 |
| chb05 | 0.484 ± 0.015 | 0.491 ± 0.040 | 0.321 ± 0.017 | 0.307 ± 0.156 | 0.660 ± 0.157 | 306 ± 141 |

Notably, **chb02 and chb03 showed the raw front-end's mildest positive
signal (AUROC ~0.59) in the earlier diagnosis** — under the feature
front-end, both of those same patients drop to ~0.47 AUROC, i.e. *below*
chance-adjacent. If the feature front-end were unlocking real signal, it
should have helped these two most; instead it did the opposite. That's
further evidence against "front-end was the bottleneck" as the full
explanation, not for it.

**Pooled (20 patient×seed results each):**

| Metric | raw (baseline) | features |
|---|---|---|
| Float balanced accuracy | 0.521 ± 0.071 | 0.488 ± 0.073 |
| Float AUROC | 0.528 ± 0.105 | 0.475 ± 0.074 |
| Float AUPRC | 0.257 ± 0.090 | 0.236 ± 0.076 |
| Float sensitivity / specificity | 0.395±0.247 / 0.647±0.209 | 0.339±0.239 / 0.637±0.234 |
| Float false alarms/hour | 318 ± 188 | 326 ± 211 |

## Reading this honestly

- **Both front-ends are flat at chance, on both splits, on this pool.** Every
  balanced-accuracy and AUROC number above sits within roughly one standard
  deviation of 0.5, and none of the four (front-end, split) combinations
  clears chance outside its own seed band. The feature front-end is not
  meaningfully different from raw — the diagnosis's central hypothesis
  ("front-end destroys the signal") predicted a clear jump when the raw
  front-end is replaced with the literature-standard feature representation.
  That jump did not happen.
- **This does not mean line length / band power / spectral entropy are bad
  features** — they're the exact representation clinical detectors use. It
  means that at 4 patients / 868 windows / a 2-of-6-channel montage trimmed
  to fit Xylo's input budget, this SNN pipeline (63 hidden neurons, 8
  timesteps, class-weighted CE, balanced-accuracy checkpoint selection) does
  not extract a usable signal from them either. The two live suspects,
  per the task's own framing, are **montage/channel choice** (only 2 of the
  6 available channels are used here, purely to fit Xylo's ceiling — a real
  seizure focus could simply not be on F7-T7/F8-T8 for a given patient) and
  **true data scale** (4 patients is small for either representation to show
  a stable signal; the same scale-sensitivity finding applies here as it did
  to the raw front-end in `docs/eeg_seizure_results.md`).
- **Per the task's explicit instruction, this is not chased further.** No
  additional features, no larger feature set, no rhythmicity-measure
  swapping, no channel-selection sweep — that would be tuning against a
  single small pool until something looks positive by chance, not a
  validated finding. The next honest step is outside this task's scope:
  either a montage/channel sweep or scaling to more patients (both were
  explicitly named as the "next suspects," not attempted here).

## What this does and doesn't establish

- **Does establish:** a complete, tested, offline-verifiable feature
  extraction module (`eia.eeg_features`) with train-only normalization,
  wired as a clean, non-destructive `eeg_frontend` option alongside the
  unchanged raw path — reusable for whatever front-end investigation comes
  next (montage sweep, more patients, or both).
- **Does NOT establish:** that the front-end was the (whole) explanation for
  the earlier ~chance result. It rules out "swap the front-end representation
  and the modality unblocks" as a standalone fix, at this data scale and
  channel budget.

## Reproducing this

```bash
pip install "eia[data,xylo]"
# (all four commands under "Run configuration" above; data/chbmit/ must
# already have the 4-patient pool cached, or these will download it first)
```
