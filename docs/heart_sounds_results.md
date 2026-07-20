# Heart-sound (PCG) classification on PhysioNet/CinC 2016 — measured results

Heart sounds is the second real-data
modality after ECG (EEG/CHB-MIT seizure detection was tried and retired —
see `docs/eeg_seizure_results.md`, banner at top). Reported on its own, never
pooled with ECG/PPG/EEG results.

## Part 0 — dataset verification (done before writing any loader code)

Verified live against the actual PhysioNet server, not assumed:

- **Access path:** WFDB-native (`.hea`/`.dat`, also `.wav`), no format
  conversion needed (unlike CHB-MIT's EDF). Streams directly via
  `wfdb.rdrecord(record, pn_dir="challenge-2016/1.0.0/training-<set>")`.
  `wfdb.get_record_list('challenge-2016')` FAILS (no top-level `RECORDS` file
  at the `1.0.0/` root) — the per-set `pn_dir=` form is required.
- **Native fs = 2000 Hz**, confirmed live (`a0001`: `sig_name=['PCG', 'ECG']`,
  `sig_len=71332`, `units=['mV','mV']`).
- **6 training sets (a-f)**, per-set contents confirmed by directly listing a
  live directory: `MD5SUMS`, `RECORDS`, `RECORDS-abnormal`, `RECORDS-normal`,
  `REFERENCE-SQI.csv`, `REFERENCE.csv`, `SHA1SUMS`, `SHA256SUMS`, plus
  per-record `.dat`/`.hea`/`.wav` triples.
- **Label convention confirmed:** `REFERENCE.csv` is `record,label` with
  label `1` = abnormal, `-1` = normal — cross-checked `a0001`'s label=1
  against its own `.hea` comment line (`# Abnormal`).
- **Quality flag:** `REFERENCE-SQI.csv` is `record,label,quality` (0/1).
  training-a has 17/409 (~4%) quality=0, skewed heavily abnormal (16/17 of
  the quality=0 recordings are abnormal). Choice made: exclude quality=0 by
  default (`min_quality=1`) — a documented Part-0 decision, not a silent
  default. `quality` is `None` (kept, not excluded) for any set whose
  `REFERENCE-SQI.csv` can't be fetched.
- **Channel selection must be by NAME, not index:** only training-a carries a
  bonus simultaneous `ECG` reference channel (confirmed by checking the first
  record of every other set — b/c/d/e/f are all single-channel `PCG` only).
  `_load_cinc2016_record` selects `rec.sig_name.index("PCG")` explicitly for
  this reason; a fixed-index loader would have silently returned the ECG
  channel, not PCG, for every training-a recording.
- **Class balance, per set (all confirmed live):**

  | set | total | normal | abnormal |
  |-----|------:|-------:|---------:|
  | a   | 409   | 117    | 292      |
  | b   | 490   | 386    | 104      |
  | c   | 31    | 7      | 24       |
  | d   | 55    | 27     | 28       |
  | e   | 2141  | 1958   | 183      |
  | f   | 114   | 80     | 34       |
  | **total** | **3240** | **2575** | **665 (~20.5%)** |

  Imbalanced, but far milder than CHB-MIT's seizure prevalence — the
  existing class-weighted-CE + balanced-accuracy machinery is applied from
  the start regardless.
- **Subject ids are NOT recoverable.** The dataset's own documentation states
  "each subject/patient may have contributed between one and six heart sound
  recordings" and "files from the same patient are unlikely to be
  numerically adjacent" — but no subject-id field exists in any distributed
  file (`RECORDS`, `REFERENCE.csv`, `REFERENCE-SQI.csv`, or the `.hea`
  header — all directly inspected). **`HeartData.groups` is therefore always
  `None`; the split is by RECORDING, not subject**, documented here and in
  the `("heart","cinc2016")` data card. Some cross-recording, same-patient
  leakage risk is possible in principle and not eliminable from this public
  release — a genuine limitation, not a bug, and consistent with common
  practice in published CinC 2016 work (patient ids are withheld).

## Part A — pipeline (two front-ends, "features" is the shipped default)

Band-pass 20-400 Hz (S1/S2 energy is low, murmurs extend higher), windowed at
`window_sec=3.0` (native 2000 Hz). Two selectable front-ends
(`HeartData.frontend`, `--heart-frontend` on `scripts/xylo_verify.py`):

- **`"raw"`** — mirrors the ECG pipeline: FFT-resample hard down to
  `resample_to=128` Xylo timesteps (the `load_mitbih` fs-matched decoupling
  pattern), per-window z-score, then the same delta ON/OFF encoding ECG
  uses. Tried FIRST, per the task spec (S1/S2 are transients, so edge-
  encoding had a real *prior* chance here, unlike EEG's sustained rhythm).
  **Measured to fail for a specific, diagnosable reason (see below) — kept
  selectable for A/B, no longer the default.**
- **`"features"` (default since this write-up)** — line length / relative
  band power (`PCG_BANDS` = low 20-100 Hz, mid 100-200 Hz, high 200-400 Hz)
  / spectral entropy, computed per sub-window at/near the NATIVE 2000 Hz
  (`eia.signal_features`, `n_subwindows=24` → 24 Xylo timesteps, each
  ~125ms of native-rate signal — enough samples per sub-window for a
  meaningful Welch PSD). `PCG_FEATURE_NAMES` = `(line_length, low, high,
  spectral_entropy)`, 4 features × 1 channel × 2 (ON/OFF delta encoding) =
  **8 of Xylo's 16 input channels**, confirmed in the mapped footprint
  (`input 8/16`) — half the budget spent, headroom to add the `mid` band or
  a second channel later. This module is the generalized, modality-agnostic
  survivor of the retired `eia.eeg_features` (recovered from git history,
  not rebuilt from scratch — same `line_length`/`relative_band_power`/
  `spectral_entropy`/`extract_window_features`/
  `normalize_features_train_only` functions, now parameterized by
  `bands`/`feature_names` instead of hardcoding EEG's bands).

### Root-cause diagnosis: why `"raw"` fails (found after the first measurement)

The first raw-frontend run measured flat chance (float balanced acc ~0.50,
AUROC ~0.51 — see below). Diagnosis: `window_sec=3.0` at native 2000 Hz is
6000 samples; `resample_to=128` FFT-resamples that down to 128 timesteps, an
**effective rate of 2000×(128/6000) ≈ 42.7 Hz — a Nyquist of ~21 Hz**. Heart
sounds live at 20-400+ Hz (S1/S2 fundamental ~20-150 Hz, murmurs extending to
~400+ Hz) — the raw-waveform downsampling band-limits the diagnostic content
away *before* the SNN ever sees it. This is the same class of lesson EEG's
front-end redesign taught (docs/eeg_frontend_results.md): a fixed few-hundred-
timestep budget forces the timestep-reduction step to happen either before or
after spectral-feature extraction, and doing it before (raw waveform) can
destroy the signal outright, whereas doing it after (per-sub-window features
at native rate, THEN reduce to a few timesteps) preserves it. The `"features"`
front-end fixes this by computing Welch PSD / line length at/near native rate
per sub-window — the timestep reduction (6000 native samples → 24 sub-windows)
happens on already-extracted spectral features, not on the raw waveform.

### Data-quality bug found and fixed along the way

The first raw-frontend measurement run's data card printed
`[warn] cinc2016: non-finite values present in X (NaN/inf)` on every single
seed — missed on first pass, then caught before trusting the result. Root
cause: a handful of CinC 2016 recordings (confirmed: `a0018`, `a0204`, and
others) have NaN samples in their raw WFDB signal that `REFERENCE-SQI.csv`'s
quality flag does **not** catch; `filtfilt` propagates a single NaN sample
across the *entire* filtered signal, silently poisoning every window from
that recording. Fixed in `load_cinc2016`: check `np.isfinite(sig).all()`
right after loading and skip the recording (same "skip failing
records/subjects, print why, never train on it" pattern already used for
network failures) — confirmed no `non-finite` warning on any subsequent run.
**This bug affected both the raw and features measurements equally** (same
loader path); re-measuring after the fix isolated the Nyquist problem above
as a second, independent, and dominant cause of the raw front-end's chance
result — the NaN bug alone was not sufficient to explain it (the raw
front-end measured chance again after the NaN fix, on clean data).

## Part A — measurement (float model first, then XyloSim, per the EEG lesson)

300 recordings pulled (seeded `max_records=300` from the 3240-recording pool,
all 6 training sets), yielding ~2100-2150 windows depending on front-end
(a few recordings dropped per the NaN fix above); class-grouped split is not
possible (no subject ids, see Part 0) so `case_level.split_data` falls back
to plain stratified train/val/test by RECORDING. 5 seeds, 5 restarts/seed,
40 epochs/restart, class-weighted CE + balanced-accuracy checkpoint
selection — the same discipline that rescued real MIT-BIH ECG from majority
collapse. Base rate (majority class) for this 300-recording pull: **0.749
normal / 0.251 abnormal** (close to the full-dataset ~20.5%, some pull-to-
pull variance from the seeded random 300-of-3240 subsample).

| front-end | float bal. acc | float AUROC | float AUPRC | float recall [normal, abnormal] |
|---|---|---|---|---|
| raw (post-NaN-fix) | 0.503 ± 0.026 | 0.517 ± 0.033 | 0.278 ± 0.021 | [0.441±0.142, 0.565±0.097] |
| **features (default)** | **0.593 ± 0.041** | **0.631 ± 0.034** | **0.385 ± 0.073** | [0.816±0.058, 0.370±0.112] |

**The features front-end clears chance; raw does not.** Balanced accuracy
0.593 is ~2.3 std devs above the 0.5 no-skill floor across 5 seeds; AUROC
0.631 ± 0.034 is ~3.9 std devs above 0.5 and consistent seed-to-seed (no
seed's AUROC point estimate fell below ~0.58 — see per-seed lines in the
raw log). Per-class recall shows genuine (if imbalanced) discrimination —
0.816 normal / 0.370 abnormal — not majority-class collapse (which would
show ~[1.0, 0.0]). Raw stayed flat at chance on both balanced accuracy and
AUROC even after the NaN fix, confirming the Nyquist diagnosis above rather
than a data-quality artifact.

XyloSim (features front-end, the shipped default):

| | balanced acc | AUROC | recall [normal, abnormal] | float-vs-XyloSim agreement |
|---|---|---|---|---|
| XyloSim | 0.512 ± 0.012 | 0.524 ± 0.030 | [0.954±0.029, 0.070±0.036] | 0.779 ± 0.064 |

XyloSim balanced accuracy/AUROC drop back to ~chance even though the float
model clearly learns — quantization erases most of the minority-class
recall specifically (0.954 vs 0.070, close to majority-only collapse on
XyloSim despite the float model's real 0.370 abnormal recall). This is the
**same shape of on-chip fidelity gap already diagnosed for ECG**
(`docs/ecg_quant_diagnosis.md`: per-timestep integer state rounding
compounding nonlinearly over the window, worse with more timesteps) — heart
sounds' features front-end uses 24 timesteps, more than ECG's real-MIT-BIH
resampled configurations, so a nontrivial gap here is consistent with, not
contradictory to, that diagnosis. Per the task's explicit ordering
("don't chase XyloSim fidelity before the float model learns"), this gap is
**not chased further here** — it's the next open engineering question for
this modality, exactly parallel to ECG's.

Mapped footprint: `input 8/16, hidden 63/1000, output 2/8` — fits easily on
one Xylo core with substantial headroom (half the input budget still free).

## Verdict

**Heart-sound classification on real CinC 2016 DOES learn** with the
`"features"` (band-power/line-length/spectral-entropy) front-end — float
balanced acc 0.593 ± 0.041 and AUROC 0.631 ± 0.034, both clearly and
consistently above chance across 5 seeds, with genuine (non-collapsed)
per-class recall. This makes heart sounds the **second genuinely-learning
real-data modality** in this repo, after ECG. The `"raw"` delta-encoded
front-end does NOT learn (flat chance, both before and after an unrelated
NaN-record bug fix) for a diagnosed, mechanistic reason: aggressive
raw-waveform downsampling (to a Xylo-sized timestep budget) drops the
Nyquist frequency below where heart-sound energy lives. **Open next step,
not attempted here:** the float-to-XyloSim on-chip fidelity gap (0.593→0.512
balanced acc, similar in shape to ECG's diagnosed gap) — a natural follow-up
using the same margin-aware-training / timestep-reduction levers explored
for ECG (`docs/ecg_quant_diagnosis.md`, `docs/ecg_quant_fixes_results.md`),
not chased in this task per its own explicit "float first" ordering.

