# EEG seizure detection on CHB-MIT — measured results (Phase 1)

Implements `docs/eeg_seizure_task.md` (MARCH "H", Phase 1). This is the
subject-independent seizure-vs-non-seizure result for the new EEG modality —
the second real-data workstream in this repo alongside ECG, reported on its
own (never pooled with ECG/PPG results).

## Outcome, stated up front

**Does NOT show cross-patient generalization at this run's data scale.**
Float balanced accuracy is 0.525 +/- 0.030 and XyloSim balanced accuracy is
0.500 +/- 0.028 across 5 seeds — both flat at chance, tightly so. AUROC
(0.549 +/- 0.050 float, 0.520 +/- 0.038 XyloSim) is only marginally above 0.5.
False-alarm rate (300-325/hour) is far above what any real seizure detector
could ship with (clinical targets are typically <1/hour).

**This is a different kind of negative result than VitalDB's**, and should
not be read the same way. VitalDB's conclusion was "the label itself does not
carry a usable signal, at any granularity tried — stop revisiting it." CHB-MIT
scalp EEG seizure detection is a well-established, decades-benchmarked
learnable problem in the literature (that is why this task picked it — see
`docs/eeg_seizure_task.md`'s framing: "strong physiology, excellent expert
labels, decades of benchmarks"). The honest interpretation here is **data
volume**, not signal absence: this run used **9 patients, 1 seizure record
each, 718 total windows** — a small fraction of CHB-MIT's ~915 hours and 198
seizures across 22-24 patients — because of a real, measured constraint (see
"Why the run is this small," below). Subject-independent seizure detection
with reasonable sensitivity typically needs many more patients and records
per patient than were used here; this result does not contradict the
literature, it just didn't reach the data volume where that signal shows up.

## Why the run is this small (a real constraint, not a shortcut)

CHB-MIT is EDF, not native WFDB — `wfdb.io.convert.edf.read_edf(...,
pn_dir=...)` streams it correctly (confirmed in Part 0), but each ~1-hour,
23-channel record is ~40 MB and, over the course of this task, download
throughput to PhysioNet varied enormously — a single record took anywhere
from ~4 to ~13+ minutes depending on when it ran, with one stretch of ~25
minutes producing only one new file. Two earlier, larger attempts (13
subjects x up to 3 records, then 9 subjects x up to 3 records) were killed
mid-run because they were on pace to take multiple hours; downloads are
cached to `data/chbmit/` (gitignored) as they complete, so this is bounded to
the network cost actually incurred, not repeated. The final run used the
already-warm cache plus the minimum additional pulls needed (`--eeg-
seizure-records 1 --eeg-nonseizure-records 0`) to get a complete, honest
multi-seed subject-independent number within a practical time budget — a
deliberate trade of data volume for wall-clock time, stated plainly rather
than hidden.

## Part 0 — dataset verification (done before writing any loader code)

- CHB-MIT is EDF on PhysioNet, no `.hea` files — `wfdb.rdrecord` (which needs
  `.hea`) 404s. `wfdb.io.convert.edf.read_edf(record+".edf", pn_dir=...)`
  reads it directly, confirmed against a live record (`chb01_03.edf`: fs=256,
  23 channels, `sig_len=921600` = exactly 1 hour).
- Channel names confirmed via live header reads: `FP1-F7, F7-T7, T7-P7,
  P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8,
  F8-T8, T8-P8, P8-O2, FZ-CZ, CZ-PZ, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8`
  (note: `T8-P8` appears twice — a real quirk of the recording, not a bug;
  our fixed montage doesn't use it, avoiding ambiguity).
- `chbXX-summary.txt` seizure interval format confirmed for both the singular
  (`Seizure Start Time:`) and numbered (`Seizure 2 Start Time:`) cases against
  live files (`chb01`, `chb03`, `chb04`, `chb06`, `chb15`).
- **A header-only probe across every subject's first record found chb12,
  chb13, chb14, chb16, chb17, chb18, chb19, chb21 all raise a "math domain
  error" from `read_edf`** — a real, reproducible limitation on those
  specific files (not per-record flakiness: their very first record fails
  too), not guessed. `datasets.DEFAULT_EEG_SUBJECTS` excludes them;
  `load_chbmit` also skips any subject/record it can't read at load time and
  prints why, so passing one of these ids is harmless (0 windows from it).
  This means **chb21 (documented as the same patient as chb01, a second
  recording session) is currently unreadable by this loader** — the
  canonicalization logic (`eeg_canonical_patient`, folds chb21 into chb01's
  group) is implemented and unit-tested on synthetic data regardless, so it's
  ready the moment chb21 becomes readable (e.g. a future wfdb version fix).

## Pipeline (hardware-first, per the task spec)

```
23 native CHB-MIT channels
  -> fixed 6-channel bipolar montage (EEG_MONTAGE — see below)
  -> 4th-order Butterworth band-pass, 0.5-25 Hz (datasets.bandpass_eeg)
  -> 4.0s windows at native 256 Hz (1024 samples)
  -> FFT-resampled to 128 Xylo timesteps (datasets.resample_windows)
  -> per-window z-score
  -> delta ON/OFF encoder, PER CHANNEL -> 12 spike channels (2 x 6)
  -> build_xylo_snn(n_in=12, n_hidden=63, n_out=2)
  -> XyloSim verify
```

**Fixed montage** (`datasets.EEG_MONTAGE`): `FP1-F7, F7-T7, FP2-F8, F8-T8,
FZ-CZ, CZ-PZ` — left+right frontotemporal pairs (a common seizure-focus
region) plus the midline central-parietal pair. Applied identically to every
subject (not patient-specific) per the task's Flag 2. **6 channels -> 12
spike channels**, leaving margin under Xylo's 16-channel ceiling (8ch would
sit exactly at the limit with zero headroom, per Flag 1) — confirmed by the
mapped footprint: `input 12/16`.

**Timesteps**: 128, down from the native 1024 samples/window (an 8x
reduction) — the `load_mitbih`-style `resample_to` decoupling, motivated by
`docs/ecg_quant_diagnosis.md`'s root-cause finding that on-chip fidelity
degrades with timestep count. Not swept against fewer/more timesteps in this
pass (noted as follow-up, same as the ECG work).

## Run configuration

9 subjects (`chb01, chb02, chb03, chb05, chb06, chb07, chb08, chb09, chb10`
— `datasets.DEFAULT_EEG_SUBJECTS`), 1 seizure record per subject, 0
non-seizure records (see "Why the run is this small"), 5 seeds x 5 restarts x
20 epochs, subject-independent `GroupShuffleSplit` (patient id).

```bash
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --eeg-seizure-records 1 --eeg-nonseizure-records 0 \
  --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300 --no-combined
```

Data card (identical every seed — same 9-subject/9-record pool; only the
train/val/test patient split and net init vary by seed):

```
samples : 718   class balance : {0: 0.752, 1: 0.248}  (540 / 178)
[warn] chbmit: high majority base rate (75.2%)
[split] case-grouped: 4 train / 2 val / 3 test patients  (no patient overlap, every seed)
```

Only 9 patients total (4/2/3 split) is a SMALL N for a subject-independent
evaluation — which specific patients land in the test set matters enormously
for a 3-patient test fold, so per-seed numbers below are expected to be noisy
by construction, not just from training-init randomness.

## Per-seed results

| Seed | Val bal acc (window, ckpt selection) | Float bal acc | Float sens/spec | XyloSim bal acc | XyloSim sens/spec | Agreement | FA/hr (float / xylo) |
|---|---|---|---|---|---|---|---|
| 0 | 0.529 | 0.512 | 0.135 / 0.889 | 0.507 | 0.192 / 0.822 | 0.909 | 100 / 160 |
| 1 | 0.603 | 0.575 | 0.627 / 0.522 | 0.501 | 0.403 / 0.600 | 0.664 | 430 / 360 |
| 2 | 0.564 | 0.490 | 0.247 / 0.733 | 0.453 | 0.312 / 0.594 | 0.829 | 240 / 365 |
| 3 | 0.619 | 0.543 | 0.413 / 0.672 | 0.541 | 0.360 / 0.722 | 0.620 | 295 / 250 |
| 4 | 0.594 | 0.507 | 0.635 / 0.378 | 0.497 | 0.400 / 0.594 | 0.747 | 560 / 365 |

**Multi-seed summary (mean +/- std over 5 seeds):**

| Metric | Float | XyloSim |
|---|---|---|
| Balanced accuracy | **0.525 +/- 0.030** | **0.500 +/- 0.028** |
| Per-class recall | [0.639+/-0.176, 0.411+/-0.200] | [0.667+/-0.092, 0.333+/-0.078] |
| Sensitivity / specificity | 0.411+/-0.200 / 0.639+/-0.176 | 0.333+/-0.078 / 0.667+/-0.092 |
| AUROC | 0.549 +/- 0.050 | 0.520 +/- 0.038 |
| AUPRC | 0.312 +/- 0.022 | 0.318 +/- 0.056 |
| False alarms / hour | 325 +/- 158 | 300 +/- 83 |
| Float vs. XyloSim agreement | — | 0.754 +/- 0.106 |

(AUPRC's baseline under 24.8% positive prevalence is 0.248 — 0.31-0.32 is only
modestly above that, not a strong signal on its own.)

## Reading the numbers honestly

- **Balanced accuracy is the decisive number, and it's flat at chance for
  both float and XyloSim** (0.525 and 0.500, both within about 1 standard
  deviation of 0.5). Sensitivity swings from 0.135 to 0.635 across seeds
  entirely because of *which 3 patients* land in the test fold each time —
  pediatric seizure semiology varies a lot patient to patient, and 3 test
  patients is nowhere near enough to average that out.
- **False-alarm rate (300-325/hour) rules this out as a usable detector as
  trained** — a real device firing every ~11 seconds is not viable regardless
  of sensitivity. This traces directly to a model with near-chance
  discrimination: pushed by class-weighted loss to not just collapse to the
  majority class, it ends up firing on a large, non-specific fraction of
  windows instead of learning the actual seizure signature.
- **Float vs. XyloSim agreement (0.754 +/- 0.106) has real seed-to-seed
  spread**, consistent with the ECG/VitalDB checkpoint-sensitivity finding
  (`docs/ecg_quant_fixes_results.md`, `docs/vitaldb_ppg_results.md`): a
  float model with near-chance, low-margin decisions is more fragile under
  quantization than a confidently discriminative one, so this number should
  not be over-read from any single seed either.
- The **pipeline mechanics are fully verified correct**: montage selection,
  band-pass, resample-to-timesteps, per-channel delta encoding to 12 spike
  channels, `n_in=12` Xylo mapping (confirmed in the footprint line, `input
  12/16`), subject-independent (patient-grouped) splitting with a hard
  leakage check, and multi-seed float/XyloSim/AUROC/AUPRC/sensitivity/
  specificity/FA-per-hour reporting. What's unverified is whether the
  *learned* net generalizes across patients at this data scale — a data
  volume question, not a plumbing question.

## What this does and doesn't establish

- **Does establish:** a complete, hardware-first, Xylo-mappable EEG pipeline
  registered as a first-class modality (`train.py` and `scripts/
  xylo_verify.py --modality eeg`), with correct subject-independent
  evaluation machinery and the clinically appropriate metric set.
- **Does NOT establish:** that this net, at 9 patients/718 windows, detects
  seizures cross-patient better than chance. Given CHB-MIT's track record in
  the literature, the most likely explanation is data volume (9 patients, 1
  record each) rather than an absence of signal in scalp EEG for this task —
  unlike VitalDB, this is NOT a "stop revisiting" conclusion.
- **Patient-specific (within-patient) secondary number**: not computed in
  this pass (optional per the task spec, gated behind time already spent on
  the download-cost investigation above). Follow-up work if revisited:
  compute it for contrast (expect it to look much better, and explicitly
  label why that's not the deployment-honest number, matching the task's
  own warning about this being the "classic CHB-MIT high-AUROC use").

## Follow-ups if this is revisited

1. **More data, patiently**: rerun with more subjects and more
   records/subject (`--eeg-seizure-records`, `--eeg-nonseizure-records`,
   `--eeg-subjects`) when there's a larger time/network budget — the
   `data/chbmit/` cache means this can be done incrementally across sessions
   rather than repeating any download.
2. **Timestep sweep**: try `--resample-to` values other than 128, following
   the ECG lesson that fewer timesteps can help XyloSim fidelity — not done
   here due to the data-volume-first priority.
3. **Investigate the `read_edf` "math domain error`** for chb12/13/14/16/
   17/18/19/21 (an 8-subject loss, including the chb01/chb21 duplicate-patient
   case) — likely a wfdb parsing edge case worth a small, separate
   investigation (e.g. comparing an `.edf` field wfdb chokes on across a
   working vs. failing file).

## Reproducing this

```bash
pip install "eia[data,xylo]"
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --eeg-seizure-records 1 --eeg-nonseizure-records 0 \
  --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300 --no-combined
```
