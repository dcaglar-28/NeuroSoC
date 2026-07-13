# EEG seizure detection on CHB-MIT — measured results (Phase 1)

> **RETIRED/PAUSED** — code removed 2026-07-13; findings preserved. EEG
> seizure detection came back ~chance under raw and feature front-ends;
> paused pending a richer montage without the 16-input Xylo ceiling (Akida
> headroom) and/or more patients. Recoverable from git history.

Implements `docs/eeg_seizure_task.md` (MARCH "H", Phase 1). Part 1 is the
subject-independent seizure-vs-non-seizure result for the new EEG modality —
the second real-data workstream in this repo alongside ECG, reported on its
own (never pooled with ECG/PPG results). Part 2 is the patient-specific
diagnostic follow-up that disambiguates why Part 1 came back ~chance.

## Outcome, stated up front

**Subject-independent does NOT show cross-patient generalization** at either
data scale tried. Float balanced accuracy is 0.525 +/- 0.030 (9 patients, 1
seizure record each) and 0.513 +/- 0.016 (5 patients, 3 records each) — both
flat at chance. AUROC (~0.53-0.55) is only marginally above 0.5. False-alarm
rate (300-325/hour) is far above what any real seizure detector could ship
with (clinical targets are typically <1/hour).

**The follow-up patient-specific diagnostic (Part 2, below) narrows down
*why*: it is ALSO flat at chance (float balanced accuracy 0.517 +/- 0.074,
AUROC 0.524 +/- 0.097) on the exact same records subject-independent was
tested on.** Per the diagnostic's own decision rule
(`docs/eeg_seizure_task.md`): patient-specific learning while
subject-independent doesn't would mean "not enough cross-patient data";
patient-specific ALSO failing on the same data — what actually happened —
points at the **front-end** (6-channel montage / 0.5-25 Hz band-pass / 128
timesteps) losing the seizure signature, not simply "too few patients." Two
of five individual patients (chb02, chb03) showed a mild positive AUROC
(~0.59-0.60) — not nothing, but not decisive either given n=5 seeds on n=2-3
records per patient. **This local result is itself still data-constrained**
(2-3 records/patient, most with only one seizure event) — see
`notebooks/02_eeg_seizure.ipynb`, built to rerun this diagnostic at a larger
scale on Colab's faster network before treating "front-end too lossy" as
final.

**This is a different kind of negative result than VitalDB's**, and should
not be read the same way. VitalDB's conclusion was "the label itself does not
carry a usable signal, at any granularity tried — stop revisiting it." CHB-MIT
scalp EEG seizure detection is a well-established, decades-benchmarked
learnable problem in the literature (that is why this task picked it — see
`docs/eeg_seizure_task.md`'s framing: "strong physiology, excellent expert
labels, decades of benchmarks"), so a genuine data-quality dead end is
unlikely — but this repo's own front-end choices (montage/band/timesteps) are
now a live suspect alongside data volume, not a settled non-issue.

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
- **Does NOT establish:** that this net, at the data scales tried, detects
  seizures better than chance — cross-patient (Part 1) or, more surprisingly,
  within-patient (Part 2) either. Unlike VitalDB, this is NOT a "stop
  revisiting" conclusion — the literature says this signal exists in scalp
  EEG — but it does mean the front-end (montage/band-pass/timesteps) is now
  an open, live question, not a settled non-issue to route around with more
  patients alone.

## Part 2 — patient-specific diagnostic (disambiguating scale vs. front-end)

Implements the follow-up in `docs/eeg_seizure_task.md`: the subject-
independent result alone can't tell "too little cross-patient data" apart
from "the front-end destroyed the seizure signal" — both look identical from
outside. The patient-specific diagnostic reuses the exact same pipeline
(montage, band-pass, resample, delta encoding, `build_xylo_snn`,
class-weighted CE, balanced-accuracy checkpoint selection) but splits WITHIN
one patient — by RECORD, never by window, so a seizure event's neighbouring
windows can't straddle train/test (`datasets.eeg_patient_specific_split`).
**This is a diagnostic upper bound, not the deployment metric** — CHB-MIT's
classic high-AUROC results in the literature are usually patient-specific and
do not reflect a device that has never seen the patient.

### Run configuration

5 patients with >=2 cached records each (`chb01, chb02, chb03, chb05, chb06`
— `chb07-10` were excluded here for having only 1 cached record, insufficient
to hold one out; see `run_eeg_patient_specific`'s eligibility check), 3
seizure + 1 non-seizure record per patient (1072 windows total), 5 seeds x 5
restarts x 20 epochs. For a fair comparison, subject-independent was ALSO
rerun on this exact same 5-patient/1072-window pool (not the original
9-patient/718-window run above) so both numbers below come from identical
data.

```bash
# patient-specific
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --split patient-specific --eeg-subjects chb01,chb02,chb03,chb05,chb06 \
  --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300

# subject-independent, same 5-patient pool, for direct comparison
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --split subject-independent --eeg-subjects chb01,chb02,chb03,chb05,chb06 \
  --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300 --no-combined
```

### Per-patient results (patient-specific, float, mean +/- std over 5 seeds)

| Patient | Balanced acc | AUROC | AUPRC | Sensitivity | Specificity | FA/hour |
|---|---|---|---|---|---|---|
| chb01 | 0.473 +/- 0.100 | 0.437 +/- 0.147 | 0.120 +/- 0.037 | 0.300 +/- 0.257 | 0.647 +/- 0.177 | 318 +/- 159 |
| chb02 | 0.565 +/- 0.032 | 0.588 +/- 0.022 | 0.312 +/- 0.025 | 0.590 +/- 0.260 | 0.540 +/- 0.228 | 414 +/- 205 |
| chb03 | 0.560 +/- 0.042 | 0.596 +/- 0.049 | 0.273 +/- 0.061 | 0.381 +/- 0.174 | 0.740 +/- 0.215 | 234 +/- 194 |
| chb05 | 0.484 +/- 0.015 | 0.491 +/- 0.040 | 0.321 +/- 0.017 | 0.307 +/- 0.156 | 0.660 +/- 0.157 | 306 +/- 141 |
| chb06 | 0.503 +/- 0.085 | 0.509 +/- 0.055 | 0.192 +/- 0.030 | 0.383 +/- 0.172 | 0.623 +/- 0.215 | 339 +/- 193 |

### Diagnostic comparison (same 5-patient/1072-window pool, float)

| Split | Balanced acc | AUROC |
|---|---|---|
| **Subject-independent** (2 train / 1 val / 2 test patients) | 0.513 +/- 0.016 | 0.527 +/- 0.030 |
| **Patient-specific** (record-held-out within each patient; pooled over 5 patients x 5 seeds = 25 results) | 0.517 +/- 0.074 | 0.524 +/- 0.097 |

### Verdict

Per `run_eeg_patient_specific`'s own decision rule
(`(patient_specific_mean - std) > 0.5 and AUROC_mean > 0.6`): **NOT MET** —
patient-specific balanced accuracy's lower band (0.517 - 0.074 = 0.443) sits
below chance, and its AUROC (0.524) doesn't clear 0.6. **Patient-specific is
ALSO ~chance on these same records.** Per the task's framework, this points
at the front-end (montage/band-pass/timestep budget) as the more likely
bottleneck, ahead of pure data volume — though the caveat below means this
isn't yet the final word.

**Caveat — this diagnostic is itself still data-constrained.** 5 patients
with 2-3 records each (mostly a single seizure event per patient) is a small
base for "does more data within one patient help" — the two patients with
the most seizure-record diversity (chb02, chb03: AUROC ~0.59-0.60) are also
the ones showing the mildest positive signal, which is at least directionally
consistent with "more within-patient data would help," not decisively
against it. `notebooks/02_eeg_seizure.ipynb` is built to rerun this exact
diagnostic with every available seizure record for these patients (CHB-MIT
has 3-7 per subject) on a faster connection — that is the run to trust before
treating "front-end too lossy" as the final answer over "still not enough
data, even within-patient."

## Follow-ups if this is revisited

1. **Run `notebooks/02_eeg_seizure.ipynb` on Colab (do this first)**: pulls
   every available seizure record for the 9 loadable subjects on a faster
   connection, runs both splits on the same larger pool, and prints the
   verdict — the most direct way to firm up "front-end too lossy" vs. "still
   not enough data" from Part 2 above. Verified to run end-to-end locally at
   reduced scale before being written up here.
2. **If the notebook confirms front-end too lossy**: revisit
   `datasets.EEG_MONTAGE` (try 8 channels — the exact Xylo ceiling), the
   band-pass range, or `resample_to` (try more than 128 timesteps, trading
   off against the ECG quantization-fidelity lesson that fewer timesteps
   usually helps on-chip agreement — worth an explicit sweep here, not
   assumed).
3. **If the notebook confirms scale problem instead**: keep pulling more
   subjects/records/subject (`--eeg-seizure-records`,
   `--eeg-nonseizure-records`, `--eeg-subjects`) — the `data/chbmit/` cache
   means this is incremental across sessions, not repeated download.
4. **Investigate the `read_edf` "math domain error"** for chb12/13/14/16/
   17/18/19/21 (an 8-subject loss, including the chb01/chb21 duplicate-patient
   case) — likely a wfdb parsing edge case worth a small, separate
   investigation (e.g. comparing an `.edf` field wfdb chokes on across a
   working vs. failing file).

## Reproducing this

```bash
pip install "eia[data,xylo]"

# subject-independent (deployment metric)
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --eeg-seizure-records 1 --eeg-nonseizure-records 0 \
  --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300 --no-combined

# patient-specific diagnostic (Part 2) -- needs >=2 cached records/patient
python scripts/xylo_verify.py --modality eeg --real --require-real \
  --split patient-specific --eeg-subjects chb01,chb02,chb03,chb05,chb06 \
  --epochs 20 --n-restarts 5 --n-seeds 5 --max-verify 300

# both splits together, at Colab scale, with the explicit verdict:
# notebooks/02_eeg_seizure.ipynb
```
