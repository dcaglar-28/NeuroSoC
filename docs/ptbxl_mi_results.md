# PTB-XL myocardial infarction detection — measured results

Deepens the ECG capability from *arrhythmia* (MIT-BIH, single-lead,
per-beat, `docs/akida_ecg_results.md`) to *myocardial infarction*
(PTB-XL, 12-lead, per-recording) — implements `docs/ptbxl_mi_task.md`.
This is a new task/dataset on the EXISTING ECG modality, not a new
modality: `EcgData`/`load_ecg`/the ECG-arrhythmia Akida path
(`build_akida_model`), heart sounds, the synthetic CRM demo, and
`rockpool_models.py` are all untouched — `MiData`/`load_ptbxl`/`load_mi`/
`build_akida_mi_model` are additive.

## Part 0 — dataset access + label mapping (verified live, not assumed)

**Access confirmed:** PTB-XL v1.0.3 is an OPEN PhysioNet dataset (the
`physionet.org/files/ptb-xl/1.0.3/` listing is directly HTTP-browsable, no
credentialed access needed — "protected/published-projects" in PhysioNet's
internal server path is just directory naming, not a gate). `wfdb.rdrecord`
streams the 100 Hz (`records100/`) version directly via `pn_dir=` (same
pattern as MIT-BIH/CinC 2016) — confirmed live: `fs=100`, `sig_len=1000`
(10 s), `n_sig=12`, `sig_name=['I','II','III','AVR','AVL','AVF','V1'..'V6']`,
units mV. Each record's exact relative path is given directly by
`ptbxl_database.csv`'s own `filename_lr` column (e.g.
`records100/15000/15000_lr`) — no need to independently derive PTB-XL's
1000-record subdirectory grouping.

**A real download bug caught before it corrupted anything:** the first
`curl` pull of `ptbxl_database.csv` silently truncated (6,221,824 of the
expected 6,594,879 bytes — a partial download that still parsed as valid-looking
CSV, just with 5,036 fewer rows: 14,932 instead of 21,799). Caught by
cross-checking the downloaded file's byte size against the directory
listing's reported size before trusting any parsed number from it — the
exact "don't trust memory, verify" discipline this project's Part-0 process
exists to enforce, this time catching a transport-layer problem rather than
a documentation-assumption one. Re-fetched with `curl -f --retry 3`;
confirmed exact byte-for-byte match (6,594,879 bytes, 21,800 lines = header
+ 21,799 records) before proceeding.

**21,799 records / 18,869 patients — confirmed exact match to the task
doc's stated counts** (`ptbxl_database.csv`, `patient_id.nunique()`).

**Label mapping (the load-bearing detail) — verified against PTB-XL's own
shipped `example_physionet.py`, not guessed:**
- `scp_codes` is a stringified `{SCP_code: likelihood}` dict per record
  (`ast.literal_eval` to parse).
- `scp_statements.csv` has a `diagnostic_class` column with exactly 5
  non-null values — NORM, MI, STTC, CD, HYP — on exactly the 44 (of 71
  total) rows where `diagnostic==1` (confirmed these two filters are
  equivalent: 44 rows have `diagnostic_class` set, 44 rows have
  `diagnostic==1`, same 44 rows).
- **Critically, the official example does NOT threshold by likelihood at
  all** — every key in `scp_codes`, regardless of its likelihood value
  (including `0.0`), counts if it's in the diagnostic-eligible set.
  Likelihood `0.0` means "present, not confidence-scored," not "absent" —
  confirmed by directly reading `example_physionet.py`'s
  `aggregate_diagnostic` function, not inferred. Adopted verbatim: see
  `datasets.scp_codes_to_superclasses`.
- **Binary MI-vs-NORM label rule (this task's own choice, stated
  explicitly):** a record is "confidently MI" (`y=1`) iff its diagnostic
  superclass SET is EXACTLY `{"MI"}`; "confidently NORM" (`y=0`) iff
  EXACTLY `{"NORM"}`. Any other set — empty, a single other superclass
  (STTC/CD/HYP alone), or ANY multi-superclass combination (including one
  that contains MI alongside something else, e.g. `{"MI","STTC"}`) — is
  EXCLUDED from this binary task, not folded into either class. See
  `datasets.mi_norm_label`.
- `validated_by_human` is `True` for 16,056/21,799 (~73.7%) records —
  reported here for completeness (the task doc explicitly asks it be
  checked) but NOT used to filter, matching the official example and the
  overwhelming majority of published PTB-XL baselines.

**Confirmed live class counts (full 21,799-record pool):** 2,532 MI-only +
9,069 NORM-only = **11,601 eligible records (~21.8% MI)** — imbalanced,
class-weighted loss + balanced-accuracy checkpoint selection applied
throughout. 2,259 unique MI patients, 8,478 unique NORM patients.

**Split — confirmed patient-respecting, used via existing machinery, not a
new splitter:** PTB-XL's shipped `strat_fold` (1–10) is confirmed live to
never split a patient across folds, and to keep the MI ratio at ~20–22% in
EVERY fold (checked all 10 individually) — folds 1–8 (9,286 records,
22.0% MI) / 9 (1,147, 20.3%) / 10 (1,168, 21.9%), zero patient overlap
across the three groups. Rather than adding a parallel fold-based splitter
for this one modality, `MiData.groups = patient_id` is set and split via
the SAME `case_level.split_data` (`GroupShuffleSplit` on patient id) every
other modality here already uses — equally patient-safe (same guarantee,
confirmed via `test_mi_no_leakage_patient_split`), reusing existing code
rather than forking a fold-aware alternative.

## Front-end and architecture

MI is a MORPHOLOGY signal (ST-segment elevation, Q waves, T-wave inversion)
— raw waveform, like ECG-arrhythmia and CRM, NOT the heart-sounds audio
filterbank. Uses all **12 leads** (MI localization needs spatial lead
information — an inferior infarct shows in II/III/AVF, an anterior one in
V1–V4 — and the committed Akida target has no 16-input ceiling to work
around, unlike Xylo). Fed as a genuinely 2-D `(12, 1000, 1)` "image" (leads
× time) to a NEW `build_akida_mi_model` — the same class of input as heart
sounds' bands × time map, NOT a reuse of ECG-arrhythmia's/CRM's
single-column `(window, 1, 1)` reshape (spatial lead structure would be
destroyed by that reshape).

Architecture (deeper than heart's, since 1000 samples is ~40× heart's 24
sub-windows): `Input(12,1000,1) uint8 -> Rescaling -> Conv2D(8, 3x3, stride
2x2) -> ReLU -> Conv2D(16, 3x3) -> MaxPool(2x2) -> ReLU -> Conv2D(32, 3x3)
-> MaxPool(2x2) -> ReLU -> Conv2D(64, 3x3) -> ReLU -> GlobalAvgPool ->
Dense(2)`. Converted CLEANLY on the first attempt — no new undocumented
Akida v2 constraints beyond the ones already found porting ECG/heart
(square kernel/stride/pool every conv layer, `post_relu_gap=True` for the
final global-avg-pooled block). ~25K float parameters.

Per-lead normalization: `MiData.X` is raw mV, never pre-normalized (same
contract as heart's filterbank features) — z-scored per lead on the TRAIN
split only, post-split, via `signal_features.normalize_features_train_only`
(heart sounds' exact function, reused verbatim — `scripts/akida_verify.py`
now gates on `modality in ("heart","mi")` rather than the heart-specific
`frontend` attribute, since `MiData` has no such field).

## Part A — measurement (float model first, then Akida-sim)

2,000 real PTB-XL recordings pulled (the loader's own seeded default subset
of the 11,601-record eligible pool), 79.1%/20.9% NORM/MI class balance this
pull (close to the full-pool ~21.8% MI). Patient-grouped split via
`case_level.split_data` (1,179 train / 295 val / 492 test patients, zero
overlap — confirmed by the script's own "case-grouped ... no case overlap"
log line), class-weighted `SparseCategoricalCrossentropy`, per-lead
train-only z-score, 3 restarts/seed selecting the best VAL-balanced-accuracy
checkpoint, **5 seeds**, 15 float epochs + 5 QAT epochs per restart, 300
held-out windows verified against the Akida sim per seed, 8-bit
weights/activations.

| seed | float bal. acc | float AUROC | float recall [NORM, MI] | Akida-sim bal. acc | agreement |
|---|---|---|---|---|---|
| 0 | 0.798 | 0.885 | [0.909, 0.688] | 0.797 | 0.967 |
| 1 | 0.749 | 0.857 | [0.642, 0.856] | 0.804 | 0.763 |
| 2 | 0.806 | 0.904 | [0.929, 0.683] | 0.808 | 0.930 |
| 3 | 0.823 | 0.928 | [0.920, 0.725] | 0.828 | 0.977 |
| 4 | 0.797 | 0.878 | [0.860, 0.734] | 0.790 | 0.950 |
| **mean ± std** | **0.795 ± 0.025** | **0.890 ± 0.024** | [0.852±0.108, 0.737±0.063] | **0.806 ± 0.013** | **0.917 ± 0.079** |

## Verdict: the float model detects MI on real PTB-XL, at the benchmarked level

**Float AUROC 0.890 ± 0.024 lands right at the literature's ~0.9+
benchmarked range for PTB-XL MI-vs-NORM** (`docs/ptbxl_mi_task.md`'s stated
expectation) — this is real, working myocardial-infarction detection on
cardiologist-labeled 12-lead ECG, not a chance result. Per-class recall is
genuine (0.852/0.737 mean, no majority collapse toward the 79%-prevalent
NORM class) and consistent across all 5 seeds (recall for MI, the harder,
minority class, never drops below 0.68 on any seed).

**The Akida fidelity gap is essentially closed here too — the THIRD
modality in a row to show this pattern** (after ECG-arrhythmia,
`docs/akida_ecg_results.md`, and heart sounds, `docs/akida_heart_results.md`):
Akida-sim balanced accuracy (0.806 ± 0.013) tracks float (0.795 ± 0.025)
closely — actually marginally higher, well within noise — with agreement
0.917 ± 0.079. Four of five seeds show agreement ≥0.93; **seed 1 is the one
seed with meaningfully lower agreement (0.763)**, driven by the same
"AUROC-holds-while-recall-shifts" pattern already diagnosed for the CRM demo
(`docs/synthetic_crm_results.md`): seed 1's float per-class recall is
[0.642, 0.856] — inverted relative to every other seed's [NORM-favoring]
pattern — while its own AUROC (0.857) stays close to the other seeds',
meaning the RANKING is still good but that particular restart's decision
threshold landed differently. Reported as measured, not re-run to chase a
cleaner number, consistent with this repo's multi-seed discipline.

This strengthens (a third data point, not a new proof) the working
hypothesis from the ECG/heart write-ups: these Akida architectures are
small **feedforward** CNNs evaluated once per window, with no per-timestep
recurrent integer state to accumulate the rounding error Xylo's LIF
dynamics were diagnosed to compound over a window
(`docs/ecg_quant_diagnosis.md`).

Mapped footprint: `(12, 1000, 1) -> (2,)`, 6 mapped Akida layers.

## Caveats (carried forward, unchanged)

- **Akida's software simulator has no confirmed bit/cycle-accurate-to-
  silicon claim** (checked 3 official BrainChip sources — see
  `docs/akida_ecg_results.md` Part 0). This measures agreement with
  BrainChip's software model, not verified silicon behavior.
- **This measures Akida 2.0 (`FPGA_v2`), not Akida Pico** — the sub-mW
  core actually committed to for the always-on biosignal cluster
  (CLAUDE.md's hardware section). Pico-specific re-characterization is
  unstarted for every modality, including this one.
- **Not evidence about STTC/ischemia, the 5-superclass multi-label task, or
  a reduced/single-lead variant** — all explicitly out of scope here, noted
  as follow-ups below.
- **Real PTB-XL, but a specific curated binary subset** — "confidently
  MI-only" vs. "confidently NORM-only" (mixed-superclass records excluded)
  is the highest-tractability framing, not the full clinical picture (real
  practice sees plenty of mixed/comorbid presentations this task
  deliberately excludes — see the label rule in Part 0).

## Reproduce

```bash
scripts/akida_docker_run.sh python scripts/akida_verify.py --modality mi --real --n-seeds 5
```

## Next steps (noted, not built here)

- STTC (ischemia) as a second binary task, and the full 5-superclass
  multi-label task (`MiData`/`load_ptbxl` already carry everything needed
  — `scp_codes_to_superclasses` returns the full set, `mi_norm_label`'s
  binary collapse is the only piece specific to this task).
- A reduced-lead / single-lead "wearable-realistic" variant — the 12-lead
  version proves the capability; a field device may not have all 12.
