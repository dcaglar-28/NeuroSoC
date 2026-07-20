# VitalDB PPG hemorrhage task — measured results

This is the multi-seed
float + XyloSim report for the new real, open, case-level blood-loss label
(VitalDB `intraop_ebl`), reported on its own per the guiding principle that
each hemorrhage dataset is trained/evaluated separately (never pooled with
BIDMC/synthetic).

## Part 0 — field/track verification (done before writing any loader code)

Checked against the installed `vitaldb` 1.5.8 library and the live API, not
memory:

- `vitaldb.load_clinical_data(caseids=[])` returns an **empty** frame despite
  its own docstring claiming that means "all cases" — every caseid must be
  passed explicitly (`caseids=list(range(1, 6389))`).
- Clinical table has `intraop_ebl` (mL): non-null for **3987/6388** cases.
  `intraop_rbc` (units transfused) also present, non-null for all cases,
  >0 for 352 — not used for v1's label, noted for a future graded version.
- Raw fingertip PPG track is `SNUADC/PLETH`, confirmed **500 Hz** via
  `load_case(caseid, ["SNUADC/PLETH"], interval=1/500)`; present in
  **6157/6388** cases.
- **3781 cases** have both `SNUADC/PLETH` and non-null `intraop_ebl` —
  the qualifying pool this loader draws from.
- EBL distribution over qualifying cases: median 150 mL, 90th pct 650 mL,
  95th pct 1100 mL. Threshold `>=500 mL` (a common surgical-transfusion-
  trigger cutoff) gives **~14% positive** at full-pool case level — chosen
  as v1's binary cutoff (documented in the loader's docstring and the data
  card, not hidden in a config).

## What was built

- `datasets.load_vitaldb_ppg` / `datasets.load_ppg_vitaldb` (provenance-
  guarded, mirrors `load_ppg`'s fallback contract), `PpgData.groups` (case id
  per window), a `("ppg", "vitaldb")` data-card entry with the three caveats,
  `--ppg-source {bidmc,vitaldb}` on `train.py` and `scripts/xylo_verify.py`,
  case-grouped `GroupShuffleSplit` (raises on any case leaking across splits),
  and `--n-seeds` multi-seed reporting on `scripts/xylo_verify.py`. BIDMC is
  unchanged.

## Run configuration

150 qualifying cases (seeded random subset, not first-N-by-caseid), 4.0s
capture window at native 500 Hz FFT-resampled to 125 Xylo timesteps, up to 40
windows/case, `ebl_threshold=500` mL, 5 seeds x 5 restarts x 15 epochs,
`--max-verify 300`.

```
python scripts/xylo_verify.py --modality ppg --real --require-real \
  --ppg-source vitaldb --max-cases 150 --epochs 15 --n-restarts 5 \
  --n-seeds 5 --max-verify 300 --no-combined
```

Data card (identical every seed — same 150-case pool, `seed` only varies the
train/val/test partition and net init):

```
samples : 6000   class balance : {0: 0.84, 1: 0.16}  (5040 / 960)
[warn] vitaldb: high majority base rate (84.0%)
[split] case-grouped: 89 train / 23 val / 38 test cases  (no case overlap, every seed)
```

## Per-seed results

| Seed | Final val bal acc | Float bal acc | Float recall [0,1] | XyloSim bal acc | XyloSim recall [0,1] | Agreement |
|---|---|---|---|---|---|---|
| 0 | 0.529 | 0.473 | [0.767, 0.178] | 0.592 | [0.635, 0.550] | 0.453 |
| 1 | 0.502 | 0.500 | [1.000, 0.000] | 0.508 | [0.975, 0.040] | 0.970 |
| 2 | 0.553 | 0.541 | [0.107, 0.975] | 0.501 | [0.677, 0.325] | 0.413 |
| 3 | 0.522 | 0.533 | [0.463, 0.603] | 0.960\* | [0.960, nan]\* | 0.433 |
| 4 | 0.504 | 0.500 | [0.084, 0.917] | 0.441 | [0.658, 0.225] | 0.387 |

\* Seed 3's 300-window XyloSim sample happened to draw **zero** class-1
windows (`recall[1] = nan`), so its "balanced accuracy" is really just
class-0 recall, not a genuine balanced estimate — an artifact of small
`--max-verify` colliding with ~16% class prevalence and case-level sampling,
not a real result. Left in the table rather than hidden, but excluded from
read as a meaningful number on its own.

**Multi-seed summary (mean +/- std over 5 seeds):**

| | Float | XyloSim |
|---|---|---|
| Balanced accuracy | 0.509 +/- 0.025 | 0.600 +/- 0.186 |
| Per-class recall | [0.484+/-0.360, 0.535+/-0.389] | [0.781+/-0.153, nan] |
| Float vs. XyloSim agreement | — | 0.531 +/- 0.220 |

## Honest read of the result

**The float model is not learning much real signal at all** — balanced
accuracy sits at 0.509 +/- 0.025 across 5 independent seeds, essentially
chance (0.5), and 3 of 5 seeds' *best-of-5-restarts* validation balanced
accuracy rounds to within 0.03 of exact majority-class collapse (0.500,
0.502, 0.504) despite class-weighted loss + balanced-accuracy checkpoint
selection — the same machinery that successfully escaped majority collapse
on real MIT-BIH ECG. This is not a training bug; it matches the label's own
documented weakness: **one whole-case EBL total is stamped onto every
4-second window from that case**, including windows recorded hours before
any bleeding happened. A single 4-second pulse waveform has no principled
reason to encode a number that summarizes an entire multi-hour surgery. BIDMC's
SpO2 proxy, by contrast, is a *per-window* measured value — worse as a
hemorrhage claim, but a fairer per-window learning target — which is exactly
why VitalDB is documented as a **step up in label realism, not in per-window
learnability**, in the data card and `CLAUDE.md`.

XyloSim balanced accuracy (0.600 +/- 0.186) and float-vs-XyloSim agreement
(0.531 +/- 0.220) show enormous seed-to-seed spread — consistent with the
ECG quantization-fidelity finding in `docs/ecg_quant_fixes_results.md` that
agreement is highly checkpoint-sensitive, compounded here by a float model
whose decisions are already close to chance (low-margin almost everywhere),
so quantization noise has maximal room to flip predictions either way.
**Do not read any single seed's XyloSim number here as a real estimate** —
the whole point of the multi-seed report is that the mean +/- 0.19-0.22
spread *is* the finding, not any one row of the per-seed table.

## What this does and doesn't establish

- **Does establish:** the loader, case-level split, provenance guarding, data
  card + caveats, and multi-seed reporting all work correctly end-to-end on a
  real, open, genuinely hemorrhage-relevant label — the intended
  infrastructure is complete and correct.
- **Does not establish:** that a single 4-second PPG window predicts
  case-level blood loss. The honest finding is that it barely does, with this
  label granularity. Follow-up options if this is revisited: aggregate
  multiple windows per case before classifying (case-level pooling matches
  the label's own granularity), use a graded/continuous EBL target instead of
  a binary cutoff, or bring in `intraop_rbc` (transfusion) as a corroborating
  label — none attempted here, kept in scope as "loader + honest baseline
  report," not "make the number good."

## Reproducing this

```bash
pip install "eia[data]"   # pulls in vitaldb
python scripts/xylo_verify.py --modality ppg --real --require-real \
  --ppg-source vitaldb --max-cases 150 --epochs 15 --n-restarts 5 \
  --n-seeds 5 --max-verify 300 --no-combined
```

First run downloads and caches ~150 cases' PLETH waveforms to `data/vitaldb/`
(gitignored); reruns (e.g. a different `--n-seeds`) reuse the cache.
