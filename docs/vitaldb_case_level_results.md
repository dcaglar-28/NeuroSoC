# VitalDB case-level hemorrhage classification — measured results

Implements `docs/vitaldb_case_level_task.md`, the matched-granularity rescue
of the per-window VitalDB result (`docs/vitaldb_ppg_results.md`: float
balanced accuracy 0.509 ± 0.025, ~chance). `intraop_ebl` is a whole-case
total, so a per-window prediction and a per-case label describe different
moments; this evaluates the SAME trained per-window SNN the honest way —
one prediction per case, pooling its per-window outputs off-chip.

## Outcome, stated up front

**~CHANCE.** Case-level balanced accuracy is 0.516 ± 0.019 across 5 seeds —
indistinguishable from the 0.5 chance line, and tighter (lower variance) than
the per-window result, not higher. AUROC (0.589 ± 0.062) is marginally above
0.5 in mean but its lower band (0.527) is only barely so, and it does not
translate into a usable classifier at the natural 0.5 probability threshold
(that's exactly what balanced accuracy measures). **Per the task's own
decision rule: do not chase this further.** VitalDB's intraoperative,
anesthesia-confounded PPG does not carry a usable blood-loss signal at the
per-window level (already established) **or** at its own label's case level
(established here). The honest next step for a flagship hemorrhage signal is
the gated LBNP dataset or the synthetic time-resolved generator — not further
VitalDB tuning.

## What this is (and isn't)

Approach (A) from the task spec: the trained per-window Xylo SNN is
unchanged — same architecture, same class-weighted loss, same balanced-
accuracy checkpoint selection used everywhere else in this repo. The only new
thing is evaluation: mean-pool the net's per-window positive-class
probability across all of a case's test windows into ONE score
(`eia.case_level.aggregate_by_group`), then classify the case. This pooling
is a **host-side** step — the Xylo core still only ever sees and classifies
individual 4-second windows; nothing about the on-chip deployment story
changes. The clinical claim here is **retrospective, whole-case** ("does this
surgery's PPG overall look like a high-blood-loss case?"), not real-time
field-hemorrhage detection.

## Run configuration

300 qualifying cases (up from 150 in the per-window run — case-level has far
fewer effective training examples, one label per case, so more cases are
needed), same window/resample settings (4.0s @ 500 Hz -> 125 Xylo timesteps,
up to 40 windows/case), `ebl_threshold=500` mL, 5 seeds x 5 restarts x 15
epochs, evaluated on the case-disjoint test split (`GroupShuffleSplit`, no
case ever appears in more than one of train/val/test).

```
python scripts/vitaldb_case_level.py --max-cases 300 --epochs 15 --n-restarts 5 --n-seeds 5
```

Data card (case-level pool, fixed across all 5 seeds — only the train/val/
test partition and net init vary by seed):

```
n_cases = 300, pos frac = 0.107  (32 positive / 268 negative cases at >=500 mL)
window-level: 12000 windows, class balance {0: 0.893, 1: 0.107}
[warn] vitaldb: class imbalance — minority class is 10.7% of samples.
[split] case-grouped: 180 train / 45 val / 75 test cases (every seed, no overlap)
```

## Per-seed results

| Seed | Val bal acc (window, ckpt selection) | Test cases (pos frac) | Case acc | Case bal acc | Case recall [0,1] | AUROC | AUPRC | Base rate |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.718 | 75 (0.120) | 0.733 | 0.513 | [0.803, 0.222] | 0.598 | 0.164 | 0.880 |
| 1 | 0.501 | 75 (0.107) | 0.107 | 0.500 | [0.000, 1.000] | 0.470 | 0.107 | 0.893 |
| 2 | 0.567 | 75 (0.173) | 0.213 | 0.494 | [0.065, 0.923] | 0.650 | 0.363 | 0.827 |
| 3 | 0.579 | 75 (0.107) | 0.480 | 0.544 | [0.463, 0.625] | 0.616 | 0.194 | 0.893 |
| 4 | 0.567 | 75 (0.173) | 0.827 | 0.530 | [0.984, 0.077] | 0.610 | 0.304 | 0.827 |

**Multi-seed summary (mean +/- std over 5 seeds):**

| Metric | Value |
|---|---|
| Case-level accuracy | 0.472 +/- 0.281 |
| Case-level balanced accuracy | **0.516 +/- 0.019** |
| Case-level per-class recall | [0.463 +/- 0.390, 0.569 +/- 0.368] |
| AUROC | 0.589 +/- 0.062 |
| AUPRC | 0.226 +/- 0.094 |
| Majority-case base rate | 0.880 (mean over seeds' test splits) |

## Reading the numbers honestly

- **Balanced accuracy (0.516 ± 0.019) is the primary, decisive number**: it is
  essentially exactly chance, and unusually *tight* across seeds — a
  consistent null result, not a noisy one. This is the metric that matters
  most because it directly answers "is this classifier better than a coin
  flip once class imbalance is accounted for," at the threshold (0.5) the
  model would actually be deployed at.
- **AUROC (0.589 ± 0.062) is marginally above 0.5** (4 of 5 seeds individually
  above it: 0.598, 0.650, 0.616, 0.610; one below at 0.470) and noticeably
  *tighter* than the wild per-window agreement spread — aggregation did
  reduce noise. But AUROC measures rank-ordering across all possible
  thresholds, not performance at the one threshold (0.5) a real classifier
  uses; a mean of 0.589 with a ±0.062 band is a **weak, marginal signal at
  best**, not a validated one, and it does not survive contact with the
  actual binary decision (balanced accuracy at 0.5 is flat at chance). Report
  it as "some rank-correlation, inconclusive," not as "the model works."
- **Case-level raw accuracy swings wildly (0.472 ± 0.281)**: seed 1 predicts
  almost every case positive (recall [0.000, 1.000], acc = 0.107 = 1 − base
  rate); seed 4 predicts almost every case negative (recall [0.984, 0.077],
  acc = 0.827 ≈ base rate). The classifier isn't finding a stable decision
  boundary — it's flipping between near-total majority-collapse and
  near-total minority-collapse depending on the random split/init, which is
  itself evidence of an absent or very weak underlying signal rather than a
  training instability to "fix": if there were a real signal, class-weighted
  training would converge toward *using* it, not toward two different
  degenerate collapses across seeds.
- Aggregation (Approach A) did what it was supposed to do methodologically —
  it fixed the label/window granularity mismatch and gave a properly
  case-disjoint, matched-granularity estimate — but the answer it returns is
  that there isn't much signal to recover this way either.

## What this does and doesn't establish

- **Does establish:** the case-level aggregation pipeline (mean-pooling,
  case-disjoint split, multi-seed balanced accuracy / recall / AUROC / AUPRC
  reporting) is correct and complete, per the task deliverable.
- **Does establish (the actual finding):** VitalDB's intraoperative PPG does
  not carry a blood-loss signal usable at either granularity tried — not
  per-window (docs/vitaldb_ppg_results.md), and not per-case (here). This is
  now a settled question for this dataset, not an open one to keep
  re-tuning.
- **Does not establish, and is explicitly out of scope per the task doc:**
  Approach (B) (graded/continuous EBL target) was NOT attempted — the task
  spec gates it on (A) showing signal first, and (A) did not. Any future
  hemorrhage-signal work on this project should look at the gated LBNP
  dataset or the synthetic time-resolved generator instead of further
  VitalDB tuning.

## Reproducing this

```bash
pip install "eia[data,xylo]"
python scripts/vitaldb_case_level.py --max-cases 300 --epochs 15 --n-restarts 5 --n-seeds 5
```

First run downloads/caches any of the 300 sampled cases not already cached
from the earlier per-window run (`data/vitaldb/`, gitignored); reruns reuse
the cache.
