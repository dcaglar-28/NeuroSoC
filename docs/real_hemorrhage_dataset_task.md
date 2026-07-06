# Task spec — add a real hemorrhage (LBNP / hypovolemia) PPG dataset

**For:** Claude Code (run in the repo, `.venv` active).
**Why:** The current real PPG path (BIDMC) uses SpO₂ desaturation as a *proxy*
for physiological compromise — not a hemorrhage signal. That is why the
Xylo-mapped model agrees ~99.7% with XyloSim but barely beats the base rate on
BIDMC: there is little real hemorrhage structure to learn. Replace the proxy
with a dataset that has a genuine central-hypovolemia label.

---

## Guiding principle (important — applies to the whole task)

**Train and evaluate each dataset separately, and surface each dataset's own
caveats explicitly. Do not try to make one model span datasets, and do not
average or reconcile results across them.**

Concretely:
- One trained model + one report per dataset (`synthetic`, `bidmc`, the new
  `lbnp`, and later ECG sets). Never pool them into a single accuracy number.
- Each dataset run must print a short **data card** up front: source, label
  definition, label rate / class balance, n_samples, sampling rate, and known
  limitations (e.g. "BIDMC: SpO₂<95% proxy, not hemorrhage"; "synthetic:
  separable by construction, expect ~1.0"; "LBNP: true central hypovolemia but
  small subject count / induced, not spontaneous bleeding").
- Each run must **alert on per-dataset red flags** rather than leaving the user
  to infer them, e.g.: class imbalance beyond a threshold; model within X% of
  the majority-class base rate (i.e. "not learning"); subject leakage between
  train/test; NaN/flatline segments dropped. Print these as explicit
  `[warn] <dataset>: ...` lines.

The point: a good result on one dataset and a weak result on another are two
separate, individually-labelled findings — not a contradiction to explain away.

---

## Step 1 — find a suitable dataset (research first)

Look for an induced central-hypovolemia PPG/pleth dataset with a graded or
binary blood-loss-relevant label. Candidates to evaluate (verify license +
availability before use):
- **Lower-Body Negative Pressure (LBNP)** studies — the standard experimental
  model for simulated hemorrhage; some PhysioNet / open datasets include PPG or
  arterial waveforms with LBNP stage labels.
- PhysioNet hypovolemia / hemorrhage / trauma waveform sets.
- Any open Compensatory-Reserve / CRM validation dataset with raw PPG.

Report back: dataset name, access method (ideally `wfdb pn_dir=...` to match the
existing streaming pattern), label definition, size, and license. **If nothing
suitably open exists, say so plainly** and stop before writing a loader — do not
substitute another proxy silently. Surfacing "no clean real hemorrhage PPG
dataset is openly available" is itself a valid, important finding.

## Step 2 — add a loader (mirror the existing pattern)

In `src/eia/datasets.py`, add `load_lbnp_ppg(...)` next to `load_bidmc_ppg`,
returning the same `PpgData` dataclass. Keep the real-loader + synthetic-fallback
convention. Add a `source` value (e.g. `"lbnp"`). Do NOT change the existing
`load_bidmc_ppg` / synthetic paths — they are separate datasets and stay as-is.

Label: map LBNP stage (or measured blood-volume decrement) to the binary
compromise label the pipeline uses, and document the exact threshold in the
docstring and the data card.

## Step 3 — wire into training + Xylo verify, per-dataset

- Add the new dataset as a selectable option wherever `--modality` / dataset
  selection lives (`train.py`, `scripts/xylo_verify.py`). Each dataset is run
  and reported on its own.
- Run, separately and reported separately:
  1. snnTorch `train.py` (research accuracy + energy sweep),
  2. `scripts/xylo_verify.py` (float vs XyloSim agreement),
  on the new LBNP dataset. Emit the per-dataset data card + red-flag warnings
  from the guiding principle.

## Step 4 — update docs

- `README.md`: add the LBNP dataset to the datasets list with its caveat.
- `CLAUDE.md`: update the "Current state" datasets line and move the "next step"
  forward once LBNP is in.

---

## Acceptance criteria

- A real (or clearly-labelled synthetic-fallback) LBNP loader returns `PpgData`
  with a documented hemorrhage-relevant label, following the existing pattern.
- Running training + Xylo verify on LBNP prints: a data card, any per-dataset
  warnings, float accuracy, XyloSim accuracy, and float-vs-XyloSim agreement —
  reported as its own result, not merged with BIDMC/synthetic.
- If no suitable open dataset is found, the deliverable is a written finding
  (candidates checked, why each was unsuitable) — not a silent proxy swap.
- Existing tests still pass; BIDMC and synthetic paths unchanged.

## Gotchas

- **Subject leakage:** split train/test by subject, not by window — LBNP sets
  have few subjects and windows within a subject are highly correlated. Warn if
  the same subject appears in both splits.
- **Label proxy creep:** if the only usable signal ends up being another proxy
  (e.g. heart-rate threshold), label it as such in the data card — don't present
  it as a validated hemorrhage label.
- **Small N / class imbalance:** LBNP datasets are small; report class balance
  and treat near-base-rate accuracy as a "not learning" warning, not a pass.

## References

- Existing loader to mirror: `src/eia/datasets.py` (`load_bidmc_ppg`).
- Xylo verify to extend: `scripts/xylo_verify.py`, `src/eia/rockpool_models.py`.
- Compensatory Reserve / hemorrhage waveform background:
  https://www.mdpi.com/1424-8220/26/8/2513  ·  https://www.ncbi.nlm.nih.gov/pubmed/25423536
- PhysioNet (dataset search + `wfdb` streaming): https://physionet.org/
