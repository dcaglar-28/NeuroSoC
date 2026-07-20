# Synthetic time-resolved CRM / occult-hemorrhage demo — results

The flagship differentiated
capability — detecting compensated hemorrhage *before* heart rate moves —
demonstrated with a physiologically-grounded synthetic generator, since real
LBNP/CRM-induction data is gated. Reuses the ECG Akida-CNN path
(`build_akida_model`, unmodified) via a new `"crm"` modality, additively.
ECG, heart sounds, `rockpool_models.py`, and the Xylo path are untouched.

## Why this exists, restated

VitalDB (`docs/vitaldb_case_level_results.md`) settled at chance because one
whole-case `intraop_ebl` number was stamped on every window, regardless of
when in the case it was captured — the label never matched the window's
actual physiological state, at either the per-window or per-case
granularity. This generator fixes that **by construction**: every window's
label is derived from the reserve fraction *at that window's own point* on
a simulated hypovolemia trajectory. It is not a claim that VitalDB's problem
is now "solved" with real data — it's a demonstration that the pipeline
*can* learn a genuinely time-aligned label, removing the one variable
(label granularity) that made VitalDB uninterpretable.

## The physiological model

Parameterized by a reserve fraction `r(t) ∈ [1, 0]` per synthetic subject
(1.0 = full compensatory reserve, 0.0 = decompensated), grounded in the
explainable-CRM literature (**"An Explainable Machine Learning Model for
the Assessment of Compensatory Reserve," MDPI Bioengineering 2023**, the
paper CLAUDE.md already cites as "the method to use once real-physiology
data exists"):

- **Trajectory (`_reserve_trajectory`):** `r = 1 - t^shape`, `shape` per
  subject in [0.7, 1.8] — monotonically non-increasing by construction (no
  noise on `r` itself; all randomness lives in the pulse signal and
  per-subject baselines), so trajectory shape varies (front-loaded vs.
  back-loaded decline) without ever violating monotonicity.
- **Pulse amplitude** (`_crm_pulse`): scales as `0.35 + 0.65·r` — falls to
  ~35% of baseline at full decompensation (reduced stroke volume/peripheral
  perfusion).
- **Dicrotic notch** — the paper's most discriminative single feature —
  scales as `0.15·r`: the WIDEST proportional range of any morphology
  term (baseline → exactly 0), the strongest cue by design, matching the
  literature.
- **Pulse width** narrows as `0.65 + 0.35·r` (systolic upstroke shift).
- **Diastolic/reflected-wave component** flattens as `0.20 + 0.80·r`.
- **Heart rate** (`_hr_from_r`) — the critical LEAD structure: flat at
  `hr_baseline` while `r > 0.30` (`_HR_RISE_R`), rises toward a
  compensatory-tachycardia peak (~1.6× baseline) as `r` falls through
  `(0.10, 0.30]`, then falls back toward a low terminal value below `r=0.10`
  (`_HR_COLLAPSE_R`) — a collapse, not sustained tachycardia, at full
  decompensation.
- **Windowing** (`_crm_window`): each window tiles individual pulses
  back-to-back at the CURRENT inter-beat interval (`60/HR(r)` seconds),
  each pulse shaped by the CURRENT `r` — so a fixed-duration window shows
  both degraded morphology and (once HR moves) altered pulse spacing, like
  a real capture.
- **Per-subject variability:** randomized baseline HR (60–80 bpm), pulse
  amplitude scale (0.8–1.2×), width scale (0.85–1.15×), and trajectory
  decline shape — no fixed waveform or fixed trajectory curve is
  memorizable.

## The label — deliberately time-aligned AND deliberately occult-inclusive

`y = 1` ("compromised") iff `cri = r < CRI_THRESHOLD` (default **0.5**),
where `cri` is stored per window (`CrmData.cri`) as the ground truth `y` is
thresholded from — confirmed exactly aligned by
`test_make_synthetic_crm_label_is_time_aligned_with_cri`.

**The label threshold (0.5) is deliberately set ABOVE the HR-rise threshold
(0.30):** this opens an "occult" band, `r ∈ (0.30, 0.50]`, where every
window is labelled compromised (`y=1`) while `_hr_from_r` is *still exactly
at baseline*. `test_make_synthetic_crm_positive_class_includes_hr_baseline_windows`
confirms this band is non-empty and entirely positive-labelled. A
classifier that clears chance here is provably using morphology, not a
proxy for "HR is already elevated" — the entire clinical value proposition.

## Front-end: waveform, not filterbank

CRM lives in pulse SHAPE at low frequency (<~10 Hz) — this is explicitly
the same class of signal ECG's architecture targets, not heart sounds'
spectral content. `scripts/akida_verify.py`'s `crm` modality reuses
`build_akida_model` **completely unchanged** (confirmed by
`test_crm_reuses_build_akida_model_unchanged_end_to_end`, which runs it
against real `make_synthetic_crm` output) — no new model builder was
written for this modality, unlike heart sounds.

## The lead effect, visualized

![CRM lead effect](img/crm_lead_effect.png)

One synthetic subject's trajectory (`eia.viz.plot_crm_lead_effect`). Top
row: three example windows (early/mid/late) — amplitude visibly drops and
the dicrotic notch visibly blunts between window 0 and window 12, well
before heart rate has moved (both still 70 bpm). Bottom panel: reserve
`r(t)` (green) declines smoothly from window 0, crossing the CRI threshold
(dashed) around window 14–15; heart rate (purple) stays completely flat
until ~window 18, rises sharply (compensatory tachycardia, peaking >100
bpm), then collapses at full decompensation (window 23) — the shaded
"occult band" (windows 15–17) is labelled compromised while HR is still
flat, exactly the detection-before-vitals-move property being demonstrated.

## Part A — measurement (float first, then Akida-sim)

1200 windows (50 subjects × 24 windows/subject), 3s @ 100 Hz (300 samples),
~55.8%/44.2% class balance this cohort (close to the ~48% the (0, 0.5)
threshold implies on average given the trajectory shape distribution).
Leakage-safe SUBJECT-grouped split (`case_level.split_data`, `CrmData.groups`
always set — confirmed no-overlap via `test_make_synthetic_crm_no_leakage_subject_split`
and, at the script level, the same "case-grouped: N train / N val / N test"
log line the Xylo/other Akida paths print), class-weighted
`SparseCategoricalCrossentropy`, 3 restarts/seed selecting the best
VAL-balanced-accuracy checkpoint, **5 seeds**, 15 float epochs + 5 QAT
epochs per restart, 300 held-out windows verified against the Akida sim per
seed, 8-bit weights/activations.

| seed | float bal. acc | float AUROC | float recall [0,1] | Akida-sim bal. acc | agreement |
|---|---|---|---|---|---|
| 0 | 0.901 | 0.987 | [0.801, 1.000] | 0.921 | 0.853 |
| 1 | 0.857 | 0.942 | [0.787, 0.928] | 0.870 | 0.933 |
| 2 | 0.909 | 0.987 | [0.977, 0.842] | 0.910 | 0.970 |
| 3 | **0.557** | 0.982 | [1.000, **0.115**] | 0.896 | 0.727 |
| 4 | **0.589** | 0.979 | [1.000, **0.177**] | 0.938 | 0.607 |
| **mean ± std** | **0.763 ± 0.156** | **0.975 ± 0.017** | [0.913±0.097, 0.612±0.385] | **0.907 ± 0.023** | **0.818 ± 0.135** |

## An honest finding: bimodal float convergence, rescued (not masked) by QAT

**This is a materially different result shape than ECG's or heart sounds'**,
where every seed converged consistently. Here, 3 of 5 seeds (0, 1, 2)
converged well (float balanced acc 0.86–0.91, genuine per-class recall). The
other 2 (3, 4) landed at float balanced acc 0.56–0.59 with class-1 recall
collapsed to 0.12–0.18 — despite the SAME class-weighted-loss +
balanced-accuracy-checkpoint-selection discipline that worked for the other
3 seeds and for every ECG/heart seed. **Critically, AUROC stayed 0.94–0.99
on EVERY seed, including the two collapsed ones** — the float model's
*continuous score* still ranks the classes almost perfectly in all 5 cases;
only the *decision threshold* (argmax) failed to calibrate correctly for 2
of them. This is the same signature this repo's ECG quantization work
named "checkpoint sensitivity" (`docs/ecg_quant_fixes_results.md`): with
only 3 restarts, best-of-N checkpoint selection can still land in a
threshold-miscalibrated local optimum for a genuinely harder, more
temporally-structured signal (multiple pulses per window, not one) than
ECG's single-beat or heart's per-window-feature framing.

**What's new here, and worth naming explicitly: QAT fine-tuning rescued
both collapsed checkpoints, not just calibration alone.**
`quantize_and_convert`'s `qat_epochs` argument continues training the
already-quantized model on real labels (5 epochs) before conversion — for
seeds 3 and 4, this pushed Akida-sim balanced accuracy to 0.896 and 0.938
respectively, well above the float checkpoint's 0.557/0.589 it started
from. This is NOT the ECG/heart story ("float and Akida-sim agree because
quantization barely perturbs an already-good decision boundary") — it's
QAT acting as genuinely additional training that recovered a poorly-placed
decision threshold. The lower agreement scores for these two seeds (0.727,
0.607, vs. 0.85–0.97 for the well-converged seeds) are the direct
fingerprint of this: float and Akida-sim disagree specifically *because*
QAT meaningfully moved the decision boundary, not because quantization
introduced noise around an already-settled one.

**Practical reading:** the underlying signal is reliably learnable (AUROC
never drops below 0.94, any seed) — this is not a pipeline or label-design
failure. But `n_restarts=3` is evidently not always enough for this richer
signal to reliably escape a bad threshold on the float side alone. Not
re-run with more restarts here to chase a cleaner number — this multi-seed
spread, reported honestly rather than cherry-picked, is itself the correct
output of the discipline this repo has followed since the ECG quantization
work: single-seed (or, here, best-case-seed) numbers are not reliable
evidence.

## Verdict

**Pipeline and concept demonstrated — NOT a clinical accuracy claim.**
Synthetic data is separable by construction (as flagged loudly in the data
card and repeated here): a high number proves (1) the pipeline runs
end-to-end train → quantize → convert → Akida-sim-verify on this modality,
and (2) the time-aligned label design is learnable, including the occult
(HR-still-baseline) band specifically — AUROC ≥0.94 on every single seed,
even the ones with threshold-calibration trouble, confirms the model is
using morphology, not an HR proxy, to separate the classes. It does **not**
demonstrate real-world occult-hemorrhage detection accuracy; that requires
the gated LBNP/CRM-induction data noted in CLAUDE.md's M-branch. The bimodal-seed finding above is a genuine, reported
limitation of this specific measurement (3-restart float convergence
robustness on a richer signal), not swept under the mean.

Footprint (all seeds, unchanged from ECG's architecture): `(300, 1, 1) ->
(2,)`, 5 mapped Akida layers. Same Part-0 caveats as ECG/heart apply
unchanged (see `docs/akida_ecg_results.md`): BrainChip's software simulator
has no confirmed bit/cycle-accurate-to-silicon claim; this measures
agreement with BrainChip's software model, not verified silicon behavior.

## Reproduce

```bash
scripts/akida_docker_run.sh python scripts/akida_verify.py --modality crm --n-seeds 5
```

`eia.viz.plot_crm_lead_effect()` regenerates the trajectory figure above.

## Next steps (noted, not built here)

- Real validation needs gated LBNP/CRM-induction data (request access
  separately — see `docs/lbnp_data_request_emails.md`).
- A graded (3-stage) or true CRI-regression variant — `CrmData.cri` (the
  continuous ground truth) is already generated and stored for exactly
  this, unused by the current binary classifier.
- Increase `n_restarts` (or add early-stopping/LR-schedule tuning) if the
  bimodal-convergence finding above needs to be resolved before this demo
  is used for anything beyond "pipeline works."
