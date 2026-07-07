# Task spec — VitalDB case-level hemorrhage classification (rescue)

**For:** Claude Code (repo, `.venv` active, VitalDB already wired in from
`docs/vitaldb_ppg_hemorrhage_task.md`, commit 760613c).

**Why:** The per-window VitalDB result was ~chance (float balanced acc
0.509 ± 0.025) and that is now understood, not a bug: `intraop_ebl` is a
**whole-case total**, so a 4-second window from hour one of a surgery that
eventually lost 600 mL is labeled "hemorrhage" even though nothing was bleeding
then. The label and the window describe different moments. The **statistically
correct** use of a per-case label is a **per-case prediction**, not a per-window
one. This task reframes VitalDB that way — the honest, matched-granularity use
of the data already in hand.

The clinical claim changes and MUST be documented: this is *retrospective,
whole-case* ("does this surgery's PPG overall look like a high-blood-loss
case?"), NOT the real-time field-hemorrhage detection the device ultimately
needs. It is a data-in-hand sanity check on whether PPG carries any case-level
blood-loss signal at all — not a flagship result.

---

## Approach (pick the simplest that respects the label granularity)

The unit of prediction is the **case**, not the window. Two options; do (A)
first, only try (B) if (A) shows signal:

**(A) Aggregate per-case features → one prediction per case.**
- For each case, run its PPG windows through the trained SNN and pool the
  per-window outputs (e.g. mean output-membrane potential, or mean predicted
  probability) into ONE case-level score, then classify the case.
- Simplest and most Xylo-honest: the on-chip net still processes short windows;
  aggregation happens off-chip over the case. Document that the aggregation step
  is NOT on the Xylo core (it's a host-side pooling of the chip's per-window
  outputs) — this matters for the deployment story.
- Split BY CASE (already have `PpgData.groups` + GroupShuffleSplit). The label is
  now naturally case-level, so there is no window/label mismatch.

**(B) Graded / continuous EBL target (only if A learns something).**
- Instead of binary >=500 mL, predict graded severity (e.g. none / moderate /
  large, or regress log-EBL) at the case level. This can extract more signal
  than a single threshold if the binary works at all.

## Measurement (same discipline as everything else)

- Report **per-case** balanced accuracy + per-class recall (positives are rare,
  ~14% of qualifying cases at >=500 mL), plus AUROC/AUPRC since it's imbalanced
  binary — over **>=5 seeds, mean ± spread**. Single-seed is noise (the ECG
  lesson).
- Compare against the trivial baseline (majority-case rate) explicitly, and warn
  if the model sits within the seed band of it (i.e. "not learning").
- Use MORE cases than the 150 in the per-window run if runtime allows — case-level
  = far fewer training examples (one per case), so it needs more cases to have
  enough cases to learn from. Report n_cases and class balance in the data card.

## Honest interpretation up front

State the expected outcomes plainly in the results doc:
- If per-case learns (balanced acc clearly above base rate across seeds): that's
  a real, if modest and retrospective, finding — PPG waveform morphology carries
  case-level blood-loss signal. Worth reporting; still NOT real-time hemorrhage.
- If per-case is ALSO ~chance: that is itself the conclusion — VitalDB's
  intraoperative, anesthesia-confounded PPG does not carry a usable blood-loss
  signal even at the granularity its label supports. Then the honest path for the
  flagship signal is the gated LBNP data (request) or the synthetic time-resolved
  generator. Do not keep tuning VitalDB past this point.

## Deliverable

- Case-level evaluation wired into `scripts/xylo_verify.py` (e.g. a
  `--ppg-agg case` mode) or a small `scripts/vitaldb_case_level.py`, reusing the
  existing loader, case groups, class-weighted loss, multi-seed loop.
- `docs/vitaldb_case_level_results.md`: n_cases, class balance, per-case balanced
  acc / recall / AUROC (mean ± spread), vs base rate, and the up-front
  interpretation above stating which of the two outcomes occurred.
- `CLAUDE.md` updated with the one-line conclusion.
- All tests green; add an offline unit test for the case-aggregation logic
  (tiny synthetic per-window scores + group ids → one score per group).

## Do NOT

- Do not pool VitalDB with BIDMC/synthetic/ECG results.
- Do not present a positive per-case result as real-time hemorrhage detection.
- Do not chase VitalDB further if per-case is also ~chance — report and stop.
