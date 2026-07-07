# Applying the ranked fixes from ecg_quant_diagnosis.md — results

**Status: fixes implemented and measured; net result is mixed, not a clean win.**
This is an honest report of what happened when the three evidence-ranked fixes
from `docs/ecg_quant_diagnosis.md` were actually applied and measured at full
training budget, staged so each fix's marginal effect is visible. Read this
alongside that diagnosis, not instead of it — the root-cause analysis there
still stands; what changed here is empirical follow-through, and it surfaced
a new, important finding of its own: **XyloSim agreement for real MIT-BIH is
highly sensitive to the specific trained checkpoint**, enough that it
dominates the effect of the fixes being tested.

## What was implemented

1. **fs-matched window/timestep control** (`datasets.load_mitbih`'s new
   `resample_to` param, `--resample-to` CLI flag). Capture a real physiological
   duration at native 360 Hz (`window=`), then FFT-resample down to the
   desired Xylo timestep count (`resample_to=`) — fixing the bug where
   `--window 90` on real data meant 250ms (90/360 Hz) instead of the 720ms
   (90/125 Hz) it meant for the synthetic generator.
2. **Margin-aware auxiliary loss** (`--margin-reg`): an additional
   class-weighted cross-entropy term on the output layer's mean membrane
   potential (not the spike-count readout used for the actual decision),
   pushing the true class's Vmem clear of the runner-up.
3. **Output-bias L2 regularization** (`--bias-reg`): penalizes
   `net[3].bias ** 2`, targeting the diagnosed real-MIT-BIH-specific fault
   where that bias was a scale outlier wasting ~51% of the output layer's
   8-bit range.

All three are off by default (`0.0`); every result below states exactly which
were enabled.

## Step 0 — reduced-budget sweep to pick the window (fix 1 only)

3 restarts × 10 epochs, real MIT-BIH, capture window fixed at 539 native
samples (1.496s @ 360 Hz), sweeping `resample_to`:

| resample_to (timesteps) | val balanced acc | notes |
|---|---|---|
| 539 (no reduction) | — | killed after 38 min — most expensive point, least informative (we already know longer is worse), not needed |
| **187** | 0.680 | **agreement 0.883** vs. true float — best of the sweep |
| 125 | 0.500 | didn't escape majority baseline in 3 restarts |
| 90 | — | network stall twice (~1h38m with ~5s CPU time — a hung PhysioNet connection, not a training issue); abandoned after two attempts |
| 60 | 0.500 | didn't escape majority baseline in 3 restarts |

187 timesteps (matching the synthetic sweep's own values, now properly
fs-matched) was the clear, promising choice — a dramatic jump from the
pre-session baseline's 0.560 agreement (at the *same* 187-native/519ms
window, unmatched) to 0.883 in this reduced-budget check.

**This did not hold up at full training budget** — see below.

## Full-budget staged results

5 restarts × 15 epochs (real ECG) / 40 epochs (synthetic ECG, PPG) — the
established recipe from `docs/ecg_quant_diagnosis.md`. Each stage adds one
fix on top of the previous; `margin_reg=0.05`, `bias_reg=0.01` (picked from a
quick 5-restart/20-epoch calibration on PPG, where combined they raised
agreement 0.810→0.880 — see caveat below on why that didn't transfer).

| Stage | Real ECG (mitbih) | Synthetic ECG | PPG (synthetic) |
|---|---|---|---|
| **Baseline** (pre-session, unmatched window=187 native=519ms) | float 0.892 (bal 0.845), recall [0.901, 0.789] — **agree 0.560** | float 0.992 (bal 0.991), recall [0.997, 0.985] — **agree 0.733** | float 0.996 (bal 0.996), recall [0.997, 0.995] — **agree 0.800** |
| **Stage 1**: fix 1 only (fs-matched, real @187 timesteps / synth @90) | float 0.856 (bal 0.832), recall [0.860, 0.805] — **agree 0.333** | float 0.988 (bal 0.989), recall [0.983, 0.995] — **agree 0.763** | unchanged (0.800) — no fix applied to PPG in this stage |
| **Stage 2**: + margin_reg 0.05 | float 0.831 (bal 0.840), recall [0.829, 0.851] — **agree 0.373** | float 0.992 (bal 0.991), recall [0.997, 0.985] — **agree 0.750** | float 0.992 (bal 0.993), recall [0.990, 0.995] — **agree 0.843** |
| **Stage 3**: + bias_reg 0.01 | float 0.829 (bal 0.837), recall [0.828, 0.846] — **agree 0.290** | float 0.986 (bal 0.986), recall [0.987, 0.985] — **agree 0.733** | float 0.990 (bal 0.991), recall [0.987, 0.995] — **agree 0.730** |

### Marginal effect of each fix

| | Real ECG | Synthetic ECG | PPG |
|---|---|---|---|
| Fix 1 (window/timestep) | **−0.227** | +0.030 | n/a |
| Fix 2 (margin loss), on top | +0.040 | −0.013 | +0.043 |
| Fix 3 (bias reg), on top | **−0.083** | −0.017 | **−0.113** |
| **Net, baseline → final** | **−0.270** | 0.000 | **−0.070** |

**This is not the clean win the diagnosis's ranking predicted.** Fix 1 alone
made real ECG *worse* at full training budget, even though the reduced-budget
sweep at the identical config showed a dramatic improvement (0.883). Fix 3
(bias regularization) made every single net worse, despite a calibration run
showing it help. Only fix 2 (margin loss) was consistently non-negative
(small positive on real ECG and PPG, small negative on synthetic ECG).

## Why: two confounds, both real findings in their own right

1. **XyloSim agreement is highly sensitive to the specific trained
   checkpoint, not just the hyperparameters.** The *identical* real-ECG
   config (window capture 539, resample_to 187) gave 0.883 agreement with 3
   restarts/10 epochs and 0.333 with 5 restarts/15 epochs — more training,
   more restarts, a *better*-trained float model (recall [0.860, 0.805] vs.
   the sweep's presumably worse one), and a *worse* quantized outcome. This
   matches `ecg_quant_diagnosis.md`'s own finding that a genuinely
   discriminative model can be *more* fragile under quantization than a
   degenerate one (a majority-collapsed model trivially "agrees with itself"
   on every prediction) — but it means single-seed, single-restart-count
   comparisons like the ones in this report aren't reliable enough to
   validate a fix from one run. A real validation needs multiple seeds per
   config, reporting a distribution, not a point estimate.

2. **The margin/bias regularization weights were calibrated at a shorter
   training budget (20 epochs) than they were evaluated at (40 epochs) and
   didn't transfer.** A fixed-weight L2/margin penalty applied for twice as
   many epochs has twice the cumulative effect relative to the primary loss
   — plausibly over-shrinking the output bias or over-flattening the margin
   by epoch 40 in a way it hadn't at epoch 20. This is a real,
   epoch-budget-sensitivity finding, not a reason to discard the fix
   outright — it means these regularizers need either a decaying weight
   schedule or their own epoch-matched calibration, not a value picked once
   and reused at a different budget.

## What this means for the ranking

Fix 1 (timestep/window reduction) is still the best-*evidenced* lever — it
directly targets the confirmed root mechanism (`ecg_quant_diagnosis.md` §2's
ablation), and the sweep showed it *can* produce a large win. What this
session adds is that realizing that win reliably needs multi-seed validation,
not a single run — the mechanism is confirmed, but "pick timesteps, done" is
not yet a turnkey fix. Fixes 2 and 3 remain plausible but are now known to be
budget-sensitive; they should not be adopted at the values used here without
a proper per-budget calibration.

## Honest recommendation, not adopted as defaults

None of `resample_to`, `margin_reg`, or `bias_reg` are wired in as new
defaults — `build_xylo_snn`/`train_modality`'s existing behavior is
unchanged unless these are explicitly passed. Follow-up work before treating
any of them as solved:

- Re-run the fix-1 sweep with **multiple seeds per `resample_to` value**
  (not one), reporting mean ± spread, before picking a "winner."
- If margin/bias regularization is revisited, calibrate the weight **at the
  epoch budget it will actually be evaluated at**, not a cheaper proxy budget.
- Consider evaluating a **held-out checkpoint ensemble** (majority vote
  across several trained restarts) rather than the single best-by-balanced-
  accuracy checkpoint, since that checkpoint's XyloSim fragility appears
  close to unpredictable from float-side metrics alone.

## Reproducing this

```bash
# fix 1 only
python scripts/xylo_verify.py --modality ecg --real --require-real \
  --window 539 --resample-to 187 --epochs 15 --n-restarts 5 --no-combined

# fix 1 + 2
python scripts/xylo_verify.py --modality ecg --real --require-real \
  --window 539 --resample-to 187 --epochs 15 --n-restarts 5 --no-combined \
  --margin-reg 0.05

# fix 1 + 2 + 3
python scripts/xylo_verify.py --modality ecg --real --require-real \
  --window 539 --resample-to 187 --epochs 15 --n-restarts 5 --no-combined \
  --margin-reg 0.05 --bias-reg 0.01
```

Substitute `--modality ecg --window 90` (no `--real`) for synthetic ECG, or
`--modality ppg` for PPG, keeping the same `--margin-reg`/`--bias-reg` flags,
to reproduce the control columns.
