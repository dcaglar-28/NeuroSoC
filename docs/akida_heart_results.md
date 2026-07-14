# Akida heart-sound port — measured results

Extends the Akida ECG slice (`docs/akida_ecg_results.md`) to heart sounds —
the second real-data modality to move onto the committed production target.
Reuses the same `eia.akida_models`/`scripts/akida_verify.py` scaffolding
(generalized additively, not forked: `to_akida_input`/`quantize_and_convert`/
`verify_against_sim` are unchanged and fully shared; only a new
`build_akida_heart_model` was added). `rockpool_models.py`, the Xylo path,
and the ECG Akida path (`build_akida_model`) are untouched.

## Outcome, stated up front

**The float→on-chip fidelity gap closes for heart sounds too, and by a
similar margin to ECG.** Xylo's heart-sound measurement (filterbank
front-end, `docs/heart_sounds_results.md`): float balanced acc 0.593 ± 0.041
→ XyloSim 0.512 ± 0.012, agreement 0.779 ± 0.064. This Akida measurement (5
seeds, real CinC 2016): float balanced acc 0.865 ± 0.020 → Akida-sim
0.867 ± 0.020 — statistically indistinguishable — **agreement 0.954 ± 0.027**.
As with ECG, the float baselines aren't the same architecture (quantized
Conv2D CNN vs. LIF SNN) so they aren't directly comparable, but the
float→on-chip *drop* is again dramatically smaller on Akida.

## Architecture: the filterbank map is a natural Conv2D fit

Heart sounds' Xylo-validated front-end (`docs/heart_sounds_results.md`'s
escalation from raw-waveform, which measured flat chance for a diagnosed
reason — the raw path is NOT offered here, see "Do NOT" below) extracts line
length / relative band power (`datasets.PCG_BANDS`) / spectral entropy per
sub-window: `datasets.PCG_FEATURE_NAMES` (4 features) × `n_subwindows=24` →
a `(4, 24)` map, bands × time. Unlike ECG's `(187,)` single-channel waveform
(which needed a `(window, 1, 1)` single-column reshape — one real spatial
axis, one phantom), **this is already genuinely 2-D**: `build_akida_heart_model`
feeds it to Akida as a `(4, 24, 1)` image with BOTH spatial axes real,
without any reshape trick.

**No new undocumented Akida v2 constraints were hit building this** — the
ones found while getting ECG to convert (square kernel/stride on the input
conv layer, square pooling on every conv layer, `Conv2D -> ReLU ->
GlobalAveragePooling2D` ordering via `post_relu_gap=True`) applied
unchanged and the architecture converted on the first attempt:
`Input(4,24,1) -> Rescaling -> Conv2D(8, 3x3, stride 1x1) -> ReLU ->
Conv2D(16, 3x3) -> MaxPool(2x2) -> ReLU -> Conv2D(32, 3x3) -> ReLU ->
GlobalAvgPool -> Dense(n_classes)`. Only one square max-pool is used (in
block2) — `n_bands=4` is small enough that a second pool would collapse the
band axis to zero; `GlobalAveragePooling2D` in block3 absorbs whatever
`(H, W)` remains regardless of exact size, so this isn't a hard limit, just
this architecture's choice. Confirmed footprint: `(4, 24, 1) -> (2,)`, 5
mapped Akida layers (`InputConv2D`/`Conv2D`/`Conv2D`/`Dense1D`/`Dequantizer`).

## Part 0 — unchanged, shared with ECG (not re-verified here)

All Part-0 findings in `docs/akida_ecg_results.md` — the container
requirement (no macOS `akida` wheel, ever), the exact pinned versions
(`akida==2.19.2`, `quantizeml==1.2.4`, `cnn2snn==2.19.2`,
`tensorflow==2.19.1`), the confirmed software-backend-with-no-hardware
behavior, and **the load-bearing caveat that BrainChip does NOT publish an
explicit bit/cycle-accurate claim for the software simulator** (checked 3
official sources, found only "a CPU implementation of the Akida
Neuromorphic Processor IP") — apply identically here and are not re-run.
**Restated because it matters for this doc's own numbers, not just ECG's:**
the "Akida-sim agreement" below is agreement with BrainChip's software
model, not a confirmed-bit-exact-to-silicon number.

## Part A — measurement (float model first, then Akida-sim)

Real CinC 2016 (`eia.datasets.load_heart(heart_frontend="features")`,
unchanged from the Xylo path — this script does not expose a `raw` option
for heart, see "Do NOT" below): 300 recordings pulled (the loader's own
seeded default), 1909 windows, ~74.9%/25.1% class balance (normal/abnormal,
this pull; close to the full-dataset ~20.5% abnormal). Leakage-safe
stratified split (no subject ids for CinC 2016, matching Xylo/heart's own
Part-0 finding), class-weighted `SparseCategoricalCrossentropy`, z-score
normalization fit on the TRAIN split only
(`signal_features.normalize_features_train_only`, applied post-split — same
leak guard as the Xylo path), 3 restarts/seed selecting the best
VAL-balanced-accuracy checkpoint, **5 seeds**, 15 float epochs + 5 QAT
epochs per restart, 300 held-out windows verified against the Akida sim per
seed, 8-bit weights/activations.

| | balanced acc | AUROC | per-class recall [normal, abnormal] |
|---|---|---|---|
| **Float** | 0.865 ± 0.020 | 0.945 ± 0.011 | [0.877±0.037, 0.853±0.067] |
| **Akida-sim** | 0.867 ± 0.020 | — | [0.918±0.018, 0.817±0.025] |

**Float vs. Akida-sim agreement: 0.954 ± 0.027.**

Per-seed agreement ranged 0.913–0.990 across all 5 seeds — every seed
comfortably above Xylo's 0.779 mean, none close to Xylo's worst-case
instability (the Xylo quantization work found single-seed agreement could
swing from 0.883 to 0.333 on nominally identical settings, per
`docs/ecg_quant_fixes_results.md`; nothing resembling that spread appears
here).

## Verdict: the gap closes for heart sounds too

**Xylo (LIF SNN, 8-bit, filterbank front-end, `docs/heart_sounds_results.md`):**
float 0.593 ± 0.041 → XyloSim 0.512 ± 0.012 → **agreement 0.779 ± 0.064**.

**Akida (quantized Conv2D CNN, 8-bit, same filterbank features, this doc):**
float 0.865 ± 0.020 → Akida-sim 0.867 ± 0.020 → **agreement 0.954 ± 0.027**
— the on-chip number is statistically indistinguishable from float, same
shape of result as ECG's.

This is now **two for two**: both modalities measured on Akida show the
float→on-chip drop shrinking dramatically relative to the equivalent Xylo
measurement. It strengthens (does not newly establish — one modality could
have been a fluke either direction) the ECG doc's hypothesis: these are
small **feedforward** CNNs evaluated once per window, with no per-timestep
recurrent integer state to accumulate rounding error over — unlike Xylo's
LIF dynamics, whose diagnosed root cause was exactly that accumulation.
Heart's Akida architecture also has a real, non-degenerate 2-D receptive
field (bands × time) where ECG's was a forced 1-D-as-2-D reshape, and the
result is essentially the same shape of fidelity outcome either way — some
evidence the effect is about **CNN vs. LIF-SNN**, not about how naturally
2-D the input happens to be.

## What this does NOT show (same caveats as ECG, restated for this modality)

- **Not a claim that Akida silicon will match this measurement** — see the
  Part-0 fidelity-claim caveat above.
- **Not an apples-to-apples "Akida beats Xylo" architecture claim** — the
  float baselines differ in both architecture (CNN vs. LIF SNN) AND
  parameter count; a fair comparison would need matched capacity, not
  attempted here. This is doubly true for heart sounds, where Akida's float
  balanced acc (0.865) is also well above Xylo's float balanced acc (0.593)
  on the identical filterbank features — the CHIP+ARCHITECTURE pairing
  changed, not the chip alone, so this is not evidence that "Akida's chip is
  better than Xylo's chip" in isolation.
- **Not evidence about Akida Pico specifically** — this measures Akida 2.0
  (`FPGA_v2`), not the sub-mW Pico core actually committed to for the
  always-on biosignal cluster (CLAUDE.md's hardware section).
- **Not evidence about TENN, on-chip learning, co-residence, or any other
  modality** — out of scope here as in the ECG slice.

## Do NOT (respected in this implementation)

- `rockpool_models.py`, the Xylo path, and `build_akida_model`/ECG's
  measured numbers are unmodified — confirmed via a synthetic ECG smoke
  test after this work, same footprint (`187,1,1) -> (2,)`) as before.
- The raw-waveform front-end is not offered for heart here —
  `scripts/akida_verify.py`'s heart loader always requests
  `heart_frontend="features"`, with no CLI flag to override it, and
  `train_and_verify` raises a clear, explicit error (not a silent fallback
  or a fabricated number) if a `HeartData` object without the features
  front-end is ever passed in (this only bites the synthetic-fallback edge
  case — see the docstring in `scripts/akida_verify.py` for that limitation).
- No fabricated numbers: every figure above came from a real 5-seed run
  against real CinC 2016 data, logged in full.

## Reproduce

```bash
scripts/akida_docker_run.sh python scripts/akida_verify.py --modality heart --real --n-seeds 5
```
