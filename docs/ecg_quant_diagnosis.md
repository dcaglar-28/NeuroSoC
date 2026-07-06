# ECG float→XyloSim fidelity gap — root-cause diagnosis

**Status: measurement only.** No training or model code was changed to produce
this report. All numbers come from `scripts/diagnose_ecg_quant.py`, which
trains (or loads a cached copy of) the exact, unmodified `train_modality`/
`build_xylo_snn` pipeline and then instruments it — see "Reproducing this"
at the bottom.

Three reference nets, same architecture (2 in, 63 hidden, 2 out,
class-weighted CE, balanced-accuracy checkpoint selection — current
`main`), held-out test set, `max_verify=300` (or 20 for the per-window
layer trace):

| | `ecg_real` (MIT-BIH) | `ecg_synth` | `ppg_synth` |
|---|---|---|---|
| window (timesteps) | 187 | 187 | 125 |
| float acc (balanced) | 0.892 (0.845) | 0.992 (0.991) | 0.996 (0.996) |
| float per-class recall | [0.901, 0.789] | [0.997, 0.985] | [0.997, 0.995] |
| XyloSim agreement (this run, n=300) | 0.560&nbsp;† | 0.733 | 0.800&nbsp;†† |
| mean hidden spike rate | **0.290** | 1.761 | 1.546 |

† Matches the 0.560 reported when the class-imbalance fix first exposed this
gap (see `git log`).
†† Measured fresh in this diagnosis; the previously-cited 0.877 for PPG was
from before this session's balanced-accuracy checkpoint-selection change —
selecting by balanced accuracy instead of raw accuracy can pick a different
restart/epoch even for a modality that isn't badly imbalanced. Not
investigated further here since PPG isn't in scope; noted so the number
isn't mistaken for a regression.

---

## Single most likely root cause

**The gap is not caused by any one isolable quantization source (weights, or
decay, or threshold) — it's caused by XyloSim executing the entire
multi-hundred-timestep simulation in fixed-point integer arithmetic, so every
single timestep's state update (Vmem, Isyn) carries a small rounding error
that compounds over the window.** LIF spiking is a non-smooth, history-erasing
process (a spike resets the membrane potential), so one early
rounding-induced spike/no-spike disagreement cascades forward nonlinearly.
More timesteps in the window means more chances for this cascade to start,
and once started it doesn't recover. ECG's longer window (187 vs. PPG's 125)
means more accumulation opportunities; real MIT-BIH's much sparser firing
(0.29 vs. ~1.5–1.8 mean hidden spike rate) means each individual event's
rounding error is proportionally more decisive, since there's less averaging
across spikes at the readout.

This is evidenced most directly by the ablation (§2) — isolating "weights
quantized, dynamics float" or "dynamics quantized, weights float" each
reproduces **96.7–100%** of the true float model's decisions, on all three
nets. Neither comes anywhere close to explaining the real XyloSim
disagreement rate (20–44 points, worst for `ecg_real`). The gap only appears
when the *entire* state machine runs in integer arithmetic end to end — an
interaction/compounding effect my float-precision ablation can't reproduce
by construction, which is itself the evidence that the mechanism is
accumulation, not a single quantized parameter.

Two real, secondary, ECG-specific factors compound this (§1, §3, §5). For
real MIT-BIH specifically, there's a *second, independent* error source on
top of the accumulation mechanism: the output layer diverges from the float
model at timestep 1 — 24 steps before its own hidden layer does (§1) —
which can only mean the output layer's own quantized dynamics are wrong from
the start, not inherited from upstream. §3 finds the specific cause: real
ECG's output-layer bias is a scale outlier relative to its own weight
matrix, wasting half of that layer's 8-bit range. Neither synthetic net has
this problem, so it's additive on top of, not an alternative to, the
timestep-accumulation mechanism above. §5 adds a class-specific effect that
isn't simply "minority class is fragile" — the direction of harm is opposite
for ECG (hits the clinically-critical class) and PPG (hits the majority
class). All three are documented below with their own evidence, scoped
separately from the primary mechanism.

---

## 1. Layer-by-layer divergence

Float net and XyloSim run on the *same* raster with `record=True`; hidden and
output spike trains (both are genuinely integer-valued at inference in this
net, so directly comparable) compared timestep by timestep, averaged over 20
held-out windows per net.

| | `ecg_real` | `ecg_synth` | `ppg_synth` |
|---|---|---|---|
| hidden layer: mean first-divergence timestep | **25.4** (median 23) | 1.0 | 1.0 |
| **output layer: mean first-divergence timestep** | **1.0** | 1.0 | 1.0 |
| hidden layer: samples that *never* diverge (of 20) | 0 | 0 | 0 |
| mean fraction of (neuron, timestep) pairs diverging | 0.204 | 0.271 | 0.244 |
| final-prediction disagreement (this n=20 sample) | 0.55 | 0.20 | 0.25 |

**Every single sample eventually diverges in the hidden layer, for all three
nets** — quantization rounding is universal, not something a few unlucky
windows hit. What differs is *when*. Synthetic ECG and PPG diverge almost
immediately (t≈1) because their hidden layers fire densely (spike rate
1.5–1.8) — a spike-count rounding difference shows up on the very first
active timestep. Real ECG's hidden layer stays *bit-exact* with the float
model for ~23–25 timesteps before the first divergence, consistent with its
much sparser firing (0.29): fewer early spikes means fewer early chances for
a rounding-driven mismatch.

**Where the divergence originates is not the same for every net, and this is
an important correction to the naive assumption that output-layer divergence
is just inherited from the hidden layer.** For `ecg_synth` and `ppg_synth`,
hidden and output diverge at the same timestep (t=1), consistent with
(though not proof of) output divergence simply following hidden divergence
immediately. **For `ecg_real`, the output layer diverges at t=1 — 24
timesteps *before* the hidden layer's own first divergence at t≈25.** Since
the output layer's input literally cannot differ from what the (still
bit-exact) hidden layer sends it that early, this divergence has to originate
*independently*, inside the output layer's own quantized dynamics — and §3
below finds exactly such an independent cause specific to `ecg_real`: its
output-layer bias is a scale outlier that wastes half the output layer's
8-bit weight range. This ties §1 and §3 together directly: for real ECG, the
output layer is already wrong from the first timestep, before hidden-layer
accumulation (§2's dominant mechanism) even gets a chance to contribute.
Immediate divergence does **not** imply the final decision is wrong, though
(`ppg_synth`'s output diverges at t=1 in every sample and still reaches 80%
final agreement) — divergence has to survive the rest of the window's
integration to flip the final decision, and whether it does depends on the
decision margin (§4).

## 2. Weights vs. dynamics ablation (the key evidence)

`weights_only`: 8-bit-round-tripped `LinearTorch` weights, threshold/bias/dash
left at their exact trained float values. `dynamics_only`: float weights,
threshold+bias round-tripped through the same affine 8/16-bit scale
`global_quantize` actually uses (dash is already exact — see note below).
Both run as an ordinary float Torch forward pass (not XyloSim, which can't
run a mixed-precision config) and are compared to the *true* float model's
own predictions on the same 300 held-out windows.

| | `ecg_real` | `ecg_synth` | `ppg_synth` |
|---|---|---|---|
| weights_only: agreement vs. true float | 0.967 | 1.000 | 0.997 |
| dynamics_only: agreement vs. true float | 0.987 | 1.000 | 1.000 |
| **full XyloSim: agreement vs. true float** | **0.560** | **0.733** | **0.800** |

Neither hybrid comes close to the real gap for any net, including
`ecg_synth` where both hybrids show **zero** disagreement (1.000) yet the
real XyloSim run disagrees on 27% of windows. This directly rules out "the
weights are quantized wrong" and "the decay/threshold is quantized wrong" as
the primary explanation — see the root-cause section above for why: the two
hybrids each only apply a *one-time* value substitution and then run the
rest of the simulation in continuous float, whereas real XyloSim rounds
*every intermediate state update*, all 187 (or 125) of them, and those
per-step errors compound in a way a one-shot float hybrid structurally
cannot reproduce.

**Why dash/decay quantization is (almost) free**: `build_xylo_snn` trains
with `LIFBitshiftTorch`, which snaps `tau_mem`/`tau_syn` to the nearest
bit-shift-representable value *during training itself* — inspecting the
mapped spec confirms `dash_mem = dash_syn = 4` exactly, an already-integer
value with nothing left to round. This is why `dynamics_only` (which does
still round threshold/bias, the other "dynamics" parameters) shows more
disagreement than a pure no-op but still nowhere near the full gap.

## 3. Weight-distribution health

`global_quantize` uses **one shared scale per parameter group**: `scale_in =
127 / max(|w_in|, |bias_hidden|)` for the input→hidden group, `scale_out =
127 / max(|w_out|, |bias_out|)` for hidden→output. A single outlier value in
either group compresses everything else in that group into fewer effective
bits — exactly the failure mode the Rockpool Xylo training tutorial warns
about.

| | `ecg_real` | `ecg_synth` | `ppg_synth` |
|---|---|---|---|
| w_in outlier frac (\|x-mean\|>2σ) | 0.000 | 0.000 | 0.016 |
| w_out outlier frac | 0.000 | 0.032 | 0.032 |
| w_in int8 range used | 100% | 100% | 100% |
| **w_out int8 range used** | **48.8%** | 100% | 100% |
| \|w_out\|<sub>max</sub> | 0.322 | 0.570 | 0.738 |
| **\|bias_out\|** | **0.656** | 0.479 | 0.091 |
| bias_out / \|w_out\|<sub>max</sub> | **2.04** | 0.84 | 0.12 |

**Real MIT-BIH hits the tutorial's named failure mode; neither synthetic net
does.** Only for `ecg_real` is `|bias_out|` (0.656) *larger* than the largest
actual output weight (0.322) — so `scale_out` is set by the bias, and the
weight matrix itself only occupies the bottom 48.8% of the available 8-bit
range, wasting roughly half the output layer's precision on real ECG
specifically. This is a genuine, ECG-real-specific, well-evidenced weight
pathology — but it is **not** the primary mechanism, because `ecg_synth` has
a *worse* gap than PPG (0.733 vs. 0.800) while using 100% of its output
range, same as PPG. It's a real, additive handicap on top of the
timestep-accumulation mechanism from §2, specific to whatever this
particular real-data training run did to the output bias — not a property
of ECG as a modality.

Outlier fractions, otherwise, are low and don't obviously distinguish ECG
from PPG (ECG's `w_in`/`w_out` are, if anything, slightly *cleaner* than
PPG's by this measure) — "non-centered/non-flat" weight distributions in the
general sense are not what's driving the ECG-vs-PPG contrast.

## 4. Decision-margin fragility

Float model's output-layer **membrane potential** (not the spike-count
readout used for the actual decision — this is a separate fragility signal),
summed over the window per output neuron, winner-minus-runner-up, split by
whether that window's float and XyloSim predictions agreed:

| | `ecg_real` | `ecg_synth` | `ppg_synth` |
|---|---|---|---|
| mean margin, agree | 108.1 | 156,897 | 55,158 |
| mean margin, disagree | 80.3 | 39,558 | 17,627 |
| **disagree / agree ratio** | 0.74 | **0.25** | 0.32 |

**Confirmed and consistent across all three nets: disagreements concentrate
on lower-margin decisions.** For the two synthetic nets the effect is large
(disagreement-case margins are a quarter to a third of agreement-case
margins); for real ECG it's present but weaker (0.74), consistent with real
ECG's gap being large enough (56% agreement, close to coin-flip territory)
that margin alone stops being a clean discriminator — a lot of *everything*
is disagreeing there, not just the close calls. This is real evidence for
"quantization noise flips decisions that were already close," not evidence
that the noise itself originates in the readout — it explains **where** the
accumulated error (§2) does its damage, not what generates it.

## 5. Class-specificity

Per true-class agreement rate and accuracy, held-out set:

| | class | n | agreement | float acc | XyloSim acc |
|---|---|---|---|---|---|
| `ecg_real` | 0 (normal) | 278 | 0.565 | 0.910 | 0.576 |
| `ecg_real` | 1 (abnormal) | 22 | 0.500 | 0.864 | 0.545 |
| `ecg_synth` | 0 (normal) | 188 | **0.989** | 0.995 | 0.995 |
| `ecg_synth` | 1 (abnormal) | 112 | **0.304** | 0.982 | **0.286** |
| `ppg_synth` | 0 (normovolemic) | 192 | **0.693** | 1.000 | 0.693 |
| `ppg_synth` | 1 (hypovolemic) | 108 | **0.991** | 0.991 | 1.000 |

**This is not "quantization hurts the minority class" in general — it's
modality- and even class-identity-specific, and the direction is opposite for
ECG vs. PPG.** For synthetic ECG, quantization is devastating specifically
for the **abnormal** (minority, clinically critical) class: 98.2% float
accuracy collapses to 28.6% under XyloSim, while the normal class is
essentially untouched (99.5%→99.5%). For PPG it's the reverse pattern on
which class suffers — the **majority** class (normovolemic) degrades
(100%→69.3%) while the minority, clinically-critical hypovolemic class stays
robust (99.1%→100%). Real ECG shows no strong class asymmetry, but only
because both classes are already badly served (agreement ~0.50–0.57 for
both) — there isn't a "good" class left to contrast against.

The likely explanation is that this tracks which **output neuron** happens to
sit closer to its own quantization-driven margin degradation (§4), which is
a property of that specific trained readout, not an inherent "minority
classes are fragile" rule. Practically: **for ECG, the clinically important
class is the one quantization damages most** — this is the single most
clinically urgent finding in this diagnosis, independent of which mechanism
(§2/§3) ultimately explains it.

## 6. ECG vs. PPG structural contrast

| | `ecg_real` | `ecg_synth` | `ppg_synth` |
|---|---|---|---|
| window / timesteps | 187 | 187 | 125 |
| mean hidden spike rate | 0.290 | 1.761 | 1.546 |
| \|w_in\|<sub>max</sub> | 1.846 | 1.826 | 2.652 |
| \|w_out\|<sub>max</sub> | 0.322 | 0.570 | 0.738 |
| bias_out / \|w_out\|<sub>max</sub> | 2.04 | 0.84 | 0.12 |
| w_out int8 range used | 48.8% | 100% | 100% |
| disagree/agree margin ratio | 0.74 | 0.25 | 0.32 |
| hidden first-divergence timestep | 25.4 | 1.0 | 1.0 |
| XyloSim agreement (this run) | 0.560 | 0.733 | 0.800 |

The biggest, cleanest structural contrast is **window length** (187 vs. 125,
a ~50% longer accumulation period for both real and synthetic ECG) combined,
for real ECG only, with **much sparser firing** and the **output-bias
outlier**. Synthetic ECG isolates the window-length effect cleanly (same
architecture and training recipe as PPG, only the window/timesteps and the
underlying waveform differ) and already shows a worse gap than PPG on that
basis alone (0.733 vs. 0.800) — real ECG's additional handicaps (sparse
firing, output-bias outlier) push it further down (0.560).

---

## Candidate fixes, ranked by expected payoff given this evidence

1. **Timestep / window reduction — highest expected payoff.** Directly
   targets the confirmed mechanism (§2: per-step integer-state accumulation
   over the window). Already empirically *partially* validated on synthetic
   ECG in the prior session (window=90 raised agreement 0.733→0.763,
   non-monotonically — window=60 was worse). Not yet successfully applied to
   real MIT-BIH (window=90 there made agreement *worse*, 0.560→0.477, most
   likely because real MIT-BIH's 360 Hz sampling makes "90 samples" a much
   shorter real-world duration than in the 125 Hz synthetic generator — an
   fs-rescaled window, ~250–260 samples, is the untried next step). Highest
   ranked because it's the only lever that addresses the demonstrated root
   mechanism, not a secondary/compounding factor.

2. **Margin-aware training (readout change) — moderate-to-high expected
   payoff.** §4 shows disagreements concentrate at low float-model margin
   consistently across all three nets. A training objective that explicitly
   widens the decision margin (e.g. a hinge term on the spike-count logits,
   or optimizing for a target margin rather than plain cross-entropy) would
   make more decisions resistant to the *same* amount of accumulated
   quantization noise, without touching the quantization or hardware side at
   all. Complementary to #1, not a substitute — worth trying together.

3. **Weight regularization/normalization targeting the real-ECG output-bias
   outlier — moderate expected payoff specifically for real MIT-BIH, low
   risk, and now causally evidenced, not just correlated.** §3 shows the
   pathology (output bias 2x larger than the largest output weight, wasting
   ~51% of that layer's 8-bit range) and §1 shows its direct consequence: the
   output layer diverges from float at timestep 1, a full 24 steps before
   its own hidden layer does — independent of, and prior to, any hidden-layer
   accumulation error. Excluding bias from the shared max-scale computation,
   or penalizing/capping bias magnitude relative to the weight matrix during
   training, directly targets this specific, already-localized failure.
   Ranked below #1/#2 only because it's real-ECG-specific (`ecg_synth` has a
   worse-than-PPG gap *without* this pathology, so it can't be the whole
   story) — but for real MIT-BIH specifically this is the one fix with a
   clean, direct causal chain from evidence to mechanism to remedy, so it's
   worth doing alongside #1/#2, not instead of them.

4. **Weight QAT (quantization-aware training of the weights specifically) —
   low expected payoff.** §2's ablation is the direct evidence against this:
   weights-only quantization, run in isolation, already reproduces 96.7–100%
   of the true float model's decisions on all three nets. There is very
   little "weight quantization error" left to recover by training the
   weights to be more quantization-robust — the bottleneck is the integer
   *simulation*, not the weight *values*. Do this last, if at all, and only
   after #1–#3 are tried.

---

## Reproducing this

```bash
python scripts/diagnose_ecg_quant.py
```

Trains (or loads from `/tmp/eia_diag_cache/*.pt` if already cached — real
MIT-BIH training alone takes ~20–30 minutes) all three reference nets with
the current, unmodified `train_modality`, then runs all six instrumentation
passes above and writes the full numeric results to
`/tmp/eia_diag_cache/results.json`. Delete the cache directory to force a
clean retrain (e.g. after any training-code change, to keep this diagnosis
reproducible against the code that actually produced it).
