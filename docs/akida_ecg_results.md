# Akida ECG port — measured results (first slice)

Implements the minimal first slice of the Akida re-target: install MetaTF,
verify the toolchain, port ONE modality (ECG), and measure whether the
on-chip fidelity gap shrinks vs. Xylo. Rockpool/XyloSim and every other
modality are untouched — this is a parallel deploy path, not a replacement.

## Outcome, stated up front

**The float→on-chip fidelity gap that XyloSim measures (float 0.845 balanced
acc → XyloSim ~0.56, a ~29-point / ~34% relative drop, see
`docs/ecg_quant_diagnosis.md`) is essentially CLOSED on Akida.** 5-seed real
MIT-BIH measurement: float balanced acc 0.928 ± 0.015 → Akida-sim balanced
acc 0.926 ± 0.039 — statistically indistinguishable — with float-vs-Akida-sim
**agreement 0.981 ± 0.007**, vs. Xylo's ~0.56 agreement on the same task.
This is a genuinely different quantization/execution model (8-bit quantized
Conv2D CNN via `quantizeml`/`cnn2snn`, not an 8-bit LIF SNN via Rockpool), so
the float baselines aren't directly comparable — but the *drop from each
path's own float baseline to its own on-chip simulator* is the apples-to-
apples number, and on that measure Akida is dramatically better here.

## Part 0 — toolchain verification (done first, in a real container)

**Environment blocker, confirmed not a version mismatch:** `akida` (the
actual execution-engine / software-simulator package — everything else in
MetaTF depends on it) has **no macOS wheel in any of its 83 PyPI releases,
ever** — confirmed by enumerating every release's file list, not just the
latest. Wheels exist only for `manylinux_2_28` (Linux aarch64/x86_64) and
`win_amd64`, cp310–cp312. This repo develops on an M-series Mac, so a Docker
container (`Dockerfile.akida`) is required — not a host-venv extra.

**Versions actually installed** (pinned exactly — see "the multi-hour build"
below for why): `akida==2.19.2`, `quantizeml==1.2.4`, `cnn2snn==2.19.2`,
`akida_models==1.14.1`, `tensorflow==2.19.1`, `tf_keras==2.19.0`. (Earlier
project notes assumed TensorFlow 2.15 based on older-quantizeml precedent —
re-verified against the actual PyPI metadata for the versions `akida==2.19.2`
pins today: `cnn2snn` hard-requires `tensorflow~=2.19.0`, not 2.15. Following
this repo's own Part-0 discipline: trust the installed/resolvable metadata
over inherited assumptions.)

**Software backend confirmed to run with no hardware:** `akida.devices()`
returns `[]` in the container (no physical device); `akida.BackendType` has
explicit `Software`/`Hardware`/`Hybrid` members; every converted model's
summary prints its sequence as `(Software)`; `cnn2snn.get_akida_version()`
resolves to `AkidaVersion.v2` (Akida 2.0) by default in this SDK version.

**The load-bearing question — simulator fidelity — checked across three
official BrainChip sources, found NO explicit claim:**
- brainchip.com/metatf-dev-tools/: describes "a processor IP simulator for
  model execution" with no fidelity/accuracy characterization.
- doc.brainchipinc.com (home): "a CPU implementation of the Akida
  Neuromorphic Processor IP" — the closest thing to a fidelity statement
  found anywhere, and it stops short of "bit-exact" or "cycle-accurate."
- doc.brainchipinc.com/user_guide/akida.html: "By default, Akida models are
  implicitly mapped on a software backend: in other words, their inference
  is computed on the host CPU." No accuracy/approximation-level statement.

**This is a materially different situation from XyloSim**, whose Rockpool
documentation explicitly states its traces match hardware exactly. **Say so
loudly, as instructed:** the "Akida-sim agreement" numbers below are
measuring agreement with BrainChip's software model of the processor, not a
BrainChip-confirmed bit-exact stand-in for silicon. This doesn't mean the
number is meaningless — the model IS fully integer-quantized (uint8/int8/
int32 in and out) before `.forward()` runs, which is architecturally
consistent with running the real quantized computation rather than a float
approximation — but it is circumstantial evidence, not a formal guarantee.

**Confirmed supported layers (Akida 2.0, via `dir(akida)` + doc site):**
`InputData`, `InputConv2D`, `Conv2D`, `Conv2DTranspose`, `Dense1D`,
`DepthwiseConv2D`, `DepthwiseConv2DTranspose`, `BufferTempConv`,
`DepthwiseBufferTempConv` (TENN's temporal layers), `Add`, `Concatenate`,
`Dequantizer`. No native Conv1D — see the architecture-fork section below.
Hardware-generation constants present in the installed SDK: `NSoC_v1`/
`NSoC_v2` (Akida 1.0), `FPGA_v2` (Akida 2.0, the current default target),
`AKD1500_v1`/`Pico_FPGA` (**Akida Pico** — the sub-mW part this project
committed to for the v1 biosignal cluster, per CLAUDE.md — present and
selectable in this SDK via `akida.virtual_devices.create_device(...,
hw_version=...)`, but NOT specifically targeted in this first slice; Akida
2.0/FPGA_v2 is what's actually simulated below, Pico-specific
re-characterization is a documented next step, not attempted here).

**Confirmed bit-widths:** `quantizeml.models.QuantizationParams` defaults to
8-bit weights/activations/outputs; BrainChip's own docs describe "8-bit or
4-bit quantized inputs and weights," with 1-bit reserved for edge/last-layer
learning. This slice uses the 8-bit default (`--weight-bits`/
`--activation-bits` on `scripts/akida_verify.py` expose 1/2/4/8 for later
work) — a documented simplification, not a claim that 8-bit is optimal.

### The multi-hour build (a real, diagnosed, and fixed problem — not the toolchain's fault)

The first container build hung on `pip install` for 2+ hours. Root-caused,
not guessed:
1. **Unconstrained version ranges across a huge combined dependency graph**
   (`eia[data,akida]` pulled `akida_models`' full model-zoo extras — opencv,
   librosa, soundata, mtcnn, imgaug, trimesh, tensorflow-datasets — plus
   `eia`'s own torch/snntorch — all with loose version ranges) gave pip's
   backtracking resolver a combinatorially large search space.
2. **`torch`, not `tensorflow`, was the actual source of a multi-GB
   accidental download.** Confirmed via PyPI metadata: `tensorflow`'s
   `nvidia-*-cu12` CUDA dependencies are gated behind an opt-in `and-cuda`
   extra (never installed by a plain `pip install tensorflow`) — so
   `tensorflow` was never the problem. `torch`'s Linux wheels, by contrast,
   unconditionally depend on `nvidia-cudnn-cu13`/`nvidia-cusparselt-cu13`/
   `nvidia-nccl-cu13`/`nvidia-nvshmem-cu13` (gated only by
   `platform_system == "Linux"`, no extra needed) — multiple GB, and
   entirely unnecessary since the Akida ECG path never imports torch (`eia`
   only imports it lazily, inside `train.py`/`rockpool_models.py`).
3. **`colima`'s `--disk` flag is NOT what backs Docker's image storage** —
   that's the separate `--root-disk` flag (default 20GiB, too small for
   TensorFlow's dependency tree). `--disk` provisions an unrelated secondary
   data volume. Recreating with `--root-disk 80` fixed a literal
   "no space left on device" build failure.
4. A ~25GB gitignored `data/` cache (prior sessions' downloaded MIT-BIH/
   CHB-MIT/CinC2016/VitalDB files) was being copied into the build context
   on every build with no `.dockerignore` — fixed by adding one.

Fixed `Dockerfile.akida`: Layer 1 installs the 6 MetaTF packages with
**exact `==` pins** (zero top-level resolver ambiguity); Layer 2 installs
`eia` itself with **`--no-deps`**, adding back only `wfdb`/`numpy`/`scipy`/
`scikit-learn`/`pytest` explicitly — torch is never installed in this
container. Rebuild with warm layer cache: seconds, not hours.

## The architecture fork

Per the task's explicit instruction for this first slice: **quantized 1-D
CNN**, not TENN. Akida 2.0 has no native Conv1D, so the ECG window (187
native-360Hz samples) is reshaped to `(187, 1, 1)` — a single-column
"image," time as the height axis — and processed with ordinary `Conv2D`
layers via `akida_models.layer_blocks.conv_block`/`dense_block` (reused, not
rebuilt — the "reuse, don't fork" principle). TENN (`BufferTempConv`/
`tenn_spatiotemporal`, a genuinely temporal Akida layer family) is confirmed
present in the SDK and is the natural next architecture to try, but is out
of scope for this minimal slice.

**Confirmed Akida v2 conversion constraints** (found empirically —
undocumented anywhere obvious, discovered via `cnn2snn.convert`'s error
messages, and load-bearing for anyone extending this architecture):
- The first (Input) conv layer's kernel AND stride must be **square**
  (`kernel_size[0] == kernel_size[1]`), even though the real width is
  always 1 — `kernel_size=(7, 7)` with `padding="same"` zero-pads the
  phantom width dimension harmlessly.
- **Every** conv layer's max-pooling must also be square, not just the
  input layer's (the first error's wording is misleading on this point).
- `Conv2D -> GlobalAveragePooling2D -> ReLU` is an INVALID layer-ordering
  pattern for conversion; the valid ordering is `Conv2D -> ReLU ->
  GlobalAveragePooling2D` (`conv_block(..., post_relu_gap=True)`).
- `quantizeml`/`cnn2snn` require the **`tf_keras`** package specifically
  (standalone Keras 2), not `tensorflow.keras` (Keras 3 under TF 2.19) —
  mixing the two raises a confusing "Could not interpret optimizer
  identifier" error with no hint about the real cause.

Final architecture (`eia.akida_models.build_akida_model`): `Input(uint8) ->
Rescaling(1/255) -> Conv2D(8, 7x7, stride 2x2) -> ReLU -> Conv2D(16, 5x5) ->
MaxPool(2x2) -> ReLU -> Conv2D(32, 3x3) -> ReLU -> GlobalAvgPool ->
Dense(n_classes)`. ~2.7K float parameters.

## Two-stage training, and a genuine QAT finding

`quantizeml.models.quantize()`'s actual signature takes calibration
`samples` but **no labels** — despite "QAT" framing in BrainChip's own
materials, calling it alone is **post-training quantization + calibration**,
not label-driven fine-tuning. Confirmed, however, that its output IS an
ordinary differentiable `tf_keras.Model` (the `Quantized*` layers are real
Keras layers with straight-through-estimator fake-quantization, not opaque
numpy transforms) — so calling `.compile()`/`.fit()` on it AGAIN with real
labels genuinely back-props through the quantization and is real QAT
fine-tuning. `eia.akida_models.quantize_and_convert`'s `qat_epochs` argument
does this (5 epochs, low LR, class-weighted, in the run below) — confirmed
working via a smoke test before trusting it in the real measurement.

## Part A — measurement (float model first, then Akida-sim, per the EEG/ECG lesson)

Real MIT-BIH (`eia.datasets.load_ecg(prefer_real=True)`, unchanged from the
Xylo path): 22,535 windows, 187 native-360Hz samples each, 92.3%/7.7% class
balance (normal/abnormal AAMI beat). Leakage-safe stratified split
(`case_level.split_data` — no subject ids for MIT-BIH beats, so plain
stratified, matching the Xylo path exactly), class-weighted
`SparseCategoricalCrossentropy`, 3 restarts/seed selecting the best
VAL-balanced-accuracy checkpoint (same discipline that rescues this dataset
from majority collapse on the Xylo path), **5 seeds**, 15 float epochs + 5
QAT epochs per restart, 300 held-out windows verified against the Akida sim
per seed, 8-bit weights/activations.

| | balanced acc | AUROC | per-class recall [normal, abnormal] |
|---|---|---|---|
| **Float** | 0.928 ± 0.015 | 0.970 ± 0.007 | [0.976±0.017, 0.879±0.041] |
| **Akida-sim** | 0.926 ± 0.039 | — | [0.982±0.010, 0.869±0.070] |

**Float vs. Akida-sim agreement: 0.981 ± 0.007.**

Per-seed agreement ranged 0.973–0.993 across all 5 seeds — tight, not one
lucky seed (the exact multi-seed discipline `docs/ecg_quant_fixes_results.md`
found necessary for the Xylo path, where a single seed's agreement swung
from 0.883 to 0.333 between runs on nominally identical settings). No such
instability observed here.

## Verdict: the gap shrinks — dramatically, on this measurement

**Xylo (LIF SNN, 8-bit, Rockpool/XyloSim, docs/ecg_quant_diagnosis.md):**
float 0.845 balanced acc → XyloSim ~0.56 → **agreement ~0.56**, a ~29-point
drop diagnosed as per-timestep integer state rounding compounding
nonlinearly over the window (worse with more timesteps, not weight bits).

**Akida (quantized Conv2D CNN, 8-bit, quantizeml/cnn2snn, this doc):** float
0.928 balanced acc → Akida-sim 0.926 balanced acc → **agreement 0.981 ±
0.007**, statistically flat — the float and on-chip numbers agree within
seed noise.

The two float baselines are NOT the same model (LIF SNN vs. quantized CNN),
so the absolute float numbers aren't a fair architecture comparison — but
the **drop from each path's own float baseline** is exactly the quantity
XyloSim/Akida-sim exist to measure, and on that quantity Akida is
dramatically better here. Plausible mechanism (not confirmed, a hypothesis
for future work): the Xylo diagnosis's root cause was *per-timestep*
integer state accumulating error *over the window* in a temporal LIF
integration; this Akida architecture has no equivalent per-timestep
recurrent integer state to accumulate error in — it's a small feedforward
CNN evaluated once per window, not once per timestep. If that mechanism is
right, it would mean the fidelity-gap problem is somewhat specific to how
Xylo's temporal LIF dynamics interact with 8-bit quantization, not a
universal property of "quantized neuromorphic inference" — worth testing
directly by porting a temporal (TENN) Akida architecture and seeing whether
its gap reopens.

## What this does NOT show

- **Not a claim that Akida silicon will match this measurement** — see the
  Part-0 fidelity-claim finding above; this is agreement with BrainChip's
  software model, not a confirmed-bit-exact simulator.
- **Not an apples-to-apples "Akida beats Xylo" architecture claim** — the
  float baselines differ (different architecture families), and this
  slice's CNN has ~2.7K parameters vs. the Xylo LIF net's smaller footprint;
  a fair comparison would need matched capacity/architecture, not attempted
  here.
- **Not evidence about Akida Pico specifically** — this measures Akida 2.0
  (`FPGA_v2`, the SDK's current default), not the sub-mW Pico core this
  project actually committed to for the always-on biosignal cluster (see
  CLAUDE.md's hardware section). Pico re-characterization is unstarted.
- **Not evidence about TENN, on-chip learning, or co-residence** — all
  explicitly out of scope for this minimal first slice.

## Reproduce

```bash
scripts/akida_docker_run.sh python scripts/akida_verify.py --real --n-seeds 5
```

`--weight-bits`/`--activation-bits` (1/2/4/8) and `--qat-epochs 0` (skip QAT,
calibration only) are available for follow-up sweeps, not run here.
