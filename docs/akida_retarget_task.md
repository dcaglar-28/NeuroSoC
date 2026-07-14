# Task spec (SHELF / FUTURE — not started) — re-target the validated pipeline onto BrainChip MetaTF / Akida

**Status:** Deferred. Do NOT start until (a) the production-silicon phase is
actually reached and (b) MetaTF tooling + an Akida target (AKD1000 SoC, Akida 2
FPGA, or Akida Pico eval) is available. This document is on the shelf so the work
is scoped when that day comes; nothing here changes the current repo.

**Why this exists:** The committed production architecture is the single-vendor
**BrainChip Akida** family (see CLAUDE.md hardware section / brief Section 4).
The current repo validates the neuromorphic-deployment *method* on SynSense
Rockpool/XyloSim because that is a mature bit-exact simulator available today.
This task ports the *validated method* — not a rebuild of the science — onto
Akida's toolchain. Keep the Xylo path intact and working throughout; Akida is a
**parallel deploy sibling**, not a replacement of the validation vehicle.

**Guiding principle (same as every prior task here):** the deploy METHOD
(train off-device → quantize → run on a bit-accurate simulator → verify against
the float model, multi-seed, per-class, leakage-safe) is chip-agnostic and
already proven. Only the toolchain and the neuron/quantization specifics change.

---

## Part 0 — VERIFY the toolchain before trusting any of the below (do this first)

My (spec author's) knowledge of MetaTF is not hands-on. Treat every API/behaviour
claim in this doc as a hypothesis to confirm against the *installed* tooling,
exactly like the VitalDB Part-0 field-name check that caught a real bug. Confirm
and write down:

- **Packages & versions:** MetaTF is TensorFlow-based, four PyPI packages —
  `akida`, `quantizeml`, `cnn2snn`, `akida_models`. Install, print versions,
  confirm import. (Docs: doc.brainchipinc.com; examples: github.com/Brainchip-Inc/akida_examples.)
- **Simulator fidelity — the load-bearing question.** MetaTF ships a "processor
  IP simulator" (`akida.Model` run in software / a virtual device). CONFIRM
  whether it is **bit/cycle-accurate to silicon** the way XyloSim is — this is
  the entire reason we can verify pre-hardware. If it is only approximate,
  say so loudly; it changes what "verified" means here. Do not assume.
- **Target/version:** which Akida generation is in scope — Akida 1.0 (AKD1000),
  Akida 2.0, or **Akida Pico** (the sub-mW always-on part we committed to for the
  v1 biosignal cluster)? Confirm what model families / layers each supports;
  Akida Pico may support a narrower set than full Akida. The v1 target is Pico.
- **Weight/activation bit-widths:** confirm supported precisions (Akida is
  typically 1/2/4-bit, vs Xylo's 8-bit) — this is the biggest reason our
  Xylo-measured fidelity numbers will NOT carry over and must be re-measured.
- **On-chip learning constraints:** confirm the edge-learning layer rules
  (last FullyConnected layer, binary weights, binary inputs, one-shot/incremental)
  — needed for the personalization demo below.
- **Supported layers / temporal support:** confirm whether `tenn_spatiotemporal`
  (Temporal Event-based Neural Networks) is available and appropriate for 1-D
  time-series, vs. a plain quantized 1-D CNN. See the architecture fork below.

## The architecture fork (decide explicitly, document the choice)

Our current model is a Rockpool `LIFBitshiftTorch` temporal SNN trained with
surrogate gradients on a 2-channel (per EEG channel) delta spike raster. Akida's
mainstream flow is **quantization-aware CNN/ViT** (spatial), not LIF-over-time.
Three ways to bridge — evaluate in this order:

1. **TENNs (`tenn_spatiotemporal`)** — Akida's temporal event-based models.
   Closest in spirit to our temporal 1-D approach (convolution over the time
   axis). Preferred if it supports low-channel 1-D biosignal windows. Try first.
2. **Quantized 1-D CNN** — reframe each window as a 1-D conv over time, QAT with
   `quantizeml`, convert with `cnn2snn`. The most Akida-native path; loses the
   explicit LIF dynamics but is the safest mapping. Fallback.
3. **NIR bridge** — export the existing net via NIR (`to_nir`, Rockpool is
   torch-only NIR) and check whether Akida/MetaTF can ingest it. Likely
   immature; investigate but don't block on it.

Document which path was chosen and why, with the Part-0 evidence behind it.

## Build (mirror the existing dual research/deploy pattern)

- Add `src/eia/akida_models.py` as a deploy sibling of `rockpool_models.py`
  (which stays untouched). Functions analogous to the Xylo ones:
  `build_akida_model(...)`, `quantize_and_convert(...)` (quantizeml → cnn2snn),
  `to_akida_sim(...)`, `verify_against_sim(...)` (float vs. Akida-sim agreement).
- **Reuse, do not fork:** `encoding.py`, `datasets.py`, `report.py`, the split
  logic (`case_level`, `eeg_patient_specific_split`), and the multi-seed /
  balanced-accuracy / per-class / provenance discipline are shared and unchanged.
- Add an `akida_verify.py` script paralleling `xylo_verify.py`: per-modality
  data card → float accuracy → Akida-sim accuracy → agreement → mapped resource
  footprint, over ≥5 seeds, mean ± spread. Same metrics per modality as today
  (ECG: balanced acc + per-class recall; EEG: AUROC/AUPRC/sensitivity/FA-per-hour).

## Re-characterize (the findings do NOT carry over — measure fresh)

- The float→sim fidelity gap, the timestep/quantization root-cause diagnosis, the
  co-residence cost, and the window-length effects are all **Xylo-toolchain-
  measured**. Re-measure every one on Akida and write a fresh results doc — do
  not copy Xylo numbers forward. Expect different behaviour: Akida's lower-bit
  weights but different (event-based CNN / TENN) state model may move the gap in
  either direction.
- **Co-residence / multimodal:** Xylo's pain (single shared 8-bit scale, ≤2 input
  synapses/neuron, no early fusion) may not apply on Akida, which is built for
  multimodal fusion. Re-evaluate whether the multi-chip-fallback question even
  survives on Akida — this could be a genuine simplification worth documenting.

## On-chip learning demo (a NEW capability Xylo could not do — make it concrete)

Akida's last-layer edge learning is a core reason we chose it and is directly
pitch-relevant. Add a small, honest demonstration: take a trained ECG (or EEG)
model, freeze the backbone, and show **incremental last-layer adaptation** to a
held-out patient / a shifted sensor condition improving that patient's metric
vs. the frozen model — the "personalize in the field" story. Keep the claim
precise (last-layer, binary-weight, few-shot), report before/after per-patient,
and note it is on-device *adaptation*, not full retraining.

## Deliverables

- `src/eia/akida_models.py`, `scripts/akida_verify.py`, both mirroring the Xylo
  siblings; Xylo path unchanged and still green.
- `docs/akida_retarget_results.md`: chosen architecture path + why, Part-0
  toolchain/fidelity findings, per-modality float-vs-Akida-sim numbers
  (multi-seed), the on-chip-learning demo result, and a co-residence re-eval.
- CLAUDE.md: update the hardware section from "future roadmap step" to "in
  progress / done" with the measured Akida fidelity, and correct any Xylo-only
  framing that the port changes.
- Offline unit tests for the new pure helpers; keep `pytest -q` green.
- Commit + push, verify the push landed (rev-parse + cat-file), report the hash.

## Do NOT

- Do NOT delete or break the Rockpool/XyloSim path — it stays as the reference
  and as valid prior evidence; Akida is added alongside.
- Do NOT trust this doc's API/behaviour claims over the installed MetaTF — Part 0
  is authoritative.
- Do NOT copy Xylo fidelity numbers into the Akida results — re-measure.
- Do NOT start this until the hardware phase and tooling access are real.
