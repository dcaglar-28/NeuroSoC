# EIA — project context for Claude Code

## What this is
EIA is a concept for a **neuromorphic field diagnostic & stabilization device**:
a rugged, offline, low-power handheld that runs many point-of-care diagnostic
tests at once on a neuromorphic processor, so a first responder in a disaster
zone can diagnose life threats and guide stabilization without power or
connectivity. It is **not** a triage/patient-prioritization tool. ICU is a
secondary market.

Organizing framework is **MARCH** (Massive hemorrhage, Airway, Respiration,
Circulation, Hypothermia/Head) — the sequence responders already use. Preventable
deaths targeted: hemorrhage (~91% of battlefield preventable deaths), airway
(~7.9%), tension pneumothorax (~1.1%).

The full project brief is `EIA_Project_Brief.docx` in this folder.

## Architecture concept (hub-and-spoke)
- **On-core (native, v1):** ECG, PPG (hemorrhage via Compensatory Reserve),
  heart/lung sounds, respiration, capnography, EEG, thermal, vision. Pure compute,
  runs as spiking neural networks on the neuromorphic chip.
- **Docked probe:** ultrasound/eFAST (transducer + beamforming is physics the
  chip can't absorb — it only interprets).
- **Cartridge module:** labs (lactate, hemoglobin, blood gas) — reagent chemistry.
- Moat = offline milliwatt-scale multimodal integration + the MARCH stabilization
  loop. Buy silicon, don't build it.

## Hardware targets (two tiers)
- **Tier 1 — SynSense Xylo (primary, now):** ultra-low-power (~30-500 uW)
  digital SNN core purpose-built for low-dimensional biosignals (ECG, PPG, EMG,
  audio). Runs the always-on on-core waveform tests. This is the v1 target.
  Constraints: LIF neurons only, ~1000 hidden neurons, few input channels, 8-bit
  weights, spike-raster input. SDK = **Rockpool**; deploy/verify via the
  **XyloSim bit-accurate simulator** (no chip needed to validate).
- **Tier 2 — BrainChip Akida (intended extension, later):** larger event-based
  processor that adds what Xylo cannot — 2-D convolution / vision (ultrasound,
  thermal, retinal), deeper multimodal fusion networks, and on-chip learning.
  Wakes intermittently for the docked-probe imaging + full MARCH fusion.
- A real device could carry both: Xylo as always-on sentinel, Akida spun up for
  imaging/fusion. Phase-0 code (1-D LIF MLPs on ECG/PPG) maps onto Xylo today.

## Xylo pipeline (Rockpool)
- `src/eia/models.py` (snnTorch) = fast local research. `src/eia/rockpool_models.py`
  = hardware-target sibling that re-expresses the same LIF net in Rockpool and
  maps -> quantizes -> runs on XyloSim.
- The neuron models match (LIF: membrane + synaptic current, subtractive reset,
  forward-Euler dt; Xylo approximates exp decay with bit-shift "dash" params).
- **Input framing (important):** Xylo allows <=16 input channels, so the delta
  ON/OFF encoding is fed as **2 channels over `window` timesteps** (samples =
  time axis), NOT flattened to 2*window features. `to_input_raster()` converts
  the encoder's (2, window) into Xylo's (window, 2). NOTE: the snnTorch model in
  `models.py` still flattens to 2*window for research speed — aligning it to the
  temporal (window, 2) framing is a good future cleanup.
- Flow (matches deploy-to-Xylo guide): resolve support module via
  `find_xylo_hdks()` else `syns61201` -> `net.as_graph()` -> `mapper(graph,
  weight_dtype='float')` -> `global_quantize(**spec)` ->
  `config_from_specification(**spec)` -> `XyloSim.from_config(config)` -> compare
  float vs bit-precise sim (`verify_against_sim`). XyloSim traces match silicon
  exactly, so it's a real pre-hardware check. Install: `pip install "eia[xylo]"`.
- Xylo-Audio 2 = `syns61201`; other variants (Audio 3, IMU) are different
  submodules with different limits — confirm against the installed Rockpool.

## Xylo training/deploy specifics (from torch-training-spiking-for-xylo)
- **`Nhidden=63` is not arbitrary:** recurrent->recurrent fanout is capped at 63
  (input->rec and rec->output fanout are unlimited). 63 = max fully-recurrent
  hidden layer. That's why `build_xylo_snn` defaults to 63.
- **`dt` must be 1-10 ms** (forward Euler). Bit-shift decay: `dash =
  round(log2(tau/dt))`; our tau=20ms,dt=1ms -> dash=4 (matches mapper output).
- **Aliases / skip connections only go toward increasing neuron id** (relevant if
  we add residual blocks for fusion).
- **Do NOT use `spike_generation_fn=PeriodicExponential`** here. The tutorial
  recommends it (tuned for its audio task: many input channels, short windows),
  but A/B'd on our net (2 in / 2 out, long single-biosignal windows) it REGRESSED
  both metrics (synthetic PPG: float 0.992->0.772, XyloSim agree 0.877->0.680).
  Kept the default `StepPWL` surrogate on the evidence.
- **Quantize gotcha:** the tutorial deletes `mapped_graph` and `dt` from the spec
  before `global_quantize(**spec)`. Our code passes the full dict (works on 3.1.0);
  if a version errors on unexpected kwargs, strip those two keys first.
- **`global_quantize` "unlucky" cases** (confirmed): weights not centered on 0, a
  few very strong weights, or a non-flat distribution -> quantized net diverges.
  This is exactly the co-residence weight-magnitude risk; normalize per sub-net or
  use `channel_quantize` when co-mapping modalities.
- Training loop that works: `Adam(net.parameters().astorch(), lr=1e-2)`, MSE (or
  CE for us), `net.reset_state()` per sample, `loss.backward()`, `step()`.

## Float->XyloSim fidelity gap — ROOT CAUSE (diagnosed, see docs/ecg_quant_diagnosis.md)
- The gap is **NOT** weight precision and **NOT** decay approximation. Ablation
  proved it: weight-only and dynamics-only float hybrids each reproduce 96.7-100%
  of float decisions — neither explains the real 20-44pt XyloSim disagreement. And
  dash/decay is quantization-EXACT (LIFBitshiftTorch snaps tau during training).
- **Real cause:** per-timestep integer STATE rounding compounding nonlinearly over
  the window — LIF's spike-reset makes errors cascade, not average out. So on-chip
  fidelity scales with NUMBER OF TIMESTEPS, not weight bits. (Corrects the older
  "bit-shift decay drift" wording in build_xylo_snn comments — same "longer window
  worse", wrong mechanism.)
- **Design principle for all on-core modalities:** keep the window/timestep count
  short. Longer windows are structurally worse (ECG 187 > PPG 125). When bringing
  a new modality, resample toward the fewest timesteps that still hold the signal.
- **Sparsity vs. fidelity tension (important):** sparser firing correlates with a
  WORSE gap (real MIT-BIH 0.29 hidden spike rate vs synthetic ~1.5-1.8). We push
  low spike rate for ENERGY; the diagnosis says very-sparse + long-window is bad
  for on-chip fidelity. Treat spike-rate as a real trade-off axis, not free.
  (Strongly supported across 3 nets but still correlational — confirm if leaned on.)
- **Real MIT-BIH has a second, independent fault:** its output-layer bias is a
  scale outlier (2x the largest output weight), wasting ~51% of that layer's 8-bit
  range; its output layer diverges from float at timestep 1 (before its hidden
  layer does). Fix = output-bias regularization, ECG-specific.
- **Fixes ranked by evidence:** (1) timestep/window reduction, fs-matched for real
  data; (2) margin-aware training (disagreements cluster at low decision margin);
  (3) output-bias regularization for real ECG; (4) weight QAT LAST — ablation shows
  little headroom there.
- Class-harm direction is modality-specific: synthetic ECG quantization devastates
  the minority abnormal-beat class (0.982->0.286); PPG degrades the majority class
  instead. Always report per-class recall, not just accuracy.

## Fixes were applied and measured — result is MIXED, not solved (see docs/ecg_quant_fixes_results.md)
- Implemented all 3: `datasets.load_mitbih(resample_to=...)` (fs-matched
  timestep control, `--resample-to` CLI flag), `--margin-reg` (Vmem-margin
  auxiliary loss), `--bias-reg` (L2 on output bias). All off by default.
- Reduced-budget sweep (3 restarts x 10 epochs) found real MIT-BIH
  resample_to=187 gave agreement 0.560->0.883 — a dramatic, promising win.
  **This did NOT reproduce at full budget** (5 restarts x 15 epochs): same
  exact config gave 0.333, WORSE than the original 0.560 baseline.
- **New key finding: XyloSim agreement for real MIT-BIH is highly sensitive
  to the specific trained checkpoint**, not just the hyperparameters/config —
  a better-trained float model (higher balanced accuracy, genuinely
  discriminative) can be MORE fragile under quantization than a worse one.
  Single-seed/single-restart-count comparisons are NOT reliable evidence for
  or against a fix; needs multi-seed validation (mean +/- spread) before
  trusting any single number, including the ones in this file.
- Full staged (fix1 -> fix1+2 -> fix1+2+3) net effect at seed=0, full budget:
  real ECG 0.560->0.290 (WORSE), synthetic ECG 0.733->0.733 (flat), PPG
  0.800->0.730 (WORSE). Margin-reg was the only consistently non-negative
  fix; bias-reg hurt every net despite a shorter-budget calibration showing
  it help (0.05/0.01 weights were calibrated at 20 epochs, evaluated at 40 —
  the mismatch likely over-applies the fixed-weight penalty at the longer
  budget).
- **Do not treat any of resample_to/margin_reg/bias_reg as solved or as new
  defaults.** Fix 1 (timestep reduction) remains the best-evidenced lever
  (it's the only one targeting the confirmed root mechanism from the
  diagnosis), but "pick timesteps, done" is not yet turnkey — validate with
  multiple seeds before relying on it. Next step if revisited: multi-seed
  sweep, epoch-matched regularizer calibration, and/or an ensemble-of-
  checkpoints readout instead of single-best-by-balanced-accuracy.

## Training
- **Gradient descent** via surrogate-gradient BPTT (spikes are non-differentiable
  at threshold, so the backward pass uses a smooth surrogate). Adam optimizer.
  Training is off-device; only fixed-weight inference runs on Xylo.
- **Torch backend everywhere** — snnTorch (research) and Rockpool's `*Torch`
  modules (deploy). Rockpool also has a JAX backend (`RateJax`/`LIFJax`, `jax_loss`,
  `jax.jit`); we standardize on Torch for one backend across both paths + MPS on
  the Mac. JAX is the escape hatch only if Torch sweeps get slow or we train big
  recurrent-fusion dynamics.
- Rockpool Torch-training rules (from the torch-training-spiking tutorial):
  - Wrap params for the optimizer: `Adam(net.parameters().astorch(), lr=...)`.
  - `net.reset_state()` (detach) between samples for correct BPTT.
  - `LIFTorch(..., learning_window=...)` is the surrogate-gradient width knob.
  - Classify via output-neuron membrane potential; `ExpSynTorch` low-pass readout
    (used for spiking *regression*) is NOT Xylo-output-mappable.
- **Time-constant floor: tau >= 10*dt** (Rockpool numerical-stability rule). This
  is the NaN trap in `build_xylo_snn`: a trainable tau driven toward 0 -> divide
  by ~0 -> NaN weights. We avoid it by freezing tau as `Constant` (0.02 = 20*dt,
  safe). Alternative if we want tau to adapt per modality: keep it trainable but
  bound it with `make_bounds`/`bounds_cost` (a loss penalty), per the
  sgd_recurrent_net tutorial.

## One-chip co-residence (confirmed from the mapper output)
- **Per-neuron config is independent.** The deploy-to-Xylo `spec` shows
  `threshold=[1,1,1,1,10,10,10,10]` in one config — thresholds, `dash_mem`,
  `dash_syn`, `bias` are per-neuron arrays. Co-resident per-modality sub-nets keep
  their own dynamics; they are NOT forced into one shared config.
- **The mapper already does block placement.** `weights_rec` is one shared hidden
  matrix with connectivity confined to blocks — so ECG on neurons 0..K, PPG on
  K..2K within the same `W_rec` is the native mechanism. XyloSim is always
  `(16,1000,8)` (whole core), so co-resident sub-nets simulate fine.
- **What IS shared:** the global `dt` (resample every modality to it), the single
  `global_quantize` scale, and `weight_shift_in/rec/out`. To co-map: build the
  COMBINED net, then map+quantize as ONE unit (a single global scale must cover
  all weights); watch weight-magnitude mismatch between sub-nets (use
  `channel_quantize` or normalize per sub-net if it hurts).
- **Fusion limit:** max 2 input synapses/hidden neuron, 1/output neuron — so a
  neuron can't read >2 input channels. Early (input-layer) cross-modality fusion
  is impossible on Xylo; fuse hidden->hidden through `W_rec`, or do heavy
  multimodal fusion on the Akida tier. Reinforces: Xylo = per-modality + light
  fusion, Akida = heavy fusion.

## Toolchain interoperability (NIR) — future option
- **NIR** (Neuromorphic Intermediate Representation) is the clean bridge for the
  `models.py` (snnTorch) vs `rockpool_models.py` (Rockpool) duplication: export a
  trained net with `to_nir()` (serializes as CubaLIF + Linear nodes) and import
  with `from_nir()`, moving one definition between toolchains — and later toward
  Akida/Loihi. Install `pip install 'rockpool[nir]'`.
- Caveats: **beta** (API may change); Rockpool NIR is **torch-only**; `from_nir`
  returns a `nirtorch.GraphExecutor` (a `torch.nn.Module`, not native Rockpool),
  but it still supports `as_graph()`, so it stays Xylo-deployable.

## Why not DynapSE-2 (evaluated, deferred)
DynapSE-2 is SynSense's analog/mixed-signal chip. Not our primary, because for a
field medical device the digital Xylo path is safer:
- **Analog mismatch** must be trained around (`percent_mismatch`,
  `mismatch_generator`, 5-30% deviation) and there is **no bit-exact simulator** —
  you verify statistically (firing-rate ratio), not exact sim==silicon like
  XyloSim. Loses the clean pre-hardware guarantee.
- 4-bit weights (vs Xylo 8-bit) via autoencoder quantization; richer but
  harder-to-control DPI/AdExpIF dynamics (AMPA/NMDA/GABA/SHUNT, adaptation).
- JAX toolchain (DynapSim), not Torch.
- Revisit only if we need rich analog temporal dynamics AND can tolerate
  mismatch-aware training. For now: Xylo = digital, bit-exact, reproducible.

## This repo = Phase-0 software prototype
Goal: prove an event-driven (spiking) pipeline can diagnose from physiological
signals at accuracy comparable to a conventional net while doing far fewer
operations — the basis for the low-power offline device. See `README.md`.

Current state:
- Two modalities working end-to-end: **ECG beat classification** and **PPG
  classification** (`--modality {ecg,ppg}` on `train.py`).
- `src/eia/`: `encoding.py` (event encoders, numpy), `energy.py` (analytical
  MAC-vs-SOP model, numpy), `datasets.py` (ECG: real MIT-BIH via wfdb + synthetic;
  PPG: real BIDMC via wfdb, real VitalDB via `vitaldb`, + synthetic — pick the
  PPG real source with `--ppg-source {bidmc,vitaldb}` on `train.py` /
  `scripts/xylo_verify.py`), `models.py` (snnTorch SNN + dense baseline,
  modality-agnostic), `device.py` (MPS/CPU), `train.py` (end-to-end demo +
  `--sweep`, both modalities).
- Unit tests pass (`pytest -q`), numpy-only.
- Verified running on the user's M-series Mac via MPS (torch 2.8), incl. `--real`
  for ECG (MIT-BIH), PPG (BIDMC), and PPG (VitalDB) with `wfdb`/`vitaldb` installed.
- **PPG has two real-data sources with different label quality.** BIDMC (still
  the default) has no hemorrhage annotation, so `--modality ppg --real` uses
  SpO2 desaturation (<95%) as a stand-in for physiological compromise — proves
  the pipeline pattern on real waveforms, not a validated hemorrhage signal.
  **VitalDB** (`--ppg-source vitaldb`) is a real, if coarse, hemorrhage-relevant
  label: case-level `intraop_ebl >= 500 mL` (significant blood loss) from open
  intraoperative monitoring — see "VitalDB PPG hemorrhage dataset" below and
  `docs/vitaldb_ppg_hemorrhage_task.md`. Neither is conscious-field-trauma
  ground truth; the synthetic PPG generator is still closest in spirit to the
  actual target (reduced amplitude + blunted dicrotic notch, mirroring
  Compensatory Reserve waveform changes). See README "Notes on rigor".

## VitalDB PPG hemorrhage dataset (real, open, case-level blood-loss label)
- `datasets.load_vitaldb_ppg` streams `SNUADC/PLETH` (confirmed 500 Hz fingertip
  pleth, present in 6157/6388 cases) and labels every window from a case by
  that case's `intraop_ebl` (confirmed field name, non-null for 3987/6388
  cases; 3781 cases have both). Label: `>=500 mL` = significant blood loss (a
  common surgical transfusion-trigger threshold; ~14% of qualifying cases).
- **`load_clinical_data(caseids=[])` returns an EMPTY frame** despite its own
  docstring claiming that means "all cases" — pass `caseids=list(range(1,6389))`
  explicitly (verified against the installed `vitaldb` 1.5.8; see Part 0 of the
  task doc before trusting field/track names from memory in future work here).
- **Case-level split, not window-level** (`PpgData.groups` = case id per
  window): many highly-correlated segments come from one case and share one
  case-level label, so `train.py`/`xylo_verify.py` use `GroupShuffleSplit` (not
  `train_test_split`) whenever `data.groups is not None`, and raise if any case
  leaks across train/val/test.
- **Caveats baked into the data card** (`report.py`, `("ppg","vitaldb")`):
  intraoperative/anesthetized (confounds vs. conscious trauma), EBL is a
  whole-case ESTIMATE not time-aligned to the bleed, and expect class
  imbalance (~14% positive at the case level, worse after small `--max-cases`
  subsetting — check the printed class balance before trusting a result).
- Downloads cache per-case to `data/vitaldb/` (gitignored); `--max-cases` caps
  how many qualifying cases a run pulls. `--n-seeds` on `xylo_verify.py` reports
  mean +/- std over seeds (float/XyloSim balanced acc, per-class recall,
  agreement) — required here per the ECG lesson that a single seed's XyloSim
  agreement is not reliable evidence on its own (see
  `docs/ecg_quant_fixes_results.md`).
- BIDMC is kept as-is, unchanged, as a secondary real-PPG dataset — VitalDB
  supersedes it only as *the* real hemorrhage-relevant label, not as the only
  real PPG source.
- **Measured (multi-seed, 150 cases, 5 seeds — see docs/vitaldb_ppg_results.md):
  the model barely learns anything from this label at the per-window level.**
  Float balanced accuracy = 0.509 +/- 0.025 (~chance) across 5 seeds; 3 of 5
  seeds' best-of-5-restarts training barely escapes exact majority-class
  collapse despite the same class-weighted-loss + balanced-accuracy-selection
  machinery that worked for real MIT-BIH ECG. Root cause is the label itself,
  not a bug: one whole-case EBL total is stamped onto every 4-second window
  from that case, including windows from hours before any bleeding — a single
  pulse has no principled reason to encode a multi-hour case summary.
  XyloSim balanced accuracy (0.600 +/- 0.186) and float-vs-XyloSim agreement
  (0.531 +/- 0.220) both have huge seed-to-seed spread, consistent with the
  ECG checkpoint-sensitivity finding above, worsened here by a float model
  whose decisions are already near-chance (low margin everywhere). VitalDB is
  a real step up in *label realism* over BIDMC's SpO2 proxy; it is not (yet)
  a step up in *per-window learnability* — case-level pooling or a graded EBL
  target are the follow-ups if this is revisited, not attempted here.
- **Case-level rescue attempted (see docs/vitaldb_case_level_task.md /
  docs/vitaldb_case_level_results.md): ALSO ~chance.** `intraop_ebl` is a
  whole-case label, so the statistically honest use of it is one prediction
  per case (`scripts/vitaldb_case_level.py`: train the same per-window SNN
  unchanged, mean-pool its per-window output probabilities into one
  host-side score per case — the Xylo core still only classifies individual
  windows). 300 cases, 5 seeds: case-level balanced accuracy 0.516 +/- 0.019
  — tight and flat at chance, not just noisy. AUROC (0.589 +/- 0.062) is
  marginally above 0.5 but doesn't survive contact with an actual 0.5
  decision threshold. **Conclusion: VitalDB's intraoperative,
  anesthesia-confounded PPG does not carry a usable blood-loss signal at
  either the per-window or the per-case granularity. Settled; do not keep
  tuning VitalDB.** The honest path to a flagship hemorrhage signal is the
  gated LBNP dataset or the synthetic time-resolved generator.

Key finding: the SNN energy advantage is **not automatic** — it requires low
spike rate + few timesteps (ops scale with `spike_rate x timesteps`). Training
now includes a sparsity penalty (`--spike-reg`) and `--sweep` traces the
accuracy/energy trade-off. Synthetic data is deliberately easy (accuracy ~1.0);
use `--real` for meaningful accuracy.

## Environment
- macOS Apple Silicon (M-series). Native arm64 Python. `.venv` in repo root.
- `pip install -e .`; run `python -m eia.train`.

## MARCH roadmap (sequenced by validated-model reachability, not MARCH order)
Deliberate sequencing decision: maximize the number of genuinely-*learning*
models on open clinical data before tackling the hardest sensing/labeling
problems. The lesson from hemorrhage is that the bottleneck is usually the
LABELS, not the model — so prioritize signals with strong physiology + excellent
open labels.
- **C (Circulation) — DONE.** ECG arrhythmia on real MIT-BIH is the one modality
  genuinely learning on real data (float balanced acc 0.845). On-chip XyloSim
  fidelity (~0.56) is the open engineering gap, not a data gap.
- **H (Head) — Phase 1 EEG pipeline built and Xylo-verified; cross-patient
  result is ~chance at the data scale run so far (data-volume-limited, not a
  data-quality dead end like VitalDB — see docs/eeg_seizure_results.md).**
  Split into two independent workstreams:
  - **EEG:** seizure detection on **CHB-MIT** (open, WFDB-via-`read_edf`,
    expert seizure labels, decades of benchmarks). Task spec:
    `docs/eeg_seizure_task.md`. Built hardware-first (23ch native → fixed
    6-channel bipolar montage → 0.5-25 Hz band-pass → resample to 128
    timesteps → delta encoder → 12 spike channels → LIF SNN, `n_in=12` →
    XyloSim; `input 12/16` in the footprint check, margin under the 16-channel
    ceiling). Subject-independent split (patient-grouped, chb21 folded into
    chb01) is wired and leakage-checked. **Measured (9 patients, 1 seizure
    record each — 718 windows, 5 seeds): float balanced acc 0.525 +/- 0.030,
    XyloSim 0.500 +/- 0.028, both flat at chance; AUROC ~0.52-0.55; false
    alarms 300-325/hour (unusable as-is).** Root cause is believed to be data
    volume (9 patients/1 record each is far below CHB-MIT's ~22-24 patients),
    not signal absence — CHB-MIT seizure detection is well-established as
    learnable in the literature, unlike VitalDB. The run was deliberately
    kept this small because PhysioNet download throughput for CHB-MIT's ~40MB
    EDF files varied wildly (4-25+ min/record) during this session; two
    larger attempts (13 then 9 subjects x up to 3 records) were killed
    mid-run as impractically slow. `data/chbmit/` caches every record
    downloaded so far — rerunning with more subjects/records via
    `--eeg-subjects`/`--eeg-seizure-records`/`--eeg-nonseizure-records` is the
    natural next step, not a re-download. Also found: `chb12/13/14/16/17/18/
    19/21` all fail to load via `wfdb.io.convert.edf.read_edf` ("math domain
    error") — a real, reproducible parser limitation, not guessed; excluded
    from `datasets.DEFAULT_EEG_SUBJECTS`. Metrics reported are
    AUROC/AUPRC/sensitivity/specificity/FA-per-hour, NOT accuracy (extreme
    imbalance), per the task spec. Caveat: CHB-MIT is pediatric epilepsy, not
    field TBI. Phase 2 = TUSZ (registration-gated) + Siena generalization; Phase 3
    = TBI spectral screening (slowing, burst suppression) — a different task.
  - **Thermal / hypothermia (later):** no open CHB-MIT-equivalent — likely a
    build-your-own-dataset problem. Deferred.
- **M (Massive hemorrhage) — PAUSED pending better physiology data.** VitalDB is
  settled ~chance (window + case level); revisit only with LBNP/CRM induced-
  hypovolemia data (request the gated Oslo/Yale sets, or build the synthetic
  time-resolved generator). The CRM feature recipe (`An Explainable ML Model for
  CRM`, MDPI Bioeng. 2023) is the method to use once real-physiology data exists.
- **R (Respiration) — PAUSED.** BIDMC already carries a respiration channel
  (currently only its PPG is used); pick up after H.
- **A (Airway) — future work.** Needs breath/lung-sound audio, vision,
  capnography — no data or model yet.

After H/EEG lands, the fusion head (ECG + the next validated modality over a
MARCH timeline) and the ultrasound-probe path remain the longer-horizon steps.

## Working style
Be concise and direct. Keep code in the `eia` package; keep notebooks thin so the
same code runs locally and on Colab.
