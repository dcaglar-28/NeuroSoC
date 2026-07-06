# Task spec — verify the PPG model on Xylo (XyloSim) before buying hardware

**For:** Claude Code (run in the repo, `.venv` active).
**Goal:** Train the hemorrhage/PPG SNN, deploy it through the Rockpool → Xylo
pipeline, and confirm the quantized network behaves identically on the
**XyloSim bit-precise simulator** as the float model. A passing check means the
network is silicon-ready without owning a Xylo HDK.

Everything below uses the helpers already in `src/eia/rockpool_models.py`.

---

## Preconditions

```bash
source .venv/bin/activate
pip install "rockpool[xylo]"     # XyloSim bit-precise simulator backend
python -c "import rockpool; print('rockpool', rockpool.__version__)"
python -c "from eia import rockpool_models; print('module imports OK')"
```

If `rockpool[xylo]` fails to build on macOS, check the Rockpool install guide
(the `xylo` extra pulls `xylosim`/`samna`): https://rockpool.ai/basics/installation.html

---

## Steps

1. **Get data + encode.** Use `eia.datasets.make_synthetic_ppg()` first (fast,
   deterministic), then repeat with `load_ppg(prefer_real=True)` (BIDMC). For
   each 1-D window: `enc = encoding.delta_encode_2ch(encoding.normalize(sig), thr)`
   then `raster = rockpool_models.to_input_raster(enc)` → shape `(window, 2)`.
   - **Why (window, 2), not flat:** Xylo allows ≤16 input channels, so the ON/OFF
     delta encoding is fed as 2 channels over `window` timesteps. See the input
     limits table in the Xylo overview:
     https://rockpool.ai/devices/xylo-overview.html

2. **Build the Rockpool net.** `net = rockpool_models.build_xylo_snn(n_hidden=63,
   n_out=2)`. This is `LinearTorch → LIFTorch → LinearTorch → LIFTorch`, the same
   LIF neuron we train in `models.py`.
   - Building networks / combinators: https://rockpool.ai/basics/standard_modules.html
   - Torch LIF training: https://rockpool.ai/tutorials/torch-training-spiking.html
   - Training specifically *for Xylo* (respecting HW constraints, dropping to
     bit-shift decay, quantization-aware tips):
     https://rockpool.ai/devices/torch-training-spiking-for-xylo.html

3. **Train.** Train `net` with a torch optimizer + cross-entropy on the readout
   spike counts, mirroring `eia.train`. Keep hidden ≤ 1000 and output ≤ 8. Add
   the same sparsity penalty idea from `train.py` (low spike rate matters for
   Xylo energy, and for staying under the per-timestep spike caps — 31 hidden,
   1 output; see the overview table).

4. **Map + quantize.** `spec = rockpool_models.map_and_quantize(net)`.
   - This runs `mapper(net.as_graph(), weight_dtype='float')` then
     `quantize_methods.global_quantize(**spec)` (8-bit weights).
   - Reference (steps 2–4): https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html
   - Graph mapping detail: https://rockpool.ai/advanced/graph_mapping.html

5. **Build XyloSim + verify.** For a batch of held-out windows, call
   `rockpool_models.verify_against_sim(net, spec, raster)` and aggregate the
   `match` field. XyloSim reproduces on-chip integer dynamics exactly, so
   agreement between the float model and XyloSim is the pre-hardware acceptance
   test (step 8 of the deploy guide shows the traces are identical to HDK).
   - XyloSim API: https://rockpool.ai/reference/_autosummary/devices.xylo.syns61201.XyloSim.html

---

## Acceptance criteria

- `rockpool[xylo]` installs; `rockpool_models` imports.
- `map_and_quantize` returns a spec with 8-bit integer weights and
  `config_from_specification(**spec)` reports `is_valid == True`.
- Over ≥200 held-out windows, **XyloSim vs. float predictions agree on a high
  fraction** (expect near-100% class-label agreement; small per-timestep trace
  differences from quantization are fine — we care about the decision).
- Report: float test accuracy, XyloSim test accuracy, and their agreement rate.
  Compare XyloSim accuracy to the snnTorch `train.py` number for the same task.

---

## Known gotchas

- **Version-specific submodule.** `rockpool_models._xylo_support()` prefers a
  connected HDK via `find_xylo_hdks()` and otherwise falls back to
  `rockpool.devices.xylo.syns61201` (Xylo-Audio 2). If the installed Rockpool
  targets a different variant (Xylo-Audio 3 / IMU), that import + the resource
  limits change — confirm against the overview table.
- **Quantization drop.** If XyloSim accuracy falls well below the float model,
  the fix is quantization-aware training (train against the bit-shift-decay /
  integer constraints), per the "training for Xylo" guide — not more epochs.
- **Spike caps.** Xylo caps spikes/timestep (31 hidden, 1 output). A dense-firing
  net can saturate these; keep the sparsity penalty on.
- **`dt` consistency.** Use the same `dt` (default 1e-3) in `build_xylo_snn` and
  any time-series handling.

## Deliverable

Add `notebooks/02_xylo_verify.ipynb` (thin, imports `eia`) or a
`scripts/xylo_verify.py` that runs steps 1–5 end to end and prints the report
table. Update `README.md` with a one-line "verify on XyloSim" command.

## Reference index

- Xylo family + resource limits: https://rockpool.ai/devices/xylo-overview.html
- Deploy quick-start (map→quantize→config→XyloSim): https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html
- Train a spiking net for Xylo: https://rockpool.ai/devices/torch-training-spiking-for-xylo.html
- LIF neuron model: https://rockpool.ai/basics/introduction_to_snns.html
- Torch training tutorial: https://rockpool.ai/tutorials/torch-training-spiking.html
- Install / extras: https://rockpool.ai/basics/installation.html
- Full API: https://rockpool.ai/reference/api.html
