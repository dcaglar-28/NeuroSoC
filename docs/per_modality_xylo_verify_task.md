# Task spec — per-modality XyloSim verification + resource footprint

**For:** Claude Code (repo, `.venv` active, `pip install "eia[xylo]"`).
**Goal:** Verify each modality's network configuration on the bit-precise
XyloSim *separately* (no hardware), and report each mapped network's resource
footprint against the Xylo limits — so we can see how modalities compose onto
chips.

Builds on `scripts/xylo_verify.py`, `src/eia/rockpool_models.py`, and the new
`src/eia/xylo_budget.py`.

These edits come straight from the official SynSense training/deploy tutorials —
apply Part 0 first, then verify.

---

## Part 0 — code fixes to apply first (from the Rockpool tutorials)

Source: "Training a spiking network to deploy to the Xylo digital SNN"
(https://rockpool.ai/devices/torch-training-spiking-for-xylo.html) and
"Training a spiking network with Torch"
(https://rockpool.ai/tutorials/torch-training-spiking.html).

1. **Add `spike_generation_fn=PeriodicExponential` to every `LIFBitshiftTorch`**
   in `build_xylo_snn`. This is the SynSense-recommended surrogate for Xylo — it
   models multiple spikes per timestep (Xylo allows up to 31 hidden spikes/step)
   and is what the official tutorial uses. Import:
   `from rockpool.nn.modules.torch.lif_torch import PeriodicExponential`.
   Expected effect: better training + higher float-vs-XyloSim agreement (the
   tutorial's whole net trains this way).

2. **Guard the quantize call.** The Xylo training tutorial does
   `del spec["mapped_graph"]; del spec["dt"]` before `global_quantize(**spec)`
   (works on a copy). Our `map_and_quantize` passes the full dict (fine on 3.1.0).
   Make it robust: try `global_quantize(**spec)`; on a TypeError about unexpected
   kwargs, strip `mapped_graph` and `dt` from a copy and retry.

3. **Training-loop hygiene** (verify `xylo_verify.py` does all of these):
   - `Adam(net.parameters().astorch(), lr=1e-2)` — `.astorch()` is required to
     hand Rockpool params to a torch optimizer; lr=1e-2 matches the tutorial.
   - `net.reset_state()` before every sample (detach) for correct BPTT.
   - Classify via output-neuron membrane potential (we do); do NOT add an
     `ExpSynTorch` readout — it is not Xylo-output-mappable.

4. **Keep `tau >= 10*dt`.** Rockpool numerical-stability rule (sgd_recurrent_net
   tutorial). We freeze `tau=Constant(0.02)` with `dt=1e-3` (= 20*dt, safe). If a
   future variant makes tau trainable, bound it with `make_bounds`/`bounds_cost`
   instead of freezing — do not let it drift toward 0 (NaN trap). Note the
   bit-shift decay Xylo actually uses: `dash = round(log2(tau/dt))` -> 4 here.

5. **`Nhidden=63` is the recurrent-fanout limit**, not a tuning choice — keep it
   as the default max hidden size; only exceed it if you drop full recurrence.

## Part A — verify each modality separately

Extend `scripts/xylo_verify.py` to take `--modality {ecg,ppg}` (default run both,
one after another) and, for **each** modality independently:

1. Load data (`load_ecg` / `load_ppg`, `--real` optional).
2. Print its **data card** first (`report.data_card(data)`), then any warnings.
3. Train the Xylo-mappable net (`build_xylo_snn`), map + quantize
   (`map_and_quantize`), build XyloSim (`to_xylo_sim`), and run
   `verify_against_sim` over a held-out set.
4. Report, **per modality, never pooled**: float acc, XyloSim acc, float-vs-sim
   agreement, and `is_valid` from `config_from_specification`.

Keep the per-dataset principle: one model, one report, one data card per
modality; do not average across modalities.

## Part B — report the mapped resource footprint

For each modality's quantized `spec`, print how much of the chip it uses, from
the mapped matrices (not just the requested sizes):
- input channels used  = `spec['weights_in'].shape[0]`
- hidden neurons used   = number of hidden units in the mapped graph
- output neurons used   = `spec['weights_out'].shape[1]`
Compare each against `xylo_budget.XYLO_MAX_*`. Then call
`xylo_budget.fits_one_chip([...])` with the actual per-modality sizes to print
whether ECG+PPG (and future modalities) co-reside on one chip and which limit
binds first.

Reference (mapped spec keys): the deploy-to-Xylo guide shows `spec` contains
`weights_in`, `weights_rec`, `weights_out`, `threshold`, `dash_mem`, etc. —
https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html

## Deliverable

- `python scripts/xylo_verify.py --modality ecg` and `--modality ppg` each print
  a data card, a XyloSim verification result, and a resource footprint.
- A one-chip composition line (from `xylo_budget.fits_one_chip`) summarising how
  many modalities fit and the binding limit.
- Update `README.md` "Verify on XyloSim" with the `--modality` usage.
- Existing tests still pass; the 4 `test_xylo_budget.py` tests too.

## Part C — verify ECG + PPG co-residence on ONE chip

Confirm two independently-trained sub-networks share one Xylo core correctly:

1. Train ECG and PPG nets separately (as in Part A).
2. Build a **combined** Rockpool net that places both sub-nets on disjoint
   neurons (concatenated 4 input channels = 2 ECG + 2 PPG; separate hidden
   blocks; 4 output neurons = 2 + 2). The mapper does block placement in the
   shared `weights_rec` automatically.
3. **Map + quantize the COMBINED net as one unit** (a single `global_quantize`
   scale must cover all weights) — do NOT quantize each sub-net separately and
   glue configs.
4. Run XyloSim on the combined config and confirm **each modality's outputs still
   agree with its own float model** (no cross-modality precision loss). If
   agreement drops for one modality, its weight magnitudes likely differ from the
   other's under the shared scale — try `channel_quantize` or normalize each
   sub-net's weights before combining.

The Xylo training tutorial names the exact failure mode to watch for here: a
single `global_quantize` scale wastes range when the combined weights are **not
centered on 0, have a few very strong outliers, or a non-flat distribution** —
precisely what mixing two independently-trained sub-nets can cause. So before
combining, log each sub-net's weight histogram; if they diverge, normalize per
sub-net or switch to `channel_quantize`.

This is confirmed feasible by the mapper output (per-neuron thresholds/dashes are
independent; `weights_rec` is block-structured), so this step is a verification,
not research.

## Part D — optional stretch: unify models.py and rockpool_models.py via NIR

Source: "Import / export between toolchains with NIR"
(https://rockpool.ai/advanced/nir_export_import.html). Goal: stop maintaining the
"same" LIF net twice (snnTorch in `models.py`, Rockpool in `rockpool_models.py`).

1. `pip install 'rockpool[nir]'`.
2. Export the trained snnTorch (or Rockpool) net with `to_nir(net)`; write with
   `nir.write(...)`. Inspect that it serializes as `CubaLIF` + `Linear` nodes with
   matching tau/threshold/weights.
3. Re-import with `from_nir(...)`. Note it returns a `nirtorch.GraphExecutor`
   (a `torch.nn.Module`, NOT a native Rockpool module) — but it still supports
   `as_graph()`, so feed it straight into `mapper(...) -> quantize -> XyloSim`.
4. Confirm the NIR-round-tripped net gives the same XyloSim result as the
   directly-built Rockpool net. If it does, NIR becomes the single source of
   truth and the duplication can be removed.

Caveats to record in the PR: NIR is **beta** (API may change) and Rockpool NIR is
**torch-only**. Keep this OPTIONAL — do not block Parts A–C on it.

## Not in scope: DynapSE-2

DynapSE (analog) was evaluated and deferred (see `CLAUDE.md`): no bit-exact
simulator, requires mismatch-aware training, 4-bit weights, JAX toolchain. Do NOT
add a DynapSE path now; Xylo stays the target.

## Gotchas

- **Two output neurons per modality is the binding limit** (8 output neurons /
  2 = 4 modalities per chip). If you reduce a modality to a single output
  neuron (1 = "compromise" vs no-spike), you double how many fit — note this in
  the report if you try it.
- Delta ON/OFF = exactly 2 input synapses per hidden neuron, which is the Xylo
  max (2). Don't add a second input projection per neuron. This also means
  **input-layer cross-modality fusion is impossible** (a neuron can't read >2
  input channels) — any on-chip fusion must go hidden->hidden via `W_rec`; heavy
  fusion belongs on Akida.
- **Shared across co-resident nets:** the global `dt`, the `global_quantize`
  scale, and `weight_shift_*`. Per-neuron thresholds/time-constants/biases stay
  independent. Map+quantize the combined net as one unit.
- Version-specific `syns61201` submodule (see `_xylo_support`), as before.
