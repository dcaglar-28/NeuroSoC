"""Xylo deployment & verification path (SynSense Rockpool).

Hardware-target sibling of `models.py`. `models.py` uses snnTorch for fast local
research; this module re-expresses the same LIF network in **Rockpool**,
SynSense's SDK, so it can be mapped, quantized, and run on the **XyloSim
bit-precise simulator** — letting you verify the full system *before* buying a
Xylo HDK. XyloSim reproduces the on-chip integer dynamics exactly (the Rockpool
docs show its traces are identical to hardware), so a passing XyloSim check is a
genuine pre-silicon acceptance test.

Why the neuron models match: Rockpool's LIF is the same one we train — membrane
potential + synaptic current, threshold with subtractive reset, forward-Euler
`dt` (rockpool.ai/basics/introduction_to_snns). On Xylo, exponential decay is
approximated by "bit-shift decay" (dash) parameters; the quantizer handles that
conversion for us.

IMPORTANT — input framing for Xylo:
    Xylo allows at most 16 SNN input channels. So we do NOT flatten the window
    into (2*window) features. Instead the delta ON/OFF encoding is fed as
    **2 input channels over `window` timesteps** — the waveform's samples are
    the time axis. This is the hardware-correct 1-D biosignal framing and stays
    well inside the 16-channel limit. (`encoding.delta_encode_2ch` returns
    (2, window); transpose to (window, 2) as the input raster.)

Xylo-Audio 2 (SYNS61201) resource limits, from the Rockpool Xylo overview:
    input channels <= 16 | hidden neurons <= 1000 | output neurons <= 8
    weights 8-bit | subtractive reset | per-neuron time constants & thresholds

Deploy flow (mirrors the "Quick-start with Xylo SNN core" guide):
    train (Torch) -> net.as_graph() -> mapper() -> global_quantize()
        -> config_from_specification() -> XyloSim.from_config() -> compare

Install:  pip install "eia[xylo]"
Docs:
  - Xylo overview:  https://rockpool.ai/devices/xylo-overview.html
  - Deploy to Xylo: https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html
  - Train for Xylo: https://rockpool.ai/devices/torch-training-spiking-for-xylo.html
"""

from __future__ import annotations

# Xylo-Audio 2 (SYNS61201) hardware limits the mapper enforces.
XYLO_MAX_INPUT_CHANNELS = 16
XYLO_MAX_HIDDEN_NEURONS = 1000
XYLO_MAX_OUTPUT_CHANNELS = 8
XYLO_WEIGHT_BITS = 8

# Our delta encoder emits 2 channels (ON, OFF); this is the Xylo input width.
N_INPUT_CHANNELS = 2


def _xylo_support():
    """Return the correct Xylo support module.

    Prefers the package matched to a connected HDK (via `find_xylo_hdks`); with
    no hardware, falls back to the Xylo-Audio 2 simulator package (syns61201),
    exactly as the deploy-to-Xylo guide recommends.
    """
    try:
        from rockpool.devices.xylo import find_xylo_hdks
        connected, support_modules, _versions = find_xylo_hdks()
        if connected:
            return support_modules[0]
    except Exception:  # noqa: BLE001 — no HDK / no samna backend -> simulator
        pass
    import rockpool.devices.xylo.syns61201 as x
    return x


def build_xylo_snn(n_hidden: int = 63, n_out: int = 2, dt: float = 1e-3):
    """Build the LIF classifier in Rockpool (Torch backend), Xylo-mappable.

    Input is the 2-channel delta raster (fed as `window` timesteps x 2 channels),
    so `n_in = 2`. One hidden LIF layer within Xylo's neuron budget, then a
    readout LIF layer over `n_out` output neurons.

    Returns an untrained Rockpool `Sequential`. Train it with the Torch backend
    like any nn.Module (surrogate gradients are built into LIFBitshiftTorch).
    """
    from rockpool.nn.modules import LIFBitshiftTorch, LinearTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.parameters import Constant

    if n_hidden > XYLO_MAX_HIDDEN_NEURONS:
        raise ValueError(
            f"n_hidden={n_hidden} exceeds Xylo budget ({XYLO_MAX_HIDDEN_NEURONS})")
    if n_out > XYLO_MAX_OUTPUT_CHANNELS:
        raise ValueError(
            f"n_out={n_out} exceeds Xylo output neurons ({XYLO_MAX_OUTPUT_CHANNELS})")

    # tau_mem/tau_syn/threshold default to *trainable* Parameters in LIFTorch.
    # A weight-scale learning rate applied to a ~20 ms time constant drives it
    # through zero within a few batches (decay math then divides by ~0 -> NaN
    # weights, which the Xylo quantizer silently casts to garbage). Fix them
    # as Constants — the same role `beta` plays as a fixed decay hyperparameter
    # in `models.py` — so only the synaptic weights (and bias, below) train.
    #
    # bias starts at 0.0 by default, which is a trap for this readout: with
    # only 2 output neurons and no per-linear bias (has_bias=False matches the
    # Xylo weight layout), a neuron that happens to net a slightly negative
    # input at init never crosses threshold, so its surrogate gradient is
    # exactly zero from step one — permanently dead, regardless of weight
    # scale or learning rate (verified empirically across many seeds/scales).
    # Starting the bias positive gives every neuron some baseline firing at
    # init, which is what actually breaks the symmetry and lets training find
    # a discriminative solution; it stays trainable so it can still move.
    #
    # LIFBitshiftTorch (not plain LIFTorch): Xylo doesn't have a continuous
    # exponential decay, it approximates it with a bit-shift ("dash"); this
    # module snaps tau_mem/tau_syn to the nearest bit-shift-representable
    # value *during training*, not just at deployment. Training against plain
    # continuous-tau LIFTorch got float accuracy near 1.0 but only ~76%
    # float-vs-XyloSim agreement after quantization — the "quantization drop"
    # this module exists to prevent (train against the constraint, not more
    # epochs against the wrong one).
    net = Sequential(
        LinearTorch((N_INPUT_CHANNELS, n_hidden), has_bias=False),
        LIFBitshiftTorch(n_hidden, tau_mem=Constant(0.02), tau_syn=Constant(0.02),
                          threshold=Constant(1.0), bias=0.5, dt=dt),
        LinearTorch((n_hidden, n_out), has_bias=False),
        LIFBitshiftTorch(n_out, tau_mem=Constant(0.02), tau_syn=Constant(0.02),
                          threshold=Constant(1.0), bias=0.5, dt=dt),
    )
    return net


def to_input_raster(spikes_2ch):
    """Convert a (2, window) delta encoding into a (window, 2) Xylo input raster.

    Use `encoding.delta_encode_2ch(encoding.normalize(signal), threshold)` to get
    the (2, window) array, then pass it here.
    """
    import numpy as np
    arr = np.asarray(spikes_2ch)
    if arr.shape[0] == N_INPUT_CHANNELS:      # (2, window) -> (window, 2)
        arr = arr.T
    return arr.astype("float32")


def map_and_quantize(net):
    """Map a trained Rockpool net to a Xylo spec and quantize to integer logic.

    Returns the quantized `spec` dict consumed by `config_from_specification`.
    Uses `global_quantize` (shared representation) as in the deploy guide.
    (`channel_quantize`, per-target-neuron scaling, was also tried on the PPG
    task and made no measurable difference to float-vs-XyloSim agreement —
    the residual gap is from the bit-shift-quantized decay dynamics over a
    long window, not the weight quantization method.)
    """
    from rockpool.transform import quantize_methods as q

    x = _xylo_support()
    spec = x.mapper(net.as_graph(), weight_dtype="float")
    spec.update(q.global_quantize(**spec))
    return spec


def to_xylo_sim(spec):
    """Build a XyloSim bit-precise simulator from a quantized spec.

    XyloSim's outputs match silicon exactly, so this is what you run to verify
    functionality without owning a Xylo chip. Returns (sim, config).
    """
    x = _xylo_support()
    config, is_valid, msg = x.config_from_specification(**spec)
    if not is_valid:
        raise RuntimeError(f"Xylo config invalid: {msg}")
    return x.XyloSim.from_config(config), config


def verify_against_sim(net, spec, input_raster):
    """Compare the float Torch model vs. the quantized Xylo bit-precise sim on the
    same input raster — the core pre-hardware acceptance check.

    Args:
        net: trained Rockpool module (float).
        spec: output of `map_and_quantize(net)`.
        input_raster: (window, 2) array of {0,1} events from `to_input_raster`.

    Returns dict with both predictions (argmax of summed output spikes) and
    whether they agree.
    """
    import numpy as np
    import torch

    sim, _config = to_xylo_sim(spec)

    net = net.reset_state()
    raster = np.asarray(input_raster, dtype="float32")
    in_float = torch.tensor(raster[None, ...])            # (batch, time, chans)
    with torch.no_grad():
        out_float, _, _ = net(in_float)
    out_float = out_float.detach().numpy()
    sim_out, _, _ = sim(raster)                           # (time, chans)

    pred_float = int(out_float.reshape(-1, out_float.shape[-1]).sum(axis=0).argmax())
    pred_xylo = int(np.asarray(sim_out).sum(axis=0).argmax())
    return {
        "pred_float": pred_float,
        "pred_xylo": pred_xylo,
        "match": pred_float == pred_xylo,
    }
