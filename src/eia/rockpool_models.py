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


def build_xylo_snn(n_hidden: int = 63, n_out: int = 2, dt: float = 1e-3,
                    n_in: int = N_INPUT_CHANNELS):
    """Build the LIF classifier in Rockpool (Torch backend), Xylo-mappable.

    Input is the delta ON/OFF raster (fed as `window` timesteps x `n_in`
    channels). `n_in` defaults to 2 — one channel's ON/OFF pair (ECG/PPG).
    Multi-channel modalities (EEG: N montage channels -> 2N spike channels
    after per-channel delta encoding) pass `n_in=2*N` explicitly; must stay
    <= `XYLO_MAX_INPUT_CHANNELS`. One hidden LIF layer within Xylo's neuron
    budget, then a readout LIF layer over `n_out` output neurons.

    Returns an untrained Rockpool `Sequential`. Train it with the Torch backend
    like any nn.Module (surrogate gradients are built into LIFBitshiftTorch).
    """
    from rockpool.nn.modules import LIFBitshiftTorch, LinearTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.parameters import Constant

    if n_in > XYLO_MAX_INPUT_CHANNELS:
        raise ValueError(
            f"n_in={n_in} exceeds Xylo budget ({XYLO_MAX_INPUT_CHANNELS})")
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
    # This still isn't bulletproof against class imbalance: on real MIT-BIH
    # ECG (7.7% minority class), 5 restarts x 15 epochs all converged to the
    # exact majority-baseline accuracy (0.923) — worse imbalance than the
    # ~30-38% minority classes this net does learn on (synthetic ECG/PPG,
    # BIDMC PPG). More restarts alone didn't fix it.
    # FIXED in scripts/xylo_verify.py: inverse-frequency class-weighted CE
    # (nn.CrossEntropyLoss(weight=...)) plus checkpoint selection by balanced
    # accuracy instead of raw accuracy (raw accuracy can't tell a
    # majority-collapsed checkpoint from a real one once one class is this
    # rare). Real MIT-BIH now reaches float per-class recall [0.90, 0.79]
    # (balanced acc 0.845) instead of [1.00, 0.00]. This exposed a real
    # XyloSim fidelity gap that raw accuracy had been masking (agreement was
    # a trivial 1.000 on the degenerate model, 0.560 on the genuine one) —
    # see the window notes below for the one lever tried against it so far.
    #
    # LIFBitshiftTorch (not plain LIFTorch): Xylo doesn't have a continuous
    # exponential decay, it approximates it with a bit-shift ("dash"); this
    # module snaps tau_mem/tau_syn to the nearest bit-shift-representable
    # value *during training*, not just at deployment. Training against plain
    # continuous-tau LIFTorch got float accuracy near 1.0 but only ~76%
    # float-vs-XyloSim agreement after quantization — the "quantization drop"
    # this module exists to prevent (train against the constraint, not more
    # epochs against the wrong one).
    #
    # ECG's quantization gap is worse than PPG's (synthetic ECG: 0.986 float
    # -> 0.637 XyloSim, vs. PPG's 0.877 agreement) — hypothesis was that ECG's
    # longer window (187 samples/timesteps vs. PPG's 125) gives bit-shift
    # decay drift more time to accumulate. A/B'd via `--window` on synthetic
    # ECG (same class-weighted training, 5 restarts x 40 epochs):
    #   window=187 (default) : float 0.992, XyloSim agree 0.733
    #   window=125            : float 0.996, XyloSim agree 0.707
    #   window=90              : float 0.988, XyloSim agree 0.763  <- best
    #   window=60              : float 0.962, XyloSim agree 0.597
    # Non-monotonic: too long lets drift accumulate, too short (60) starves
    # the net of enough integration time to hold a robust decision — 90 is a
    # sweet spot, not "shorter is always better". Confirmed this does NOT
    # transfer to real MIT-BIH as-is: at window=90, real-ECG agreement got
    # *worse* (0.560 at window=187 -> 0.477 at window=90). Likely cause: real
    # MIT-BIH is sampled at 360 Hz vs. the synthetic generator's 125 Hz, so
    # the same sample count is a very different real-world duration in each
    # (90 samples = 250ms real vs. 720ms synthetic) — "shorter window" as a
    # timestep-count lever doesn't carry across datasets with different `fs`
    # without rescaling by it. Left real MIT-BIH at the default window=187;
    # `--window` stays available for further probing (e.g. try ~250-260
    # samples on real data, matching synthetic's window=90 in wall-clock time
    # via the fs ratio) but that's future work, not done here.
    #
    # NOT using spike_generation_fn=PeriodicExponential: the official Xylo
    # training tutorial recommends it (it's tuned for its own audio task —
    # many input channels, short windows), but on this net (2 in, 2 out, long
    # single-biosignal windows) it measurably regressed both metrics we
    # actually care about, A/B'd on synthetic PPG, 5 restarts x 40 epochs each,
    # otherwise identical settings:
    #   default (StepPWL)     : float acc 0.992, XyloSim agreement 0.877
    #   PeriodicExponential   : float acc 0.772, XyloSim agreement 0.680
    # Kept the default surrogate on the evidence, not the tutorial's prior.
    net = Sequential(
        LinearTorch((n_in, n_hidden), has_bias=False),
        LIFBitshiftTorch(n_hidden, tau_mem=Constant(0.02), tau_syn=Constant(0.02),
                          threshold=Constant(1.0), bias=0.5, dt=dt),
        LinearTorch((n_hidden, n_out), has_bias=False),
        LIFBitshiftTorch(n_out, tau_mem=Constant(0.02), tau_syn=Constant(0.02),
                          threshold=Constant(1.0), bias=0.5, dt=dt),
    )
    return net


def build_combined_xylo_snn(nets: list, dt: float = 1e-3):
    """Combine independently-trained per-modality nets onto one Xylo core.

    Each `nets[i]` must be a `build_xylo_snn`-shaped Sequential (already
    trained). Returns one bigger Sequential with block-diagonal input/output
    weights — modality i's hidden units (and output neurons) only ever see
    modality i's input channels and only ever drive modality i's outputs —
    which is exactly how the mapper places independently-trained sub-nets in
    Xylo's single shared `weights_rec`: per-neuron dynamics stay independent,
    but the whole combined net gets ONE `global_quantize` scale (see
    `map_and_quantize`), so this must be quantized as one unit, not per-block.
    """
    import torch
    from rockpool.nn.modules import LIFBitshiftTorch, LinearTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.parameters import Constant

    n_ins = [net[0].shape[0] for net in nets]
    n_hiddens = [net[0].shape[1] for net in nets]
    n_outs = [net[2].shape[1] for net in nets]
    total_in, total_hidden, total_out = sum(n_ins), sum(n_hiddens), sum(n_outs)

    if total_in > XYLO_MAX_INPUT_CHANNELS:
        raise ValueError(
            f"combined input channels={total_in} exceeds Xylo budget "
            f"({XYLO_MAX_INPUT_CHANNELS})")
    if total_hidden > XYLO_MAX_HIDDEN_NEURONS:
        raise ValueError(
            f"combined hidden neurons={total_hidden} exceeds Xylo budget "
            f"({XYLO_MAX_HIDDEN_NEURONS})")
    if total_out > XYLO_MAX_OUTPUT_CHANNELS:
        raise ValueError(
            f"combined output neurons={total_out} exceeds Xylo budget "
            f"({XYLO_MAX_OUTPUT_CHANNELS})")

    w_in = torch.zeros(total_in, total_hidden)
    w_out = torch.zeros(total_hidden, total_out)
    hidden_bias = torch.zeros(total_hidden)
    output_bias = torch.zeros(total_out)

    in_off = hid_off = out_off = 0
    for net, n_in, n_hid, n_out in zip(nets, n_ins, n_hiddens, n_outs):
        w_in[in_off:in_off + n_in, hid_off:hid_off + n_hid] = net[0].weight.detach()
        w_out[hid_off:hid_off + n_hid, out_off:out_off + n_out] = net[2].weight.detach()
        # LIFBitshiftTorch's bias here is a single scalar shared across the
        # whole layer (see build_xylo_snn); broadcast it across this
        # sub-net's block of the combined per-neuron bias vector.
        hidden_bias[hid_off:hid_off + n_hid] = net[1].bias.detach()
        output_bias[out_off:out_off + n_out] = net[3].bias.detach()
        in_off += n_in
        hid_off += n_hid
        out_off += n_out

    combined = Sequential(
        LinearTorch((total_in, total_hidden), has_bias=False, weight=w_in),
        LIFBitshiftTorch(total_hidden, tau_mem=Constant(0.02), tau_syn=Constant(0.02),
                          threshold=Constant(1.0), bias=hidden_bias, dt=dt),
        LinearTorch((total_hidden, total_out), has_bias=False, weight=w_out),
        LIFBitshiftTorch(total_out, tau_mem=Constant(0.02), tau_syn=Constant(0.02),
                          threshold=Constant(1.0), bias=output_bias, dt=dt),
    )
    return combined, n_ins, n_hiddens, n_outs


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


def map_and_quantize(net, method: str = "global"):
    """Map a trained Rockpool net to a Xylo spec and quantize to integer logic.

    Returns the quantized `spec` dict consumed by `config_from_specification`.

    Args:
        method: "global" (default, one shared scale for the whole network, as
            in the deploy guide) or "channel" (per-target-neuron scaling).
            For a single small net the two made no measurable difference
            (tried on the PPG task). For a *combined* multi-modality net
            (`build_combined_xylo_snn`), where two independently-trained
            sub-nets' weight magnitudes can differ, "global" wastes range on
            whichever sub-net is smaller — this is the exact failure mode
            `docs/per_modality_xylo_verify_task.md` Part C warns about, and
            "channel" is the documented fix.
    """
    from rockpool.transform import quantize_methods as q

    x = _xylo_support()
    spec = x.mapper(net.as_graph(), weight_dtype="float")
    quantize_fn = q.channel_quantize if method == "channel" else q.global_quantize
    try:
        quantized = quantize_fn(**spec)
    except TypeError:
        # Some Rockpool versions' global_quantize doesn't accept every key the
        # mapper puts in `spec` (e.g. `mapped_graph`, `dt`) — the official
        # Xylo training tutorial strips those two before quantizing. Our
        # installed version's global_quantize has a **kwargs catch-all so this
        # never actually triggers, but keep the fallback for older/newer ones.
        stripped = {k: v for k, v in spec.items() if k not in ("mapped_graph", "dt")}
        quantized = quantize_fn(**stripped)
    spec.update(quantized)
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

    Returns dict with both predictions (argmax of summed output spikes),
    whether they agree, and the raw summed-output-spike vectors
    (`out_float`/`out_xylo`, shape (n_out,)) — callers that need a
    continuous score (e.g. AUROC/AUPRC under class imbalance, which argmax
    predictions alone can't support) derive it from these rather than
    re-running the sim.
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

    out_float_sum = out_float.reshape(-1, out_float.shape[-1]).sum(axis=0)
    out_xylo_sum = np.asarray(sim_out).sum(axis=0)
    pred_float = int(out_float_sum.argmax())
    pred_xylo = int(out_xylo_sum.argmax())
    return {
        "pred_float": pred_float,
        "pred_xylo": pred_xylo,
        "out_float": out_float_sum,
        "out_xylo": out_xylo_sum,
        "match": pred_float == pred_xylo,
    }
