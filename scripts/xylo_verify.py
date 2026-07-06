"""Pre-hardware acceptance check: train each modality's SNN, deploy it
through the Rockpool -> Xylo pipeline, and confirm the quantized network
agrees with the float model on XyloSim (the bit-precise simulator).

Verifies each modality *separately* (own data card, own XyloSim check, own
mapped resource footprint), then — when both modalities are run — combines
the two independently-trained nets onto one Xylo core and confirms neither
modality's decisions are corrupted by co-residence and shared quantization.

See `docs/xylo_verification_task.md` (per-modality basics) and
`docs/per_modality_xylo_verify_task.md` (this script's spec) for the full
task write-ups, including the two empirical findings that shape the training
code below: this net's spike-count readout only trains reliably with a
positive initial bias (see `rockpool_models.build_xylo_snn`), and the
official tutorial's `PeriodicExponential` surrogate measurably regresses
both accuracy and XyloSim agreement on this specific 2-in/2-out net (kept on
the evidence, not the tutorial's prior — same file).

Run from the repo root (needs `pip install "eia[xylo]"`):

    python scripts/xylo_verify.py                     # ecg + ppg, synthetic, + one-chip check
    python scripts/xylo_verify.py --modality ppg       # just PPG
    python scripts/xylo_verify.py --real               # real MIT-BIH / BIDMC (needs wfdb + network)
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from eia import encoding, report, rockpool_models as rm, xylo_budget as xb
from eia.datasets import load_ecg, load_ppg

_LOADERS = {"ecg": load_ecg, "ppg": load_ppg}
_HIDDEN_KEY = "1_LIFBitshiftTorch"


def _encode_batch(X: np.ndarray, threshold: float) -> np.ndarray:
    """Delta-encode each window into a Xylo input raster. -> (n, window, 2)."""
    out = np.empty((X.shape[0], X.shape[1], 2), dtype=np.float32)
    for i in range(X.shape[0]):
        enc = encoding.delta_encode_2ch(encoding.normalize(X[i]), threshold)
        out[i] = rm.to_input_raster(enc)
    return out


def _eval(net, R, y_t):
    net.reset_state()
    with torch.no_grad():
        out, _state, rec = net(R, record=True)
    preds = out.sum(dim=1).argmax(dim=1)
    acc = (preds == y_t).float().mean().item()
    return acc, rec[_HIDDEN_KEY]["spikes"].mean().item(), preds


def train_modality(modality: str, real: bool, epochs: int, n_hidden: int,
                    threshold: float, batch_size: int, lr: float,
                    spike_reg: float, seed: int, n_restarts: int):
    """Part A (train half): data card + multi-restart training with
    val-accuracy checkpoint selection (see build_xylo_snn for why this net
    needs restarts). Returns the trained net plus held-out tensors, reused
    both for this modality's own XyloSim check and Part C's combined check.
    """
    data = _LOADERS[modality](prefer_real=real)
    report.data_card(data)
    n_classes = int(np.asarray(data.y).max()) + 1

    # Held out for final reporting — never used for model selection below.
    Xfit, Xte, yfit, yte = train_test_split(
        data.X, data.y, test_size=0.25, random_state=seed, stratify=data.y)
    Xtr, Xval, ytr, yval = train_test_split(
        Xfit, yfit, test_size=0.2, random_state=seed, stratify=yfit)

    Rtr = torch.tensor(_encode_batch(Xtr, threshold))
    Rval = torch.tensor(_encode_batch(Xval, threshold))
    Rte = torch.tensor(_encode_batch(Xte, threshold))
    ytr_t = torch.tensor(ytr)
    yval_t = torch.tensor(yval)
    yte_t = torch.tensor(yte)
    print(f"[encoding] {modality}: mean input event rate = "
          f"{float((Rtr != 0).float().mean()):.3f}")

    lossfn = nn.CrossEntropyLoss()
    n = Rtr.shape[0]
    best_val_acc, best_state = -1.0, None
    for restart in range(n_restarts):
        torch.manual_seed(seed * 100 + restart)
        net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=n_classes)
        # `.astorch()` hands Rockpool's Parameter dict to a torch optimizer
        # as the flat leaf-Parameter iterable it expects (see build_xylo_snn
        # docstring for why bias is NOT frozen like tau/threshold).
        opt = torch.optim.Adam(net.parameters().astorch(), lr=lr)
        for ep in range(epochs):
            idx = torch.randperm(n)
            for k in range(0, n, batch_size):
                bidx = idx[k:k + batch_size]
                opt.zero_grad()
                net.reset_state()
                out, _state, rec = net(Rtr[bidx], record=True)
                logits = out.sum(dim=1)  # sum output spikes over time -> (batch, n_out)
                spk_rate = rec[_HIDDEN_KEY]["spikes"].mean()
                loss = lossfn(logits, ytr_t[bidx]) + spike_reg * spk_rate
                loss.backward()
                opt.step()
            val_acc, _, _ = _eval(net, Rval, yval_t)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(net.state_dict())
        print(f"[train] {modality} restart {restart}: "
              f"best-so-far val accuracy = {best_val_acc:.3f}")

    net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=n_classes)
    net.load_state_dict(best_state)
    print(f"[train] {modality}: final best val accuracy = {best_val_acc:.3f}")

    float_acc, mean_hidden_rate, float_preds = _eval(net, Rte, yte_t)
    spec = rm.map_and_quantize(net)
    _config, is_valid, msg = rm._xylo_support().config_from_specification(**spec)
    print(f"[xylo] {modality}: config_from_specification is_valid={is_valid} ({msg})")

    return {
        "modality": modality, "source": data.source, "net": net, "spec": spec,
        "n_hidden": n_hidden, "n_out": n_classes,
        "Rte": Rte, "yte_t": yte_t, "float_preds": float_preds,
        "float_acc": float_acc, "mean_hidden_rate": mean_hidden_rate,
        "is_valid": is_valid,
    }


def verify_modality(result: dict, max_verify: int) -> dict:
    """Part A (verify half): run XyloSim over held-out windows, print report."""
    net, spec, Rte, yte_t = result["net"], result["spec"], result["Rte"], result["yte_t"]
    n_verify = min(max_verify, Rte.shape[0])
    matches, xylo_correct = 0, 0
    for i in range(n_verify):
        res = rm.verify_against_sim(net, spec, Rte[i].numpy())
        matches += int(res["match"])
        xylo_correct += int(res["pred_xylo"] == int(yte_t[i]))
    agreement_rate = matches / n_verify
    xylo_acc = xylo_correct / n_verify

    modality, source = result["modality"], result["source"]
    header = f" XYLO VERIFICATION -- {modality.upper()} ({source}) "
    print(f"\n{header:=^60}")
    print(f"Held-out windows verified: {n_verify} / {Rte.shape[0]}")
    print(f"Float model test accuracy: {result['float_acc']:.3f}")
    print(f"XyloSim test accuracy    : {xylo_acc:.3f}")
    print(f"Float vs. XyloSim agree  : {agreement_rate:.3f}")
    print(f"Mean hidden spike rate   : {result['mean_hidden_rate']:.3f}")
    print("=" * 60)

    result.update(xylo_acc=xylo_acc, agreement_rate=agreement_rate, n_verify=n_verify)
    return result


def report_footprint(result: dict) -> "xb.Modality":
    """Part B: mapped resource footprint (from the actual mapped matrices,
    not just the requested sizes) vs. Xylo's hardware limits."""
    spec = result["spec"]
    n_in_used = spec["weights_in"].shape[0]
    n_hidden_used = spec["weights_in"].shape[1]
    n_out_used = spec["weights_out"].shape[1]
    print(f"\n[footprint] {result['modality']}: "
          f"input {n_in_used}/{xb.XYLO_MAX_INPUT_CHANNELS}  "
          f"hidden {n_hidden_used}/{xb.XYLO_MAX_HIDDEN_NEURONS}  "
          f"output {n_out_used}/{xb.XYLO_MAX_OUTPUT_CHANNELS}")
    return xb.Modality(result["modality"], n_in=n_in_used,
                        n_hidden=n_hidden_used, n_out=n_out_used)


def verify_combined(results: list, max_verify: int) -> None:
    """Part C: combine the ALREADY-trained per-modality nets onto one chip,
    quantize as ONE unit, and confirm each modality's decisions still agree
    with its own independently-verified standalone float model — i.e. that
    co-residence + shared quantization doesn't corrupt either modality.

    Uses `global_quantize` (the Part A default). `channel_quantize` was A/B'd
    here too, on the theory that two independently-trained sub-nets rarely
    share a weight scale (the exact failure mode the task doc names for this
    step) — but it didn't reliably help: on ECG+PPG it left ECG's combined
    agreement unchanged (0.630) and made PPG's *worse* (0.573 vs 0.650 under
    global_quantize). Combining two sub-nets under a fixed 8-bit weight
    budget measurably degrades fidelity vs. either standalone net regardless
    of quantization method here — a genuine finding to report, not a bug to
    chase further; see the run report for both modalities' numbers.
    """
    nets = [r["net"] for r in results]
    combined, n_ins, _n_hiddens, n_outs = rm.build_combined_xylo_snn(nets)
    spec = rm.map_and_quantize(combined)
    _config, is_valid, msg = rm._xylo_support().config_from_specification(**spec)
    print(f"\n[xylo] combined: config_from_specification is_valid={is_valid} ({msg})")

    sim, _config = rm.to_xylo_sim(spec)
    total_in = sum(n_ins)

    print(f"\n{' ONE-CHIP CO-RESIDENCE CHECK ':=^60}")
    in_off = out_off = 0
    for result, n_in, n_out in zip(results, n_ins, n_outs):
        Rte, float_preds = result["Rte"], result["float_preds"]
        out_slice = slice(out_off, out_off + n_out)
        n_verify = min(max_verify, Rte.shape[0])

        agree = 0
        for i in range(n_verify):
            raster = Rte[i].numpy()
            padded = np.zeros((raster.shape[0], total_in), dtype="float32")
            padded[:, in_off:in_off + n_in] = raster

            sim_out, _, _ = sim(padded)

            pred_combined_xylo = int(
                np.asarray(sim_out)[:, out_slice].sum(axis=0).argmax())
            agree += int(pred_combined_xylo == int(float_preds[i]))

        agreement_vs_standalone = agree / n_verify
        print(f"{result['modality']}: combined-XyloSim vs. standalone-float "
              f"agreement = {agreement_vs_standalone:.3f}  "
              f"({n_verify} windows)")
        result["combined_agreement_vs_standalone"] = agreement_vs_standalone
        in_off += n_in
        out_off += n_out
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Per-modality XyloSim verification")
    ap.add_argument("--modality", choices=["ecg", "ppg", "both"], default="both")
    ap.add_argument("--real", action="store_true",
                     help="use real data (MIT-BIH for ecg, BIDMC for ppg)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-hidden", type=int, default=63)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--spike-reg", type=float, default=2e-2)
    ap.add_argument("--n-restarts", type=int, default=5)
    ap.add_argument("--max-verify", type=int, default=300,
                     help="cap on how many held-out windows to run through XyloSim")
    ap.add_argument("--no-combined", action="store_true",
                     help="skip the Part C one-chip co-residence check")
    args = ap.parse_args()

    modalities = ["ecg", "ppg"] if args.modality == "both" else [args.modality]

    results = []
    for modality in modalities:
        result = train_modality(
            modality, real=args.real, epochs=args.epochs, n_hidden=args.n_hidden,
            threshold=args.threshold, batch_size=128, lr=1e-2,
            spike_reg=args.spike_reg, seed=0, n_restarts=args.n_restarts)
        result = verify_modality(result, max_verify=args.max_verify)
        results.append(result)

    mods = [report_footprint(r) for r in results]
    print(f"\n{xb.fits_one_chip(mods)}")

    if len(results) == 2 and not args.no_combined:
        verify_combined(results, max_verify=args.max_verify)


if __name__ == "__main__":
    main()
