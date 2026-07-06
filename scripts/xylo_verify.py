"""Pre-hardware acceptance check: train the PPG SNN, deploy it through the
Rockpool -> Xylo pipeline, and confirm the quantized network agrees with the
float model on XyloSim (the bit-precise simulator). See
`docs/xylo_verification_task.md` for the full task spec.

Run from the repo root (needs `pip install "eia[xylo]"`):

    python scripts/xylo_verify.py                 # synthetic PPG, quick
    python scripts/xylo_verify.py --real          # real BIDMC PPG (needs wfdb + network)
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from eia import encoding, rockpool_models as rm
from eia.datasets import load_ppg


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
    acc = (out.sum(dim=1).argmax(dim=1) == y_t).float().mean().item()
    return acc, rec["1_LIFBitshiftTorch"]["spikes"].mean().item()


def run(real: bool = False, epochs: int = 40, n_hidden: int = 63,
        threshold: float = 0.25, batch_size: int = 128, lr: float = 1e-2,
        spike_reg: float = 2e-2, max_verify: int = 300, seed: int = 0,
        n_restarts: int = 5):
    np.random.seed(seed)

    data = load_ppg(prefer_real=real)
    print(f"[data] source={data.source}  X={data.X.shape}  "
          f"pos_frac={data.y.mean():.2f}  fs={data.fs}Hz")

    # Held out for final reporting (float acc, XyloSim acc/agreement) — never
    # used for model selection below.
    Xfit, Xte, yfit, yte = train_test_split(
        data.X, data.y, test_size=0.25, random_state=seed, stratify=data.y)
    # A further split off the fit set, used only to pick the best training
    # checkpoint (see note below on why this net needs that).
    Xtr, Xval, ytr, yval = train_test_split(
        Xfit, yfit, test_size=0.2, random_state=seed, stratify=yfit)

    Rtr = torch.tensor(_encode_batch(Xtr, threshold))
    Rval = torch.tensor(_encode_batch(Xval, threshold))
    Rte = torch.tensor(_encode_batch(Xte, threshold))
    ytr_t = torch.tensor(ytr)
    yval_t = torch.tensor(yval)
    yte_t = torch.tensor(yte)
    print(f"[encoding] mean input event rate = {float((Rtr != 0).float().mean()):.3f}")

    # ------------------------------------------------------------------ #
    # Train the Xylo-mappable net (Rockpool Torch backend).
    #
    # This 2-output-neuron, no-per-linear-bias readout is prone to a
    # dead-neuron collapse (see build_xylo_snn) that a positive initial bias
    # avoids but doesn't fully cure — training is noisy, seed-sensitive, and
    # can wander back toward the majority-class solution in later epochs.
    # We restart from several initialisations and, within each, checkpoint on
    # a held-out val split (disjoint from the final test set); the best
    # val-accuracy checkpoint across all restarts is kept. This is standard
    # model selection, not test-set tuning — the test set below is untouched
    # until the very end.
    # ------------------------------------------------------------------ #
    lossfn = nn.CrossEntropyLoss()
    n = Rtr.shape[0]
    best_val_acc, best_state = -1.0, None
    for restart in range(n_restarts):
        torch.manual_seed(seed * 100 + restart)
        net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=2)
        # Rockpool overrides `.parameters()` to return a nested Parameter dict
        # (for family-based inspection); go through the torch.nn.Module base
        # method to get the flat iterable of leaf Parameters a torch optimizer
        # needs.
        opt = torch.optim.Adam(torch.nn.Module.parameters(net), lr=lr)
        for ep in range(epochs):
            idx = torch.randperm(n)
            for k in range(0, n, batch_size):
                bidx = idx[k:k + batch_size]
                opt.zero_grad()
                net.reset_state()
                out, _state, rec = net(Rtr[bidx], record=True)
                logits = out.sum(dim=1)  # sum output spikes over time -> (batch, 2)
                spk_rate = rec["1_LIFBitshiftTorch"]["spikes"].mean()
                loss = lossfn(logits, ytr_t[bidx]) + spike_reg * spk_rate
                loss.backward()
                opt.step()
            val_acc, _ = _eval(net, Rval, yval_t)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(net.state_dict())
        print(f"[train] restart {restart}: best-so-far val accuracy = {best_val_acc:.3f}")

    net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=2)
    net.load_state_dict(best_state)
    print(f"[train] final best val accuracy = {best_val_acc:.3f}")

    # ------------------------------------------------------------------ #
    # Float-model test accuracy (best checkpoint, held-out test set).
    # ------------------------------------------------------------------ #
    float_acc, mean_hidden_rate = _eval(net, Rte, yte_t)

    # ------------------------------------------------------------------ #
    # Map -> quantize -> XyloSim, verify per held-out window.
    # ------------------------------------------------------------------ #
    spec = rm.map_and_quantize(net)
    config, is_valid, msg = rm._xylo_support().config_from_specification(**spec)
    print(f"[xylo] config_from_specification is_valid={is_valid} ({msg})")

    n_verify = min(max_verify, Rte.shape[0])
    matches, xylo_correct = 0, 0
    for i in range(n_verify):
        res = rm.verify_against_sim(net, spec, Rte[i].numpy())
        matches += int(res["match"])
        xylo_correct += int(res["pred_xylo"] == int(yte_t[i]))
    agreement_rate = matches / n_verify
    xylo_acc = xylo_correct / n_verify

    print("\n===== XYLO VERIFICATION REPORT =====")
    print(f"Data source              : {data.source}")
    print(f"Held-out windows verified: {n_verify} / {Rte.shape[0]}")
    print(f"Float model test accuracy: {float_acc:.3f}")
    print(f"XyloSim test accuracy    : {xylo_acc:.3f}")
    print(f"Float vs. XyloSim agree  : {agreement_rate:.3f}")
    print(f"Mean hidden spike rate   : {mean_hidden_rate:.3f}")
    print("=====================================")
    return {
        "source": data.source, "float_acc": float_acc, "xylo_acc": xylo_acc,
        "agreement_rate": agreement_rate, "n_verify": n_verify,
    }


def main():
    ap = argparse.ArgumentParser(description="Verify the PPG SNN on XyloSim")
    ap.add_argument("--real", action="store_true", help="use real BIDMC PPG data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-hidden", type=int, default=63)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--spike-reg", type=float, default=2e-2)
    ap.add_argument("--max-verify", type=int, default=300,
                     help="cap on how many held-out windows to run through XyloSim")
    args = ap.parse_args()
    run(real=args.real, epochs=args.epochs, n_hidden=args.n_hidden,
        threshold=args.threshold, spike_reg=args.spike_reg,
        max_verify=args.max_verify)


if __name__ == "__main__":
    main()
