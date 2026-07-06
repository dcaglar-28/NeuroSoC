"""End-to-end Phase-0 demo: train the SNN + baseline on ECG or PPG, compare
accuracy and estimated energy.

Run from the repo root:

    python -m eia.train                        # ECG, synthetic data, quick
    python -m eia.train --modality ppg         # PPG, synthetic data
    python -m eia.train --real                 # try real data (needs wfdb + network)
    python -m eia.train --epochs 15 --device cpu

Everything torch-related lives here and in models.py, imported lazily.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import encoding, energy, report
from .datasets import load_ecg, load_ppg
from .device import get_device
from .models import build_baseline, build_snn

_LOADERS = {"ecg": load_ecg, "ppg": load_ppg}


def _encode_batch(X: np.ndarray, threshold: float) -> np.ndarray:
    """Delta-encode each window into 2 channels. -> (n, 2, window) float32."""
    out = np.empty((X.shape[0], 2, X.shape[1]), dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = encoding.delta_encode_2ch(encoding.normalize(X[i]), threshold)
    return out


def run(real: bool = False, epochs: int = 10, hidden: int = 128,
        timesteps: int = 20, threshold: float = 0.25, batch_size: int = 128,
        lr: float = 1e-3, spike_reg: float = 5e-2, device_pref: str = "auto",
        seed: int = 0, verbose: bool = True, modality: str = "ecg",
        require_real: bool = False):
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(device_pref)
    print(f"[device] {device}")

    data = _LOADERS[modality](prefer_real=real, require_real=require_real)
    card = report.data_card(data, verbose=False)
    # Guards the exact bug this repo hit once already: a data object loaded
    # for one modality silently fed to training/reporting for another (see
    # notebooks/01_ecg_snn.ipynb history). Cheap and always-on, not just in
    # debug builds — this is a correctness invariant, not a bare `assert`.
    report.assert_provenance(card, data, modality)
    print(f"[data] modality={modality}  source={data.source}  "
          f"provenance={card.provenance}  X={data.X.shape}  "
          f"pos_frac={data.y.mean():.2f}  fs={data.fs}Hz")

    Xtr, Xte, ytr, yte = train_test_split(
        data.X, data.y, test_size=0.25, random_state=seed, stratify=data.y)

    # Tensors for the dense baseline (raw windows).
    Xtr_t = torch.tensor(Xtr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    yte_t = torch.tensor(yte, device=device)

    # Spike-encoded tensors for the SNN.
    Xtr_s = torch.tensor(_encode_batch(Xtr, threshold), device=device)
    Xte_s = torch.tensor(_encode_batch(Xte, threshold), device=device)
    print(f"[encoding] mean input event rate = "
          f"{float((Xtr_s != 0).float().mean()):.3f}")

    n_classes = int(data.y.max()) + 1
    window = data.X.shape[1]

    def batches(n):
        idx = torch.randperm(n)
        for k in range(0, n, batch_size):
            yield idx[k:k + batch_size]

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #
    base = build_baseline(window, hidden, n_classes).to(device)
    opt = torch.optim.Adam(base.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        base.train()
        for bidx in batches(Xtr_t.shape[0]):
            opt.zero_grad()
            loss = lossfn(base(Xtr_t[bidx]), ytr_t[bidx])
            loss.backward()
            opt.step()
    base.eval()
    with torch.no_grad():
        base_acc = (base(Xte_t).argmax(1) == yte_t).float().mean().item()

    # ------------------------------------------------------------------ #
    # SNN
    # ------------------------------------------------------------------ #
    net = build_snn(window, hidden, n_classes, timesteps).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        for bidx in batches(Xtr_s.shape[0]):
            opt.zero_grad()
            logits, spk_rate = net(Xtr_s[bidx])
            # Task loss + sparsity penalty: pushing the firing rate down is what
            # earns the event-driven energy advantage.
            loss = lossfn(logits, ytr_t[bidx]) + spike_reg * spk_rate
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    net.eval()
    with torch.no_grad():
        logits, spk_rate = net(Xte_s)
        mean_rate = spk_rate.item()
        snn_acc = (logits.argmax(1) == yte_t).float().mean().item()

    # ------------------------------------------------------------------ #
    # Energy comparison (per single inference)
    # ------------------------------------------------------------------ #
    energy_report = energy.compare(net.layer_sizes, timesteps, avg_spike_rate=mean_rate)

    if verbose:
        print("\n===== RESULTS =====")
        print(f"Baseline (dense)  accuracy : {base_acc:.3f}")
        print(f"SNN (event-driven) accuracy: {snn_acc:.3f}")
        print(f"SNN mean hidden spike rate : {mean_rate:.3f}")
        print(f"SNN train time             : {train_time:.1f}s ({epochs} epochs)")
        print("-------------------")
        print(energy_report)
        print("===================")
    return {
        "baseline_acc": base_acc, "snn_acc": snn_acc, "mean_rate": mean_rate,
        "energy_ratio": energy_report.energy_ratio, "source": data.source,
        "timesteps": timesteps, "threshold": threshold, "modality": modality,
    }


def sweep(real: bool = False, epochs: int = 10, device_pref: str = "auto",
          timesteps_grid=(10, 20, 40), threshold_grid=(0.15, 0.25, 0.4),
          spike_reg: float = 5e-2, modality: str = "ecg",
          require_real: bool = False):
    """Trace the accuracy vs. energy trade-off across encoder/timestep settings.

    This is the core Phase-0 deliverable: show how sparse and how few timesteps
    you can go while keeping diagnostic accuracy — and where the event-driven
    pipeline becomes cheaper than the dense baseline.
    """
    print(f"{'timesteps':>9} {'threshold':>9} {'snn_acc':>8} {'base_acc':>8} "
          f"{'spike_rate':>10} {'energy_x':>9}")
    print("-" * 60)
    rows = []
    for T in timesteps_grid:
        for th in threshold_grid:
            r = run(real=real, epochs=epochs, timesteps=T, threshold=th,
                    spike_reg=spike_reg, device_pref=device_pref, verbose=False,
                    modality=modality, require_real=require_real)
            rows.append(r)
            marker = "  <- SNN cheaper" if r["energy_ratio"] > 1 else ""
            print(f"{T:>9} {th:>9.2f} {r['snn_acc']:>8.3f} {r['baseline_acc']:>8.3f} "
                  f"{r['mean_rate']:>10.3f} {r['energy_ratio']:>8.1f}x{marker}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="EIA Phase-0 SNN demo")
    ap.add_argument("--modality", default="ecg", choices=["ecg", "ppg"],
                    help="which physiological signal to classify")
    ap.add_argument("--real", action="store_true",
                    help="try real data (MIT-BIH for ecg, BIDMC for ppg)")
    ap.add_argument("--require-real", action="store_true",
                    help="raise instead of silently falling back to synthetic "
                         "if real data fails to load (implies --real)")
    ap.add_argument("--sweep", action="store_true",
                    help="trace the accuracy/energy trade-off across settings")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--spike-reg", type=float, default=5e-2,
                    help="sparsity penalty weight (higher = fewer spikes)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()
    real = args.real or args.require_real
    if args.sweep:
        sweep(real=real, epochs=args.epochs, device_pref=args.device,
              spike_reg=args.spike_reg, modality=args.modality,
              require_real=args.require_real)
    else:
        run(real=real, epochs=args.epochs, hidden=args.hidden,
            timesteps=args.timesteps, threshold=args.threshold,
            spike_reg=args.spike_reg, device_pref=args.device,
            modality=args.modality, require_real=args.require_real)


if __name__ == "__main__":
    main()
