"""Measurement-only diagnostic for the ECG float->XyloSim fidelity gap.

Does NOT change training or model code. Trains (or loads cached) nets via the
existing, unmodified `train_modality`/`verify_modality` from `xylo_verify.py`,
then instruments float vs XyloSim behaviour layer by layer to find WHERE and
WHY they diverge. Writes results to /tmp/eia_diag_cache/results.json for
`docs/ecg_quant_diagnosis.md` to cite.

Run from repo root: python scripts/diagnose_ecg_quant.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from xylo_verify import train_modality, per_class_recall, balanced_accuracy  # noqa: E402

from eia import rockpool_models as rm  # noqa: E402

CACHE_DIR = "/tmp/eia_diag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(CACHE_DIR, "results.json")

HIDDEN_KEY = "1_LIFBitshiftTorch"
OUT_KEY = "3_LIFBitshiftTorch"


# --------------------------------------------------------------------------- #
# Train-or-load cache
# --------------------------------------------------------------------------- #
def get_or_train(key: str, **kwargs):
    path = os.path.join(CACHE_DIR, f"{key}.pt")
    if os.path.exists(path):
        print(f"[cache] loading {key} from {path}")
        blob = torch.load(path, weights_only=False)
    else:
        print(f"[cache] training {key} (not cached yet)...")
        result = train_modality(**kwargs)
        blob = {
            "state_dict": result["net"].state_dict(),
            "n_hidden": result["n_hidden"], "n_out": result["n_out"],
            "Rte": result["Rte"], "yte_t": result["yte_t"],
            "float_preds": result["float_preds"],
            "float_acc": result["float_acc"], "float_bal_acc": result["float_bal_acc"],
            "float_recalls": result["float_recalls"],
            "mean_hidden_rate": result["mean_hidden_rate"],
            "modality": result["modality"], "source": result["source"],
        }
        torch.save(blob, path)
    net = rm.build_xylo_snn(n_hidden=blob["n_hidden"], n_out=blob["n_out"])
    net.load_state_dict(blob["state_dict"])
    return net, blob


# --------------------------------------------------------------------------- #
# 1. Layer-by-layer float vs XyloSim divergence
# --------------------------------------------------------------------------- #
def layer_divergence(net, spec, raster: np.ndarray) -> dict:
    """Run float net + XyloSim on the same raster with record=True; compare
    hidden-layer and output-layer spike trains elementwise (both are
    genuinely integer-valued in this net, so directly comparable — Vmem/Isyn
    are on different numeric scales between float and int and are reported
    separately, dequantized by the same weight scale for a fair look)."""
    sim, _config = rm.to_xylo_sim(spec)

    net.reset_state()
    x = torch.tensor(raster[None, ...], dtype=torch.float32)
    with torch.no_grad():
        out_f, _state, rec_f = net(x, record=True)
    out_s, _state_s, rec_s = sim(raster, record=True)

    hidden_f = rec_f[HIDDEN_KEY]["spikes"][0].detach().numpy()
    hidden_s = rec_s["Spikes"]
    out_f_spikes = out_f[0].detach().numpy()

    hidden_diff = hidden_f != hidden_s
    hidden_diverge_t = np.where(hidden_diff.any(axis=1))[0]
    out_diff = out_f_spikes != out_s
    out_diverge_t = np.where(out_diff.any(axis=1))[0]

    return {
        "n_timesteps": int(hidden_f.shape[0]),
        "hidden_first_divergence_t": int(hidden_diverge_t[0]) if len(hidden_diverge_t) else None,
        "hidden_frac_timesteps_diverging": float(hidden_diff.any(axis=1).mean()),
        "hidden_frac_neuron_timesteps_diverging": float(hidden_diff.mean()),
        "output_first_divergence_t": int(out_diverge_t[0]) if len(out_diverge_t) else None,
        "output_frac_timesteps_diverging": float(out_diff.any(axis=1).mean()),
        "final_pred_float": int(out_f_spikes.sum(axis=0).argmax()),
        "final_pred_xylo": int(out_s.sum(axis=0).argmax()),
    }


# --------------------------------------------------------------------------- #
# 2. Weights-only vs dynamics-only quantization ablation
# --------------------------------------------------------------------------- #
def _quant_dequant(x: np.ndarray, scale: float) -> np.ndarray:
    return np.round(x * scale).astype(np.int64).astype(np.float64) / scale


def _compute_scales(net) -> tuple:
    w_in = net[0].weight.detach().numpy()
    w_out = net[2].weight.detach().numpy()
    bias_h = net[1].bias.detach().numpy()
    bias_o = net[3].bias.detach().numpy()
    max_w = max(np.abs(w_in).max(), np.abs(bias_h).max())
    max_w_out = max(np.abs(w_out).max(), np.abs(bias_o).max())
    return 127.0 / max_w, 127.0 / max_w_out


def build_hybrid(net, mode: str):
    """mode: 'weights_only' (8-bit weights, float dynamics) or
    'dynamics_only' (float weights, 8/16-bit threshold+bias+dash) — isolates
    each quantization source in a plain float Torch forward pass (XyloSim
    itself can't run partial/mixed-precision configs)."""
    scale_in, scale_out = _compute_scales(net)
    hybrid = copy.deepcopy(net)
    with torch.no_grad():
        if mode == "weights_only":
            hybrid[0].weight.data = torch.tensor(
                _quant_dequant(net[0].weight.detach().numpy(), scale_in), dtype=torch.float32)
            hybrid[2].weight.data = torch.tensor(
                _quant_dequant(net[2].weight.detach().numpy(), scale_out), dtype=torch.float32)
        elif mode == "dynamics_only":
            hybrid[1].bias.data = torch.tensor(
                _quant_dequant(net[1].bias.detach().numpy(), scale_in), dtype=torch.float32)
            hybrid[3].bias.data = torch.tensor(
                _quant_dequant(net[3].bias.detach().numpy(), scale_out), dtype=torch.float32)
            hybrid[1].threshold = torch.tensor(
                _quant_dequant(net[1].threshold.detach().numpy(), scale_in), dtype=torch.float32)
            hybrid[3].threshold = torch.tensor(
                _quant_dequant(net[3].threshold.detach().numpy(), scale_out), dtype=torch.float32)
        else:
            raise ValueError(mode)
    return hybrid


def _spike_pred(net, R) -> torch.Tensor:
    net.reset_state()
    with torch.no_grad():
        out, _state, _rec = net(R)
    return out.sum(dim=1).argmax(dim=1)


def ablation(net, Rte, yte_t, n_verify: int, n_classes: int) -> dict:
    R = Rte[:n_verify]
    y = yte_t[:n_verify]
    true_preds = _spike_pred(net, R)

    out = {}
    for mode in ("weights_only", "dynamics_only"):
        hybrid = build_hybrid(net, mode)
        hybrid_preds = _spike_pred(hybrid, R)
        out[mode] = {
            "acc_vs_labels": float((hybrid_preds == y).float().mean().item()),
            "bal_acc_vs_labels": balanced_accuracy(hybrid_preds, y, n_classes),
            "agreement_vs_true_float": float((hybrid_preds == true_preds).float().mean().item()),
        }
    return out


# --------------------------------------------------------------------------- #
# 3. Per-layer weight distribution health
# --------------------------------------------------------------------------- #
def weight_health(w: np.ndarray, scale: float, name: str) -> dict:
    w = w.ravel().astype(np.float64)
    w_int = np.round(w * scale).astype(np.int64)
    dequant = w_int.astype(np.float64) / scale
    quant_err = np.abs(w - dequant)
    std = w.std() + 1e-12
    return {
        "name": name, "n": int(w.size),
        "mean": float(w.mean()), "std": float(w.std()),
        "min": float(w.min()), "max": float(w.max()), "abs_max": float(np.abs(w).max()),
        "skew": float(((w - w.mean()) ** 3).mean() / std ** 3),
        "outlier_frac_beyond_2std": float((np.abs(w - w.mean()) > 2 * std).mean()),
        "mean_quant_error": float(quant_err.mean()),
        "mean_quant_error_pct_of_absmax": float(quant_err.mean() / (np.abs(w).max() + 1e-12) * 100),
        "int8_bins_used": int(len(np.unique(w_int))),
        "int8_range_used_pct": float(np.abs(w_int).max() / 127 * 100),
    }


def weight_report(net) -> dict:
    scale_in, scale_out = _compute_scales(net)
    w_in = net[0].weight.detach().numpy()
    w_out = net[2].weight.detach().numpy()
    return {
        "w_in": weight_health(w_in, scale_in, "w_in (input->hidden)"),
        "w_out": weight_health(w_out, scale_out, "w_out (hidden->output)"),
        "scale_in": float(scale_in), "scale_out": float(scale_out),
    }


# --------------------------------------------------------------------------- #
# 4 & 5. Per-sample margins, agreement, class breakdown (one pass, shared)
# --------------------------------------------------------------------------- #
def per_sample_analysis(net, spec, Rte, yte_t, n_verify: int) -> dict:
    margins, float_preds, xylo_preds, true_classes = [], [], [], []
    for i in range(n_verify):
        raster = Rte[i].numpy()
        net.reset_state()
        x = torch.tensor(raster[None, ...], dtype=torch.float32)
        with torch.no_grad():
            out, _state, rec = net(x, record=True)
        # Decision rule stays spike-count sum (matches the rest of the
        # pipeline); Vmem margin is a SEPARATE fragility diagnostic, not the
        # decision itself, per the requested analysis.
        pred_float = int(out.sum(dim=1).argmax(dim=1).item())
        vmem_out = rec[OUT_KEY]["vmem"][0].sum(dim=0).numpy()
        sorted_vmem = np.sort(vmem_out)[::-1]
        margin = float(sorted_vmem[0] - sorted_vmem[1])

        res = rm.verify_against_sim(net, spec, raster)
        margins.append(margin)
        float_preds.append(pred_float)
        xylo_preds.append(res["pred_xylo"])
        true_classes.append(int(yte_t[i]))

    margins = np.array(margins)
    float_preds = np.array(float_preds)
    xylo_preds = np.array(xylo_preds)
    true_classes = np.array(true_classes)
    agree = float_preds == xylo_preds

    result = {
        "n_verify": n_verify,
        "margin_mean_agree": float(margins[agree].mean()) if agree.any() else None,
        "margin_median_agree": float(np.median(margins[agree])) if agree.any() else None,
        "margin_mean_disagree": float(margins[~agree].mean()) if (~agree).any() else None,
        "margin_median_disagree": float(np.median(margins[~agree])) if (~agree).any() else None,
        "n_agree": int(agree.sum()), "n_disagree": int((~agree).sum()),
        "by_class": {},
    }
    for c in sorted(np.unique(true_classes)):
        mask = true_classes == c
        n_c = int(mask.sum())
        result["by_class"][int(c)] = {
            "n": n_c,
            "agreement_rate": float(agree[mask].mean()) if n_c else None,
            "float_acc": float((float_preds[mask] == c).mean()) if n_c else None,
            "xylo_acc": float((xylo_preds[mask] == c).mean()) if n_c else None,
        }
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_full_diagnosis(key: str, net, blob, n_verify: int = 300, n_layer_samples: int = 20):
    print(f"\n{'=' * 70}\nDIAGNOSING: {key} ({blob['modality']}/{blob['source']})\n{'=' * 70}")
    Rte, yte_t = blob["Rte"], blob["yte_t"]
    n_classes = blob["n_out"]
    n_verify = min(n_verify, Rte.shape[0])
    spec = rm.map_and_quantize(net)

    # 1. layer divergence, averaged over a sample of windows
    layer_stats = [layer_divergence(net, spec, Rte[i].numpy()) for i in range(n_layer_samples)]
    hidden_first = [s["hidden_first_divergence_t"] for s in layer_stats if s["hidden_first_divergence_t"] is not None]
    output_first = [s["output_first_divergence_t"] for s in layer_stats if s["output_first_divergence_t"] is not None]
    n_hidden_never_diverge = sum(1 for s in layer_stats if s["hidden_first_divergence_t"] is None)
    layer_summary = {
        "n_samples": n_layer_samples,
        "n_hidden_never_diverges": n_hidden_never_diverge,
        "hidden_first_divergence_t_mean": float(np.mean(hidden_first)) if hidden_first else None,
        "hidden_first_divergence_t_median": float(np.median(hidden_first)) if hidden_first else None,
        "output_first_divergence_t_mean": float(np.mean(output_first)) if output_first else None,
        "mean_hidden_frac_neuron_timesteps_diverging": float(np.mean([s["hidden_frac_neuron_timesteps_diverging"] for s in layer_stats])),
        "final_pred_disagreement_rate": float(np.mean([s["final_pred_float"] != s["final_pred_xylo"] for s in layer_stats])),
    }
    print(f"[1] layer divergence: hidden first-diverges at t={layer_summary['hidden_first_divergence_t_mean']}, "
          f"{n_hidden_never_diverge}/{n_layer_samples} samples never diverge in hidden layer")

    # 2. ablation
    ablation_stats = ablation(net, Rte, yte_t, n_verify, n_classes)
    print(f"[2] ablation: weights_only agree-with-float={ablation_stats['weights_only']['agreement_vs_true_float']:.3f}, "
          f"dynamics_only agree-with-float={ablation_stats['dynamics_only']['agreement_vs_true_float']:.3f}")

    # 3. weight health
    weight_stats = weight_report(net)
    print(f"[3] weight health: w_in outlier_frac={weight_stats['w_in']['outlier_frac_beyond_2std']:.3f}, "
          f"w_out outlier_frac={weight_stats['w_out']['outlier_frac_beyond_2std']:.3f}")

    # 4 & 5. margins, agreement, class breakdown
    sample_stats = per_sample_analysis(net, spec, Rte, yte_t, n_verify)
    print(f"[4/5] margin (agree)={sample_stats['margin_mean_agree']}, "
          f"margin (disagree)={sample_stats['margin_mean_disagree']}, by_class={sample_stats['by_class']}")

    return {
        "modality": blob["modality"], "source": blob["source"],
        "float_acc": blob["float_acc"], "float_bal_acc": blob["float_bal_acc"],
        "float_recalls": blob["float_recalls"], "mean_hidden_rate": blob["mean_hidden_rate"],
        "window": int(Rte.shape[1]),
        "layer_divergence": layer_summary,
        "ablation": ablation_stats,
        "weights": weight_stats,
        "per_sample": sample_stats,
    }


if __name__ == "__main__":
    print("=== Training / loading the three reference nets ===")
    ecg_real_net, ecg_real = get_or_train(
        "ecg_real", modality="ecg", real=True, require_real=True,
        epochs=15, n_hidden=63, threshold=0.25, batch_size=128, lr=1e-2,
        spike_reg=2e-2, seed=0, n_restarts=5)

    ecg_synth_net, ecg_synth = get_or_train(
        "ecg_synth", modality="ecg", real=False,
        epochs=40, n_hidden=63, threshold=0.25, batch_size=128, lr=1e-2,
        spike_reg=2e-2, seed=0, n_restarts=5)

    ppg_synth_net, ppg_synth = get_or_train(
        "ppg_synth", modality="ppg", real=False,
        epochs=40, n_hidden=63, threshold=0.25, batch_size=128, lr=1e-2,
        spike_reg=2e-2, seed=0, n_restarts=5)

    print("\nAll three reference nets ready. Running full diagnosis...")
    all_results = {}
    all_results["ecg_real"] = run_full_diagnosis("ecg_real", ecg_real_net, ecg_real)
    all_results["ecg_synth"] = run_full_diagnosis("ecg_synth", ecg_synth_net, ecg_synth)
    all_results["ppg_synth"] = run_full_diagnosis("ppg_synth", ppg_synth_net, ppg_synth)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")
