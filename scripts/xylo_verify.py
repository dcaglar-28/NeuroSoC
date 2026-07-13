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
    python scripts/xylo_verify.py --modality ppg --real --ppg-source vitaldb \
        --n-seeds 5                                    # real VitalDB blood-loss label, multi-seed
        (needs `pip install "eia[data]"`; see docs/vitaldb_ppg_hemorrhage_task.md)
    python scripts/xylo_verify.py --modality heart --real --n-seeds 5
        # real PhysioNet/CinC 2016 heart sounds, multi-seed — see docs/heart_sounds_task.md
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

from eia import case_level, encoding, report, rockpool_models as rm, xylo_budget as xb
from eia.datasets import load_ecg, load_heart, load_ppg, load_ppg_vitaldb

_LOADERS = {"ecg": load_ecg, "ppg": load_ppg, "heart": load_heart}
_PPG_SOURCE_LOADERS = {"bidmc": load_ppg, "vitaldb": load_ppg_vitaldb}
_HIDDEN_KEY = "1_LIFBitshiftTorch"
_OUT_KEY = "3_LIFBitshiftTorch"


def _encode_batch(X: np.ndarray, threshold: float) -> np.ndarray:
    """Delta-encode each window into a Xylo input raster.

    2-D X (n, window) — single-channel ECG/PPG/heart -> (n, window, 2) (one
    ON/OFF pair). 3-D X (n, channels, window) — a future multi-channel
    modality -> each channel gets its own delta-encoded ON/OFF pair,
    concatenated along the channel axis -> (n, window, 2*channels). Must
    stay <= 16 (Xylo's input-channel ceiling) — the caller picks `channels`
    accordingly. Currently dead for every registered modality (all are 2-D).
    """
    if X.ndim == 2:
        out = np.empty((X.shape[0], X.shape[1], 2), dtype=np.float32)
        for i in range(X.shape[0]):
            enc = encoding.delta_encode_2ch(encoding.normalize(X[i]), threshold)
            out[i] = rm.to_input_raster(enc)
        return out
    if X.ndim == 3:
        n, n_ch, window = X.shape
        out = np.empty((n, window, 2 * n_ch), dtype=np.float32)
        for i in range(n):
            chans = [rm.to_input_raster(
                encoding.delta_encode_2ch(encoding.normalize(X[i, c]), threshold))
                for c in range(n_ch)]
            out[i] = np.concatenate(chans, axis=1)  # (window, 2) x n_ch -> (window, 2*n_ch)
        return out
    raise ValueError(f"unsupported X.ndim={X.ndim} (expected 2 or 3)")


def per_class_recall(preds, y_t, n_classes: int) -> list:
    """Recall for each class: fraction of that class's true examples the
    model got right. Plain accuracy hides imbalance (a model that always
    predicts the majority class scores high accuracy but 0 recall on the
    minority) — recall per class is what actually shows that."""
    recalls = []
    for c in range(n_classes):
        mask = y_t == c
        n_c = int(mask.sum().item())
        recalls.append(float((preds[mask] == c).float().mean().item()) if n_c else float("nan"))
    return recalls


def balanced_accuracy(preds, y_t, n_classes: int) -> float:
    """Mean per-class recall — the standard imbalance-robust accuracy metric.
    Unlike raw accuracy, a majority-only classifier scores ~1/n_classes here,
    not ~0.92 on a 92/8 split."""
    recalls = [r for r in per_class_recall(preds, y_t, n_classes) if not np.isnan(r)]
    return float(np.mean(recalls)) if recalls else 0.0


def _class_weights(y_t, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss, computed from the
    training split. A majority class at 92.3% otherwise dominates the
    (unweighted) gradient enough that this net's checkpoints never escape the
    majority-baseline solution — confirmed empirically on real MIT-BIH (5
    restarts x 15 epochs, always exactly 0.923 = the base rate)."""
    counts = torch.bincount(y_t, minlength=n_classes).float().clamp(min=1)
    weight = counts.sum() / (n_classes * counts)
    return weight


def _margin_loss(vmem_mean: torch.Tensor, y: torch.Tensor, lossfn: nn.Module) -> torch.Tensor:
    """Auxiliary loss pushing the true class's output membrane potential
    clear of the runner-up — the docs/ecg_quant_diagnosis.md finding is that
    float-vs-XyloSim disagreements concentrate on samples where the float
    model's own Vmem margin (winner minus runner-up) is already small, i.e.
    quantization noise flips only the *already-close* decisions. Reusing the
    same class-weighted CrossEntropyLoss on Vmem (mean over time, not sum —
    keeps the logit scale sane for softmax) is a standard, scale-robust way
    to widen that margin: softmax-CE directly penalizes small margins, and
    training on it as a small AUXILIARY term (not the primary objective —
    that stays the spike-count readout XyloSim actually uses) avoids the
    earlier-found failure mode of training on Vmem as the primary signal
    (float accuracy soared but XyloSim agreement got much worse, since a
    vmem-primary net can "hide" its decision below spiking threshold in a
    way that doesn't survive quantization)."""
    vmem_mean = vmem_mean.mean(dim=1)  # (batch, time, n_out) -> (batch, n_out)
    return lossfn(vmem_mean, y)


def _eval(net, R, y_t, n_classes: int):
    net.reset_state()
    with torch.no_grad():
        out, _state, rec = net(R, record=True)
    logits = out.sum(dim=1)  # (n, n_classes) summed output spikes
    preds = logits.argmax(dim=1)
    acc = (preds == y_t).float().mean().item()
    bal_acc = balanced_accuracy(preds, y_t, n_classes)
    recalls = per_class_recall(preds, y_t, n_classes)
    return acc, bal_acc, recalls, rec[_HIDDEN_KEY]["spikes"].mean().item(), preds, logits


def train_modality(modality: str, real: bool, epochs: int, n_hidden: int,
                    threshold: float, batch_size: int, lr: float,
                    spike_reg: float, seed: int, n_restarts: int,
                    require_real: bool = False, loader_kwargs: dict | None = None,
                    margin_reg: float = 0.0, bias_reg: float = 0.0,
                    loader=None, split_fn=None, card_verbose: bool = True):
    """Part A (train half): data card + multi-restart training with
    balanced-accuracy checkpoint selection (see build_xylo_snn for why this
    net needs restarts). Returns the trained net plus held-out tensors,
    reused both for this modality's own XyloSim check and Part C's combined
    check.

    Loss is class-weighted (inverse training-split frequency) and checkpoints
    are selected by balanced accuracy, not raw accuracy: on real MIT-BIH ECG
    (92.3% majority class), unweighted CE + raw-accuracy selection converged
    to the exact majority-baseline accuracy every single restart (5 restarts
    x 15 epochs) — raw accuracy can't tell a majority-collapsed solution from
    a genuinely discriminative one when one class is this rare, so neither
    the loss nor the selection metric can be allowed to only look at it.

    `margin_reg` and `bias_reg` are two of the fixes from
    `docs/ecg_quant_diagnosis.md`, both off by default: `margin_reg` adds the
    Vmem-margin auxiliary loss (see `_margin_loss`); `bias_reg` penalizes the
    output layer's bias magnitude (L2), targeting the diagnosed real-MIT-BIH-
    specific failure where that bias is a scale outlier that wastes ~51% of
    the output layer's 8-bit weight range in `global_quantize`. EXPERIMENTAL:
    at full training budget these did not reliably improve XyloSim agreement
    (see `docs/ecg_quant_fixes_results.md`) — real-MIT-BIH agreement turned
    out to be highly sensitive to the specific trained checkpoint, and the
    regularizer weights that helped at a shorter calibration budget did not
    transfer to a longer one. Not recommended as defaults without a
    multi-seed validation pass; kept available for further experimentation.
    """
    loader = loader or _LOADERS[modality]
    data = loader(prefer_real=real, require_real=require_real,
                  **(loader_kwargs or {}))
    card = report.data_card(data, verbose=card_verbose)
    # Guards the exact bug this repo hit once already: a data object loaded
    # for one modality silently fed to training/reporting for another (see
    # notebooks/01_ecg_snn.ipynb history). Cheap and always-on.
    report.assert_provenance(card, data, modality)
    n_classes = int(np.asarray(data.y).max()) + 1

    # Held out for final reporting — never used for model selection below.
    # Case-grouped when `data.groups` is set (VitalDB subject-independent),
    # plain-stratified otherwise — see `eia.case_level.split_data`. Callers
    # needing a different split pass `split_fn(data, seed) -> same 9-tuple
    # shape` to override this.
    split_fn = split_fn or case_level.split_data
    Xtr, Xval, Xte, ytr, yval, yte, _groups_tr, _groups_val, _groups_te = \
        split_fn(data, seed)

    # Feature front-end (docs/heart_sounds_task.md's escalation path,
    # originally built for EEG's — retired — feature front-end): X holds RAW
    # feature values out of the loader (line length is in raw signal units,
    # band power is a [0,1] fraction, spectral entropy is [0,1] -- wildly
    # different scales). Z-score fit on Xtr ONLY, applied to Xval/Xte, here
    # (post-split) rather than in the loader (pre-split) -- fitting on the
    # loader's full pool would leak val/test statistics into training.
    if getattr(data, "frontend", "raw") == "features":
        from eia import signal_features
        Xtr, Xval, Xte = signal_features.normalize_features_train_only(Xtr, Xval, Xte)

    Rtr = torch.tensor(_encode_batch(Xtr, threshold))
    Rval = torch.tensor(_encode_batch(Xval, threshold))
    Rte = torch.tensor(_encode_batch(Xte, threshold))
    ytr_t = torch.tensor(ytr)
    yval_t = torch.tensor(yval)
    yte_t = torch.tensor(yte)
    print(f"[encoding] {modality}: mean input event rate = "
          f"{float((Rtr != 0).float().mean()):.3f}")

    n_in = Rtr.shape[-1]  # 2 for single-channel ECG/PPG/heart, 2*n_channels for a multi-channel modality
    class_weight = _class_weights(ytr_t, n_classes)
    print(f"[train] {modality}: class weights = {class_weight.tolist()}")
    lossfn = nn.CrossEntropyLoss(weight=class_weight)
    n = Rtr.shape[0]
    best_val_bal_acc, best_state = -1.0, None
    for restart in range(n_restarts):
        torch.manual_seed(seed * 100 + restart)
        net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=n_classes, n_in=n_in)
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
                if margin_reg > 0:
                    loss = loss + margin_reg * _margin_loss(
                        rec[_OUT_KEY]["vmem"], ytr_t[bidx], lossfn)
                if bias_reg > 0:
                    loss = loss + bias_reg * (net[3].bias ** 2)
                loss.backward()
                opt.step()
            _val_acc, val_bal_acc, _val_recalls, _, _, _val_logits = \
                _eval(net, Rval, yval_t, n_classes)
            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_state = copy.deepcopy(net.state_dict())
        print(f"[train] {modality} restart {restart}: "
              f"best-so-far val balanced accuracy = {best_val_bal_acc:.3f}")

    net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=n_classes, n_in=n_in)
    net.load_state_dict(best_state)
    print(f"[train] {modality}: final best val balanced accuracy = {best_val_bal_acc:.3f}")

    float_acc, float_bal_acc, float_recalls, mean_hidden_rate, float_preds, float_logits = \
        _eval(net, Rte, yte_t, n_classes)
    spec = rm.map_and_quantize(net)
    _config, is_valid, msg = rm._xylo_support().config_from_specification(**spec)
    print(f"[xylo] {modality}: config_from_specification is_valid={is_valid} ({msg})")

    # Real-world seconds spanned by one window, from the (possibly resampled)
    # window length and the data's own (possibly rescaled) fs — generic
    # across modalities; only `_binary_extra_metrics`'s false-alarms-per-hour
    # metric uses it (eeg originally, now heart too).
    window_sec = float(Xte.shape[-1] / data.fs)

    return {
        "modality": modality, "source": data.source, "net": net, "spec": spec,
        "n_hidden": n_hidden, "n_out": n_classes, "n_in": n_in,
        "n_samples": int(data.X.shape[0]), "provenance": card.provenance,
        "requested_real": bool(data.requested_real), "window_sec": window_sec,
        "Rte": Rte, "yte_t": yte_t, "float_preds": float_preds,
        "float_logits": float_logits,
        "float_acc": float_acc, "float_bal_acc": float_bal_acc,
        "float_recalls": float_recalls, "mean_hidden_rate": mean_hidden_rate,
        "is_valid": is_valid,
    }


def _binary_extra_metrics(y_t, preds, scores: np.ndarray, window_sec: float) -> dict:
    """Sensitivity/specificity/AUROC/AUPRC/false-alarms-per-hour for an
    imbalanced binary modality (originally added for EEG seizure detection,
    docs/eeg_seizure_task.md; reused by heart-sound abnormal/normal since
    that's imbalanced too, ~20.5% abnormal) — reported INSTEAD OF accuracy
    since per-class recall alone doesn't give a threshold-free ranking
    measure, hence AUROC/AUPRC. `scores` is a continuous positive-class score
    (softmax probability for the float model, normalized spike-count share
    for XyloSim, since XyloSim has no continuous membrane readout) used only
    for AUROC/AUPRC ranking, not the classification decision itself.
    false-alarms/hour assumes a continuous monitoring stream (meaningful for
    EEG; printed for heart too but less clinically load-bearing there — a
    single per-recording classification, not continuous monitoring).
    """
    y_np = y_t.numpy() if hasattr(y_t, "numpy") else np.asarray(y_t)
    preds_np = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
    n_pos, n_neg = int((y_np == 1).sum()), int((y_np == 0).sum())
    sensitivity = float(((preds_np == 1) & (y_np == 1)).sum() / n_pos) if n_pos else float("nan")
    specificity = float(((preds_np == 0) & (y_np == 0)).sum() / n_neg) if n_neg else float("nan")
    fp = int(((preds_np == 1) & (y_np == 0)).sum())
    neg_hours = (n_neg * window_sec) / 3600.0
    fa_per_hour = float(fp / neg_hours) if neg_hours > 0 else float("nan")
    auroc = auprc = float("nan")
    if len(np.unique(y_np)) > 1:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(y_np, scores))
        auprc = float(average_precision_score(y_np, scores))
    return {"sensitivity": sensitivity, "specificity": specificity,
            "auroc": auroc, "auprc": auprc, "fa_per_hour": fa_per_hour}


def verify_modality(result: dict, max_verify: int) -> dict:
    """Part A (verify half): run XyloSim over held-out windows, print report.

    Operates purely on the `result` dict `train_modality` produced — the
    trained `net`, its `spec`, and the held-out `Rte`/`yte_t` all came from
    the one `data` object `train_modality` loaded and already checked with
    `report.assert_provenance`, so there is no second load here that could
    diverge from it. This function only reports; it doesn't re-fetch data.
    """
    net, spec, Rte, yte_t = result["net"], result["spec"], result["Rte"], result["yte_t"]
    n_classes = result["n_out"]
    n_verify = min(max_verify, Rte.shape[0])
    matches = 0
    xylo_preds = torch.empty(n_verify, dtype=torch.long)
    xylo_out = np.empty((n_verify, n_classes), dtype=np.float64)
    for i in range(n_verify):
        res = rm.verify_against_sim(net, spec, Rte[i].numpy())
        matches += int(res["match"])
        xylo_preds[i] = res["pred_xylo"]
        xylo_out[i] = res["out_xylo"]
    yte_slice = yte_t[:n_verify]
    xylo_acc = (xylo_preds == yte_slice).float().mean().item()
    xylo_bal_acc = balanced_accuracy(xylo_preds, yte_slice, n_classes)
    xylo_recalls = per_class_recall(xylo_preds, yte_slice, n_classes)
    agreement_rate = matches / n_verify

    modality, source = result["modality"], result["source"]
    header = f" XYLO VERIFICATION -- {modality.upper()} ({source}) "
    print(f"\n{header:=^60}")
    print(f"Modality / source        : {modality} / {source}")
    print(f"Provenance               : {result['provenance']}")
    print(f"Total samples (dataset)  : {result['n_samples']}")
    print(f"Held-out windows verified: {n_verify} / {Rte.shape[0]}")
    print(f"Float model accuracy     : {result['float_acc']:.3f}  "
          f"(balanced: {result['float_bal_acc']:.3f})")
    print(f"Float per-class recall   : {[f'{r:.3f}' for r in result['float_recalls']]}")
    print(f"XyloSim accuracy         : {xylo_acc:.3f}  (balanced: {xylo_bal_acc:.3f})")
    print(f"XyloSim per-class recall : {[f'{r:.3f}' for r in xylo_recalls]}")
    print(f"Float vs. XyloSim agree  : {agreement_rate:.3f}")
    print(f"Mean hidden spike rate   : {result['mean_hidden_rate']:.3f}")

    result.update(xylo_acc=xylo_acc, xylo_bal_acc=xylo_bal_acc,
                   xylo_recalls=xylo_recalls, agreement_rate=agreement_rate,
                   n_verify=n_verify)

    if modality == "heart" and n_classes == 2:
        # Accuracy is misleading under heart's moderate imbalance (~20.5%
        # abnormal) — report the threshold-free ranking metrics instead
        # (docs/heart_sounds_task.md; this helper originally shipped for
        # EEG's more extreme imbalance, see _binary_extra_metrics).
        window_sec = result["window_sec"]
        float_probs = torch.softmax(
            result["float_logits"][:n_verify], dim=1)[:, 1].numpy()
        xylo_scores = xylo_out[:, 1] / (xylo_out.sum(axis=1) + 1e-8)
        float_extra = _binary_extra_metrics(yte_slice, result["float_preds"][:n_verify],
                                             float_probs, window_sec)
        xylo_extra = _binary_extra_metrics(yte_slice, xylo_preds, xylo_scores, window_sec)
        print(f"Float  sens/spec/AUROC/AUPRC/FA-per-hr: "
              f"{float_extra['sensitivity']:.3f} / {float_extra['specificity']:.3f} / "
              f"{float_extra['auroc']:.3f} / {float_extra['auprc']:.3f} / "
              f"{float_extra['fa_per_hour']:.2f}")
        print(f"XyloSim sens/spec/AUROC/AUPRC/FA-per-hr: "
              f"{xylo_extra['sensitivity']:.3f} / {xylo_extra['specificity']:.3f} / "
              f"{xylo_extra['auroc']:.3f} / {xylo_extra['auprc']:.3f} / "
              f"{xylo_extra['fa_per_hour']:.2f}")
        result.update(float_extra_metrics=float_extra, xylo_extra_metrics=xylo_extra)

    print("=" * 60)
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


def _print_multiseed_summary(modality: str, seed_results: list) -> None:
    """Mean +/- std over seeds for float/XyloSim balanced accuracy, per-class
    recall, and agreement. The ECG quantization-fidelity work
    (docs/ecg_quant_fixes_results.md) found a single seed's XyloSim agreement
    is not reliable evidence — the identical config swung from 0.883 to 0.333
    agreement between runs — so any dataset this matters for reports a seed
    band here, not a point estimate."""
    n = len(seed_results)

    def _stat(key):
        vals = np.array([r[key] for r in seed_results], dtype=float)
        return vals.mean(), vals.std()

    def _stat_recalls(key):
        arr = np.array([r[key] for r in seed_results], dtype=float)  # (n_seeds, n_classes)
        return arr.mean(axis=0), arr.std(axis=0)

    fb_m, fb_s = _stat("float_bal_acc")
    xb_m, xb_s = _stat("xylo_bal_acc")
    ag_m, ag_s = _stat("agreement_rate")
    fr_m, fr_s = _stat_recalls("float_recalls")
    xr_m, xr_s = _stat_recalls("xylo_recalls")

    header = f" MULTI-SEED SUMMARY -- {modality.upper()} ({seed_results[0]['source']}) "
    print(f"\n{header:=^60}")
    print(f"n_seeds                  : {n}")
    print(f"Float balanced acc       : {fb_m:.3f} +/- {fb_s:.3f}")
    print(f"Float per-class recall   : "
          f"{[f'{m:.3f}+/-{s:.3f}' for m, s in zip(fr_m, fr_s)]}")
    print(f"XyloSim balanced acc     : {xb_m:.3f} +/- {xb_s:.3f}")
    print(f"XyloSim per-class recall : "
          f"{[f'{m:.3f}+/-{s:.3f}' for m, s in zip(xr_m, xr_s)]}")
    print(f"Float vs. XyloSim agree  : {ag_m:.3f} +/- {ag_s:.3f}")

    if "float_extra_metrics" in seed_results[0]:
        def _stat_extra(net_key, metric_key):
            vals = np.array([r[net_key][metric_key] for r in seed_results], dtype=float)
            return float(np.nanmean(vals)), float(np.nanstd(vals))

        for net_key, label in (("float_extra_metrics", "Float"),
                                ("xylo_extra_metrics", "XyloSim")):
            sens_m, sens_s = _stat_extra(net_key, "sensitivity")
            spec_m, spec_s = _stat_extra(net_key, "specificity")
            auroc_m, auroc_s = _stat_extra(net_key, "auroc")
            auprc_m, auprc_s = _stat_extra(net_key, "auprc")
            fa_m, fa_s = _stat_extra(net_key, "fa_per_hour")
            print(f"{label:8} sensitivity/specificity : "
                  f"{sens_m:.3f}+/-{sens_s:.3f} / {spec_m:.3f}+/-{spec_s:.3f}")
            print(f"{label:8} AUROC/AUPRC             : "
                  f"{auroc_m:.3f}+/-{auroc_s:.3f} / {auprc_m:.3f}+/-{auprc_s:.3f}")
            print(f"{label:8} false alarms/hour       : {fa_m:.2f}+/-{fa_s:.2f}")
    print("=" * 60)



def main():
    ap = argparse.ArgumentParser(description="Per-modality XyloSim verification")
    ap.add_argument("--modality", choices=["ecg", "ppg", "heart", "both"], default="both")
    ap.add_argument("--real", action="store_true",
                     help="use real data (MIT-BIH for ecg, BIDMC for ppg, "
                          "CinC 2016 for heart)")
    ap.add_argument("--require-real", action="store_true",
                     help="raise instead of silently falling back to synthetic "
                          "if real data fails to load (implies --real)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-hidden", type=int, default=63)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--spike-reg", type=float, default=2e-2)
    ap.add_argument("--n-restarts", type=int, default=5)
    ap.add_argument("--max-verify", type=int, default=300,
                     help="cap on how many held-out windows to run through XyloSim")
    ap.add_argument("--no-combined", action="store_true",
                     help="skip the Part C one-chip co-residence check")
    ap.add_argument("--window", type=int, default=None,
                     help="override the generator's/capture window length. For "
                          "synthetic data this IS the Xylo timestep count. For "
                          "real MIT-BIH this is the NATIVE (360 Hz) capture "
                          "length in samples — a physiological duration, not a "
                          "timestep budget; pair with --resample-to to pick the "
                          "Xylo timestep count independently (see "
                          "docs/ecg_quant_diagnosis.md and datasets.load_mitbih).")
    ap.add_argument("--resample-to", type=int, default=None,
                     help="real MIT-BIH only: FFT-resample each captured beat "
                          "down to this many Xylo timesteps, rescaling fs to "
                          "match — the fs-matched fix for the window/timestep "
                          "vs. XyloSim-agreement trade-off (see "
                          "docs/ecg_quant_diagnosis.md). No effect on synthetic "
                          "data or on PPG. EXPERIMENTAL: a reduced-budget sweep "
                          "showed a large win (resample_to=187) but it did not "
                          "reproduce at full training budget in a single seed=0 "
                          "run — see docs/ecg_quant_fixes_results.md before "
                          "trusting any one result.")
    ap.add_argument("--margin-reg", type=float, default=0.0,
                     help="weight on the Vmem-margin auxiliary loss (0 = off). "
                          "Targets the diagnosed fragility: float-vs-XyloSim "
                          "disagreements concentrate on low-margin decisions. "
                          "EXPERIMENTAL — see docs/ecg_quant_fixes_results.md.")
    ap.add_argument("--bias-reg", type=float, default=0.0,
                     help="weight on an L2 penalty on the output layer's bias "
                          "(0 = off). Targets the diagnosed real-MIT-BIH-"
                          "specific fault where that bias is a scale outlier "
                          "wasting ~51%% of the output layer's 8-bit range. "
                          "EXPERIMENTAL — did not transfer across training "
                          "budgets, see docs/ecg_quant_fixes_results.md.")
    ap.add_argument("--ppg-source", choices=["bidmc", "vitaldb"], default="bidmc",
                     help="which real dataset backs the ppg modality (ignored "
                          "for ecg). vitaldb = case-level intraop_ebl blood-"
                          "loss label (see docs/vitaldb_ppg_hemorrhage_task.md); "
                          "bidmc = SpO2-desaturation proxy (unchanged default).")
    ap.add_argument("--max-cases", type=int, default=None,
                     help="vitaldb only: subset of qualifying cases to pull "
                          "(default: load_vitaldb_ppg's own default of 30).")
    ap.add_argument("--n-seeds", type=int, default=1,
                     help="repeat train+verify over this many seeds and report "
                          "mean +/- std float/XyloSim balanced acc, per-class "
                          "recall, and agreement — a single seed's XyloSim "
                          "agreement is not reliable evidence on its own (see "
                          "docs/ecg_quant_fixes_results.md's checkpoint-"
                          "sensitivity finding).")
    ap.add_argument("--heart-frontend", choices=["raw", "features"], default="features",
                     help="heart only: 'features' (default — line length / "
                          "relative band power / spectral entropy per "
                          "sub-window at/near native 2000 Hz) or 'raw' "
                          "(delta-encoded waveform, downsampled to a few "
                          "Xylo timesteps -- measured to band-limit away the "
                          "20-400+ Hz heart-sound content: ~chance, see "
                          "docs/heart_sounds_results.md; kept selectable "
                          "for A/B, not the default).")
    args = ap.parse_args()
    real = args.real or args.require_real
    loader_kwargs = {}
    if args.window is not None:
        loader_kwargs["window"] = args.window
    if args.resample_to is not None:
        loader_kwargs["resample_to"] = args.resample_to

    ppg_loader_kwargs = dict(loader_kwargs)
    if args.ppg_source == "vitaldb":
        ppg_loader_kwargs.pop("window", None)  # vitaldb uses window_sec, not window
        if args.max_cases is not None:
            ppg_loader_kwargs["max_cases"] = args.max_cases

    heart_loader_kwargs = dict(loader_kwargs)
    heart_loader_kwargs.pop("window", None)  # cinc2016 uses window_sec, not window
    heart_loader_kwargs["heart_frontend"] = args.heart_frontend  # always explicit, never relies
                                                                  # on load_cinc2016's own default
    if args.heart_frontend == "features":
        heart_loader_kwargs.pop("resample_to", None)  # features use n_subwindows, not resample_to

    n_seeds = max(1, args.n_seeds)

    modalities = ["ecg", "ppg"] if args.modality == "both" else [args.modality]

    results = []
    for modality in modalities:
        loader, mod_loader_kwargs = None, loader_kwargs
        if modality == "ppg":
            loader = _PPG_SOURCE_LOADERS[args.ppg_source]
            mod_loader_kwargs = ppg_loader_kwargs
        elif modality == "heart":
            mod_loader_kwargs = heart_loader_kwargs

        seed_results = []
        for s in range(n_seeds):
            result = train_modality(
                modality, real=real, epochs=args.epochs, n_hidden=args.n_hidden,
                threshold=args.threshold, batch_size=128, lr=1e-2,
                spike_reg=args.spike_reg, seed=s, n_restarts=args.n_restarts,
                require_real=args.require_real, loader_kwargs=mod_loader_kwargs,
                margin_reg=args.margin_reg, bias_reg=args.bias_reg, loader=loader)
            result = verify_modality(result, max_verify=args.max_verify)
            seed_results.append(result)

        if n_seeds > 1:
            _print_multiseed_summary(modality, seed_results)
        results.append(seed_results[-1])

    mods = [report_footprint(r) for r in results]
    print(f"\n{xb.fits_one_chip(mods)}")

    if len(results) == 2 and not args.no_combined:
        verify_combined(results, max_verify=args.max_verify)


if __name__ == "__main__":
    main()
