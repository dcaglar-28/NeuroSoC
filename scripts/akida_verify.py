"""Pre-hardware acceptance check for the Akida path — ECG only, first slice
(docs/akida_retarget_task.md). Parallels `xylo_verify.py`'s per-modality
train -> quantize -> verify-against-sim -> footprint structure, but for
BrainChip MetaTF (`eia.akida_models`) instead of Rockpool/XyloSim. Reuses
`datasets.load_ecg`/`load_mitbih`, `report`, and `case_level.split_data` —
the exact same data/split/provenance discipline as the Xylo path.

**Linux only** (`akida` has no macOS wheel — see Dockerfile.akida). Run
inside the container:

    scripts/akida_docker_run.sh python scripts/akida_verify.py --real --n-seeds 5
    scripts/akida_docker_run.sh python scripts/akida_verify.py --n-seeds 2   # synthetic, quick

See `src/eia/akida_models.py` for the confirmed Akida v2 layer constraints
(square kernel/stride/pool, valid block-ordering patterns) this script's
model relies on, and `docs/akida_ecg_results.md` for the measured results,
the Xylo-gap comparison, and the Part-0 simulator-fidelity finding.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from eia import case_level, report
from eia.datasets import load_ecg


def per_class_recall(preds: np.ndarray, y: np.ndarray, n_classes: int) -> list:
    """Recall for each class — see `xylo_verify.per_class_recall` (same logic,
    duplicated rather than imported so the Akida and Xylo verify scripts stay
    fully independent scripts, not implicitly coupled through script imports).
    """
    recalls = []
    for c in range(n_classes):
        mask = y == c
        n_c = int(mask.sum())
        recalls.append(float((preds[mask] == c).mean()) if n_c else float("nan"))
    return recalls


def balanced_accuracy(preds: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    recalls = [r for r in per_class_recall(preds, y, n_classes) if not np.isnan(r)]
    return float(np.mean(recalls)) if recalls else 0.0


def _class_weight_dict(y: np.ndarray, n_classes: int) -> dict:
    """Inverse-frequency class weights as a {class: weight} dict — the format
    `tf_keras.Model.fit(class_weight=...)` expects (vs. Xylo path's tensor)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.clip(counts, 1, None)
    weight = counts.sum() / (n_classes * counts)
    return {c: float(weight[c]) for c in range(n_classes)}


def train_and_verify(data, seed: int, epochs: int, qat_epochs: int, n_restarts: int,
                      max_verify: int, weight_bits: int, activation_bits: int) -> dict:
    """Train (float) -> quantize+QAT-fine-tune -> convert -> verify-against-
    Akida-sim for one seed. Restarts pick the best VAL balanced accuracy
    float checkpoint (mirrors `xylo_verify.train_modality`'s restart logic —
    this net is small and can also land in a majority-collapsed local
    optimum, same reason the Xylo path restarts).
    """
    import tensorflow as tf
    import tf_keras

    from eia import akida_models as am

    card = report.data_card(data, verbose=(seed == 0))
    report.assert_provenance(card, data, "ecg")
    n_classes = int(np.asarray(data.y).max()) + 1

    Xtr, Xval, Xte, ytr, yval, yte, _gtr, _gval, _gte = case_level.split_data(data, seed)
    Xtr_u8 = am.to_akida_input(Xtr)
    Xval_u8 = am.to_akida_input(Xval)
    Xte_u8 = am.to_akida_input(Xte)
    class_weight = _class_weight_dict(ytr, n_classes)
    print(f"[train] ecg(akida): class weights = {class_weight}")

    window = Xtr.shape[-1]
    best_val_bal_acc, best_weights = -1.0, None
    for restart in range(n_restarts):
        tf.random.set_seed(seed * 100 + restart)
        model = am.build_akida_model(window=window, n_classes=n_classes)
        model.compile(
            optimizer="adam",
            loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        model.fit(Xtr_u8, ytr, epochs=epochs, batch_size=32, class_weight=class_weight,
                  verbose=0)
        val_preds = np.asarray(model.predict(Xval_u8, verbose=0)).argmax(axis=-1)
        val_bal_acc = balanced_accuracy(val_preds, yval, n_classes)
        print(f"[train] ecg(akida) restart {restart}: val balanced accuracy = {val_bal_acc:.3f}")
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_weights = copy.deepcopy(model.get_weights())

    model = am.build_akida_model(window=window, n_classes=n_classes)
    model.set_weights(best_weights)
    model.compile(
        optimizer="adam",
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    print(f"[train] ecg(akida): final best val balanced accuracy = {best_val_bal_acc:.3f}")

    float_logits = np.asarray(model.predict(Xte_u8, verbose=0))
    float_preds = float_logits.argmax(axis=-1)
    float_acc = float((float_preds == yte).mean())
    float_bal_acc = balanced_accuracy(float_preds, yte, n_classes)
    float_recalls = per_class_recall(float_preds, yte, n_classes)

    calib = Xtr_u8[:min(1024, Xtr_u8.shape[0])]
    _qmodel, akida_model = am.quantize_and_convert(
        model, calib, weight_bits=weight_bits, activation_bits=activation_bits,
        qat_epochs=qat_epochs, qat_X=Xtr_u8, qat_y=ytr, qat_class_weight=class_weight)

    n_verify = min(max_verify, Xte_u8.shape[0])
    res = am.verify_against_sim(model, akida_model, Xte_u8[:n_verify])
    yte_slice = yte[:n_verify]
    akida_preds = res["pred_akida"]
    akida_acc = float((akida_preds == yte_slice).mean())
    akida_bal_acc = balanced_accuracy(akida_preds, yte_slice, n_classes)
    akida_recalls = per_class_recall(akida_preds, yte_slice, n_classes)
    agreement_rate = float((res["pred_float"][:n_verify] == akida_preds).mean())

    auroc = auprc = float("nan")
    if n_classes == 2 and len(np.unique(yte)) > 1:
        from scipy.special import softmax
        from sklearn.metrics import average_precision_score, roc_auc_score
        float_probs = softmax(float_logits, axis=-1)[:, 1]
        auroc = float(roc_auc_score(yte, float_probs))
        auprc = float(average_precision_score(yte, float_probs))

    return {
        "source": data.source, "provenance": card.provenance,
        "n_samples": int(data.X.shape[0]), "n_verify": n_verify,
        "float_acc": float_acc, "float_bal_acc": float_bal_acc,
        "float_recalls": float_recalls, "float_auroc": auroc, "float_auprc": auprc,
        "akida_acc": akida_acc, "akida_bal_acc": akida_bal_acc,
        "akida_recalls": akida_recalls, "agreement_rate": agreement_rate,
        "weight_bits": weight_bits, "activation_bits": activation_bits,
        # Footprint: (window, 1, 1) uint8 in -> n_classes logits out, and the
        # mapped Akida layer count (InputConv2D/Conv2D/Dense1D/Dequantizer --
        # see docs/akida_ecg_results.md for the full per-layer breakdown from
        # `akida_model.summary()`). No hard channel/neuron BUDGET check here
        # unlike Xylo's xylo_budget.py -- Akida 2.0's mesh sizing constraints
        # weren't investigated in this first slice (out of scope, see
        # docs/akida_ecg_results.md "What this does NOT show").
        "footprint_input_shape": (window, 1, 1),
        "footprint_output_shape": (n_classes,),
        "footprint_n_akida_layers": len(akida_model.layers),
    }


def print_result(r: dict) -> None:
    header = f" AKIDA VERIFICATION -- ECG ({r['source']}) "
    print(f"\n{header:=^60}")
    print(f"Provenance                : {r['provenance']}")
    print(f"Samples (dataset / verify): {r['n_samples']} / {r['n_verify']}")
    print(f"Float accuracy (balanced) : {r['float_acc']:.3f} ({r['float_bal_acc']:.3f})")
    print(f"Float per-class recall    : {[f'{x:.3f}' for x in r['float_recalls']]}")
    print(f"Float AUROC / AUPRC       : {r['float_auroc']:.3f} / {r['float_auprc']:.3f}")
    print(f"Akida-sim accuracy (bal.) : {r['akida_acc']:.3f} ({r['akida_bal_acc']:.3f})")
    print(f"Akida-sim per-class recall: {[f'{x:.3f}' for x in r['akida_recalls']]}")
    print(f"Float vs. Akida-sim agree : {r['agreement_rate']:.3f}")
    print(f"Quantization (w/a bits)   : {r['weight_bits']}/{r['activation_bits']}")
    print(f"Footprint (in -> out)     : {r['footprint_input_shape']} -> "
          f"{r['footprint_output_shape']}, {r['footprint_n_akida_layers']} mapped Akida layers")
    print("=" * 60)


def print_multiseed_summary(results: list) -> None:
    def _stat(key):
        vals = np.array([r[key] for r in results], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    def _stat_recalls(key):
        arr = np.array([r[key] for r in results], dtype=float)
        return arr.mean(axis=0), arr.std(axis=0)

    fb = _stat("float_bal_acc")
    xb = _stat("akida_bal_acc")
    ag = _stat("agreement_rate")
    auroc = _stat("float_auroc")
    fr_m, fr_s = _stat_recalls("float_recalls")
    xr_m, xr_s = _stat_recalls("akida_recalls")

    print(f"\n{' MULTI-SEED SUMMARY -- ECG (AKIDA) ':=^60}")
    print(f"n_seeds                  : {len(results)}")
    print(f"Float balanced acc       : {fb[0]:.3f} +/- {fb[1]:.3f}")
    print(f"Float AUROC              : {auroc[0]:.3f} +/- {auroc[1]:.3f}")
    print(f"Float per-class recall   : {[f'{m:.3f}+/-{s:.3f}' for m, s in zip(fr_m, fr_s)]}")
    print(f"Akida-sim balanced acc   : {xb[0]:.3f} +/- {xb[1]:.3f}")
    print(f"Akida-sim per-class recall: {[f'{m:.3f}+/-{s:.3f}' for m, s in zip(xr_m, xr_s)]}")
    print(f"Float vs. Akida-sim agree: {ag[0]:.3f} +/- {ag[1]:.3f}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Akida ECG verification (first slice)")
    ap.add_argument("--real", action="store_true", help="use real MIT-BIH (needs wfdb + network)")
    ap.add_argument("--require-real", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--qat-epochs", type=int, default=5,
                     help="QAT fine-tune epochs on the quantized model (0 = calibration only)")
    ap.add_argument("--n-restarts", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--max-verify", type=int, default=300)
    ap.add_argument("--weight-bits", type=int, default=8, choices=[1, 2, 4, 8])
    ap.add_argument("--activation-bits", type=int, default=8, choices=[1, 2, 4, 8])
    args = ap.parse_args()
    real = args.real or args.require_real
    n_seeds = max(1, args.n_seeds)

    results = []
    for s in range(n_seeds):
        data = load_ecg(prefer_real=real, require_real=args.require_real)
        r = train_and_verify(
            data, seed=s, epochs=args.epochs, qat_epochs=args.qat_epochs,
            n_restarts=args.n_restarts, max_verify=args.max_verify,
            weight_bits=args.weight_bits, activation_bits=args.activation_bits)
        print_result(r)
        results.append(r)

    if n_seeds > 1:
        print_multiseed_summary(results)


if __name__ == "__main__":
    main()
