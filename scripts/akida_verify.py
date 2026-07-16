"""Pre-hardware acceptance check for the Akida path — ECG arrhythmia, ECG
myocardial infarction (PTB-XL), heart sounds, and the synthetic CRM/occult-
hemorrhage demo (docs/akida_retarget_task.md, docs/synthetic_crm_task.md,
docs/ptbxl_mi_task.md). Parallels `xylo_verify.py`'s per-modality train ->
quantize -> verify-against-sim -> footprint structure, but for BrainChip
MetaTF (`eia.akida_models`) instead of Rockpool/XyloSim. Reuses
`datasets.load_ecg`/`load_heart`/`load_crm`/`load_mi`, `signal_features`
(heart's filterbank front-end, also reused for mi's per-lead norm),
`report`, and `case_level.split_data` — the exact same data/split/
provenance discipline as the Xylo path.

**Linux only** (`akida` has no macOS wheel — see Dockerfile.akida). Run
inside the container:

    scripts/akida_docker_run.sh python scripts/akida_verify.py --real --n-seeds 5
    scripts/akida_docker_run.sh python scripts/akida_verify.py --n-seeds 2   # ecg, synthetic, quick
    scripts/akida_docker_run.sh python scripts/akida_verify.py --modality heart --real --n-seeds 5
    scripts/akida_docker_run.sh python scripts/akida_verify.py --modality crm --n-seeds 5
    scripts/akida_docker_run.sh python scripts/akida_verify.py --modality mi --real --n-seeds 5

Heart sounds is ALWAYS run with the filterbank front-end
(`heart_frontend="features"`, `datasets.PCG_FEATURE_NAMES`/`PCG_BANDS`) —
the raw waveform front-end measured flat chance on both Xylo and (see
docs/akida_heart_results.md) the reason it isn't offered as a CLI choice
here; there is no `--heart-frontend raw` escape hatch by design. CRM is
ALWAYS synthetic (`--real`/`--require-real` are accepted for CLI symmetry
but `load_crm` has no real branch — see its docstring) and ALWAYS uses the
ECG-style raw-waveform front-end (`build_akida_model`, reused unchanged),
NOT heart's filterbank — CRM is a low-frequency pulse-MORPHOLOGY signal,
not a spectral one; see docs/synthetic_crm_results.md. MI (PTB-XL) is also
morphology, so ALSO raw-waveform (never the filterbank) — but genuinely
2-D (12 leads x time, `build_akida_mi_model`), not a reuse of ECG-
arrhythmia's single-column architecture; see docs/ptbxl_mi_task.md.

See `src/eia/akida_models.py` for the confirmed Akida v2 layer constraints
(square kernel/stride/pool, valid block-ordering patterns) this script's
models rely on, `docs/akida_ecg_results.md` for the ECG-arrhythmia measured
results, `docs/akida_heart_results.md` for heart sounds',
`docs/synthetic_crm_results.md` for the CRM demo's, and
`docs/ptbxl_mi_results.md` for MI's — including the Xylo-gap comparison and
the Part-0 simulator-fidelity finding, shared across modalities, and (CRM
only) the explicit synthetic/non-clinical caveat.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from eia import case_level, report, signal_features
from eia.datasets import load_crm, load_ecg, load_heart, load_mi


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


def _build_model_for(modality: str, data_shape: tuple, n_classes: int):
    """Dispatch to the right `eia.akida_models` builder. `data_shape` is
    `Xtr.shape[1:]` -- `(window,)` for ecg AND crm (crm reuses
    `build_akida_model`'s waveform-over-time architecture unchanged: CRM is
    a low-frequency pulse-MORPHOLOGY signal, the same class of input ECG's
    architecture was built for, not heart's spectral filterbank map),
    `(n_features, n_subwindows)` for heart (post-split, pre-`to_akida_input`,
    matching `HeartData.X`'s `(n, n_features, n_subwindows)` convention),
    `(n_leads, n_samples)` for mi (matching `MiData.X`'s `(n, 12, 1000)`
    convention -- genuinely 2-D like heart's, not a reuse of ecg/crm's
    single-column reshape, since MI needs spatial lead information)."""
    from eia import akida_models as am

    if modality in ("ecg", "crm"):
        return am.build_akida_model(window=data_shape[0], n_classes=n_classes)
    if modality == "heart":
        n_features, n_subwindows = data_shape
        return am.build_akida_heart_model(
            n_bands=n_features, n_subwindows=n_subwindows, n_classes=n_classes)
    if modality == "mi":
        n_leads, n_samples = data_shape
        return am.build_akida_mi_model(
            n_leads=n_leads, n_samples=n_samples, n_classes=n_classes)
    raise ValueError(f"unknown modality {modality!r}")


def train_and_verify(data, modality: str, seed: int, epochs: int, qat_epochs: int,
                      n_restarts: int, max_verify: int, weight_bits: int,
                      activation_bits: int) -> dict:
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
    report.assert_provenance(card, data, modality)
    n_classes = int(np.asarray(data.y).max()) + 1

    if modality == "heart" and getattr(data, "frontend", "raw") != "features":
        # A real, documented gap, not a bug to silently paper over: the
        # synthetic heart-sound fallback (`make_synthetic_heart`) is always
        # raw-waveform-shaped (`(n, window)`), but `build_akida_heart_model`
        # only supports the filterbank shape (raw is banned for heart here
        # anyway -- it measured flat chance, see docs/heart_sounds_results.md).
        # This only bites when real CinC 2016 fails to load and silently
        # falls back to synthetic (`--real` without `--require-real`); use
        # `--require-real` to fail loudly instead, or pass `--real` with
        # working network access.
        raise RuntimeError(
            f"heart Akida path requires the filterbank front-end "
            f"(data.frontend='features'), got {data.frontend!r} -- "
            "make_synthetic_heart() (the real-data fallback) is raw-only "
            "and unsupported here. Pass --real (needs network) or "
            "--require-real to fail loudly instead of silently falling "
            "back to an unusable synthetic shape.")

    Xtr, Xval, Xte, ytr, yval, yte, _gtr, _gval, _gte = case_level.split_data(data, seed)

    # Per-CHANNEL z-score fit on Xtr ONLY, applied to Xval/Xte, here (post-
    # split, not in the loader) so val/test statistics can't leak into
    # training. Two callers need this, for the same underlying reason
    # (channels on heterogeneous scales, unlike ecg/crm's single channel):
    # heart's filterbank front-end (line length / band power / spectral
    # entropy are wildly different scales) and mi's 12 raw-mV leads
    # (`MiData.X` is never pre-normalized — see its docstring). Reused
    # verbatim, not modality-specific despite the name/module origin.
    if modality == "mi" or getattr(data, "frontend", "raw") == "features":
        Xtr, Xval, Xte = signal_features.normalize_features_train_only(Xtr, Xval, Xte)

    Xtr_u8 = am.to_akida_input(Xtr)
    Xval_u8 = am.to_akida_input(Xval)
    Xte_u8 = am.to_akida_input(Xte)
    class_weight = _class_weight_dict(ytr, n_classes)
    print(f"[train] {modality}(akida): class weights = {class_weight}")

    data_shape = Xtr.shape[1:]
    best_val_bal_acc, best_weights = -1.0, None
    for restart in range(n_restarts):
        tf.random.set_seed(seed * 100 + restart)
        model = _build_model_for(modality, data_shape, n_classes)
        model.compile(
            optimizer="adam",
            loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        model.fit(Xtr_u8, ytr, epochs=epochs, batch_size=32, class_weight=class_weight,
                  verbose=0)
        val_preds = np.asarray(model.predict(Xval_u8, verbose=0)).argmax(axis=-1)
        val_bal_acc = balanced_accuracy(val_preds, yval, n_classes)
        print(f"[train] {modality}(akida) restart {restart}: val balanced accuracy = {val_bal_acc:.3f}")
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_weights = copy.deepcopy(model.get_weights())

    model = _build_model_for(modality, data_shape, n_classes)
    model.set_weights(best_weights)
    model.compile(
        optimizer="adam",
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    print(f"[train] {modality}(akida): final best val balanced accuracy = {best_val_bal_acc:.3f}")

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
        # docs/akida_ecg_results.md/docs/akida_heart_results.md "What this
        # does NOT show").
        "modality": modality,
        "footprint_input_shape": Xtr_u8.shape[1:],
        "footprint_output_shape": (n_classes,),
        "footprint_n_akida_layers": len(akida_model.layers),
    }


def print_result(r: dict) -> None:
    header = f" AKIDA VERIFICATION -- {r['modality'].upper()} ({r['source']}) "
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

    modality = results[0]["modality"]
    print(f"\n{f' MULTI-SEED SUMMARY -- {modality.upper()} (AKIDA) ':=^60}")
    print(f"n_seeds                  : {len(results)}")
    print(f"Float balanced acc       : {fb[0]:.3f} +/- {fb[1]:.3f}")
    print(f"Float AUROC              : {auroc[0]:.3f} +/- {auroc[1]:.3f}")
    print(f"Float per-class recall   : {[f'{m:.3f}+/-{s:.3f}' for m, s in zip(fr_m, fr_s)]}")
    print(f"Akida-sim balanced acc   : {xb[0]:.3f} +/- {xb[1]:.3f}")
    print(f"Akida-sim per-class recall: {[f'{m:.3f}+/-{s:.3f}' for m, s in zip(xr_m, xr_s)]}")
    print(f"Float vs. Akida-sim agree: {ag[0]:.3f} +/- {ag[1]:.3f}")
    print("=" * 60)


def _load_data(modality: str, real: bool, require_real: bool):
    if modality == "ecg":
        return load_ecg(prefer_real=real, require_real=require_real)
    if modality == "heart":
        # ALWAYS the filterbank front-end -- raw delta measured flat chance
        # on both Xylo and Akida-CNN-relevant inputs (see
        # docs/heart_sounds_results.md); no CLI escape hatch to "raw" here.
        return load_heart(prefer_real=real, require_real=require_real,
                           heart_frontend="features")
    if modality == "crm":
        # ALWAYS synthetic -- load_crm has no real branch (real LBNP/CRM-
        # induction data is gated, see docs/synthetic_crm_task.md); --real
        # is accepted for CLI symmetry (recorded honestly in
        # `requested_real`) and --require-real raises immediately.
        return load_crm(prefer_real=real, require_real=require_real)
    if modality == "mi":
        return load_mi(prefer_real=real, require_real=require_real)
    raise ValueError(f"unknown modality {modality!r}")


def main():
    ap = argparse.ArgumentParser(
        description="Akida verification (ECG arrhythmia / ECG MI / heart sounds / CRM)")
    ap.add_argument("--modality", choices=["ecg", "heart", "crm", "mi"], default="ecg")
    ap.add_argument("--real", action="store_true",
                     help="use real data (MIT-BIH for ecg, CinC 2016 for heart, PTB-XL "
                          "for mi) — needs network. No effect for crm (always synthetic).")
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
        data = _load_data(args.modality, real, args.require_real)
        r = train_and_verify(
            data, modality=args.modality, seed=s, epochs=args.epochs,
            qat_epochs=args.qat_epochs, n_restarts=args.n_restarts,
            max_verify=args.max_verify, weight_bits=args.weight_bits,
            activation_bits=args.activation_bits)
        print_result(r)
        results.append(r)

    if n_seeds > 1:
        print_multiseed_summary(results)


if __name__ == "__main__":
    main()
