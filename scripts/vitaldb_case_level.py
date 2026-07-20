"""Case-level VitalDB hemorrhage classification — the matched-granularity
rescue of the per-window result. See `docs/vitaldb_case_level_results.md` and
`docs/vitaldb_ppg_results.md` (the ~chance per-window result this responds
to): `intraop_ebl` is a single whole-case total stamped onto every window
from that case, so a per-window prediction and a per-case label describe
different moments — the statistically honest use of the label is one
prediction per case.

Approach (A) from the task spec: train the SAME per-window Xylo-mappable SNN
used elsewhere in this repo (identical class-weighted loss + balanced-
accuracy checkpoint selection, unchanged), then — only at evaluation time —
mean-pool its per-window output probabilities into ONE score per case
(`eia.case_level.aggregate_by_group`) and classify at the case level. That
pooling is a HOST-SIDE step: the Xylo core still only ever sees and
classifies individual short windows; nothing about the on-chip net changes.

CLINICAL CLAIM (must not be overstated): retrospective, whole-case ("does
this surgery's PPG overall look like a high-blood-loss case?"), NOT the
real-time field-hemorrhage detection the device ultimately needs.

Run (needs `pip install "eia[data,xylo]"`):

    python scripts/vitaldb_case_level.py --max-cases 300 --n-seeds 5
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xylo_verify import (  # noqa: E402  (sibling script, not a package import)
    _class_weights, _encode_batch, balanced_accuracy,
)

from eia import case_level, report, rockpool_models as rm  # noqa: E402
from eia.datasets import load_vitaldb_ppg  # noqa: E402

_HIDDEN_KEY = "1_LIFBitshiftTorch"


def train_window_model(data, seed: int, epochs: int, n_hidden: int,
                        threshold: float, batch_size: int, lr: float,
                        spike_reg: float, n_restarts: int) -> dict:
    """Train the per-window SNN exactly as `xylo_verify.train_modality` does
    (same class-weighted loss, restarts, balanced-accuracy checkpoint
    selection — unchanged) on a case-disjoint split, then return the trained
    net's per-window positive-class probability on the held-out TEST windows
    plus their case ids, ready for case-level aggregation. Nothing about
    training is case-aware; only the split (case-grouped, so no case's
    windows straddle train/val/test) and the post-hoc evaluation are.
    """
    n_classes = int(np.asarray(data.y).max()) + 1
    Xtr, Xval, Xte, ytr, yval, yte, _gtr, _gval, groups_te = \
        case_level.split_data(data, seed)

    Rtr = torch.tensor(_encode_batch(Xtr, threshold))
    Rval = torch.tensor(_encode_batch(Xval, threshold))
    Rte = torch.tensor(_encode_batch(Xte, threshold))
    ytr_t = torch.tensor(ytr)
    yval_t = torch.tensor(yval)

    class_weight = _class_weights(ytr_t, n_classes)
    print(f"[train] vitaldb (case-level): class weights = {class_weight.tolist()}")
    lossfn = nn.CrossEntropyLoss(weight=class_weight)
    n = Rtr.shape[0]
    best_val_bal_acc, best_state = -1.0, None
    for restart in range(n_restarts):
        torch.manual_seed(seed * 100 + restart)
        net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=n_classes)
        opt = torch.optim.Adam(net.parameters().astorch(), lr=lr)
        for ep in range(epochs):
            idx = torch.randperm(n)
            for k in range(0, n, batch_size):
                bidx = idx[k:k + batch_size]
                opt.zero_grad()
                net.reset_state()
                out, _state, rec = net(Rtr[bidx], record=True)
                logits = out.sum(dim=1)
                spk_rate = rec[_HIDDEN_KEY]["spikes"].mean()
                loss = lossfn(logits, ytr_t[bidx]) + spike_reg * spk_rate
                loss.backward()
                opt.step()
            net.reset_state()
            with torch.no_grad():
                val_out, _state, _rec = net(Rval)
            val_preds = val_out.sum(dim=1).argmax(dim=1)
            val_bal_acc = balanced_accuracy(val_preds, yval_t, n_classes)
            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_state = copy.deepcopy(net.state_dict())
        print(f"[train] vitaldb (case-level) restart {restart}: "
              f"best-so-far val balanced accuracy (window-level) = {best_val_bal_acc:.3f}")

    net = rm.build_xylo_snn(n_hidden=n_hidden, n_out=n_classes)
    net.load_state_dict(best_state)
    print(f"[train] vitaldb (case-level): final best val balanced accuracy "
          f"(window-level) = {best_val_bal_acc:.3f}")

    net.reset_state()
    with torch.no_grad():
        out, _state, _rec = net(Rte)
    logits = out.sum(dim=1)
    probs_pos = torch.softmax(logits, dim=1)[:, 1].numpy()
    return {"probs_pos": probs_pos, "yte": yte, "groups_te": groups_te}


def evaluate_case_level(window_result: dict) -> dict:
    """Mean-pool the trained net's per-window test-set probabilities into one
    score per case (`case_level.aggregate_by_group`), then compute case-level
    balanced accuracy, per-class recall, AUROC, and AUPRC against the
    case-level majority-class base rate. This is the ONLY place aggregation
    happens — training above is unchanged, ordinary window-level training."""
    probs_pos, yte, groups_te = (
        window_result["probs_pos"], window_result["yte"], window_result["groups_te"])
    case_ids, case_scores = case_level.aggregate_by_group(probs_pos, groups_te)
    case_y = case_level.aggregate_labels_by_group(yte, groups_te, case_ids)

    case_preds = (case_scores >= 0.5).astype(int)
    acc = float((case_preds == case_y).mean())
    recalls = []
    for c in (0, 1):
        mask = case_y == c
        recalls.append(float((case_preds[mask] == c).mean()) if mask.sum() else float("nan"))
    bal_acc = float(np.nanmean(recalls))

    auroc = auprc = float("nan")
    if len(np.unique(case_y)) > 1:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(case_y, case_scores))
        auprc = float(average_precision_score(case_y, case_scores))

    base_rate = float(max((case_y == 0).mean(), (case_y == 1).mean()))
    return {
        "n_test_cases": int(case_ids.size), "test_pos_frac": float(case_y.mean()),
        "acc": acc, "bal_acc": bal_acc, "recalls": recalls,
        "auroc": auroc, "auprc": auprc, "base_rate": base_rate,
    }


def run_seed(data, seed: int, epochs: int, n_hidden: int, threshold: float,
             batch_size: int, lr: float, spike_reg: float, n_restarts: int) -> dict:
    window_result = train_window_model(
        data, seed, epochs, n_hidden, threshold, batch_size, lr, spike_reg, n_restarts)
    case_result = evaluate_case_level(window_result)
    print(f"\n{' CASE-LEVEL RESULT -- seed ' + str(seed) + ' ':=^60}")
    print(f"Test cases               : {case_result['n_test_cases']}  "
          f"(pos frac {case_result['test_pos_frac']:.3f})")
    print(f"Case-level accuracy      : {case_result['acc']:.3f}  "
          f"(balanced: {case_result['bal_acc']:.3f})")
    print(f"Case-level per-class recall: {[f'{r:.3f}' for r in case_result['recalls']]}")
    print(f"AUROC / AUPRC            : {case_result['auroc']:.3f} / {case_result['auprc']:.3f}")
    print(f"Majority-case base rate  : {case_result['base_rate']:.3f}  "
          f"(a model at 0.5 balanced acc / 0.5 AUROC has learned nothing)")
    print("=" * 60)
    return case_result


def print_multiseed_summary(seed_results: list) -> None:
    n = len(seed_results)

    def _stat(key):
        vals = np.array([r[key] for r in seed_results], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    def _stat_recalls(idx):
        vals = np.array([r["recalls"][idx] for r in seed_results], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    bal_m, bal_s = _stat("bal_acc")
    acc_m, acc_s = _stat("acc")
    auroc_m, auroc_s = _stat("auroc")
    auprc_m, auprc_s = _stat("auprc")
    r0_m, r0_s = _stat_recalls(0)
    r1_m, r1_s = _stat_recalls(1)
    base_rate = seed_results[0]["base_rate"]

    print(f"\n{' MULTI-SEED CASE-LEVEL SUMMARY (VitalDB) ':=^60}")
    print(f"n_seeds                  : {n}")
    print(f"Case-level accuracy      : {acc_m:.3f} +/- {acc_s:.3f}")
    print(f"Case-level balanced acc  : {bal_m:.3f} +/- {bal_s:.3f}")
    print(f"Case-level per-class recall: [{r0_m:.3f}+/-{r0_s:.3f}, "
          f"{r1_m:.3f}+/-{r1_s:.3f}]")
    print(f"AUROC                    : {auroc_m:.3f} +/- {auroc_s:.3f}")
    print(f"AUPRC                    : {auprc_m:.3f} +/- {auprc_s:.3f}")
    print(f"Majority-case base rate  : {base_rate:.3f}  "
          f"(chance = 0.5 balanced acc / 0.5 AUROC, not the base rate itself)")
    learned = (bal_m - bal_s) > 0.5 and (auroc_m - auroc_s) > 0.5
    verdict = ("LEARNED SOMETHING: balanced acc and AUROC both clear 0.5 "
               "outside the seed band — a real, if modest and retrospective, "
               "case-level blood-loss signal in this PPG data.") if learned else (
        "~CHANCE: balanced acc and/or AUROC do not clearly clear 0.5 across "
        "seeds. Per docs/vitaldb_case_level_results.md: this dataset's PPG does "
        "not carry a usable blood-loss signal even at its label's own "
        "granularity. Do not chase this further — the honest next step for a "
        "flagship hemorrhage signal is LBNP (gated) or the synthetic "
        "time-resolved generator, not more VitalDB tuning.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Case-level VitalDB hemorrhage classification")
    ap.add_argument("--max-cases", type=int, default=300,
                     help="qualifying VitalDB cases to use (more than the "
                          "150-case per-window run: case-level has far fewer "
                          "training examples, one label per case).")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-hidden", type=int, default=63)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--spike-reg", type=float, default=2e-2)
    ap.add_argument("--n-restarts", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--require-real", action="store_true", default=True,
                     help="this script only makes sense on real VitalDB data; "
                          "always on (kept as a flag for symmetry with the "
                          "rest of the repo's CLIs).")
    args = ap.parse_args()

    # Load ONCE — the case pool must stay fixed across seeds so only the
    # train/val/test partition and net init vary (matches the per-window
    # multi-seed run's design in scripts/xylo_verify.py).
    data = load_vitaldb_ppg(max_cases=args.max_cases)
    card = report.data_card(data)
    report.assert_provenance(card, data, "ppg")

    case_ids_all = np.unique(data.groups)
    case_y_all = case_level.aggregate_labels_by_group(data.y, data.groups, case_ids_all)
    print(f"\n[data] vitaldb case-level pool: {case_ids_all.size} cases, "
          f"pos frac = {case_y_all.mean():.3f} "
          f"(counts {dict(zip(*np.unique(case_y_all, return_counts=True)))})")

    seed_results = [
        run_seed(data, seed, args.epochs, args.n_hidden, args.threshold,
                  128, 1e-2, args.spike_reg, args.n_restarts)
        for seed in range(args.n_seeds)
    ]
    print_multiseed_summary(seed_results)


if __name__ == "__main__":
    main()
