"""Case-level aggregation and case-grouped splitting.

For datasets where the ground-truth label is inherently case-level but the
model consumes short windows (VitalDB: `intraop_ebl` is a single whole-case
total stamped onto every window from that case — see
`docs/vitaldb_case_level_task.md`), the statistically honest evaluation is one
prediction per case, not one per window. This module holds the two pure,
torch-free pieces of that: pooling per-window scores into per-case scores, and
splitting a windowed dataset by case (not window) so no case straddles
train/val/test.

Pure NumPy + scikit-learn — importable without torch, so the aggregation
logic is unit-testable on a tiny synthetic array with no model/network/GPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def aggregate_by_group(scores: np.ndarray, groups: np.ndarray):
    """Mean-pool per-window scores into one score per group (case).

    This is a HOST-SIDE pooling step, not something the Xylo core does — the
    chip still only ever sees and classifies individual short windows; this
    function represents the aggregation a device/host would do across a
    case's sequence of per-window chip outputs.

    Args:
        scores: (n_windows,) float — a per-window continuous score (e.g. the
            softmax probability of the positive class).
        groups: (n_windows,) — case/group id per window, same length.

    Returns:
        (case_ids, case_scores): `case_ids` sorted ascending (unique groups),
        `case_scores[i] = mean(scores[groups == case_ids[i]])`.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    if scores.shape[0] != groups.shape[0]:
        raise ValueError(
            f"scores and groups must be the same length, got "
            f"{scores.shape[0]} and {groups.shape[0]}")
    case_ids = np.unique(groups)
    case_scores = np.array([scores[groups == c].mean() for c in case_ids])
    return case_ids, case_scores


def aggregate_labels_by_group(y: np.ndarray, groups: np.ndarray,
                               case_ids: np.ndarray) -> np.ndarray:
    """Per-case label, for each id in `case_ids`, verified consistent across
    every window of that case. Raises if a case has more than one distinct
    window label — that would mean a case/window bookkeeping bug upstream,
    since a case-level label (like `intraop_ebl`) is one number per case by
    construction.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    case_ids = np.asarray(case_ids)
    case_y = np.empty(case_ids.shape[0], dtype=y.dtype)
    for i, c in enumerate(case_ids):
        labels = np.unique(y[groups == c])
        if labels.size != 1:
            raise ValueError(
                f"case {c!r} has inconsistent window labels {labels.tolist()} "
                "— case-level label must be uniform across a case's windows.")
        case_y[i] = labels[0]
    return case_y


def split_data(data, seed: int):
    """Case/subject-grouped split when `data.groups` is set (many
    highly-correlated windows per case sharing one label — must not straddle
    train/val/test), plain stratified split otherwise. Returns window-level
    arrays plus the group id for each split's windows (None when `data.groups`
    is None), and prints the per-split case counts (raising on any overlap)
    so grouping is visible in the log, not just trusted to the splitter.

    Shared by `scripts/xylo_verify.py` and `scripts/vitaldb_case_level.py` so
    both use the exact same case-disjoint splitting logic.
    """
    groups = getattr(data, "groups", None)
    if groups is None:
        Xfit, Xte, yfit, yte = train_test_split(
            data.X, data.y, test_size=0.25, random_state=seed, stratify=data.y)
        Xtr, Xval, ytr, yval = train_test_split(
            Xfit, yfit, test_size=0.2, random_state=seed, stratify=yfit)
        return Xtr, Xval, Xte, ytr, yval, yte, None, None, None

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    fit_idx, te_idx = next(gss1.split(data.X, data.y, groups=groups))
    Xfit, Xte = data.X[fit_idx], data.X[te_idx]
    yfit, yte = data.y[fit_idx], data.y[te_idx]
    groups_fit, groups_te = groups[fit_idx], groups[te_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(gss2.split(Xfit, yfit, groups=groups_fit))
    Xtr, Xval = Xfit[tr_idx], Xfit[val_idx]
    ytr, yval = yfit[tr_idx], yfit[val_idx]
    groups_tr, groups_val = groups_fit[tr_idx], groups_fit[val_idx]

    tr_cases, val_cases = set(groups_tr.tolist()), set(groups_val.tolist())
    te_cases = set(groups_te.tolist())
    overlap = (tr_cases & val_cases) | (tr_cases & te_cases) | (val_cases & te_cases)
    print(f"[split] case-grouped: {len(tr_cases)} train / {len(val_cases)} val "
          f"/ {len(te_cases)} test cases"
          + (f"  [warn] OVERLAP: {overlap}" if overlap else "  (no case overlap)"))
    if overlap:
        raise RuntimeError(f"case-level leakage across splits: {overlap}")
    return Xtr, Xval, Xte, ytr, yval, yte, groups_tr, groups_val, groups_te
