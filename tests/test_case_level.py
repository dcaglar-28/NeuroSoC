"""Case-aggregation logic: tiny synthetic per-window scores + group ids ->
one score per group. No torch/network — see docs/vitaldb_case_level_task.md."""

import numpy as np
import pytest

from eia import case_level


def test_aggregate_by_group_means_within_group():
    scores = np.array([0.1, 0.3, 0.8, 1.0])
    groups = np.array([1, 1, 2, 2])
    case_ids, case_scores = case_level.aggregate_by_group(scores, groups)
    assert case_ids.tolist() == [1, 2]
    assert case_scores.tolist() == pytest.approx([0.2, 0.9])


def test_aggregate_by_group_handles_uneven_group_sizes():
    scores = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    groups = np.array([5, 5, 5, 5, 7])
    case_ids, case_scores = case_level.aggregate_by_group(scores, groups)
    assert case_ids.tolist() == [5, 7]
    assert case_scores.tolist() == pytest.approx([0.25, 1.0])


def test_aggregate_by_group_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        case_level.aggregate_by_group(np.array([0.1, 0.2]), np.array([1]))


def test_aggregate_labels_by_group_reads_uniform_case_label():
    y = np.array([0, 0, 1, 1, 1])
    groups = np.array([10, 10, 20, 20, 20])
    case_ids = np.array([10, 20])
    case_y = case_level.aggregate_labels_by_group(y, groups, case_ids)
    assert case_y.tolist() == [0, 1]


def test_aggregate_labels_by_group_raises_on_inconsistent_case_label():
    y = np.array([0, 1])  # same case, two different labels -> bug upstream
    groups = np.array([1, 1])
    with pytest.raises(ValueError):
        case_level.aggregate_labels_by_group(y, groups, np.array([1]))


def test_split_data_no_groups_falls_back_to_stratified_split():
    n = 40
    X = np.arange(n).reshape(n, 1).astype("float32")
    y = np.array([0, 1] * (n // 2))

    class _D:
        pass

    d = _D()
    d.X, d.y, d.groups = X, y, None
    Xtr, Xval, Xte, ytr, yval, yte, gtr, gval, gte = case_level.split_data(d, seed=0)
    assert gtr is None and gval is None and gte is None
    assert Xtr.shape[0] + Xval.shape[0] + Xte.shape[0] == n


def test_split_data_with_groups_has_no_case_overlap():
    rng = np.random.default_rng(0)
    n_cases = 20
    windows_per_case = 5
    groups = np.repeat(np.arange(n_cases), windows_per_case)
    case_labels = (np.arange(n_cases) % 2 == 0).astype("int64")
    y = case_labels[groups]
    X = rng.normal(size=(groups.size, 3)).astype("float32")

    class _D:
        pass

    d = _D()
    d.X, d.y, d.groups = X, y, groups
    Xtr, Xval, Xte, ytr, yval, yte, gtr, gval, gte = case_level.split_data(d, seed=1)
    tr_cases, val_cases, te_cases = set(gtr.tolist()), set(gval.tolist()), set(gte.tolist())
    assert not (tr_cases & val_cases)
    assert not (tr_cases & te_cases)
    assert not (val_cases & te_cases)
    assert tr_cases | val_cases | te_cases == set(range(n_cases))
