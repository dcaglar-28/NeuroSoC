"""Data-provenance tests: confirm --real / require_real never silently
substitute the wrong dataset (see docs on the audit that added this file)."""

from unittest.mock import patch

import numpy as np
import pytest

from eia import datasets
from eia.datasets import EcgData, PpgData


def _fake_ecg(n=10):
    return EcgData(X=np.zeros((n, 187), dtype="float32"),
                    y=np.zeros(n, dtype="int64"), fs=360.0, source="mitbih")


def _fake_ppg(n=10):
    return PpgData(X=np.zeros((n, 125), dtype="float32"),
                    y=np.zeros(n, dtype="int64"), fs=125.0, source="bidmc")


def test_ecg_default_is_synthetic_and_not_requested_real():
    d = datasets.load_ecg(prefer_real=False, n_samples=20)
    assert d.source == "synthetic"
    assert d.requested_real is False


def test_ecg_real_success_sets_requested_real_and_real_source():
    with patch.object(datasets, "load_mitbih", return_value=_fake_ecg()):
        d = datasets.load_ecg(prefer_real=True)
        assert d.source == "mitbih"
        assert d.requested_real is True


def test_ecg_real_failure_falls_back_but_marks_requested_real():
    with patch.object(datasets, "load_mitbih", side_effect=RuntimeError("no network")):
        d = datasets.load_ecg(prefer_real=True, require_real=False, n_samples=20)
        # source is synthetic (the fallback), but requested_real=True records
        # that this is a FALLBACK, not a deliberate synthetic request.
        assert d.source == "synthetic"
        assert d.requested_real is True


def test_ecg_require_real_raises_instead_of_falling_back():
    with patch.object(datasets, "load_mitbih", side_effect=RuntimeError("no network")):
        with pytest.raises(RuntimeError, match="require-real"):
            datasets.load_ecg(prefer_real=True, require_real=True)


def test_ecg_require_real_without_prefer_real_is_a_usage_error():
    with pytest.raises(ValueError):
        datasets.load_ecg(prefer_real=False, require_real=True)


def test_ppg_default_is_synthetic_and_not_requested_real():
    d = datasets.load_ppg(prefer_real=False, n_samples=20)
    assert d.source == "synthetic"
    assert d.requested_real is False


def test_ppg_real_success_sets_requested_real_and_real_source():
    with patch.object(datasets, "load_bidmc_ppg", return_value=_fake_ppg()):
        d = datasets.load_ppg(prefer_real=True)
        assert d.source == "bidmc"
        assert d.requested_real is True


def test_ppg_real_failure_falls_back_but_marks_requested_real():
    with patch.object(datasets, "load_bidmc_ppg", side_effect=RuntimeError("no network")):
        d = datasets.load_ppg(prefer_real=True, require_real=False, n_samples=20)
        assert d.source == "synthetic"
        assert d.requested_real is True


def test_ppg_require_real_raises_instead_of_falling_back():
    with patch.object(datasets, "load_bidmc_ppg", side_effect=RuntimeError("no network")):
        with pytest.raises(RuntimeError, match="require-real"):
            datasets.load_ppg(prefer_real=True, require_real=True)
