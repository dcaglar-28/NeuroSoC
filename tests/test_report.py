import numpy as np
import pytest

from eia import report
from eia.datasets import make_synthetic_ppg, make_synthetic_ecg


def test_data_card_basic_fields():
    d = make_synthetic_ppg(n_samples=500, seed=0)
    card = report.data_card(d, verbose=False)
    assert card.modality == "ppg"
    assert card.source == "synthetic"
    assert card.n_samples == 500
    assert 0.0 <= card.majority_base_rate <= 1.0
    assert set(card.class_counts) <= {0, 1}


def test_small_n_warns():
    d = make_synthetic_ecg(n_samples=50, seed=1)
    card = report.data_card(d, verbose=False)
    assert any("small dataset" in w for w in card.warnings)


def _base_rate(d):
    y = np.asarray(d.y)
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / y.size)


def test_not_learning_warns_when_model_at_base_rate():
    d = make_synthetic_ppg(n_samples=1000, abnormal_frac=0.5, seed=2)
    card = report.data_card(d, model_acc=_base_rate(d), verbose=False)
    assert any("NOT learning" in w for w in card.warnings)


def test_high_base_rate_warns():
    d = make_synthetic_ecg(n_samples=1000, abnormal_frac=0.1, seed=3)
    card = report.data_card(d, verbose=False)
    assert any("majority base rate" in w for w in card.warnings)


def test_str_renders():
    d = make_synthetic_ppg(n_samples=300, seed=4)
    s = str(report.data_card(d, verbose=False))
    assert "DATA CARD" in s and "label" in s


def test_provenance_synthetic_requested():
    d = make_synthetic_ppg(n_samples=100, seed=0)
    card = report.data_card(d, verbose=False)
    assert card.provenance == "SYNTHETIC (requested)"


def test_provenance_synthetic_fallback():
    d = make_synthetic_ecg(n_samples=100, seed=0)
    d.requested_real = True  # simulate: real load failed, this is the fallback
    card = report.data_card(d, verbose=False)
    assert "FALLBACK" in card.provenance


def test_provenance_real():
    d = make_synthetic_ppg(n_samples=100, seed=0)
    d.source = "bidmc"  # simulate a real-sourced object for this check
    card = report.data_card(d, verbose=False)
    assert card.provenance == "REAL (bidmc)"


def test_assert_provenance_passes_when_consistent():
    d = make_synthetic_ppg(n_samples=50, seed=0)
    card = report.data_card(d, verbose=False)
    report.assert_provenance(card, d, "ppg")  # must not raise


def test_assert_provenance_raises_on_modality_mismatch():
    d = make_synthetic_ppg(n_samples=50, seed=0)
    card = report.data_card(d, verbose=False)
    with pytest.raises(ValueError, match="modality"):
        report.assert_provenance(card, d, "ecg")


def test_assert_provenance_raises_on_source_mismatch():
    """The exact bug class this guards against: a data card built from one
    dataset, but a (since-mutated, or swapped-out) data object with a
    different source is what actually gets trained/verified on."""
    d = make_synthetic_ppg(n_samples=50, seed=0)
    card = report.data_card(d, verbose=False)
    d.source = "bidmc"  # data object no longer matches the card built from it
    with pytest.raises(ValueError, match="source"):
        report.assert_provenance(card, d, "ppg")
