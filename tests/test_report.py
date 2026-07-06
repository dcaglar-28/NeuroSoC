import numpy as np

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
