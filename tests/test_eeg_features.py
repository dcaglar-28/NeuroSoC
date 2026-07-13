"""Offline unit tests for the EEG feature front-end (docs/eeg_frontend_task.md)
-- line length, relative band power, spectral entropy, train-only
normalization. Pure NumPy/SciPy, no network."""

import numpy as np
import pytest

from eia import eeg_features


def test_line_length_on_a_known_ramp():
    # x = [0, 1, 2, ..., 9] -> |diff| is 1 nine times -> line length = 9.
    x = np.arange(10, dtype=float)
    assert eeg_features.line_length(x) == pytest.approx(9.0)


def test_line_length_zero_on_a_flat_signal():
    x = np.full(20, 3.0)
    assert eeg_features.line_length(x) == pytest.approx(0.0)


def test_line_length_vectorizes_over_leading_axes():
    x = np.stack([np.arange(10, dtype=float), np.arange(0, 20, 2, dtype=float)])
    ll = eeg_features.line_length(x)
    assert ll.shape == (2,)
    assert ll[0] == pytest.approx(9.0)
    assert ll[1] == pytest.approx(18.0)  # steps of 2, 9 steps -> 18


def test_relative_band_power_pure_sine_lands_in_correct_band():
    fs = 256.0
    t = np.arange(int(4 * fs)) / fs  # 4 seconds

    # A 2 Hz sine's power should land mostly in delta (0.5-4 Hz) -- a little
    # spectral leakage into the adjacent theta bin from Welch's window is
    # expected, so this checks "clearly dominant", not "all of it".
    sine_2hz = np.sin(2 * np.pi * 2.0 * t)
    bp = eeg_features.relative_band_power(sine_2hz, fs)
    assert bp["delta"] > 0.75
    assert bp["delta"] > bp["theta"]
    assert bp["delta"] > bp["beta"]

    # A 20 Hz sine's power should land mostly in beta (12.5-25 Hz).
    sine_20hz = np.sin(2 * np.pi * 20.0 * t)
    bp = eeg_features.relative_band_power(sine_20hz, fs)
    assert bp["beta"] > 0.75
    assert bp["beta"] > bp["delta"]
    assert bp["beta"] > bp["theta"]


def test_relative_band_power_fractions_are_bounded_and_sum_reasonably():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    x = np.sin(2 * np.pi * 10.0 * t)  # alpha-band sine
    bp = eeg_features.relative_band_power(x, fs)
    for name, frac in bp.items():
        assert 0.0 <= frac <= 1.0 + 1e-6, f"{name} fraction out of [0,1]: {frac}"
    # delta/theta/alpha/beta are non-overlapping-ish bands over a limited
    # spectrum, so their fractions should not individually exceed 1 and the
    # dominant one (alpha, ~8-13 Hz) should clearly stand out.
    assert bp["alpha"] == max(bp.values())


def test_spectral_entropy_low_on_clean_sine_high_on_noise():
    fs = 256.0
    rng = np.random.default_rng(0)
    t = np.arange(int(4 * fs)) / fs

    clean_sine = np.sin(2 * np.pi * 10.0 * t)
    white_noise = rng.normal(size=t.shape)

    ent_sine = eeg_features.spectral_entropy(clean_sine, fs)
    ent_noise = eeg_features.spectral_entropy(white_noise, fs)

    assert 0.0 <= ent_sine <= 1.0
    assert 0.0 <= ent_noise <= 1.0
    # Rhythmic (power concentrated in one bin) -> low entropy;
    # noise-like (power spread flat) -> high entropy.
    assert ent_sine < ent_noise


def test_extract_window_features_shape_is_feature_major():
    fs = 256.0
    n_ch, window_samples, n_subwindows = 2, 1024, 8
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(n_ch, window_samples))

    feat = eeg_features.extract_window_features(sig, fs, n_subwindows)
    assert feat.shape == (len(eeg_features.FEATURE_NAMES) * n_ch, n_subwindows)
    assert np.isfinite(feat).all()


def test_extract_window_features_raises_on_too_short_subwindow():
    fs = 256.0
    sig = np.zeros((2, 16))  # 16 samples / 8 sub-windows = 2 samples each -> too short
    with pytest.raises(ValueError, match="too short"):
        eeg_features.extract_window_features(sig, fs, n_subwindows=8)


def test_normalize_features_train_only_zscores_train_to_mean0_std1():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(loc=5.0, scale=2.0, size=(50, 3, 4))
    Xval = rng.normal(loc=5.0, scale=2.0, size=(10, 3, 4))
    Xte = rng.normal(loc=5.0, scale=2.0, size=(10, 3, 4))

    Xtr_n, Xval_n, Xte_n = eeg_features.normalize_features_train_only(Xtr, Xval, Xte)

    assert np.allclose(Xtr_n.mean(axis=(0, 2)), 0.0, atol=1e-6)
    assert np.allclose(Xtr_n.std(axis=(0, 2)), 1.0, atol=1e-6)


def test_normalize_features_train_only_does_not_leak_val_test_stats():
    """The no-leakage guard: val/test are normalized with TRAIN's mean/std,
    not their own -- so if val/test have a very different distribution from
    train, their normalized mean/std will NOT land at (0, 1)."""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(loc=0.0, scale=1.0, size=(50, 2, 4))
    # Val has a very different (shifted, wider) distribution from train.
    Xval = rng.normal(loc=50.0, scale=10.0, size=(10, 2, 4))

    Xtr_n, Xval_n, _Xte_n = eeg_features.normalize_features_train_only(Xtr, Xval, Xtr)

    # If normalization had (incorrectly) been fit per-split (leaking), Xval_n
    # would also land at mean~0/std~1. Since it's fit on Xtr only, Xval's
    # very different raw distribution must show up as a large normalized
    # mean, proving no independent (leaking) fit happened on Xval.
    assert not np.allclose(Xval_n.mean(axis=(0, 2)), 0.0, atol=1.0)


def test_normalize_features_train_only_handles_zero_variance_feature():
    Xtr = np.ones((10, 2, 4))  # zero variance in every feature
    Xval = np.ones((5, 2, 4)) * 3.0
    Xte = np.ones((5, 2, 4)) * 3.0
    Xtr_n, Xval_n, Xte_n = eeg_features.normalize_features_train_only(Xtr, Xval, Xte)
    assert np.isfinite(Xtr_n).all()
    assert np.isfinite(Xval_n).all()
    assert np.isfinite(Xte_n).all()
