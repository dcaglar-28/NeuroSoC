"""Offline unit tests for the modality-agnostic windowed-signal feature
front-end (`eia.signal_features`) -- line length, relative band power,
spectral entropy, train-only normalization. Pure NumPy/SciPy, no network.

Originally exercised via EEG's bands (docs/eeg_frontend_task.md, retired);
these tests use an arbitrary example band dict since the module itself makes
no domain assumptions (see docs/heart_sounds_task.md for the current user,
heart-sound murmur detection)."""

import numpy as np
import pytest

from eia import signal_features

_BANDS = {"low": (0.5, 4.0), "mid": (4.0, 13.0), "high": (12.5, 25.0)}
_FEATURE_NAMES = ("line_length", "low", "high", "spectral_entropy")


def test_line_length_on_a_known_ramp():
    # x = [0, 1, 2, ..., 9] -> |diff| is 1 nine times -> line length = 9.
    x = np.arange(10, dtype=float)
    assert signal_features.line_length(x) == pytest.approx(9.0)


def test_line_length_zero_on_a_flat_signal():
    x = np.full(20, 3.0)
    assert signal_features.line_length(x) == pytest.approx(0.0)


def test_line_length_vectorizes_over_leading_axes():
    x = np.stack([np.arange(10, dtype=float), np.arange(0, 20, 2, dtype=float)])
    ll = signal_features.line_length(x)
    assert ll.shape == (2,)
    assert ll[0] == pytest.approx(9.0)
    assert ll[1] == pytest.approx(18.0)  # steps of 2, 9 steps -> 18


def test_relative_band_power_pure_sine_lands_in_correct_band():
    fs = 256.0
    t = np.arange(int(4 * fs)) / fs  # 4 seconds

    # A 2 Hz sine's power should land mostly in "low" (0.5-4 Hz) -- a little
    # spectral leakage into the adjacent bin from Welch's window is
    # expected, so this checks "clearly dominant", not "all of it".
    sine_2hz = np.sin(2 * np.pi * 2.0 * t)
    bp = signal_features.relative_band_power(sine_2hz, fs, _BANDS)
    assert bp["low"] > 0.75
    assert bp["low"] > bp["mid"]
    assert bp["low"] > bp["high"]

    # A 20 Hz sine's power should land mostly in "high" (12.5-25 Hz).
    sine_20hz = np.sin(2 * np.pi * 20.0 * t)
    bp = signal_features.relative_band_power(sine_20hz, fs, _BANDS)
    assert bp["high"] > 0.75
    assert bp["high"] > bp["low"]
    assert bp["high"] > bp["mid"]


def test_relative_band_power_fractions_are_bounded_and_sum_reasonably():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    x = np.sin(2 * np.pi * 10.0 * t)  # "mid"-band sine
    bp = signal_features.relative_band_power(x, fs, _BANDS)
    for name, frac in bp.items():
        assert 0.0 <= frac <= 1.0 + 1e-6, f"{name} fraction out of [0,1]: {frac}"
    assert bp["mid"] == max(bp.values())


def test_spectral_entropy_low_on_clean_sine_high_on_noise():
    fs = 256.0
    rng = np.random.default_rng(0)
    t = np.arange(int(4 * fs)) / fs

    clean_sine = np.sin(2 * np.pi * 10.0 * t)
    white_noise = rng.normal(size=t.shape)

    ent_sine = signal_features.spectral_entropy(clean_sine, fs)
    ent_noise = signal_features.spectral_entropy(white_noise, fs)

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

    feat = signal_features.extract_window_features(
        sig, fs, n_subwindows, _FEATURE_NAMES, _BANDS)
    assert feat.shape == (len(_FEATURE_NAMES) * n_ch, n_subwindows)
    assert np.isfinite(feat).all()


def test_extract_window_features_single_channel():
    fs = 2000.0
    window_samples, n_subwindows = 6000, 24
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(1, window_samples))

    feat = signal_features.extract_window_features(
        sig, fs, n_subwindows, _FEATURE_NAMES, _BANDS)
    assert feat.shape == (len(_FEATURE_NAMES), n_subwindows)
    assert np.isfinite(feat).all()


def test_extract_window_features_raises_on_too_short_subwindow():
    fs = 256.0
    sig = np.zeros((2, 16))  # 16 samples / 8 sub-windows = 2 samples each -> too short
    with pytest.raises(ValueError, match="too short"):
        signal_features.extract_window_features(sig, fs, 8, _FEATURE_NAMES, _BANDS)


def test_normalize_features_train_only_zscores_train_to_mean0_std1():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(loc=5.0, scale=2.0, size=(50, 3, 4))
    Xval = rng.normal(loc=5.0, scale=2.0, size=(10, 3, 4))
    Xte = rng.normal(loc=5.0, scale=2.0, size=(10, 3, 4))

    Xtr_n, Xval_n, Xte_n = signal_features.normalize_features_train_only(Xtr, Xval, Xte)

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

    Xtr_n, Xval_n, _Xte_n = signal_features.normalize_features_train_only(Xtr, Xval, Xtr)

    # If normalization had (incorrectly) been fit per-split (leaking), Xval_n
    # would also land at mean~0/std~1. Since it's fit on Xtr only, Xval's
    # very different raw distribution must show up as a large normalized
    # mean, proving no independent (leaking) fit happened on Xval.
    assert not np.allclose(Xval_n.mean(axis=(0, 2)), 0.0, atol=1.0)


def test_normalize_features_train_only_handles_zero_variance_feature():
    Xtr = np.ones((10, 2, 4))  # zero variance in every feature
    Xval = np.ones((5, 2, 4)) * 3.0
    Xte = np.ones((5, 2, 4)) * 3.0
    Xtr_n, Xval_n, Xte_n = signal_features.normalize_features_train_only(Xtr, Xval, Xte)
    assert np.isfinite(Xtr_n).all()
    assert np.isfinite(Xval_n).all()
    assert np.isfinite(Xte_n).all()
