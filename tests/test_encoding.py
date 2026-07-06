import numpy as np

from eia import encoding


def test_delta_encode_flat_signal_no_events():
    sig = np.ones(100)
    spikes = encoding.delta_encode(sig, threshold=0.05)
    assert encoding.event_count(spikes) == 0


def test_delta_encode_ramp_emits_up_events():
    sig = np.linspace(0, 1, 100)
    spikes = encoding.delta_encode(sig, threshold=0.1)
    assert (spikes > 0).sum() > 0
    assert (spikes < 0).sum() == 0  # monotonically rising -> no down events


def test_delta_encode_2ch_shape_and_channels():
    sig = np.sin(np.linspace(0, 6.28, 200))
    ch = encoding.delta_encode_2ch(sig, threshold=0.1)
    assert ch.shape == (2, 200)
    assert ch.max() <= 1 and ch.min() >= 0


def test_event_rate_is_sparse_for_smooth_signal():
    sig = np.sin(np.linspace(0, 6.28, 500))
    spikes = encoding.delta_encode(encoding.normalize(sig), threshold=0.2)
    assert encoding.event_rate(spikes) < 0.5  # sparser than dense sampling


def test_normalize_unit_std():
    sig = np.random.default_rng(0).normal(5, 3, size=1000)
    out = encoding.normalize(sig)
    assert abs(out.mean()) < 1e-6
    assert abs(out.std() - 1.0) < 1e-6
