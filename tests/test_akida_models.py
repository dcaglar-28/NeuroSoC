"""Offline unit tests for `eia.akida_models` (docs/akida_retarget_task.md).

`to_akida_input` is pure NumPy and always runs. Everything that touches the
actual MetaTF toolchain (`build_akida_model`/`quantize_and_convert`/
`verify_against_sim`) is skip-guarded with `pytest.importorskip("akida")` --
`akida` has no macOS wheel (see Dockerfile.akida), so those tests SKIP on the
host Mac this repo is developed on and RUN for real inside the Akida
container (`scripts/akida_docker_run.sh pytest -q`)."""

import numpy as np
import pytest

from eia import akida_models as am


def test_to_akida_input_shape_and_dtype():
    X = np.random.default_rng(0).normal(size=(10, 187)).astype("float32")
    out = am.to_akida_input(X)
    assert out.shape == (10, 187, 1, 1)
    assert out.dtype == np.uint8


def test_to_akida_input_range_is_0_255():
    X = np.random.default_rng(0).normal(loc=5.0, scale=3.0, size=(20, 50))
    out = am.to_akida_input(X)
    assert out.min() >= 0
    assert out.max() <= 255
    # Min-max per window should hit both ends of the range for a
    # non-degenerate window.
    assert out.max() == 255
    assert out.min() == 0


def test_to_akida_input_constant_window_does_not_nan():
    X = np.full((3, 20), 7.0)
    out = am.to_akida_input(X)
    assert np.isfinite(out).all()
    assert out.dtype == np.uint8


def test_to_akida_input_per_window_normalization():
    # Two windows with very different scales should each independently span
    # [0, 255] -- confirms normalization is per-window, not global.
    X = np.stack([np.linspace(0, 1, 10), np.linspace(100, 200, 10)])
    out = am.to_akida_input(X)
    assert out[0].min() == 0 and out[0].max() == 255
    assert out[1].min() == 0 and out[1].max() == 255


def test_to_akida_input_3d_shape_and_dtype():
    # Heart's filterbank map: (n, n_features, n_subwindows) -> (n, H, W, 1).
    X = np.random.default_rng(0).normal(size=(5, 4, 24)).astype("float32")
    out = am.to_akida_input(X)
    assert out.shape == (5, 4, 24, 1)
    assert out.dtype == np.uint8


def test_to_akida_input_3d_normalizes_per_row_not_per_sample():
    # Row 0 (e.g. line_length, large raw scale) and row 1 (e.g. a [0,1]
    # band-power fraction) should each independently span [0, 255] --
    # confirms the per-row (per-feature-channel), not whole-sample, min-max.
    X = np.zeros((1, 2, 8))
    X[0, 0] = np.linspace(1000.0, 2000.0, 8)   # a "large-scale" feature row
    X[0, 1] = np.linspace(0.0, 1.0, 8)          # a "[0,1]-scale" feature row
    out = am.to_akida_input(X)
    assert out[0, 0].min() == 0 and out[0, 0].max() == 255
    assert out[0, 1].min() == 0 and out[0, 1].max() == 255


def test_build_akida_model_shapes():
    pytest.importorskip("akida")
    model = am.build_akida_model(window=187, n_classes=2)
    assert model.input_shape == (None, 187, 1, 1)
    assert model.output_shape == (None, 2)


def test_build_akida_heart_model_shapes():
    pytest.importorskip("akida")
    model = am.build_akida_heart_model(n_bands=4, n_subwindows=24, n_classes=2)
    assert model.input_shape == (None, 4, 24, 1)
    assert model.output_shape == (None, 2)


def test_heart_quantize_and_convert_and_verify_roundtrip():
    """Same smoke test as ECG's, for the heart model + its 2-D (bands x
    time) input -- confirms the Akida v2 conversion constraints
    (square kernel/stride/pool, valid layer ordering) also hold for this
    genuinely-2-D input shape, not just ECG's single-column one."""
    pytest.importorskip("akida")
    import tf_keras

    rng = np.random.default_rng(0)
    X = am.to_akida_input(rng.normal(size=(32, 4, 24)).astype("float32"))
    y = rng.integers(0, 2, size=32)

    model = am.build_akida_heart_model(n_bands=4, n_subwindows=24, n_classes=2)
    model.compile(optimizer="adam",
                   loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=1, batch_size=16, verbose=0)

    _qmodel, akida_model = am.quantize_and_convert(model, X[:16], num_samples=16, batch_size=8)
    res = am.verify_against_sim(model, akida_model, X[:8])

    assert res["pred_float"].shape == (8,)
    assert res["pred_akida"].shape == (8,)
    assert 0.0 <= res["agreement_rate"] <= 1.0


def test_quantize_and_convert_and_verify_roundtrip():
    """Small end-to-end smoke test: build -> (trivially) train -> quantize ->
    convert -> verify_against_sim, on tiny random data. Confirms the whole
    pipeline is wired correctly -- not a fidelity claim (see
    docs/akida_ecg_results.md for the real measurement)."""
    pytest.importorskip("akida")
    import tf_keras

    rng = np.random.default_rng(0)
    X = am.to_akida_input(rng.normal(size=(32, 187)).astype("float32"))
    y = rng.integers(0, 2, size=32)

    model = am.build_akida_model(window=187, n_classes=2)
    model.compile(optimizer="adam",
                   loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=1, batch_size=16, verbose=0)

    _qmodel, akida_model = am.quantize_and_convert(model, X[:16], num_samples=16, batch_size=8)
    res = am.verify_against_sim(model, akida_model, X[:8])

    assert res["pred_float"].shape == (8,)
    assert res["pred_akida"].shape == (8,)
    assert 0.0 <= res["agreement_rate"] <= 1.0


def test_quantize_and_convert_qat_fine_tune_runs():
    pytest.importorskip("akida")
    import tf_keras

    rng = np.random.default_rng(0)
    X = am.to_akida_input(rng.normal(size=(32, 187)).astype("float32"))
    y = rng.integers(0, 2, size=32)

    model = am.build_akida_model(window=187, n_classes=2)
    model.compile(optimizer="adam",
                   loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=1, batch_size=16, verbose=0)

    _qmodel, akida_model = am.quantize_and_convert(
        model, X[:16], num_samples=16, batch_size=8, qat_epochs=1, qat_X=X, qat_y=y)
    out = akida_model.forward(X[:4])
    assert out.shape[0] == 4


def test_crm_reuses_build_akida_model_unchanged_end_to_end():
    """CRM does NOT get its own builder (docs/synthetic_crm_task.md: reuse
    the ECG waveform -> Akida CNN path, not heart's filterbank) -- this
    confirms `build_akida_model` (unmodified) converts and runs on the
    real `make_synthetic_crm` data shape, end to end."""
    pytest.importorskip("akida")
    import tf_keras

    from eia.datasets import make_synthetic_crm

    d = make_synthetic_crm(n_subjects=3, windows_per_subject=8, window_sec=1.0,
                            fs=50.0, seed=0)
    X = am.to_akida_input(d.X)
    y = d.y

    model = am.build_akida_model(window=d.X.shape[1], n_classes=2)
    assert model.input_shape == (None, d.X.shape[1], 1, 1)
    model.compile(optimizer="adam",
                   loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    _qmodel, akida_model = am.quantize_and_convert(model, X[:12], num_samples=12, batch_size=8)
    res = am.verify_against_sim(model, akida_model, X[:8])
    assert res["pred_float"].shape == (8,)
    assert res["pred_akida"].shape == (8,)
    assert 0.0 <= res["agreement_rate"] <= 1.0


def test_shockable_reuses_build_akida_model_unchanged_end_to_end():
    """Shockable-rhythm (VF/VT) does NOT get its own builder (like CRM,
    docs/shockable_rhythm_task.md: reuse the ECG waveform -> Akida CNN path,
    not heart's/mi's genuinely-2-D filterbank/lead map -- VF/VT is a
    single-lead morphology/rhythm signal) -- confirms `build_akida_model`
    (unmodified) converts and runs on the real `make_synthetic_shockable`
    data shape, end to end."""
    pytest.importorskip("akida")
    import tf_keras

    from eia.datasets import make_synthetic_shockable

    d = make_synthetic_shockable(n_samples=24, window_sec=5.0, fs=50.0, seed=0)
    X = am.to_akida_input(d.X)
    y = d.y

    model = am.build_akida_model(window=d.X.shape[1], n_classes=2)
    assert model.input_shape == (None, d.X.shape[1], 1, 1)
    model.compile(optimizer="adam",
                   loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    _qmodel, akida_model = am.quantize_and_convert(model, X[:12], num_samples=12, batch_size=8)
    res = am.verify_against_sim(model, akida_model, X[:8])
    assert res["pred_float"].shape == (8,)
    assert res["pred_akida"].shape == (8,)
    assert 0.0 <= res["agreement_rate"] <= 1.0


def test_build_akida_mi_model_shapes():
    pytest.importorskip("akida")
    model = am.build_akida_mi_model(n_leads=12, n_samples=1000, n_classes=2)
    assert model.input_shape == (None, 12, 1000, 1)
    assert model.output_shape == (None, 2)


def test_mi_quantize_and_convert_and_verify_roundtrip():
    """Same smoke test as heart's, for the MI model + its genuinely-2-D
    (12 leads x 1000 samples) input -- confirms the Akida v2 conversion
    constraints hold at this much larger time-axis size too, not just
    heart's 24-sample one."""
    pytest.importorskip("akida")
    import tf_keras

    rng = np.random.default_rng(0)
    X = am.to_akida_input(rng.normal(size=(16, 12, 1000)).astype("float32"))
    y = rng.integers(0, 2, size=16)

    model = am.build_akida_mi_model(n_leads=12, n_samples=1000, n_classes=2)
    model.compile(optimizer="adam",
                   loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    _qmodel, akida_model = am.quantize_and_convert(model, X[:8], num_samples=8, batch_size=4)
    res = am.verify_against_sim(model, akida_model, X[:4])

    assert res["pred_float"].shape == (4,)
    assert res["pred_akida"].shape == (4,)
    assert 0.0 <= res["agreement_rate"] <= 1.0


def test_mi_reuses_signal_features_normalize_and_real_generator_shape_end_to_end():
    """MiData (from make_synthetic_mi -- the fallback, but same shape as
    real PTB-XL) round-trips through to_akida_input + build_akida_mi_model
    with real data-shaped input, confirming the full mi pipeline's pieces
    fit together, not just the shape-parametrized smoke test above."""
    pytest.importorskip("akida")
    import tf_keras

    from eia.datasets import make_synthetic_mi

    d = make_synthetic_mi(n_samples=16, n_leads=12, window=1000, seed=0)
    X = am.to_akida_input(d.X)
    y = d.y

    model = am.build_akida_mi_model(n_leads=d.X.shape[1], n_samples=d.X.shape[2], n_classes=2)
    assert model.input_shape == (None, 12, 1000, 1)
    model.compile(optimizer="adam",
                   loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    _qmodel, akida_model = am.quantize_and_convert(model, X[:8], num_samples=8, batch_size=4)
    res = am.verify_against_sim(model, akida_model, X[:4])
    assert res["pred_float"].shape == (4,)
    assert res["pred_akida"].shape == (4,)
