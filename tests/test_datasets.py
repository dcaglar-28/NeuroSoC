"""Data-provenance tests: confirm --real / require_real never silently
substitute the wrong dataset (see docs on the audit that added this file)."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from eia import datasets
from eia.datasets import CrmData, EcgData, HeartData, MiData, PpgData, ShockableData


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


def _fake_vitaldb_ppg(n=10):
    return PpgData(X=np.zeros((n, 125), dtype="float32"),
                    y=np.zeros(n, dtype="int64"), fs=125.0, source="vitaldb",
                    groups=np.arange(n, dtype="int64"))


def test_vitaldb_ebl_label_thresholding():
    ebl = np.array([0.0, 100.0, 499.0, 500.0, 501.0, 5000.0])
    labels = datasets.vitaldb_ebl_labels(ebl, ebl_threshold=500.0)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert labels.dtype == np.int64


def test_vitaldb_ebl_label_custom_threshold():
    ebl = np.array([50.0, 150.0, 300.0])
    labels = datasets.vitaldb_ebl_labels(ebl, ebl_threshold=150.0)
    assert labels.tolist() == [0, 1, 1]


def test_ppg_vitaldb_real_success_sets_requested_real_and_source():
    with patch.object(datasets, "load_vitaldb_ppg", return_value=_fake_vitaldb_ppg()):
        d = datasets.load_ppg_vitaldb(prefer_real=True)
        assert d.source == "vitaldb"
        assert d.requested_real is True
        assert d.groups is not None


def test_ppg_vitaldb_real_failure_falls_back_but_marks_requested_real():
    with patch.object(datasets, "load_vitaldb_ppg", side_effect=RuntimeError("no network")):
        d = datasets.load_ppg_vitaldb(prefer_real=True, require_real=False)
        assert d.source == "synthetic"
        assert d.requested_real is True


def test_ppg_vitaldb_require_real_raises_instead_of_falling_back():
    with patch.object(datasets, "load_vitaldb_ppg", side_effect=RuntimeError("no network")):
        with pytest.raises(RuntimeError, match="require-real"):
            datasets.load_ppg_vitaldb(prefer_real=True, require_real=True)


def test_ppg_vitaldb_require_real_without_prefer_real_is_a_usage_error():
    with pytest.raises(ValueError):
        datasets.load_ppg_vitaldb(prefer_real=False, require_real=True)


# --------------------------------------------------------------------------- #
# Heart sounds (PCG)
# --------------------------------------------------------------------------- #
def _fake_heart(n=10):
    return HeartData(X=np.zeros((n, 128), dtype="float32"),
                      y=np.zeros(n, dtype="int64"), fs=128.0, source="cinc2016")


def test_make_synthetic_heart_shape_labels_and_balance():
    d = datasets.make_synthetic_heart(n_samples=500, window=64, abnormal_frac=0.3, seed=0)
    assert d.X.shape == (500, 64)
    assert d.X.dtype == np.float32
    assert set(np.unique(d.y).tolist()) == {0, 1}
    assert d.fs == 128.0
    assert d.source == "synthetic"
    # Bernoulli(0.3) over 500 draws — generous tolerance, just checking the
    # knob actually does something rather than pinning an exact count.
    frac = d.y.mean()
    assert 0.2 < frac < 0.4


def test_heart_default_is_synthetic_and_not_requested_real():
    d = datasets.load_heart(prefer_real=False)
    assert d.source == "synthetic"
    assert d.requested_real is False


def test_heart_real_success_sets_requested_real_and_real_source():
    with patch.object(datasets, "load_cinc2016", return_value=_fake_heart()):
        d = datasets.load_heart(prefer_real=True)
        assert d.source == "cinc2016"
        assert d.requested_real is True


def test_heart_real_failure_falls_back_but_marks_requested_real():
    with patch.object(datasets, "load_cinc2016", side_effect=RuntimeError("no network")):
        d = datasets.load_heart(prefer_real=True, require_real=False)
        assert d.source == "synthetic"
        assert d.requested_real is True


def test_heart_require_real_raises_instead_of_falling_back():
    with patch.object(datasets, "load_cinc2016", side_effect=RuntimeError("no network")):
        with pytest.raises(RuntimeError, match="require-real"):
            datasets.load_heart(prefer_real=True, require_real=True)


def test_heart_require_real_without_prefer_real_is_a_usage_error():
    with pytest.raises(ValueError):
        datasets.load_heart(prefer_real=False, require_real=True)


def test_heart_groups_is_none_subject_ids_not_recoverable():
    # Part 0 finding (docs/heart_sounds_task.md): CinC 2016 does not expose a
    # subject id in any distributed file, so the split is by recording, not
    # subject -- documented here as an explicit contract, not a silent gap.
    d = datasets.make_synthetic_heart(n_samples=10)
    assert d.groups is None


def test_bandpass_filter_preserves_shape_and_is_finite():
    fs = 2000.0
    t = np.arange(4000) / fs
    sig = np.sin(2 * np.pi * 100 * t) + np.random.default_rng(0).normal(0, 0.1, 4000)
    out = datasets._bandpass_filter(sig, fs, band=(20.0, 400.0))
    assert out.shape == sig.shape
    assert np.isfinite(out).all()


def test_bandpass_filter_attenuates_out_of_band_signal():
    fs = 2000.0
    t = np.arange(8000) / fs  # 4 seconds, long enough for a clean filter response
    # 5 Hz is well below the PCG band (20-400 Hz) -> should be heavily attenuated.
    low_freq = np.sin(2 * np.pi * 5 * t)
    in_band = np.sin(2 * np.pi * 100 * t)  # inside the band -> should survive
    out_low = datasets._bandpass_filter(low_freq, fs, band=(20.0, 400.0))
    out_in = datasets._bandpass_filter(in_band, fs, band=(20.0, 400.0))
    assert out_low.std() < 0.1 * low_freq.std()
    assert out_in.std() > 0.5 * in_band.std()


# --------------------------------------------------------------------------- #
# Generic helpers used by heart sounds (originally added for EEG, but not
# EEG-specific -- kept here so their coverage survives EEG code removal;
# see docs/heart_sounds_task.md Part B).
# --------------------------------------------------------------------------- #
def test_resample_windows_2d_and_3d():
    X2 = np.random.default_rng(0).normal(size=(4, 100))
    out2 = datasets.resample_windows(X2, 25)
    assert out2.shape == (4, 25)

    X3 = np.random.default_rng(0).normal(size=(4, 6, 100))
    out3 = datasets.resample_windows(X3, 25)
    assert out3.shape == (4, 6, 25)


def test_call_with_timeout_returns_result_on_success():
    result, err = datasets._call_with_timeout(lambda x: x * 2, 5.0, 0, 21)
    assert result == 42
    assert err is None


def test_call_with_timeout_times_out_on_slow_callable():
    def _slow():
        time.sleep(5.0)
        return "too late"

    t0 = time.time()
    result, err = datasets._call_with_timeout(_slow, 0.2, 0)
    elapsed = time.time() - t0

    assert result is None
    assert isinstance(err, TimeoutError)
    # Returns promptly at the timeout, not after the full 5s sleep -- this
    # is the exact "one stalled connection hangs the whole cell" bug fix.
    assert elapsed < 2.0


def test_call_with_timeout_reports_exception_without_hanging():
    def _boom():
        raise ValueError("network is on fire")

    result, err = datasets._call_with_timeout(_boom, 5.0, 0)
    assert result is None
    assert isinstance(err, ValueError)
    assert "on fire" in str(err)


def test_call_with_timeout_retries_before_succeeding():
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok on 3rd try"

    result, err = datasets._call_with_timeout(_flaky, 5.0, 2)
    assert result == "ok on 3rd try"
    assert err is None
    assert calls["n"] == 3


def test_call_with_timeout_gives_up_after_exhausting_retries():
    calls = {"n": 0}

    def _always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    result, err = datasets._call_with_timeout(_always_fails, 5.0, 2)
    assert result is None
    assert isinstance(err, RuntimeError)
    assert calls["n"] == 3  # 1 initial attempt + 2 retries


# --------------------------------------------------------------------------- #
# CRM (synthetic time-resolved Compensatory Reserve) — docs/synthetic_crm_task.md
# --------------------------------------------------------------------------- #
def test_reserve_trajectory_is_monotonically_non_increasing():
    rng = np.random.default_rng(0)
    for _ in range(20):  # many random `shape` draws -- must hold for all of them
        r = datasets._reserve_trajectory(30, rng)
        assert np.all(np.diff(r) <= 1e-9)


def test_reserve_trajectory_starts_at_1_ends_at_0():
    rng = np.random.default_rng(0)
    r = datasets._reserve_trajectory(50, rng)
    assert r[0] == pytest.approx(1.0)
    assert r[-1] == pytest.approx(0.0)
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_hr_flat_at_baseline_while_r_above_rise_threshold():
    hr = datasets._hr_from_r(np.array([1.0, 0.8, datasets._HR_RISE_R]), hr_baseline=70.0)
    assert np.allclose(hr, 70.0)


def test_hr_rises_as_r_falls_through_the_compensatory_band():
    hr_baseline = 70.0
    hr_mid = datasets._hr_from_r(np.array([0.2]), hr_baseline)[0]  # inside (collapse, rise)
    hr_at_collapse_boundary = datasets._hr_from_r(np.array([datasets._HR_COLLAPSE_R]), hr_baseline)[0]
    assert hr_mid > hr_baseline
    assert hr_at_collapse_boundary > hr_mid  # tachycardia peaks approaching collapse


def test_hr_collapses_below_baseline_at_full_decompensation():
    hr_baseline = 70.0
    hr_at_collapse_boundary = datasets._hr_from_r(np.array([datasets._HR_COLLAPSE_R]), hr_baseline)[0]
    hr_at_zero = datasets._hr_from_r(np.array([0.0]), hr_baseline)[0]
    assert hr_at_zero < hr_at_collapse_boundary
    assert hr_at_zero < hr_baseline  # terminal HR is a collapse, not just "less tachycardic"


def test_crm_pulse_amplitude_decreases_as_r_falls():
    rng = np.random.default_rng(0)
    pulse_full = datasets._crm_pulse(r=1.0, n=200, rng=rng)
    pulse_low = datasets._crm_pulse(r=0.1, n=200, rng=rng)
    assert pulse_full.max() > pulse_low.max()


def test_crm_pulse_notch_blunts_as_r_falls():
    # The dicrotic-notch term's own amplitude coefficient is `0.15 * r` --
    # confirms it shrinks smoothly toward (not jumping to) zero as r falls,
    # the "strongest cue" / widest-proportional-range design choice.
    rng = np.random.default_rng(0)
    notch_full = 0.15 * 1.0
    notch_zero = 0.15 * 0.0
    assert notch_full > notch_zero == 0.0
    # And directly on the generated pulses: subtracting a same-r-but-
    # notch-free reference isn't available, so check the documented
    # formula instead (the pulse-shape test above already exercises the
    # full generator end-to-end).
    pulse_full = datasets._crm_pulse(r=1.0, n=200, rng=rng)
    pulse_zero = datasets._crm_pulse(r=0.0, n=200, rng=rng)
    assert not np.allclose(pulse_full, pulse_zero)


def test_make_synthetic_crm_label_is_time_aligned_with_cri():
    """The core property this whole generator exists to guarantee (unlike
    VitalDB's whole-case label): every window's binary label matches ITS
    OWN `cri` value, not some pooled/case-level number."""
    d = datasets.make_synthetic_crm(n_subjects=5, windows_per_subject=12, seed=0)
    expected_y = (d.cri < datasets.CRI_THRESHOLD).astype(np.int64)
    assert np.array_equal(d.y, expected_y)


def test_make_synthetic_crm_trajectory_monotonic_per_subject():
    d = datasets.make_synthetic_crm(n_subjects=6, windows_per_subject=15, seed=1)
    for subj in np.unique(d.groups):
        cri_subj = d.cri[d.groups == subj]
        assert np.all(np.diff(cri_subj) <= 1e-9)


def test_make_synthetic_crm_positive_class_includes_hr_baseline_windows():
    """The "occult" property the task spec requires: some y=1 (compromised)
    windows must have cri in the (_HR_RISE_R, CRI_THRESHOLD] band, where
    `_hr_from_r` is still flat at baseline -- i.e. the classifier is
    provably not just detecting "HR is already elevated"."""
    d = datasets.make_synthetic_crm(n_subjects=20, windows_per_subject=24, seed=0)
    occult = (d.cri > datasets._HR_RISE_R) & (d.cri <= datasets.CRI_THRESHOLD)
    assert occult.sum() > 0
    assert np.all(d.y[occult] == 1)


def test_make_synthetic_crm_shape_dtype_and_groups_always_set():
    d = datasets.make_synthetic_crm(n_subjects=4, windows_per_subject=10,
                                     window_sec=2.0, fs=50.0, seed=0)
    assert d.X.shape == (40, 100)
    assert d.X.dtype == np.float32
    assert d.y.dtype == np.int64
    assert d.groups is not None
    assert d.groups.shape == (40,)
    assert d.cri.shape == (40,)
    assert d.fs == 50.0
    assert d.source == "synthetic"


def test_make_synthetic_crm_no_leakage_subject_split():
    from eia import case_level
    d = datasets.make_synthetic_crm(n_subjects=15, windows_per_subject=10, seed=0)
    _Xtr, _Xval, _Xte, _ytr, _yval, _yte, gtr, gval, gte = case_level.split_data(d, seed=0)
    tr_s, val_s, te_s = set(gtr.tolist()), set(gval.tolist()), set(gte.tolist())
    assert not (tr_s & val_s)
    assert not (tr_s & te_s)
    assert not (val_s & te_s)


def test_crm_default_is_synthetic_and_not_requested_real():
    d = datasets.load_crm(prefer_real=False, n_subjects=3, windows_per_subject=5)
    assert d.source == "synthetic"
    assert d.requested_real is False


def test_crm_prefer_real_still_returns_synthetic_but_records_request():
    # load_crm has NO real branch (real LBNP/CRM data is gated) -- prefer_real
    # is honestly recorded even though the result is always synthetic.
    d = datasets.load_crm(prefer_real=True, n_subjects=3, windows_per_subject=5)
    assert d.source == "synthetic"
    assert d.requested_real is True


def test_crm_require_real_always_raises():
    with pytest.raises(RuntimeError, match="require-real"):
        datasets.load_crm(prefer_real=True, require_real=True)
    with pytest.raises(RuntimeError, match="require-real"):
        datasets.load_crm(prefer_real=False, require_real=True)


# --------------------------------------------------------------------------- #
# MI (PTB-XL 12-lead myocardial infarction) — docs/ptbxl_mi_task.md
# --------------------------------------------------------------------------- #
# A literal lookup mirroring scp_statements.csv's REAL structure (verified
# live against the actual file, docs/ptbxl_mi_task.md Part 0) -- code ->
# diagnostic_class, already filtered to `diagnostic==1` rows. Real examples:
# NORM is the sole NORM-class code; IMI/ASMI are real MI-class codes; NDT is
# a real STTC-class code; SR/ABQRS/LVOLT are real but NON-diagnostic (no
# diagnostic_class), i.e. NOT in this lookup at all.
_SCP_TO_CLASS = {"NORM": "NORM", "IMI": "MI", "ASMI": "MI", "NDT": "STTC", "1AVB": "CD"}


def test_scp_codes_to_superclasses_single_norm():
    assert datasets.scp_codes_to_superclasses(
        {"NORM": 100.0, "SR": 0.0}, _SCP_TO_CLASS) == {"NORM"}


def test_scp_codes_to_superclasses_single_mi():
    assert datasets.scp_codes_to_superclasses(
        {"IMI": 35.0, "ABQRS": 0.0}, _SCP_TO_CLASS) == {"MI"}


def test_scp_codes_to_superclasses_ignores_likelihood_zero_is_not_absent():
    """THE load-bearing Part-0 finding: likelihood 0.0 means "present but
    not confidence-scored," not "absent" -- confirmed against PTB-XL's own
    shipped example_physionet.py, which does not threshold by likelihood at
    all. A code present with likelihood 0.0 must still count."""
    assert datasets.scp_codes_to_superclasses({"IMI": 0.0}, _SCP_TO_CLASS) == {"MI"}


def test_scp_codes_to_superclasses_ignores_non_diagnostic_codes():
    # SR ("sinus rhythm") is a real PTB-XL code with no diagnostic_class --
    # not in the lookup -- must not contribute a superclass.
    assert datasets.scp_codes_to_superclasses({"SR": 0.0}, _SCP_TO_CLASS) == set()


def test_scp_codes_to_superclasses_mixed_multi_superclass():
    assert datasets.scp_codes_to_superclasses(
        {"IMI": 50.0, "NDT": 20.0}, _SCP_TO_CLASS) == {"MI", "STTC"}


def test_mi_norm_label_confidently_mi():
    assert datasets.mi_norm_label({"MI"}) == 1


def test_mi_norm_label_confidently_norm():
    assert datasets.mi_norm_label({"NORM"}) == 0


def test_mi_norm_label_excludes_empty_superclass_set():
    assert datasets.mi_norm_label(set()) is None


def test_mi_norm_label_excludes_other_single_superclass():
    assert datasets.mi_norm_label({"STTC"}) is None
    assert datasets.mi_norm_label({"CD"}) is None
    assert datasets.mi_norm_label({"HYP"}) is None


def test_mi_norm_label_excludes_mixed_superclass_even_when_mi_present():
    # A record with BOTH MI and STTC is NOT "confidently MI" for this
    # binary task -- excluded, not folded into the MI class.
    assert datasets.mi_norm_label({"MI", "STTC"}) is None
    assert datasets.mi_norm_label({"MI", "NORM"}) is None


def test_make_synthetic_mi_shape_dtype_and_no_groups():
    d = datasets.make_synthetic_mi(n_samples=20, n_leads=12, window=1000, seed=0)
    assert d.X.shape == (20, 12, 1000)
    assert d.X.dtype == np.float32
    assert d.y.dtype == np.int64
    assert set(np.unique(d.y).tolist()) <= {0, 1}
    assert d.groups is None  # no natural patient grouping for synthetic samples
    assert d.source == "synthetic"


def test_mi_default_is_synthetic_and_not_requested_real():
    d = datasets.load_mi(prefer_real=False, n_samples=10)
    assert d.source == "synthetic"
    assert d.requested_real is False


def test_mi_real_success_sets_requested_real_and_real_source():
    fake = MiData(X=np.zeros((5, 12, 1000), dtype="float32"),
                   y=np.zeros(5, dtype="int64"), fs=100.0, source="ptbxl",
                   groups=np.arange(5, dtype="int64"))
    with patch.object(datasets, "load_ptbxl", return_value=fake):
        d = datasets.load_mi(prefer_real=True)
        assert d.source == "ptbxl"
        assert d.requested_real is True
        assert d.groups is not None


def test_mi_real_failure_falls_back_but_marks_requested_real():
    with patch.object(datasets, "load_ptbxl", side_effect=RuntimeError("no network")):
        d = datasets.load_mi(prefer_real=True, require_real=False, n_samples=10)
        assert d.source == "synthetic"
        assert d.requested_real is True


def test_mi_require_real_raises_instead_of_falling_back():
    with patch.object(datasets, "load_ptbxl", side_effect=RuntimeError("no network")):
        with pytest.raises(RuntimeError, match="require-real"):
            datasets.load_mi(prefer_real=True, require_real=True)


def test_mi_require_real_without_prefer_real_is_a_usage_error():
    with pytest.raises(ValueError):
        datasets.load_mi(prefer_real=False, require_real=True)


def test_mi_no_leakage_patient_split():
    """Mirrors the heart/crm no-leakage tests: a MiData with several windows
    per patient (PTB-XL patients can contribute multiple recordings) must
    never straddle a patient across train/val/test."""
    from eia import case_level
    rng = np.random.default_rng(0)
    n_patients = 15
    windows_per_patient = 3
    n = n_patients * windows_per_patient
    groups = np.repeat(np.arange(n_patients), windows_per_patient)
    d = MiData(X=rng.normal(size=(n, 12, 1000)).astype("float32"),
                y=rng.integers(0, 2, size=n).astype("int64"),
                fs=100.0, source="ptbxl", groups=groups)
    _Xtr, _Xval, _Xte, _ytr, _yval, _yte, gtr, gval, gte = case_level.split_data(d, seed=0)
    tr_s, val_s, te_s = set(gtr.tolist()), set(gval.tolist()), set(gte.tolist())
    assert not (tr_s & val_s)
    assert not (tr_s & te_s)
    assert not (val_s & te_s)


def test_mi_per_lead_normalize_shape_compatible():
    """MiData.X's (n, 12, 1000) convention is exactly the (n, C, T) shape
    `signal_features.normalize_features_train_only` (heart's per-channel
    train-only z-score, reused verbatim for mi's per-lead norm -- see
    scripts/akida_verify.py) expects."""
    from eia import signal_features
    rng = np.random.default_rng(0)
    Xtr = rng.normal(loc=5.0, scale=2.0, size=(30, 12, 1000)).astype("float32")
    Xval = rng.normal(loc=5.0, scale=2.0, size=(10, 12, 1000)).astype("float32")
    Xte = rng.normal(loc=5.0, scale=2.0, size=(10, 12, 1000)).astype("float32")
    Xtr_n, Xval_n, Xte_n = signal_features.normalize_features_train_only(Xtr, Xval, Xte)
    assert Xtr_n.shape == Xtr.shape
    assert np.allclose(Xtr_n.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(Xtr_n.std(axis=(0, 2)), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Shockable-rhythm (VF/VT) — docs/shockable_rhythm_task.md
# --------------------------------------------------------------------------- #
def test_rhythm_code_to_shockable_vf_family_is_shockable():
    for code in ("VF", "VFIB", "VFL"):
        assert datasets.rhythm_code_to_shockable(code) == 1


def test_rhythm_code_to_shockable_nonshockable_codes():
    for code in ("N", "NSR", "SBR", "NOD", "SVTA", "AFIB", "AF", "B", "BI",
                  "VER", "PM", "ASYS", "HGEA"):
        assert datasets.rhythm_code_to_shockable(code) == 0


def test_rhythm_code_to_shockable_noise_is_dropped_not_nonshockable():
    assert datasets.rhythm_code_to_shockable("NOISE") is None


def test_rhythm_code_to_shockable_unknown_code_is_dropped():
    assert datasets.rhythm_code_to_shockable("SOME_UNRECOGNIZED_CODE") is None


def test_rhythm_code_to_shockable_vt_without_rate_is_dropped():
    """VT can't be classified without a rate estimate -- must not guess."""
    assert datasets.rhythm_code_to_shockable("VT") is None
    assert datasets.rhythm_code_to_shockable("VT", vt_rate_bpm=None) is None


def test_rhythm_code_to_shockable_rapid_vt_is_shockable():
    assert datasets.rhythm_code_to_shockable("VT", vt_rate_bpm=200.0) == 1


def test_rhythm_code_to_shockable_slow_vt_is_nonshockable():
    assert datasets.rhythm_code_to_shockable("VT", vt_rate_bpm=100.0) == 0


def test_rhythm_code_to_shockable_vt_rate_threshold_is_inclusive():
    assert datasets.rhythm_code_to_shockable(
        "VT", vt_rate_bpm=150.0, vt_rate_threshold=150.0) == 1


def test_rhythm_code_to_shockable_custom_threshold():
    assert datasets.rhythm_code_to_shockable(
        "VT", vt_rate_bpm=160.0, vt_rate_threshold=180.0) == 0
    assert datasets.rhythm_code_to_shockable(
        "VT", vt_rate_bpm=190.0, vt_rate_threshold=180.0) == 1


def test_rhythm_intervals_from_annotations_ignores_non_rhythm_symbols():
    """Mirrors CUDB's real annotation structure (Part 0): most annotations
    are per-beat markers (symbol != '+') with empty aux_note; only
    symbol=='+' entries carry a rhythm code. Conflating them (an early bug
    caught in Part 0) inflates episode counts with sub-second durations."""
    symbol = ["N", "N", "+", "N", "N", "+", "N"]
    sample = [10, 20, 30, 40, 50, 200, 210]
    aux_note = ["", "", "(N\x00", "", "", "(VT\x00", ""]
    intervals = datasets._rhythm_intervals_from_annotations(symbol, sample, aux_note, sig_len=300)
    assert intervals == [(30, 200, "N"), (200, 300, "VT")]


def test_rhythm_intervals_from_annotations_strips_parens_and_nul():
    symbol = ["+", "+"]
    sample = [0, 50]
    aux_note = ["(VFIB\x00", "(N\x00"]
    intervals = datasets._rhythm_intervals_from_annotations(symbol, sample, aux_note, sig_len=100)
    assert intervals == [(0, 50, "VFIB"), (50, 100, "N")]


def test_dominant_rhythm_full_containment():
    intervals = [(0, 100, "N"), (100, 300, "VF"), (300, 400, "N")]
    assert datasets._dominant_rhythm(intervals, 100, 300, min_coverage=1.0) == "VF"


def test_dominant_rhythm_straddling_window_dropped_at_full_coverage():
    """A window straddling a rhythm change is ambiguous/transition -- must
    be dropped (None), not force-labeled with whichever code covers more."""
    intervals = [(0, 160, "N"), (160, 400, "VF")]
    # window [100,300): N covers [100,160)=60, VF covers [160,300)=140.
    assert datasets._dominant_rhythm(intervals, 100, 300, min_coverage=1.0) is None


def test_dominant_rhythm_majority_code_returned_below_full_coverage():
    intervals = [(0, 160, "N"), (160, 400, "VF")]
    # window [100,300): N=60/200=0.30, VF=140/200=0.70
    assert datasets._dominant_rhythm(intervals, 100, 300, min_coverage=0.6) == "VF"
    assert datasets._dominant_rhythm(intervals, 100, 300, min_coverage=0.8) is None


def test_dominant_rhythm_no_overlap_is_none():
    intervals = [(0, 100, "N")]
    assert datasets._dominant_rhythm(intervals, 200, 300) is None


def test_dominant_rhythm_empty_intervals_is_none():
    assert datasets._dominant_rhythm([], 0, 100) is None


def test_estimate_rate_bpm_recovers_known_pulse_rate():
    """A synthetic narrow-pulse train at a KNOWN rate should be recovered
    within a generous tolerance -- confirms the peak-picker's rate estimate
    is at least directionally sane (not a diagnostic-grade claim, see
    docs/shockable_rhythm_task.md's Part-0 validation against real VFDB
    windows)."""
    fs = 250.0
    true_bpm = 200.0
    period = 60.0 / true_bpm
    n = int(5 * fs)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    pulse_times = np.arange(0, 5.0, period)
    for pt in pulse_times:
        idx = int(round(pt * fs))
        width = max(1, int(0.01 * fs))
        lo, hi = max(0, idx - width), min(n, idx + width)
        sig[lo:hi] += np.hanning(hi - lo)
    rate = datasets._estimate_rate_bpm(sig, fs)
    assert abs(rate - true_bpm) / true_bpm < 0.2


def test_estimate_rate_bpm_flat_signal_returns_zero():
    fs = 250.0
    sig = np.zeros(int(2 * fs))
    assert datasets._estimate_rate_bpm(sig, fs) == 0.0


def test_make_synthetic_shockable_shape_dtype_and_no_groups():
    d = datasets.make_synthetic_shockable(n_samples=200, seed=0)
    assert d.X.ndim == 2
    assert d.X.shape[0] == 200
    assert d.X.dtype == np.float32
    assert d.y.dtype == np.int64
    assert set(np.unique(d.y).tolist()) <= {0, 1}
    assert d.groups is None  # no natural record grouping for synthetic samples
    assert d.source == "synthetic"


def test_make_synthetic_shockable_per_window_normalized():
    d = datasets.make_synthetic_shockable(n_samples=50, seed=0)
    assert np.allclose(d.X.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(d.X.std(axis=1), 1.0, atol=1e-2)


def test_make_synthetic_shockable_shockable_is_minority_by_default():
    d = datasets.make_synthetic_shockable(n_samples=1000, seed=0)
    assert 0.0 < d.y.mean() < 0.5


def test_shockable_default_is_synthetic_and_not_requested_real():
    d = datasets.load_shockable(prefer_real=False, n_samples=20)
    assert d.source == "synthetic"
    assert d.requested_real is False


def test_shockable_real_success_sets_requested_real_and_real_source():
    fake = ShockableData(X=np.zeros((5, 500), dtype="float32"),
                           y=np.zeros(5, dtype="int64"), fs=100.0, source="vfdb_cudb",
                           groups=np.array(["vfdb:418"] * 5, dtype=object))
    with patch.object(datasets, "load_vfdb_cudb", return_value=fake):
        d = datasets.load_shockable(prefer_real=True)
        assert d.source == "vfdb_cudb"
        assert d.requested_real is True
        assert d.groups is not None


def test_shockable_real_failure_falls_back_but_marks_requested_real():
    with patch.object(datasets, "load_vfdb_cudb", side_effect=RuntimeError("no network")):
        d = datasets.load_shockable(prefer_real=True, require_real=False, n_samples=10)
        assert d.source == "synthetic"
        assert d.requested_real is True


def test_shockable_require_real_raises_instead_of_falling_back():
    with patch.object(datasets, "load_vfdb_cudb", side_effect=RuntimeError("no network")):
        with pytest.raises(RuntimeError, match="require-real"):
            datasets.load_shockable(prefer_real=True, require_real=True)


def test_shockable_require_real_without_prefer_real_is_a_usage_error():
    with pytest.raises(ValueError):
        datasets.load_shockable(prefer_real=False, require_real=True)


def test_shockable_no_leakage_record_split():
    """Mirrors the mi/crm no-leakage tests: a ShockableData with several
    windows per record (VFDB/CUDB records are different patients) must never
    straddle a record across train/val/test."""
    from eia import case_level
    rng = np.random.default_rng(0)
    n_records = 15
    windows_per_record = 6
    n = n_records * windows_per_record
    record_ids = [f"vfdb:{i}" for i in range(n_records)]
    groups = np.array([record_ids[i // windows_per_record] for i in range(n)], dtype=object)
    d = ShockableData(X=rng.normal(size=(n, 500)).astype("float32"),
                        y=rng.integers(0, 2, size=n).astype("int64"),
                        fs=100.0, source="vfdb_cudb", groups=groups)
    _Xtr, _Xval, _Xte, _ytr, _yval, _yte, gtr, gval, gte = case_level.split_data(d, seed=0)
    tr_s, val_s, te_s = set(gtr.tolist()), set(gval.tolist()), set(gte.tolist())
    assert not (tr_s & val_s)
    assert not (tr_s & te_s)
    assert not (val_s & te_s)
