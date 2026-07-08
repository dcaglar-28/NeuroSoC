"""Offline unit tests for the EEG/CHB-MIT pieces (montage selection,
band-pass, resample-to-timesteps, summary parsing, patient-alias grouping,
within-patient split, provenance guarding) — no network. See
docs/eeg_seizure_task.md."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

from eia import case_level, datasets, report
from eia.datasets import EegData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_select_montage_shape_and_order():
    sig = np.arange(20).reshape(5, 4).astype(float)  # 5 samples, 4 channels
    sig_name = ["A", "B", "C", "D"]
    out = datasets.select_montage(sig, sig_name, ["C", "A"])
    assert out.shape == (5, 2)
    assert np.array_equal(out[:, 0], sig[:, 2])  # "C"
    assert np.array_equal(out[:, 1], sig[:, 0])  # "A"


def test_select_montage_raises_on_missing_channel():
    sig = np.zeros((5, 2))
    with pytest.raises(ValueError, match="missing montage"):
        datasets.select_montage(sig, ["A", "B"], ["A", "Z"])


def test_bandpass_eeg_preserves_shape_and_is_finite():
    rng = np.random.default_rng(0)
    fs = 256.0
    t = np.arange(512) / fs
    sig = np.stack([np.sin(2 * np.pi * 10 * t) + rng.normal(0, 0.1, 512)
                     for _ in range(3)], axis=1)  # (512, 3)
    out = datasets.bandpass_eeg(sig, fs, band=(0.5, 25.0))
    assert out.shape == sig.shape
    assert np.isfinite(out).all()


def test_bandpass_eeg_attenuates_out_of_band_signal():
    fs = 256.0
    t = np.arange(2560) / fs  # 10 seconds, long enough for a clean filter response
    # 100 Hz is well outside the (0.5, 25) Hz seizure band -> should be heavily attenuated.
    high_freq = np.sin(2 * np.pi * 100 * t)[:, None]
    low_freq = np.sin(2 * np.pi * 5 * t)[:, None]  # inside the band -> should survive
    out_high = datasets.bandpass_eeg(high_freq, fs)
    out_low = datasets.bandpass_eeg(low_freq, fs)
    assert out_high.std() < 0.1 * high_freq.std()
    assert out_low.std() > 0.5 * low_freq.std()


def test_resample_windows_2d_and_3d():
    X2 = np.random.default_rng(0).normal(size=(4, 100))
    out2 = datasets.resample_windows(X2, 25)
    assert out2.shape == (4, 25)

    X3 = np.random.default_rng(0).normal(size=(4, 6, 100))
    out3 = datasets.resample_windows(X3, 25)
    assert out3.shape == (4, 6, 25)


def test_eeg_canonical_patient_merges_chb21_into_chb01():
    assert datasets.eeg_canonical_patient("chb21") == "chb01"
    assert datasets.eeg_canonical_patient("chb01") == "chb01"
    assert datasets.eeg_canonical_patient("chb05") == "chb05"


def test_parse_chbmit_summary_singular_seizure_format():
    text = """Data Sampling Rate: 256 Hz

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_05.edf
File Start Time: 15:43:19
File End Time: 16:43:19
Number of Seizures in File: 0
"""
    parsed = datasets.parse_chbmit_summary(text)
    assert parsed["chb01_03"] == [(2996, 3036)]
    assert parsed["chb01_05"] == []


def test_parse_chbmit_summary_numbered_seizure_format():
    text = """File Name: chb04_08.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 1679 seconds
Seizure 1 End Time: 1781 seconds
Seizure 2 Start Time: 3782 seconds
Seizure 2 End Time: 3898 seconds
"""
    parsed = datasets.parse_chbmit_summary(text)
    assert parsed["chb04_08"] == [(1679, 1781), (3782, 3898)]


def test_label_seizure_window_overlap_logic():
    seizures = [(100, 140)]
    assert datasets._label_seizure_window(96, 100, seizures) == 0   # ends exactly at onset
    assert datasets._label_seizure_window(100, 104, seizures) == 1  # starts at onset
    assert datasets._label_seizure_window(130, 134, seizures) == 1  # fully inside
    assert datasets._label_seizure_window(139, 143, seizures) == 1  # straddles offset
    assert datasets._label_seizure_window(140, 144, seizures) == 0  # starts exactly at offset
    assert datasets._label_seizure_window(0, 4, seizures) == 0      # far before


def _fake_eeg(n=10):
    return EegData(X=np.zeros((n, 6, 128), dtype="float32"),
                    y=np.zeros(n, dtype="int64"), fs=32.0, source="chbmit",
                    groups=np.array(["chb01"] * n))


def test_eeg_real_success_sets_requested_real_and_source():
    with patch.object(datasets, "load_chbmit", return_value=_fake_eeg()):
        d = datasets.load_eeg(prefer_real=True)
        assert d.source == "chbmit"
        assert d.requested_real is True
        assert d.groups is not None


def test_eeg_real_failure_falls_back_but_marks_requested_real():
    with patch.object(datasets, "load_chbmit", side_effect=RuntimeError("no network")):
        d = datasets.load_eeg(prefer_real=True, require_real=False)
        assert d.source == "synthetic"
        assert d.requested_real is True


def test_eeg_require_real_raises_instead_of_falling_back():
    with patch.object(datasets, "load_chbmit", side_effect=RuntimeError("no network")):
        with pytest.raises(RuntimeError, match="require-real"):
            datasets.load_eeg(prefer_real=True, require_real=True)


def test_eeg_require_real_without_prefer_real_is_a_usage_error():
    with pytest.raises(ValueError):
        datasets.load_eeg(prefer_real=False, require_real=True)


def test_make_synthetic_eeg_shape_and_labels():
    d = datasets.make_synthetic_eeg(n_samples=50, window=128, seed=0)
    assert d.X.shape == (50, len(datasets.EEG_MONTAGE), 128)
    assert set(np.unique(d.y).tolist()) <= {0, 1}
    assert d.source == "synthetic"


def test_data_card_reports_correct_window_for_3d_eeg_X():
    d = datasets.make_synthetic_eeg(n_samples=20, window=128, fs=32.0, seed=0)
    card = report.data_card(d, verbose=False)
    assert card.window == 128
    assert card.duration_s == pytest.approx(128 / 32.0)


def test_group_split_merges_chb21_and_chb01_no_leakage():
    """Simulates the loader's canonicalization: windows nominally from
    subject chb21 must land in the SAME group as chb01's, so a group-split
    never lets one patient's data straddle train/test under two ids."""
    rng = np.random.default_rng(0)
    raw_subjects = np.array(
        (["chb01"] * 10) + (["chb21"] * 10) + (["chb05"] * 10)
        + (["chb08"] * 10) + (["chb10"] * 10))
    canonical_groups = np.array([datasets.eeg_canonical_patient(s) for s in raw_subjects])
    assert set(canonical_groups.tolist()) == {"chb01", "chb05", "chb08", "chb10"}  # chb21 folded in

    y = (canonical_groups == "chb05").astype("int64")  # arbitrary consistent-per-group label
    X = rng.normal(size=(50, 6, 16)).astype("float32")

    class _D:
        pass

    d = _D()
    d.X, d.y, d.groups = X, y, canonical_groups
    Xtr, Xval, Xte, ytr, yval, yte, gtr, gval, gte = case_level.split_data(d, seed=0)
    tr_groups, val_groups, te_groups = set(gtr.tolist()), set(gval.tolist()), set(gte.tolist())
    assert not (tr_groups & val_groups)
    assert not (tr_groups & te_groups)
    assert not (val_groups & te_groups)


def _fake_patient_eeg(record_labels: dict, n_windows_per_record: int = 10,
                       patient: str = "chb01", other_patient_one_record: bool = True):
    """Build a tiny synthetic EegData for one patient with several records,
    each record a mix of the two classes (mirrors real CHB-MIT: a record
    with a seizure also has plenty of non-seizure windows either side of
    it). `record_labels`: {record_name: pos_frac} — fraction of that
    record's windows labelled seizure (1).
    """
    rng = np.random.default_rng(0)
    X_list, y_list, groups_list, rec_list = [], [], [], []
    for rec_name, pos_frac in record_labels.items():
        n_pos = round(n_windows_per_record * pos_frac)
        labels = [1] * n_pos + [0] * (n_windows_per_record - n_pos)
        for lab in labels:
            X_list.append(rng.normal(size=(6, 16)).astype("float32"))
            y_list.append(lab)
            groups_list.append(patient)
            rec_list.append(rec_name)

    if other_patient_one_record:
        # A second patient with only 1 record -- should be ineligible.
        for lab in [0, 1, 0, 0]:
            X_list.append(rng.normal(size=(6, 16)).astype("float32"))
            y_list.append(lab)
            groups_list.append("chb99")
            rec_list.append("chb99_01")

    return EegData(
        X=np.stack(X_list), y=np.array(y_list, dtype="int64"), fs=32.0,
        source="chbmit", groups=np.array(groups_list, dtype="<U8"),
        record_ids=np.array(rec_list, dtype="<U16"))


def test_eeg_patient_specific_split_no_leakage():
    d = _fake_patient_eeg({"chb01_01": 0.3, "chb01_02": 0.3, "chb01_03": 0.0,
                            "chb01_04": 0.0})
    split = datasets.eeg_patient_specific_split(d, "chb01", seed=0)
    assert split is not None
    Xtr, Xval, Xte, ytr, yval, yte, rec_tr, rec_val, rec_te = split

    te_records = set(rec_te.tolist())
    tr_records = set(rec_tr.tolist())
    # The TEST records must never also appear in TRAIN (val is carved from
    # train windows, so it's allowed to share records with train by design).
    assert not (te_records & tr_records)
    assert len(te_records) >= 1
    assert len(tr_records) >= 1


def test_eeg_patient_specific_split_both_classes_in_train_and_test():
    d = _fake_patient_eeg({"chb01_01": 0.3, "chb01_02": 0.3, "chb01_03": 0.0,
                            "chb01_04": 0.0})
    split = datasets.eeg_patient_specific_split(d, "chb01", seed=0)
    assert split is not None
    _Xtr, _Xval, _Xte, ytr, _yval, yte = split[:6]
    assert set(np.unique(ytr).tolist()) == {0, 1}
    assert set(np.unique(yte).tolist()) == {0, 1}


def test_eeg_patient_specific_split_returns_none_for_single_record_patient():
    d = _fake_patient_eeg({"chb01_01": 0.3}, other_patient_one_record=False)
    assert datasets.eeg_patient_specific_split(d, "chb01", seed=0) is None


def test_eeg_patient_specific_split_returns_none_for_unknown_patient():
    d = _fake_patient_eeg({"chb01_01": 0.3, "chb01_02": 0.3})
    assert datasets.eeg_patient_specific_split(d, "chb42", seed=0) is None


def test_eeg_patient_specific_split_skips_when_no_permutation_balances_classes():
    # Only ONE record has any positive windows at all -> whichever record
    # holds the positives, the other side of the split has zero of them.
    d = _fake_patient_eeg({"chb01_01": 1.0, "chb01_02": 0.0, "chb01_03": 0.0})
    assert datasets.eeg_patient_specific_split(d, "chb01", seed=0) is None


def test_eeg_eligible_patients_requires_at_least_two_records():
    from xylo_verify import _eeg_eligible_patients

    d = _fake_patient_eeg({"chb01_01": 0.3, "chb01_02": 0.3})
    eligible = _eeg_eligible_patients(d)
    assert "chb01" in eligible
    assert "chb99" not in eligible  # only 1 record for chb99
