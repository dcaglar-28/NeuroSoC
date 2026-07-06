from eia import xylo_budget as xb


def test_output_neurons_bind_first():
    # Default modality = 2 in / 63 hidden / 2 out.
    # input cap = 16//2 = 8, hidden cap = 1000//63 = 15, output cap = 8//2 = 4.
    r = xb.fits_one_chip(xb.default_modalities(["ecg"]))
    assert r.binding == "output neurons"
    assert r.max_modalities == 4


def test_four_modalities_fit_one_chip():
    r = xb.fits_one_chip(xb.default_modalities(["ecg", "ppg", "resp", "audio"]))
    assert r.fits
    assert r.total_out == 8 and r.total_out <= xb.XYLO_MAX_OUTPUT_CHANNELS


def test_five_modalities_overflow_outputs():
    r = xb.fits_one_chip(xb.default_modalities(["ecg", "ppg", "resp", "audio", "eeg"]))
    assert not r.fits
    assert r.total_out > xb.XYLO_MAX_OUTPUT_CHANNELS


def test_report_str():
    s = str(xb.fits_one_chip(xb.default_modalities(["ecg", "ppg"])))
    assert "one-chip budget" in s and "binding" in s
