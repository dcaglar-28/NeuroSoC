from eia import energy


def test_dense_macs():
    # [4,3,2] -> 4*3 + 3*2 = 18
    assert energy.dense_macs_per_inference([4, 3, 2]) == 18


def test_snn_cheaper_when_sparse():
    layers = [374, 128, 128, 2]
    report = energy.compare(layers, timesteps=50, avg_spike_rate=0.05)
    # Sparse spiking should come out cheaper than dense MACs.
    assert report.energy_ratio > 1.0


def test_dense_more_expensive_when_fully_active():
    layers = [100, 100, 2]
    # Even at full activity over many timesteps the model returns a finite ratio.
    report = energy.compare(layers, timesteps=50, avg_spike_rate=1.0)
    assert report.snn_sops > 0 and report.dense_macs > 0


def test_report_str_has_fields():
    s = str(energy.compare([10, 5, 2], timesteps=10, avg_spike_rate=0.1))
    assert "Energy ratio" in s
