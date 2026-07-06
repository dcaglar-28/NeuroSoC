# EIA — Phase-0 software prototype

Event-driven multimodal edge diagnostics. This repo is the **Phase-0** proof: a
laptop-runnable simulation showing that an *event-driven* (spiking) pipeline can
diagnose from physiological signals at accuracy comparable to a conventional
network, while processing far fewer operations — the basis for the low-power,
offline field device described in the project brief.

Two modalities run end-to-end — **ECG beat classification** and **PPG /
hemorrhage-proxy classification** — each with the same event encoder, spiking
neural network, conventional baseline, and analytical energy comparison.
Additional modalities (heart & lung sounds, EEG) plug into the same pattern.

## What's here

```
src/eia/
  encoding.py         # delta / level-crossing event encoders (NumPy, no torch)
  energy.py           # analytical MAC-vs-SOP energy model (NumPy, no torch)
  datasets.py         # ECG (MIT-BIH) + PPG (BIDMC) loaders, real (wfdb) or synthetic
  report.py           # data_card(): per-dataset summary + red-flag warnings (NumPy)
  viz.py              # plots: waveforms per class, delta encoding scheme, class balance
  models.py           # spiking classifier (snnTorch) + conventional baseline
  rockpool_models.py  # hardware-target sibling: Rockpool -> Xylo mapping/quantization
  device.py           # picks MPS (Apple Silicon) / CUDA / CPU
  train.py            # end-to-end demo: trains both, prints accuracy + energy
scripts/
  xylo_verify.py      # trains the PPG net and verifies it on the XyloSim bit-precise simulator
notebooks/
  01_ecg_snn.ipynb   # thin: data card + waveform/encoding plots + train (local or Colab)
tests/          # unit tests for the numpy-only modules (run without torch)
```

The design rule: **logic lives in the package, notebooks stay thin.** The same
code runs in VSCode locally and in Colab — nothing is duplicated.

## Setup (macOS, Apple Silicon)

Use the native arm64 Python (Homebrew or python.org), not x86 under Rosetta.

```bash
cd eia
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # installs torch, snntorch, etc. (arm64 builds)
```

## Run it

```bash
python -m eia.train                            # ECG, synthetic data — no download, quick
python -m eia.train --modality ppg             # PPG, synthetic data
python -m eia.train --real                     # real MIT-BIH (needs: pip install 'eia[data]' + network)
python -m eia.train --modality ppg --real      # real BIDMC PPG & Respiration dataset
python -m eia.train --sweep                    # trace the accuracy vs. energy trade-off
python -m eia.train --modality ppg --sweep --real
```

Expected output: baseline vs. SNN accuracy, the SNN's mean spike rate, and an
energy report (dense MACs vs. spiking SOPs, and the ratio).

### The energy advantage is not automatic

A spiking network is only cheaper than a dense one when it is **sparse** (few
neurons firing) and uses **few timesteps**. Operations scale with
`spike_rate x timesteps`, so a dense-firing SNN over many timesteps does *more*
work, not less. Two levers control this:

- `--threshold` — higher delta threshold => fewer input events => sparser activity.
- `--spike-reg` — penalty that pushes the firing rate down during training.

`--sweep` runs a grid over timesteps and thresholds and marks the settings where
the event-driven pipeline actually beats the dense baseline. Producing that
accuracy/energy curve — how sparse you can go while keeping diagnostic accuracy
— is the real Phase-0 result.

## Verify on XyloSim (pre-hardware check)

Before buying a Xylo HDK, confirm each modality's SNN survives being mapped,
8-bit quantized, and run on **XyloSim** — the bit-precise simulator whose
traces match real silicon — and check how the modalities compose onto one
chip:

```bash
pip install "eia[xylo]"                     # pulls in rockpool[xylo] (XyloSim, samna)
python scripts/xylo_verify.py               # ecg + ppg, synthetic, + one-chip co-residence check
python scripts/xylo_verify.py --modality ppg           # just PPG
python scripts/xylo_verify.py --modality ecg --real    # just ECG, real MIT-BIH (needs wfdb + network)
python scripts/xylo_verify.py --real                   # both, real MIT-BIH + BIDMC
python scripts/xylo_verify.py --no-combined            # skip the Part C one-chip check
```

For each modality: a data card, then float-model accuracy, XyloSim accuracy,
and their agreement rate, then the mapped resource footprint (input/hidden/
output neurons used vs. Xylo's limits). When both modalities run, also prints
a **one-chip composition line** (`xylo_budget.fits_one_chip`) and a
**co-residence check** — combines the two independently-trained nets onto one
Xylo core (block-diagonal weights, single shared quantization) and confirms
each modality's decisions still agree with its own standalone float model.

See `docs/xylo_verification_task.md` and `docs/per_modality_xylo_verify_task.md`
for the full specs and known gotchas — this exact readout (2 output neurons,
spike-count logits) is prone to a dead-neuron collapse that only a positive
initial bias fixes, the official tutorial's `PeriodicExponential` surrogate
measurably regresses this net (kept on the evidence, not the tutorial's
prior), and co-residence measurably costs some XyloSim fidelity vs. either
modality's standalone number under a shared 8-bit weight budget. All
documented in `rockpool_models.py` and `scripts/xylo_verify.py`.

## Run the tests

```bash
pip install pytest
pytest -q                  # numpy-only tests; no torch or network needed
```

## Inspect the data + encoding

Before training, look at what you're feeding the model. `report.data_card()`
prints a per-dataset summary with warnings; `viz` draws the waveforms and the
delta encoding.

```python
from eia.datasets import load_ppg
from eia import report, viz

data = load_ppg(prefer_real=False)
report.data_card(data)                 # source, label, class balance, warnings
viz.plot_waveforms(data)               # example windows per class
viz.plot_encoding(data.X[0], threshold=0.2, fs=data.fs)   # the ON/OFF spike raster
```

The notebook `notebooks/01_ecg_snn.ipynb` walks through all of this. To refresh
its saved outputs (data card + plots + results) from the command line:

```bash
pip install jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/01_ecg_snn.ipynb
```

Then commit/push so the Colab copy (which clones from GitHub) picks it up.

## Local vs. Colab

Develop locally — an M-series Mac runs this comfortably on the MPS GPU, with a
faster edit/run loop than Colab. `notebooks/01_ecg_snn.ipynb` also runs on Colab
(free GPU) as a fallback for heavier models; edit the `REPO` URL in its setup
cell after you push to GitHub. Some spiking ops lack MPS kernels and fall back to
CPU automatically — harmless at this scale; pass `--device cpu` if you hit an
unsupported-op error.

## Push to GitHub

```bash
git init
git add .
git commit -m "EIA Phase-0: ECG event-driven prototype"
git branch -M main
git remote add origin https://github.com/USERNAME/eia.git
git push -u origin main
```

## Notes on rigor

- **Synthetic data** exists so the pipeline runs the moment you clone it. Real
  claims should use `--real` (MIT-BIH for ECG, BIDMC for PPG) and, as you add
  modalities, the corresponding public datasets (PTB-XL, PhysioNet CinC 2016
  heart sounds, ICBHI respiratory sounds, CHB-MIT EEG).
- **PPG label is a proxy, not a hemorrhage label.** BIDMC (ICU patients, no
  induced hypovolemia) has no blood-loss annotation, so `--modality ppg --real`
  labels windows by SpO2 desaturation (< 95%) as an accessible real-data stand-in
  for physiological compromise. It proves the binary-window pipeline pattern on
  real waveforms; it is not a validated hemorrhage/Compensatory-Reserve signal.
  The synthetic PPG generator (`make_synthetic_ppg`) is closer in spirit to the
  actual target: reduced pulse amplitude + blunted dicrotic notch, the
  waveform-shape changes CRI relies on. A real hemorrhage claim needs an LBNP
  (lower-body negative pressure) or induced-hypovolemia PPG dataset.
- **Energy is analytical**, not measured — MAC/SOP counts times per-op energy
  constants (`energy.py`, documented and swappable). Phase 2 replaces this with
  real FPGA power measurements.
