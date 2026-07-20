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

**Hardware note:** the committed production target is the **BrainChip Akida**
family (single vendor, MetaTF toolchain); **XyloSim** (below) is the current
bit-exact validation vehicle, not the production silicon — see CLAUDE.md's
"Hardware target" section for the full decision record.

## What's here

```
src/eia/
  encoding.py         # delta / level-crossing event encoders (NumPy, no torch)
  energy.py           # analytical MAC-vs-SOP energy model (NumPy, no torch)
  datasets.py         # ECG (MIT-BIH) + PPG (BIDMC, VitalDB) + EEG (CHB-MIT) loaders, real or synthetic
  report.py           # data_card(): per-dataset summary + red-flag warnings (NumPy)
  viz.py              # plots: waveforms per class, delta encoding scheme, class balance
  models.py           # spiking classifier (snnTorch) + conventional baseline
  rockpool_models.py  # hardware-target sibling: Rockpool -> Xylo mapping/quantization
  device.py           # picks MPS (Apple Silicon) / CUDA / CPU
  train.py            # end-to-end demo: trains both, prints accuracy + energy
scripts/
  xylo_verify.py      # trains the PPG net and verifies it on the XyloSim bit-precise simulator
notebooks/
  03_akida_ecg.ipynb  # thin: Akida ECG port, first slice (Linux/Colab only)
  04_mvp_pitch.ipynb  # self-contained MVP pitch/validation notebook (numpy+matplotlib only)
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
python -m eia.train --modality ppg --real --ppg-source vitaldb  # real VitalDB blood-loss label
python -m eia.train --sweep                    # trace the accuracy vs. energy trade-off
python -m eia.train --modality ppg --sweep --real
python -m eia.train --modality eeg --real      # real CHB-MIT seizure detection
python -m eia.train --real --require-real      # fail loudly instead of silently using synthetic
                                                # if the real dataset can't be loaded
```

Every run prints a `[data] ... provenance=...` line (or, in `xylo_verify.py`, a
full data card) so you can always see from the output alone whether a run
actually used real data, synthetic by request, or synthetic as a fallback.
`--real` alone still falls back to synthetic if the real dataset can't load —
but that fallback always prints a `[warn]` line naming why, never silently.
Add `--require-real` to turn that fallback into a hard error instead.

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
python scripts/xylo_verify.py --real --require-real    # fail loudly, don't fall back, if real data can't load
python scripts/xylo_verify.py --no-combined            # skip the Part C one-chip check
python scripts/xylo_verify.py --modality ecg --window 90  # override the encoded window/timestep count
python scripts/xylo_verify.py --modality ppg --real --ppg-source vitaldb \
    --n-seeds 5                                # real VitalDB blood-loss label, multi-seed report
python scripts/xylo_verify.py --modality eeg --real --require-real \
    --n-seeds 5                                # real CHB-MIT seizure detection, subject-independent
```

For each modality: a data card, then float-model accuracy, XyloSim accuracy,
and their agreement rate, then the mapped resource footprint (input/hidden/
output neurons used vs. Xylo's limits). When both modalities run, also prints
a **one-chip composition line** (`xylo_budget.fits_one_chip`) and a
**co-residence check** — combines the two independently-trained nets onto one
Xylo core (block-diagonal weights, single shared quantization) and confirms
each modality's decisions still agree with its own standalone float model.

Training uses class-weighted cross-entropy and selects checkpoints by
**balanced accuracy**, not raw accuracy — on real MIT-BIH's 92.3%/7.7% split,
raw accuracy can't distinguish a majority-collapsed model from a genuine one.
The report prints both accuracy and balanced accuracy, plus **per-class
recall**, for the float model and XyloSim separately, so a degenerate model
can't hide behind a high headline number. See `rockpool_models.build_xylo_snn`
for the full writeup, including why a shorter encoding window helps synthetic
ECG's XyloSim fidelity but does *not* transfer as-is to real MIT-BIH (different
sampling rate — same window length is a different real-world duration).

See `rockpool_models.py` and `scripts/xylo_verify.py`
for the full specs and known gotchas — this exact readout (2 output neurons,
spike-count logits) is prone to a dead-neuron collapse that only a positive
initial bias fixes, the official tutorial's `PeriodicExponential` surrogate
measurably regresses this net (kept on the evidence, not the tutorial's
prior), and co-residence measurably costs some XyloSim fidelity vs. either
modality's standalone number under a shared 8-bit weight budget. All
documented in `rockpool_models.py` and `scripts/xylo_verify.py`.

## Verify on Akida (committed production target)

BrainChip **Akida** (MetaTF toolchain) is the committed production silicon
(see CLAUDE.md's "Hardware target"); this repo also validates on Rockpool/
XyloSim (above) because it's a mature bit-exact simulator available today.
See `docs/akida_ecg_results.md` for the first, minimal slice (ECG only);
myocardial infarction, heart sounds, the synthetic CRM demo, and shockable-
rhythm detection have since ported onto the same path (see each modality's
own `docs/*_results.md`) — the Xylo path above stays untouched throughout.

**Linux only — needs a container.** BrainChip's `akida` package (the actual
execution-engine / software simulator) publishes no macOS wheel, in any of
its PyPI releases, ever — confirmed against the full release history, not
assumed. Build and run the dev container (native `linux/arm64` on an
M-series Mac via Docker/Colima, no emulation; also builds on `linux/amd64`):

```bash
# one-time: a container runtime, if you don't already have one
brew install docker colima && colima start --cpu 4 --memory 8 --root-disk 80
#   ^ root-disk matters: colima's default (20GiB) is too small for
#     tensorflow + the rest of MetaTF's dependency tree; --disk alone is a
#     SEPARATE data volume, not what Docker's image storage actually uses.

scripts/akida_docker_run.sh                                   # interactive shell
scripts/akida_docker_run.sh pytest -q                          # tests run for real here (skipped on macOS)
scripts/akida_docker_run.sh python scripts/akida_verify.py --real --n-seeds 5
```

(`docker-compose.akida.yml` is the compose-plugin equivalent, if you have it.)

For ECG: a data card, then the **float** model's balanced accuracy/per-class
recall/AUROC (train first, same class-weighted-loss + balanced-accuracy-
checkpoint-selection discipline as the Xylo path), then the **Akida
software-simulator's** accuracy and its agreement with the float model, then
the mapped input-channel footprint. See `docs/akida_ecg_results.md` for the
full write-up: the confirmed Akida v2 layer constraints that shaped the
architecture (square kernel/stride/pool on every conv layer, specific valid
layer-ordering patterns), the Part-0 finding that BrainChip does **not**
publish an explicit bit/cycle-accurate claim for the software simulator the
way SynSense does for XyloSim, and the measured comparison against Xylo's
known ECG fidelity gap (float 0.845 balanced acc → XyloSim ~0.56 agreement).

`src/eia/akida_models.py` (deploy sibling of `rockpool_models.py`, which
stays untouched) builds a small quantized Conv2D-over-time CNN — Akida 2.0
has no native Conv1D, so the ECG window is treated as a `(window, 1, 1)`
single-column "image," the simplest mapping for this first slice (TENN,
Akida's genuinely temporal layer family, is the noted alternative for later).

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

This snippet is directly runnable (`python3` or a Jupyter kernel) — the current
notebooks are `notebooks/03_akida_ecg.ipynb` (the live Akida ECG port; Linux/Colab
only, see "Verify on Akida" below) and `notebooks/04_mvp_pitch.ipynb` (a
self-contained pitch/validation walkthrough — numpy + matplotlib only, no repo
clone needed).

## Local vs. Colab

Develop locally — an M-series Mac runs `python -m eia.train` comfortably on the
MPS GPU, with a faster edit/run loop than Colab. Some spiking ops lack MPS
kernels and fall back to CPU automatically — harmless at this scale; pass
`--device cpu` if you hit an unsupported-op error. The Akida path
(`notebooks/03_akida_ecg.ipynb`, `scripts/akida_verify.py`) is Linux-only (no
macOS wheel for `akida`, ever) — run it on Colab or the repo's Docker container,
not locally on macOS.

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
- **PPG has two real-data sources now, with different caveats.** BIDMC (default,
  `--modality ppg --real`, ICU patients, no induced hypovolemia) has no
  blood-loss annotation, so it labels windows by SpO2 desaturation (< 95%) as an
  accessible real-data stand-in for physiological compromise — proves the
  binary-window pipeline pattern on real waveforms, not a validated hemorrhage
  signal. **VitalDB** (`--ppg-source vitaldb`) adds a genuine, if coarse, blood-
  loss label: case-level estimated blood loss (`intraop_ebl >= 500 mL` =
  significant) from open intraoperative surgical monitoring — the first *open*
  dataset in this repo with an actual hemorrhage-relevant label. It is **not**
  conscious-hemorrhage ground truth: patients are anesthetized (confounds PPG),
  EBL is a whole-case estimate (not time-aligned to the bleed), and it's split
  **by case, not window** (`PpgData.groups`) since many correlated windows share
  one case-level label. See `docs/vitaldb_ppg_results.md` for the full
  write-up and Part-0 field-name verification. BIDMC stays as a secondary real-
  PPG dataset (kept, not deleted). The synthetic PPG generator
  (`make_synthetic_ppg`) remains closest in spirit to the actual target: reduced
  pulse amplitude + blunted dicrotic notch, the waveform-shape changes CRI
  relies on. An LBNP (lower-body negative pressure) dataset — true induced
  central hypovolemia in conscious subjects — is still the intended future
  addition; VitalDB does not replace that need, it replaces the SpO2 proxy for
  a real (if coarse) hemorrhage label.
- **Energy is analytical**, not measured — MAC/SOP counts times per-op energy
  constants (`energy.py`, documented and swappable). Phase 2 replaces this with
  real FPGA power measurements.
