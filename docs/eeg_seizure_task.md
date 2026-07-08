# Task spec — EEG seizure detection on CHB-MIT (MARCH "H", Phase 1)

**For:** Claude Code (repo, `.venv` active, `wfdb` installed via `eia[data]`).

**Why this, why now:** ECG (Circulation) is the one modality with a real,
genuinely-learning model. Hemorrhage (M) is paused: its bottleneck was the
*labels*, not the model (VitalDB settled ~chance at window and case level — see
`docs/vitaldb_case_level_results.md`). Seizure detection is the opposite
situation — **strong physiology, excellent expert labels, decades of benchmarks**
— so it is the cheapest path to a *second* validated real-data model. This task
builds it and, in doing so, proves the on-core EEG path for the MARCH "H" (Head)
branch.

**Scope of this task = Phase 1 only:** CHB-MIT, subject-independent seizure
vs. non-seizure. TUSZ/Siena generalization (Phase 2) and TBI spectral screening
(Phase 3) are noted at the end but NOT built here.

**MARCH framing (record it, don't re-litigate it):**
H splits into two independent workstreams — **EEG** (seizure now; altered
mental status / TBI screening later) and **Thermal** (hypothermia, later, no
CHB-MIT-equivalent open dataset exists). This task is the EEG/seizure milestone
only. Clinical-fit caveat to state in the data card: CHB-MIT is *pediatric
epilepsy monitoring*, not *field TBI* — seizure detection is a legitimate,
learnable component of "Head" (post-traumatic seizures, altered mental status),
not trauma ground truth. This is a much milder caveat than VitalDB's because the
task actually learns.

---

## Dataset

CHB-MIT Scalp EEG Database (PhysioNet, `pn_dir="chbmit"`, WFDB-compatible).
- 23 pediatric subjects (22 unique patients; **chb01 and chb21 are the same
  patient** — treat as one group to avoid leakage), ages 1.5–22.
- 23-channel scalp EEG, **256 Hz**, 16-bit, ~915 hours, ~198 annotated seizures.
- Seizure onset/offset annotations per record (`.seizures` / summary files).
- Fully open — no registration or DUA (unlike TUSZ, see Phase 2).

Mirror the `load_mitbih` pattern: stream a subset of records via `wfdb`, raise
`ImportError`/`RuntimeError` if unavailable, and expose a `load_eeg(prefer_real,
require_real, ...)` wrapper with synthetic fallback so tests/notebooks run
offline. Start with a handful of records for iteration, parameterize the record
list, and cache locally.

---

## Design the pipeline HARDWARE-FIRST (this is the point)

Do NOT build a 23-channel detector and shrink later. Design to the Xylo input
budget from the start:

```
23 EEG channels
      │  fixed bipolar montage, subject-independent channel selection
      ▼
  N channels  (N <= 8)
      │  band-pass (seizure band, ~0.5–25 Hz)
      ▼
  resample / window  (few timesteps — see below)
      │  delta ON/OFF encoder (doubles channel count)
      ▼
  2N spike channels  (must be <= 16)
      │
  LIF SNN  (build_xylo_snn / Rockpool)
      ▼
  seizure probability  ->  XyloSim verify
```

### Flag 1 — the channel ceiling is exact.
Xylo allows **≤16 input channels**, and the delta encoder emits ON+OFF, i.e.
**2 channels per EEG channel**. So `8 EEG channels × 2 = 16` sits *exactly* on
the limit with **zero headroom**. **Start at N=6 (→12 spike channels)** to leave
margin, confirm it maps, then try N=8 only if 6 underperforms. Report the mapped
input-channel count against the limit in the footprint line, as the other
modalities do.

### Flag 2 — channel selection must be a FIXED montage.
A field device has never seen this patient, so you cannot use patient-specific
focal channels. Pick a **fixed subset of the standard bipolar longitudinal
("double-banana") montage** applied identically to every subject. Document the
exact channel list. (A patient-specific channel set would inflate results and is
not the deployment story.)

---

## THREE non-negotiable methodology rules (each is a prior lesson)

### 1. Subject-independent split is the HEADLINE. (VitalDB case-leakage lesson.)
- Split **by patient** with `GroupShuffleSplit` on a `groups` = patient-id array
  (the repo already does group-splitting when `data.groups` is set — reuse it).
  Merge chb01/chb21 into one group.
- **Never** let windows from one patient appear in both train and test. Also
  never split correlated *windows* from the same recording/seizure across the
  boundary — group at the patient level and the recording level is contained
  within it.
- Patient-specific detection (train and test within a patient — CHB-MIT's classic
  high-AUROC use) may be reported as an *optional secondary* number, clearly
  labelled, but the **headline metric is cross-patient**. Raise if any patient
  leaks across splits, exactly like the VitalDB case check.

### 2. Extreme imbalance → the right metrics, not accuracy. (MIT-BIH lesson.)
Seizures are a tiny fraction of 915 hours — imbalance is far worse than MIT-BIH's
7.7%. Reuse the existing machinery: inverse-frequency **class-weighted CE** +
checkpoint selection by **balanced accuracy**. Report, per model (float and
XyloSim) and over **≥5 seeds, mean ± spread**:
- **AUROC and AUPRC** (AUPRC matters most under heavy imbalance),
- **sensitivity** (seizure recall) and **specificity**,
- **false alarms / hour** — the standard clinical seizure-detector metric; a
  field device that cries wolf is useless,
- per-class recall.
Compare every number to the trivial base rate and warn if the model sits inside
the seed band of it. Single-seed numbers are noise (the ECG checkpoint-
sensitivity finding).

### 3. Few timesteps. (ECG quantization root-cause lesson.)
EEG at 256 Hz over a multi-second window = hundreds–thousands of samples =
far too many Xylo timesteps; on-chip fidelity degrades with timestep count
(`docs/ecg_quant_diagnosis.md`: per-timestep integer state rounding compounds).
Band-pass to the seizure band first (kills the need for 256 Hz), then use the
`load_mitbih` `resample_to` pattern: **capture the physiological window at native
fs, then resample to the fewest timesteps that still hold the seizure
signature.** Decouple "seconds of EEG" from "Xylo timesteps." Report float-vs-
XyloSim agreement as a function of timestep count if time allows.

---

## Repo integration

- `datasets.py`: add an `EegData` dataclass mirroring `PpgData` (X, y, fs,
  source ∈ {"chbmit","synthetic"}, `requested_real`, **`groups`** = patient id).
  Add `load_chbmit(...)`, a synthetic `make_synthetic_eeg(...)` fallback, and
  `load_eeg(prefer_real, require_real, ...)`. Register `"eeg"` in `train.py`'s
  `_LOADERS` and add it to `--modality` choices there and in
  `scripts/xylo_verify.py`.
- `report.py`: add the `("eeg","chbmit")` data card with the caveats above
  (pediatric-epilepsy-not-TBI; class balance; subject-independent requirement).
- Keep the delta-encode + `build_xylo_snn` + `verify_against_sim` flow unchanged;
  only the front-end (montage select + band-pass + resample) and `n_in = 2N` are
  new. `n_out = 2` (seizure / non-seizure) is unchanged.
- Everything torch/scipy-heavy stays importable-lazily; numpy-only tests must
  still pass without torch.

## Deliverables

- Loader + model wiring above; `--modality eeg` runs end-to-end in `train.py`
  and `scripts/xylo_verify.py --modality eeg --real --n-seeds 5`.
- `docs/eeg_seizure_results.md`: records used, channel montage, n patients,
  class balance, window/fs/timesteps, float + XyloSim AUROC/AUPRC/sensitivity/
  specificity/FA-per-hour/per-class-recall (mean ± spread over ≥5 seeds), the
  subject-independent-vs-patient-specific gap, and an honest up-front
  interpretation (does it learn cross-patient, yes/no, vs base rate).
- `CLAUDE.md`: one-line status under the H/EEG entry.
- Offline unit test for the new pieces (montage selection shape, band-pass,
  resample-to-timesteps, group-split no-leakage on tiny synthetic EEG).

## Do NOT

- Do NOT headline a patient-specific (within-patient) number — cross-patient is
  the deployment-honest metric.
- Do NOT split by window across patients or recordings (leakage).
- Do NOT feed all 23 channels or exceed 16 spike channels.
- Do NOT report accuracy as the headline under this imbalance.
- Do NOT start on TUSZ/Siena or TBI here — Phase 1 is CHB-MIT only.

---

## Later (noted, NOT this task)

- **Phase 2 — generalization:** TUH Seizure Corpus (TUSZ; research standard,
  adults, many seizure types — but **registration + data-use agreement gated**,
  not fully open like CHB-MIT, budget for that) and Siena Scalp EEG (small,
  clean, good cross-dataset check).
- **Phase 3 — TBI screening (different task, not seizure):** lightweight spectral
  markers responders actually care about — spectral slowing, alpha suppression,
  theta/delta ratio, burst suppression. Needs its own labels; do not conflate
  with seizure detection.
- **Thermal / hypothermia:** separate H workstream, later; no open
  CHB-MIT-equivalent — likely a build-your-own-dataset problem.
