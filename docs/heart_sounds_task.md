# Task spec — heart-sound (PCG) classification + retire EEG

**For:** Claude Code (repo, `.venv`). Two parts: (A) add heart sounds as the new
second real-data modality, mirroring the proven ECG pipeline; (B) retire the EEG
modality code (keep its findings docs). Lung sounds are noted as the NEXT step,
not built here.

**Why heart sounds:** ECG is the one modality genuinely learning on real data;
EEG seizure detection came back ~chance under two different front-ends (settled —
see docs/eeg_frontend_results.md), because cross-patient seizure detection is a
hard task, not because the chip can't run EEG. Heart-sound classification is the
opposite: an *audio-class 1-D signal* (the exact neuromorphic-audio sweet spot),
a well-benchmarked task with abundant labels, and MARCH-relevant (cardiac
auscultation is an on-core test). Highest-confidence path to a genuinely-learning
second modality for the MVP.

---

## Part 0 — verify dataset access before building (do first, don't trust memory)

Same discipline that caught real bugs in VitalDB/CHB-MIT. Confirm and write down:

- **Dataset:** PhysioNet/CinC Challenge 2016 heart-sound (phonocardiogram, PCG)
  database. Confirm the access path (PhysioNet `challenge-2016` content; via
  `wfdb` `pn_dir=` if it streams, else direct download of the training sets
  a–f), the file format (.wav + .hea), the native sampling rate (~2000 Hz), and
  the **label file structure** (per-recording normal/abnormal label; note the
  challenge's `-1`/`1` = normal/abnormal convention, and whether an "unsure" or
  recording-quality flag exists — decide how to handle it, document the choice).
- **Grouping for leakage:** confirm whether multiple recordings come from the
  same subject and whether subject ids are recoverable. If so, split BY SUBJECT
  (GroupShuffleSplit, like VitalDB cases) to avoid leakage; if only per-recording
  labels exist, split by recording and say so. Record which in the data card.
- **Class balance:** report the normal/abnormal ratio (it is imbalanced) so the
  class-weighted-loss + balanced-accuracy machinery is applied from the start.

## Part A — build the modality (mirror the ECG pipeline, the proven pattern)

- `datasets.py`: add a real loader (`load_cinc2016` / `load_heart`) + a synthetic
  fallback (`make_synthetic_heart` — stylised S1/S2 + optional murmur band-noise)
  + a `load_heart(prefer_real, require_real, ...)` wrapper with the same
  provenance contract as `load_ecg` (`requested_real`, `[warn]` on fallback,
  `groups` set when subject ids exist). Register `"heart"` in the `_LOADERS` dict
  and `--modality` choices in `train.py` and `scripts/xylo_verify.py`.
- **Preprocessing / windowing:** band-pass to the PCG band (~20–400 Hz; S1/S2
  energy is low, murmurs higher), segment into fixed-length windows (or cardiac-
  cycle segments if easy), z-score per window, and **resample to a short Xylo
  timestep budget** (the `load_mitbih` `resample_to` pattern — few timesteps,
  fs-matched; heart sounds at 2000 Hz must be downsampled hard).
- **Front-end — start simple, escalate only if needed.** Baseline = the same
  delta ON/OFF encoding ECG uses (S1/S2 are transients, so edge-encoding has a
  real chance here, unlike EEG's sustained rhythm). BUT the *discriminative*
  signal for abnormality is often the murmur, which is spectral — so if the raw-
  delta float model underperforms, add a filterbank/band-power front-end (the
  retired `eeg_features` filterbank logic can be salvaged/adapted rather than
  rebuilt). Try raw-delta first for speed; document if you escalate.
- `report.py`: add the `("heart","cinc2016")` / `("heart","synthetic")` data
  cards with label definition, class balance, and provenance.

## Part A — measurement (same discipline as ECG; FLOAT model first)

- Class-weighted CE + checkpoint selection by **balanced accuracy** (imbalanced).
  Leakage-safe split (by subject if available). ≥5 seeds, mean ± spread.
- Report balanced accuracy, **per-class recall**, and AUROC/AUPRC for the FLOAT
  model FIRST — that's "does it learn." Only then XyloSim accuracy + float-vs-sim
  agreement + the mapped footprint (inputs/hidden/outputs vs. Xylo limits). This
  ordering is the hard-won EEG lesson: don't chase on-chip fidelity before the
  float model learns.
- Success bar: float balanced acc clearly above base rate with genuine per-class
  recall (CinC 2016 is a learnable benchmark, so this should happen — if it
  doesn't, check Part 0 label handling before anything else).

## Part B — retire EEG (declutter, but preserve the record)

- **Remove** the EEG *code*: `EegData`, `load_chbmit`/`load_eeg`/
  `make_synthetic_eeg`, `eia/eeg_features.py`, `eeg_patient_specific_split`, the
  EEG montage/band-pass/resample helpers, the `"eeg"` registration in `train.py`
  and `scripts/xylo_verify.py`, the `("eeg",*)` data cards in `report.py`, and
  the EEG unit tests. Remove only EEG-specific code — do NOT break shared modules
  (`encoding.py`, shared dataclasses, `report.py`, the `xylo_verify` structure,
  the `case_level` split used by other modalities). `pytest -q` stays green.
- **KEEP** the EEG findings docs as honest record: `docs/eeg_seizure_results.md`,
  `docs/eeg_frontend_results.md` (and the task specs). Add a one-line banner at
  the top of each: "RETIRED/PAUSED — code removed <date>; findings preserved.
  EEG seizure detection came back ~chance under raw and feature front-ends;
  paused pending a richer montage without the 16-input Xylo ceiling (Akida
  headroom) and/or more patients. Recoverable from git history."
- Update `CLAUDE.md`: mark the H/EEG entry as retired/paused (keep the one-line
  findings summary, note the code was removed and why), and update the modality
  list + roadmap so the second modality is now heart sounds.

## Deliverables

- Heart-sound loader + modality wiring + data card + `xylo_verify` support; runs
  end-to-end: `python -m eia.train --modality heart --real` and
  `scripts/xylo_verify.py --modality heart --real --n-seeds 5`.
- `docs/heart_sounds_results.md`: Part-0 access/label findings, n recordings,
  class balance, float balanced acc / per-class recall / AUROC (multi-seed),
  XyloSim agreement + footprint, and an explicit verdict (does it learn?).
- EEG code retired per Part B; EEG findings docs kept + bannered; CLAUDE.md updated.
- Offline unit tests for the new pure pieces (synthetic heart generator shape,
  windowing/resample, band-pass, no-leakage subject split). `pytest -q` green.
- Commit (ideally two commits: "add heart-sound modality" and "retire EEG code,
  keep findings"), push origin main, verify the push landed (rev-parse +
  cat-file), report the hash.

## Next step (note only — do NOT build now)

**Lung sounds (ICBHI Respiratory Sound Database)** is the documented next
modality after heart sounds: also audio-class, benchmarked, and it covers MARCH
**Respiration/Airway** (a new MARCH letter rather than a second cardiac test).
Record it in CLAUDE.md's roadmap as the planned follow-on.

## Do NOT

- Do NOT touch the ECG code — it is the working modality and the MVP/pitch anchor.
- Do NOT delete the EEG *findings* docs — only the EEG code.
- Do NOT chase XyloSim fidelity before the heart-sound float model learns.
- Do NOT build lung sounds in this task — note it as next only.
