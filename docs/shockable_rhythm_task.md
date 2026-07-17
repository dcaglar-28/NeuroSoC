# Task spec — shockable-rhythm (VF/VT) detection from ECG

**For:** Claude Code (repo, `.venv`; `eia-akida` container for the Akida measurement).
Adds shockable-rhythm detection — the defibrillate-or-not decision (AED-in-a-box) —
as another ECG task on the committed Akida path. Reuses the ECG-Akida scaffolding
(same as arrhythmia and MI). Keep the ECG-arrhythmia, MI, heart-sound, and existing
Akida paths untouched; add additively.

**Why:** highest-value / lowest-cost next modality. It reuses the exact Akida
waveform path that's already delivered three times, and it's a *differentiated*
signal — no manual analog (a pulse tells you "no pulse," not the rhythm), and it's
the acute cardiac-arrest decision. It completes the ECG capability: arrhythmia +
MI + **cardiac arrest / shockable rhythm**. It also feeds the shock-etiology fusion
(a shockable rhythm is a distinct branch from cardiogenic/hemorrhagic/obstructive).

---

## Part 0 — verify datasets + the shockable label rule (do first; don't guess)

- **Datasets (both PhysioNet, open, `wfdb`):** the MIT-BIH Malignant Ventricular
  Ectopy Database (**VFDB**, `pn_dir="vfdb"`, 22 records, 250 Hz, 2 channels) and
  the Creighton University Ventricular Tachyarrhythmia Database (**CUDB**,
  `pn_dir="cudb"`, 35 records, 250 Hz). Confirm access/stream, sampling rate, and
  channel count. Cache to `data/vfdb/`, `data/cudb/` (gitignored).
- **Rhythm annotations — the load-bearing detail:** these datasets annotate rhythm
  *episodes* (e.g. `[VF`, `[VT`, `[N`, `(VFIB`, `(VT`, `(N`, asystole, noise), not
  per-beat labels. Confirm the exact annotation format and rhythm codes present
  before mapping anything. Verify against the datasets' own documentation, not
  memory.
- **Shockable definition (adopt a standard, cite it):** the widely-used rule
  (de Bruin et al. / AHA-aligned) is **SHOCKABLE = ventricular fibrillation (VF)
  or rapid ventricular tachycardia (VT above a rate threshold, commonly
  ~150–180 bpm); NON-SHOCKABLE = everything else** (normal sinus, other
  supraventricular rhythms, slow VT, asystole, noise). State the exact rule you
  adopt and how you handle VT rate and noise/asystole segments.
- Report class balance (shockable is the minority) and total windows.

## Task framing

- **Unit = a fixed-length analysis window**, mirroring how an AED works — a
  ~5–8 s segment classified shockable vs. non-shockable (pick the duration in
  Part 0 against what the annotations support; document it). Label each window by
  the rhythm covering it; drop ambiguous/transition windows or windows dominated
  by annotated noise.
- **Binary: shockable vs. non-shockable.** (A finer VF-vs-VT split is a possible
  follow-up; don't build it now.)

## Front-end + model (reuse the ECG waveform path)

- This is a **morphology/rhythm** signal (VF = disorganized, no clear QRS; VT =
  wide, fast, regular QRS), so use the **raw-waveform path** like ECG-arrhythmia
  and MI — NOT the audio filterbank. Band-pass to the ECG band, z-score per window,
  resample to a short Akida timestep budget (the `resample_to` pattern; 250 Hz over
  a multi-second window is many samples — downsample sensibly).
- Reuse `build_akida_model` (single-lead) — or the `(leads, time)` variant if using
  both VFDB channels. Respect the known Akida v2 conv constraints. quantizeml QAT →
  cnn2snn convert → Akida sim.

## Measurement (float first; AED-standard metrics)

- Split **by record** (GroupShuffleSplit on record id — records are patients;
  never split a record's windows across train/test). Class-weighted loss,
  balanced-accuracy checkpoint selection, ≥5 seeds.
- Report the FLOAT model first: **sensitivity (shockable recall), specificity
  (non-shockable recall)**, balanced accuracy, AUROC, per-class recall — sensitivity
  and specificity are the AED-standard metrics (the AHA performance goals are
  sensitivity ≥90% for VF and specificity ≥95%; report against those). THEN
  Akida-sim + float-vs-sim agreement + footprint. Carry the standard caveats (sim
  not confirmed bit-exact; Akida 2.0 FPGA, not Pico).
- Calibration: shockable-rhythm detection is a well-benchmarked task, so expect the
  float model to learn strongly; if it's near chance, re-check the Part-0 rhythm-code
  mapping and window labeling before tuning.

## Deliverables

- `datasets` loader (`load_vfdb`/`load_cudb` or a combined `load_shockable`) + a
  `"shockable"` (or `"vf"`) modality + synthetic fallback, provenance-labeled;
  wired into `scripts/akida_verify.py`. Runs:
  `scripts/akida_docker_run.sh python scripts/akida_verify.py --modality shockable --real --n-seeds 5`.
- `report.py` data card with the shockable rule, class balance, and caveats.
- `docs/shockable_rhythm_results.md`: Part-0 access/rhythm-code findings, n records/
  windows, class balance, window duration, float sensitivity/specificity/AUROC
  (multi-seed) vs the AHA goals, Akida-sim + agreement + footprint, verdict, caveats.
- CLAUDE.md: note ECG extended to shockable-rhythm under Circulation.
- Offline unit tests (rhythm-code→shockable mapping on a literal annotation list,
  window-labeling edge cases, no-leakage record split, per-window norm shape).
  Skip-guard akida imports. pytest -q green host + container.
- Commit, push origin main, verify push landed (rev-parse + cat-file), report hash.

## Do NOT

- Do NOT touch the ECG-arrhythmia, MI, heart-sound, rockpool, or existing Akida
  paths — extend additively.
- Do NOT use the audio filterbank — this is waveform morphology.
- Do NOT split a record's windows across train/test (leakage).
- Do NOT guess the rhythm-code → shockable mapping — verify against the datasets'
  documentation and adopt a cited standard.
- Do NOT chase Akida-sim fidelity before the float model detects shockable rhythm.
- Do NOT build the VF-vs-VT sub-split now — note as future.
