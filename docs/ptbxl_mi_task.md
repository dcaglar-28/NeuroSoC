# Task spec — MI / STEMI detection from 12-lead ECG (PTB-XL)

**For:** Claude Code (repo, `.venv`; `eia-akida` container for the Akida measurement).
Deepens the ECG capability from *arrhythmia* (MIT-BIH) to **myocardial infarction /
ischemia** — the acute cardiac killer, expert-only to read, with no manual analog
(a pulse cannot diagnose a heart attack). Reuses the ECG Akida-CNN path. This is a
new *task/dataset on the existing modality*, not a new modality — keep heart sounds,
the existing ECG-arrhythmia path, rockpool_models.py, and other Akida paths untouched;
add additively.

**Why:** MI is differentiated exactly where the device should be — it does what a
field responder can't do bare-handed. PTB-XL is large, cardiologist-labeled, open,
and benchmarked, so it's high-value AND high-tractability, and it compounds your
strongest existing result rather than starting from zero.

---

## Part 0 — verify PTB-XL access + label mapping (do first; don't trust memory)

- **Dataset:** PTB-XL (PhysioNet, `physionet.org/content/ptb-xl/`) — 21,799 records
  / 18,869 patients, 12-lead, 10 s, with 100 Hz (`records100/`) and 500 Hz
  (`records500/`) versions. Confirm the download/stream mechanism (wfdb `pn_dir`
  or a direct download + cache to `data/ptbxl/`, gitignored). Use the **100 Hz**
  version (1000 samples/record) — plenty for MI morphology and lighter.
- **Label mapping (the load-bearing detail):** diagnoses live in
  `ptbxl_database.csv`'s `scp_codes` (a dict of SCP code → likelihood). Map each SCP
  code to its diagnostic **superclass** via `scp_statements.csv`
  (`diagnostic_class`). The 5 superclasses: NORM, **MI**, STTC (ST/T change /
  ischemia), CD, HYP. Confirm this join and how likelihood/`validated_by_human` are
  used before trusting any label. Report the exact rule you adopt.
- **Split:** PTB-XL ships a recommended 10-fold `strat_fold` that **respects
  patients** (no patient across folds) — use it (folds 1–8 train, 9 val, 10 test),
  or GroupShuffleSplit on `patient_id`. Never split by record across a patient.
- Report class balance for the chosen task.

## Task framing (start binary, highest-value)

- **Binary first: MI vs NORM** — records confidently labeled MI (superclass) vs
  confidently NORM. Most tractable, and it *is* the heart-attack-detection claim.
- Note as follow-ups (do NOT build now): STTC (ischemia) as a second binary, and
  the full 5-superclass multi-label task.

## Leads + front-end

- This is a **morphology** signal (ST-segment elevation, Q waves, T-wave inversion),
  so use the **raw-waveform path** (like ECG-arrhythmia / CRM), NOT the audio
  filterbank.
- **Use the 12 leads** (MI localization needs spatial lead information, and the
  committed Akida target has no 16-input ceiling). Feed as a `(leads, time)` 2-D map
  → Akida Conv2D — the same natural 2-D fit as heart's bands×time. Normalize
  per-lead. (Note a reduced-lead / single-lead "wearable-realistic" variant as a
  future follow-up — don't build it now; the 12-lead version proves the capability.)
- Reuse the Akida CNN scaffolding (`build_akida_model` / `build_akida_heart_model`
  pattern — generalize or add a sibling for the (leads, time) input); respect the
  Akida v2 conv constraints (square kernel/stride/pool, valid layer ordering).

## Pipeline + measurement (float first)

- `datasets`: add `load_ptbxl` / a `"mi"` (or `"ecg_mi"`) task path + synthetic
  fallback, provenance-labeled. Wire into `scripts/akida_verify.py`.
- Patient-level split (PTB-XL folds), class-weighted loss if imbalanced,
  balanced-accuracy checkpoint selection, per-class recall, per-lead norm fit on
  TRAIN only, ≥5 seeds. Report the **FLOAT model first** (balanced acc / per-class
  recall / **AUROC** — MI detection is usually reported by AUROC) — does it detect
  MI on real PTB-XL — THEN Akida-sim balanced acc + float-vs-sim agreement +
  footprint. Carry the standard caveats (sim not confirmed bit-exact; Akida 2.0 FPGA
  not Pico).
- Calibration: PTB-XL MI-vs-NORM is a benchmarked task (published AUROC ~0.9+), so
  expect the float model to learn well; if it's near chance, re-check the Part-0
  label mapping and lead handling before tuning.

## Deliverables

- `load_ptbxl` + `"mi"` task + `akida_verify` support (runs:
  `scripts/akida_docker_run.sh python scripts/akida_verify.py --modality mi --real --n-seeds 5`);
  `report.py` data card for `("mi","ptbxl")` / synthetic with the label-rule +
  caveats.
- `docs/ptbxl_mi_results.md`: Part-0 access/label findings, n records/patients,
  class balance, lead set used, float + Akida-sim (multi-seed), footprint, explicit
  verdict (does it detect MI on the committed target?), caveats.
- CLAUDE.md: note ECG deepened from arrhythmia → MI/ischemia under Circulation.
- Offline unit tests (SCP→superclass mapping on a literal dict, patient-level no-leak
  split, per-lead norm shape, label-rule edge cases). Skip-guard akida imports.
  pytest -q green on host (akida tests skip) + container (run).
- Commit, push origin main, verify the push landed (rev-parse + cat-file), report hash.

## Do NOT

- Do NOT touch the ECG-arrhythmia path, heart sounds, rockpool_models.py, or the
  existing Akida paths — extend additively.
- Do NOT use the audio filterbank front-end — MI is morphology (raw waveform).
- Do NOT split by record across a patient — use PTB-XL's patient-respecting folds.
- Do NOT build the multi-label / STTC / reduced-lead variants in this task — note
  as future.
- Do NOT chase Akida-sim fidelity before the float model detects MI (float-first).
- Do NOT guess the SCP→superclass mapping — verify it against scp_statements.csv.
