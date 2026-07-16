# Task spec — lung-sound classification (ICBHI 2017), third real-data modality

**For:** Claude Code (repo, `.venv` for host; the `eia-akida` Docker container for
anything importing `akida`). Adds lung sounds as the third real-data modality and
the first to cover MARCH **Respiration / Airway** (ECG + heart sounds are both
Circulation-cluster). Mirrors the heart-sound work; reuses the proven filterbank →
quantized-Conv2D → Akida recipe. Keep ECG, heart sounds, the Xylo path, and the
existing Akida paths untouched.

**Why lung sounds, why now:** ECG and heart sounds both learn on real data and
deploy on Akida with almost no on-chip loss (0.928/0.926 and 0.865/0.867). Lung
sounds is the same *audio-class* problem shape, so the recipe transfers directly —
but it adds a new MARCH category (Respiration/Airway) rather than a third cardiac
test. Highest-tractability way to broaden the MARCH story.

**Scope decision — go straight to the Akida-CNN path, skip the Xylo/SNN detour.**
Akida is the committed production target and the CNN recipe both deploys faithfully
and learns better than the LIF SNN did. For a *new* modality there's no reason to
build the Xylo/SNN path first (heart sounds did, only because it predated the Akida
decision). Build lung sounds directly for the Akida path: `datasets` loader +
filterbank front-end + `scripts/akida_verify.py --modality lung`, measuring float +
Akida-sim. (A Xylo/SNN path can be added later only if a co-residence or
SOP-energy-baseline analysis needs it — not now.)

---

## Part 0 — verify dataset ACCESS first (this is the new-risk vs heart/ECG)

Unlike MIT-BIH and CinC 2016, **ICBHI 2017 is NOT on PhysioNet/wfdb** — it can't be
streamed with `wfdb.rdrecord`. Confirm and document, before building the loader:

- **Source + format:** the official ICBHI 2017 Respiratory Sound Database (the
  BHI challenge site, or a stable mirror). Confirm the download mechanism (a single
  large ~4 GB archive of `.wav` + per-recording `.txt` annotation files, plus a
  patient-diagnosis file and the official train/test list). Document the exact
  source URL and cache-once-to-`data/icbhi/` approach (gitignored, like the other
  data caches). If it needs a manual download / login, say so clearly and STOP with
  instructions rather than guessing a URL.
- **Annotation format:** each recording's `.txt` lists breathing cycles as
  `start end crackles(0/1) wheezes(0/1)`. Confirm and parse this.
- **Filename metadata:** `patientID_recordingIndex_chestLocation_acqMode_device.wav`
  — the **patient ID is the first field and IS recoverable** (unlike CinC 2016).
  Use it for a patient-level leakage-safe split.
- **Sampling rates are mixed** (4 kHz / 10 kHz / 44.1 kHz across recordings) —
  confirm and resample everything to one common rate before feature extraction.
- **Class balance:** report the normal-vs-adventitious cycle ratio (it's imbalanced;
  crackle/wheeze cycles are common). Print it in the data card.

## Task framing (start simple)

- **Unit = the annotated breathing cycle.** Extract each cycle, band-pass to the
  respiratory-sound band (~50–2500 Hz; crackles are transient broadband, wheezes
  tonal ~100–1000 Hz), window/pad to a fixed length.
- **Binary first: normal vs. adventitious** (adventitious = crackle and/or wheeze
  present in the cycle) — mirrors heart's binary framing, most tractable. Note the
  4-class version (normal / crackle / wheeze / both) as a follow-up; do NOT build it
  now.

## Front-end + model (reuse the proven heart recipe)

- **Filterbank front-end, NOT raw waveform** — the same lesson heart sounds and EEG
  taught: audio/spectral signals need spectral features. Reuse the heart filterbank
  logic (band-power / line-length / spectral-entropy, or a log-mel front-end)
  computed over sub-windows at the native rate → a `(n_features, n_subwindows)`
  bands×time map. Salvage/generalize the existing feature code; don't rebuild it.
- **Akida model:** the bands×time map is a natural 2-D Conv2D input, exactly like
  heart. Reuse `build_akida_heart_model`'s pattern (generalize it or add a sibling)
  — the Akida v2 constraints found earlier (square kernel/stride/pool, valid layer
  ordering) still apply. quantizeml QAT → cnn2snn convert → Akida software sim.

## Measurement (same discipline; FLOAT first)

- **Patient-level split** via GroupShuffleSplit on patient ID (ICBHI has them — a
  rigor upgrade over CinC's recording-level split). Raise if any patient leaks
  across train/test.
- Class-weighted loss, checkpoint selection by **balanced accuracy**, per-class
  recall, feature normalization fit on TRAIN only, ≥5 seeds mean ± spread.
- Report the FLOAT model first (balanced acc / per-class recall / AUROC) — does it
  learn on real ICBHI — THEN Akida-sim balanced acc + float-vs-Akida-sim agreement
  + mapped footprint. Carry the standard caveats: Akida sim not confirmed
  bit-exact; CNN-vs-SNN if compared; Akida 2.0 FPGA, not Pico.

**Honest calibration (state it in the results doc):** lung sounds is a *harder*
audio task than heart sounds — more class imbalance, and crackle/wheeze detection
varies with recording device and chest location, so published benchmarks are more
modest. Expect it to learn (audio-class, benchmarked), but likely not as cleanly as
heart's 0.865. If the float model is ~chance, check Part-0 access/annotation
parsing and the resample rate before tuning anything.

## Deliverables

- `datasets` loader (`load_icbhi` / `load_lung`) + synthetic fallback + filterbank
  front-end; `"lung"` wired into `scripts/akida_verify.py --modality lung`. Runs:
  `scripts/akida_docker_run.sh python scripts/akida_verify.py --modality lung --real --n-seeds 5`.
- `docs/lung_sounds_results.md`: Part-0 access/annotation findings, n recordings /
  patients / cycles, class balance, float vs Akida-sim (multi-seed), footprint,
  explicit verdict (does it learn on the committed target?), caveats.
- `report.py` data card for `("lung","icbhi")` / `("lung","synthetic")`.
- CLAUDE.md: add lung sounds under MARCH Respiration/Airway with the result.
- Offline unit tests for the new pure pieces (annotation parser on a literal
  string, filename→patient-id parse, cycle extraction shape, no-leakage patient
  split, filterbank on a known signal). Skip-guard anything importing `akida`.
  pytest -q green on host (akida tests skip) and in container (they run).
- Commit, push origin main, verify the push landed (rev-parse + cat-file), report
  the hash.

## Do NOT

- Do NOT touch ECG, heart sounds, `rockpool_models.py`, or the existing Akida paths
  (`build_akida_model`/`build_akida_heart_model`) — extend additively.
- Do NOT use a raw-waveform front-end (it's chance for audio — filterbank only).
- Do NOT split by cycle/recording across patients (leakage — split by patient).
- Do NOT chase Akida-sim fidelity before the float model learns (float-first).
- Do NOT build the 4-class version or a Xylo/SNN path in this task — note as future.
- Do NOT fabricate a number or guess a download URL — if access needs a manual step,
  stop and report it.
