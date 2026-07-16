# Task spec — synthetic time-resolved CRM / occult-hemorrhage generator

**For:** Claude Code (repo, `.venv`; `eia-akida` container for the Akida measurement).
Builds the flagship differentiated capability — occult-hemorrhage detection via
Compensatory Reserve from the PPG waveform — as a *demonstrable* result, using a
physiologically-grounded synthetic generator (real LBNP data is gated; requests are
out separately). Reuses the ECG Akida-CNN path. Keep ECG, heart sounds, and the
existing paths untouched; add additively.

**Why this, and why it's not a cop-out:** VitalDB is settled dead because one
whole-case blood-loss number was stamped on every window — the label never matched
the window's physiological state. This generator **fixes that by construction**: it
produces a PPG signal evolving along a hypovolemia trajectory, and each window's
label is the reserve state *at that moment*. So it's a genuine test of whether the
pipeline can track occult volume loss, honestly synthetic. The differentiated claim
it demonstrates: detect compensated hemorrhage *before* heart rate and blood
pressure move — the one thing a responder taking a pulse cannot do.

---

## The physiological model (ground it in the CRM/LBNP literature, cite it)

Parameterize a subject's state by a **reserve fraction `r` ∈ [1, 0]**: 1.0 =
normovolemic (full compensatory reserve), 0.0 = decompensation. A synthetic
"subject" is a trajectory `r(t)` decreasing over time (with randomized rate). PPG
pulse morphology is a function of `r` (the CRM waveform relationships — reference
the explainable-CRM paper, MDPI Bioeng. 2023, for which features change):

- **Pulse amplitude** decreases as `r` falls (reduced stroke volume / peripheral perfusion).
- **Dicrotic notch** is prominent at `r≈1`, progressively **blunts then disappears**
  as `r` falls — the signature CRM morphology change; make it the strongest cue.
- **Pulse width / systolic upstroke time** narrows/shifts with `r`.
- **Diastolic (reflected-wave) component** flattens with `r`.
- **Heart rate:** near-baseline while `r` is high, rising (compensatory tachycardia)
  only as `r` gets low, then collapse at decompensation. CRITICAL: the morphology
  changes must lead the HR change — early-trajectory windows should have shifted
  morphology while HR is still near-normal. That lead is the entire CRM value prop
  and the demo must exhibit it.
- **Per-subject variability + noise:** randomize each subject's baseline morphology,
  baseline HR, trajectory rate, and add realistic noise (+ optional motion artifact),
  so the model can't memorize one fixed shape.

## The label — time-aligned (this is the whole point)

Each generated window carries the reserve value at its point on the trajectory:
- Primary: a **Compensatory Reserve Index** `CRI = r` ∈ [0,1] (the 0–100 "fuel
  gauge" the real CipherOx device shows).
- For the existing classifier pipeline, the loader discretizes CRI to a configurable
  label (default **binary**: `CRI < threshold` = "compromised reserve"), with the
  positive class deliberately including **early, HR-still-normal** windows — so a
  successful classifier is provably detecting occult loss before vitals move. Note a
  graded (3-stage) or true CRI-regression variant as follow-ups.

State explicitly in the data card: the label matches each window's physiological
state (unlike VitalDB) — that alignment is the reason this is a valid pipeline test.

## Front-end — this is a MORPHOLOGY signal, NOT audio

Important distinction from heart/lung sounds: CRM lives in the PPG **pulse shape**
(amplitude, notch, width, slopes) at low frequency (<~10 Hz), not in high-frequency
spectral content. So the right front-end is the **ECG-style raw-waveform path**
(normalized pulse window → Akida CNN over time), NOT the filterbank. Reuse
`build_akida_model`'s waveform pattern, not `build_akida_heart_model`'s bands×time
map. (Optional alternative: compute the CRM hand-features — pulse amplitude, upstroke
time, notch timing, area, slopes — and feed those; note it, don't build it first.)

## Pipeline + measurement (reuse ECG Akida path; float first)

- `datasets`: add `make_synthetic_crm(...)` (the generator) + a load path
  (`load_crm` / a `"crm"` modality) returning the same dataclass shape, provenance
  = **synthetic** (clearly labeled). No download.
- Wire `"crm"` into `scripts/akida_verify.py`, reusing the ECG waveform → Akida CNN.
- Same discipline: class-weighted loss (if imbalanced), balanced-accuracy selection,
  per-class recall, ≥5 seeds, split by synthetic **subject** (not window — windows
  from one trajectory are correlated), float first then Akida-sim + agreement.
- **Calibration caveat, stated loudly (like the other synthetic cards):** synthetic
  data is separable by construction, so expect high accuracy — this proves the
  PIPELINE and that the time-aligned label is learnable, it is **NOT a clinical
  accuracy claim.** The value is demonstrating the capability end-to-end and that
  the granularity problem is fixed; real accuracy needs the gated LBNP data.

## Deliverables

- `make_synthetic_crm` + `"crm"` modality in `datasets`, wired into
  `scripts/akida_verify.py` (waveform front-end); `report.py` data card for
  `("crm","synthetic")` with the caveats above and the time-aligned-label note.
- `docs/synthetic_crm_results.md`: the physiological model + literature grounding,
  the trajectory/label design, float + Akida-sim (multi-seed) on the synthetic set,
  and an explicit, honest verdict (pipeline + concept demonstrated; not clinical).
- A small visualization (or documented example) showing a trajectory: waveform
  morphology shifting while HR stays flat, then HR rising — the CRM lead effect.
- CLAUDE.md: update M (Massive hemorrhage) — synthetic time-resolved demo built;
  real-data validation pending gated LBNP.
- Offline unit tests: trajectory monotonicity, label time-alignment (a window's
  label matches its `r`), morphology-vs-`r` relationships (amplitude decreases,
  notch blunts), no-leakage subject split. pytest -q green on host + container.
- Commit, push origin main, verify push landed (rev-parse + cat-file), report hash.

## Do NOT

- Do NOT present any synthetic number as clinical hemorrhage-detection accuracy —
  the data card and results doc must state it proves the pipeline/concept only.
- Do NOT stamp one label across a whole trajectory — the per-window time-aligned
  label is the entire reason this exists (the VitalDB lesson).
- Do NOT use the audio filterbank front-end — CRM is morphology (waveform path).
- Do NOT touch ECG/heart/rockpool/existing Akida paths; add additively.
- Do NOT chase Akida-sim fidelity before the float model works (float-first).
