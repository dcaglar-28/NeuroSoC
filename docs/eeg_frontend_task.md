# Task spec — EEG front-end redesign: feature-based front-end for seizure detection

**For:** Claude Code (repo, `.venv`, CHB-MIT already wired; the 4-patient /
12-record pool is already cached in `data/chbmit/` — no new download needed).

**Why:** The diagnostic is settled (docs/eeg_seizure_results.md + the Colab run):
subject-independent AND patient-specific are both ~chance, and the **float** model
(plain PyTorch, no chip, no quantization) is at chance — so the seizure signal is
being destroyed by the **front-end**, before the network, not by data volume or
hardware. Root cause: we delta/level-crossing-encode the **raw waveform**, which
captures *edges* (perfect for ECG's QRS transient) but discards the *sustained
rhythmic activity and spectral-power shift* that define a seizure. This task
replaces that front-end with the representation clinical detectors actually use.

**What real scalp-EEG detectors do (the design basis):** classical detectors
(Gotman; Persyst Reveal) never feed a network the raw trace — they extract a
small set of physiologically-grounded features and detect *sustained rhythmic
evolution*. The best-evidenced features are **line length**, **relative band
power** (esp. ~12.5–25 Hz), and a **rhythmicity** measure. This is also a double
win for the neuromorphic target: a few feature channels over few sub-windows is
both more learnable (proven: raw isn't) and lower power (fewer inputs × timesteps).

---

## The redesign

Replace the raw-waveform → delta-encode front-end with, per channel, per short
**sub-window** across the analysis window:

- **Line length:** sum of |x[t] − x[t−1]| over the sub-window (amplitude+frequency
  complexity — the top single feature).
- **Relative band power:** Welch PSD → power in δ (0.5–4), θ (4–8), α (8–13),
  β (13–25) Hz as a fraction of total power (relative = amplitude-invariant).
  Keep the ~12.5–25 Hz band explicitly (best-performing per the literature).
- **Rhythmicity:** one measure — spectral entropy (low = rhythmic), or peak-PSD-
  to-mean ratio, or dominant-frequency stability across sub-windows. Pick one,
  document why.

Compute these over sliding sub-windows (e.g. a 4 s window → 8–16 sub-windows of
0.25–0.5 s) → a low-dimensional **feature time-series** (features × channels ×
sub-windows). Sub-windows are the timesteps: few, and they still carry the
seizure's temporal evolution. Then spike-encode the (slowly-varying) feature
envelopes and feed the SNN. Normalize each feature (z-score) with stats fit on
**train only**, applied to val/test — leaking normalization across the split is a
silent bug; guard it.

### The binding constraint — input budget (design choice, document it)
Features × channels can blow past Xylo's ≤16 inputs fast (6 ch × 6 features = 36).
Start **compact enough to fit Xylo** as the constrained baseline — e.g. trim to
~4 channels × (line length + 2 bands + rhythmicity ≈ 4 features) = 16, or pool a
feature across channels (Gotman's "≥2 channels rhythmic" is itself a cross-channel
aggregate). Document the chosen feature × channel layout and the mapped input
count against the 16 ceiling. Note in the results that the **committed Akida
target loosens this ceiling**, so the feature set can expand there later — a point
in favor of the single-vendor Akida decision.

## Keep it a clean A/B — don't rip out the old path
Add the feature front-end as a selectable option (e.g. `eeg_frontend={"raw",
"features"}`), leaving the raw-delta path intact as the baseline. The whole value
here is the comparison: same data, same SNN, same splits, only the front-end
differs.

## Measurement — the ONE number that matters first
Re-run BOTH splits (subject-independent and patient-specific, `--split`) on the
SAME cached 4-patient pool, ≥5 seeds, mean ± spread. **The decisive metric is the
FLOAT model** (AUROC/AUPRC/sensitivity/specificity/FA-per-hour + balanced acc) —
because the diagnosis says the float model is starved by the front-end. So:

- If the feature front-end lets the **float** model clear chance (AUROC clearly
  > 0.5, balanced acc above base rate, on the same data the raw front-end failed
  on) → the diagnosis is confirmed and the modality is unblocked. THEN scale up
  patients/records for a real result, and only then look at XyloSim on-chip
  agreement (secondary — fidelity matters only once the float model learns).
- If the **float** model is STILL ~chance with the feature front-end → the front-
  end wasn't the whole story; next suspects are montage choice and true data
  scale. Report that plainly and stop, don't keep tuning features blindly.

Report the feature front-end vs. raw-delta baseline side by side.

## Deliverables

- Feature extraction as a pure, offline-testable module (numpy/scipy) — line
  length, relative band power, rhythmicity — plus the sub-window framing and the
  spike-encoding of feature envelopes. Wire it behind the `eeg_frontend` option
  in `load_chbmit` / the encode path used by `train.py` and `scripts/xylo_verify.py`.
- `docs/eeg_frontend_results.md`: feature × channel layout + input count vs. 16,
  chosen rhythmicity measure, float feature-vs-raw comparison on both splits
  (mean ± spread), and an explicit verdict (did the float model clear chance?).
- CLAUDE.md: update the H/EEG status with the outcome.
- Offline unit tests: line length on a known ramp; relative band power on a pure
  sine at a known frequency landing in the right band; rhythmicity high on a clean
  sine vs. low on noise; no-leakage check on the train-only normalization.
- Commit + push, verify the push landed (rev-parse + cat-file), report the hash.

## Do NOT

- Do NOT pull new CHB-MIT data — iterate on the cached 4-patient pool for speed;
  scale up only after the feature front-end clears chance on the float model.
- Do NOT delete the raw-delta front-end — it's the A/B baseline.
- Do NOT leak feature-normalization stats across train/val/test.
- Do NOT jump to XyloSim/on-chip fidelity until the float model learns — that
  order was the whole lesson of this diagnostic.
- Do NOT expand to a huge feature set to force a result — start with the proven
  small set; if it doesn't move the float model, that's the finding.
