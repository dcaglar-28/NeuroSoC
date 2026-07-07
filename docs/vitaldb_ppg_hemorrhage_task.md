# Task spec — VitalDB PPG hemorrhage task (real, open blood-loss label)

**For:** Claude Code (repo, `.venv` active).
**Why:** Replace the BIDMC **SpO2 proxy** with a **real** hemorrhage-relevant PPG
label. VitalDB is fully open (PhysioNet, `pip install vitaldb`), has fingertip
PPG + ABP + ECG waveforms, and a real blood-loss label (`intraop_ebl`,
estimated blood loss). This is the first *open* dataset that actually carries a
hemorrhage label — no data-use agreement needed.

Follow the guiding principle from `docs/real_hemorrhage_dataset_task.md`
(train/report each dataset separately, print a data card + warnings, never pool)
and the multi-seed / short-timestep lessons below.

---

## Part 0 — research + verify BEFORE coding (don't assume field names)

VitalDB track and clinical-field names are specific; confirm them against the
installed library, not memory:
- Install: `pip install vitaldb`. Add `vitaldb` to the `[data]` extra in pyproject.
- Case list + clinical info: fetch the clinical parameters table (e.g. the
  public `https://api.vitaldb.net/cases` CSV) and confirm the exact **estimated
  blood loss** field name (likely `intraop_ebl`, mL) and any transfusion fields.
- Waveform tracks: confirm the fingertip **PPG/pleth** track name (likely
  `SNUADC/PLETH` or `Solar8000/PLETH`) and its sampling rate (documented 500 Hz).
- Determine which cases actually HAVE both the PLETH track AND a recorded EBL
  value — not all cases have all tracks. Print how many qualify.

If any of these can't be confirmed, STOP and report what's missing rather than
guessing a field name.

## Part 1 — loader

Add `load_vitaldb_ppg(...)` in `src/eia/datasets.py`, mirroring `load_bidmc_ppg`,
returning `PpgData` with `source="vitaldb"` and the provenance plumbing
(`requested_real`, `[warn]` on fallback, honored by `require_real`).

- **Signal:** fingertip PPG/pleth waveform, segmented into fixed windows (start
  with a few seconds per segment, matching the per-pulse framing PPG already
  uses). Cache downloaded waveforms under `data/` (gitignored) so re-runs are
  cheap — VitalDB pulls are large.
- **Label (v1, binary, documented as coarse):** derive from case-level
  `intraop_ebl`: e.g. class 1 = significant blood loss (EBL >= a documented
  threshold, or transfused), class 0 = minimal (EBL below a low threshold).
  Put the exact threshold in the docstring and the data card. A graded version
  can come later; keep v1 binary.
- **Resampling:** downsample the 500 Hz PPG toward a Xylo-friendly rate / short
  timestep count (use the same `resample_to`-style decoupling as `load_mitbih`:
  capture the physiological duration, then pick the Xylo timestep budget
  independently). Fewer timesteps is better for on-chip fidelity — see the
  root-cause section in `CLAUDE.md`.
- **Split BY CASE, not by segment.** All segments from one surgical case must be
  in the same split. Warn on any case-level leakage.

## Part 2 — data card + honest caveats

The `report.data_card` for VitalDB must state plainly:
- source = vitaldb, label = "intraop_ebl >= <threshold> mL (case-level)".
- **Caveats (print them):** (1) intraoperative, ANESTHETIZED patients — anesthesia,
  vasopressors, and surgical context confound PPG vs. conscious field trauma;
  (2) EBL is an ESTIMATE and a WHOLE-CASE total, so the label is coarse and NOT
  time-aligned to the moment of bleeding; (3) likely heavy class imbalance
  (most surgeries are low-EBL) — use the existing class-weighted loss +
  balanced-accuracy selection, and report per-class recall.

This is a real hemorrhage-relevant label and a big step up from the SpO2 proxy —
but it is NOT conscious-hemorrhage ground truth. Say so in the card.

## Part 3 — train, verify, measure (multi-seed from the start)

- Run `train.py` and `scripts/xylo_verify.py` on VitalDB as its own modality/
  dataset, reported separately (never pooled with BIDMC/synthetic).
- **MULTI-SEED:** report float and XyloSim balanced accuracy + per-class recall +
  float-vs-sim agreement as **mean +/- spread over >=5 seeds**. The ECG lesson:
  single-seed XyloSim agreement is noise; a result only counts if it clears the
  seed band.
- Keep the model Xylo-mappable (2-channel delta input, <=63 hidden, short window).

## Deliverable

- `load_vitaldb_ppg` returning `PpgData(source="vitaldb")` with a documented,
  real blood-loss label and case-level split.
- A data card printing the label definition + the three caveats above.
- Multi-seed float + XyloSim results for VitalDB, reported on its own.
- README + CLAUDE.md updated: VitalDB added to the datasets list; note it as the
  real (open) hemorrhage label that supersedes the BIDMC SpO2 proxy for the
  hemorrhage task (keep BIDMC as a secondary real-PPG dataset, don't delete it).
- All existing tests green; add a test for the VitalDB label thresholding logic
  (can run on a tiny synthetic EBL array, no network).

## Gotchas

- **Network + size:** VitalDB waveform pulls are large; cache in `data/`
  (gitignored), and support a `--max-cases` / subset arg so a quick run doesn't
  download everything.
- **Not all cases have PLETH or EBL** — filter to cases with both; report the
  count and the class balance.
- **Case-level leakage** is the #1 risk (many segments per case) — split by case.
- **Don't silently fall back to synthetic** if VitalDB fails — honor
  `require_real` and print a `[warn]` naming the reason, like the other loaders.

## Reference

- VitalDB on PhysioNet: https://physionet.org/content/vitaldb/1.0.0/
- VitalDB paper (Nature Sci Data): https://www.nature.com/articles/s41597-022-01411-5
- Existing loader to mirror: `src/eia/datasets.py` (`load_bidmc_ppg`, `load_mitbih`
  for the `resample_to` timestep-decoupling pattern).
- Root-cause / short-timestep design principle: `CLAUDE.md` fidelity-gap section.
