# Shockable-rhythm (VF/VT) detection — measured results

Deepens the ECG capability under Circulation to a third task: *arrhythmia*
(MIT-BIH, single-lead, per-beat, `docs/akida_ecg_results.md`) → *myocardial
infarction* (PTB-XL, 12-lead, per-recording, `docs/ptbxl_mi_results.md`) →
**shockable rhythm** (VFDB + CUDB, single-lead, per-window rhythm episode) —
implements `docs/shockable_rhythm_task.md`. This is a new task/dataset on a
new binary decision (defibrillate or not), reusing the EXISTING ECG-Akida
waveform scaffolding: ECG-arrhythmia, MI, heart sounds, the synthetic CRM
demo, and `rockpool_models.py` are all untouched — `ShockableData`/
`load_vfdb_cudb`/`load_shockable`/the reused `build_akida_model` are additive.

## Part 0 — dataset access + rhythm-code mapping (verified live, not assumed)

**Access confirmed:** both datasets are open PhysioNet/`wfdb` sources, no
credentialed access needed. VFDB (MIT-BIH Malignant Ventricular Ectopy
Database, `pn_dir="vfdb"`) — confirmed live via `wfdb.get_record_list`: 22
records (`418`–`615`), 250 Hz, 2 channels (both named `"ECG"`; only channel 0
used, for shape consistency with CUDB's single channel). CUDB (Creighton
University Ventricular Tachyarrhythmia Database, `pn_dir="cudb"`): 35 records
(`cu01`–`cu35`), 250 Hz, 1 channel.

**Rhythm annotations are episode markers, not per-beat labels — confirmed
live, and a real bug caught along the way.** In both datasets' `.atr` files,
rhythm changes are recorded as annotations with `symbol == '+'` and the
actual rhythm code in `aux_note` (parenthesis-prefixed, NUL-padded, e.g.
`'(VF\x00'`). CUDB's `.atr` ALSO carries ~950 per-beat `symbol='N'`
annotations per record with EMPTY `aux_note` (a generic beat marker — CUDB's
own documentation states "all beats are labelled normal (although many are
ectopic)", i.e. not diagnostically informative per-beat). An early Part-0
pass conflated these with rhythm annotations (no `symbol` filter), which
inflated the apparent CUDB "episode" count to ~20,000 with sub-second
durations — caught by checking `ann.symbol`, not just `aux_note`, before
trusting any duration statistic. The final loader (`_rhythm_intervals_from_
annotations`) filters to `symbol=='+'` only.

**Rhythm codes actually present** (confirmed live, cross-checked against
PhysioNet's own VFDB documentation, which independently spells out every
code found): VFDB has `AFIB, ASYS, B, BI, HGEA, N, NOD, NOISE, NSR, PM, SBR,
SVTA, VER, VF, VFIB, VFL, VT`. CUDB has the smaller set `AF, N, VF, VT`.

**Shockable rule adopted (de Bruin et al. / AHA-aligned, per
`docs/shockable_rhythm_task.md`):** SHOCKABLE = `VF`/`VFIB` (fibrillation) or
`VFL` (ventricular flutter, clinically VF-equivalent), or `VT` at/above
`VT_RATE_THRESHOLD_BPM = 150` (the low end of the commonly-cited 150–180 bpm
range, chosen for sensitivity — missing a shockable rhythm is worse than an
unnecessary shock advisory). NON-SHOCKABLE = everything else: normal sinus
(`N`/`NSR`), sinus brady (`SBR`), nodal (`NOD`), SVTA, AFIB/AF, bigeminy-
adjacent codes (`B`/`BI`), ventricular escape (`VER`), paced (`PM`),
high-grade ectopy short of sustained VT/VF (`HGEA`), asystole (`ASYS` —
clinically non-shockable: an AED correctly advises no shock for a flat
line), and slow VT. `NOISE` is DROPPED, not folded into non-shockable — it's
a signal-quality state, not a rhythm.

**VT rate has no usable annotation — estimated per window from the raw
waveform instead.** VFDB carries NO per-beat annotations at all (`ann.symbol`
is `{'+'}` only, confirmed across every VFDB record), so a VT episode's rate
can't be read off annotation timing. `_estimate_rate_bpm` computes it per
WINDOW from the raw waveform (5–25 Hz band-pass + peak-picking on the squared
signal, 200 ms minimum peak spacing). Validated in Part 0 against known-VT
windows: measured 173–275 bpm across VFDB record 421's 50 VT episodes and
across a long 885 s VT episode in record 427, all comfortably above
threshold; a sanity check against normal-rhythm 'N' windows in the same
record measured 103–144 bpm, correctly well below. A window whose estimated
rate can't be computed (fewer than 2 detected peaks) is dropped, not guessed
into either class — an early version of the loader had a real bug here
(the "can't estimate" sentinel, `0.0` bpm, was passed straight into the
rate-threshold comparison and silently classified as slow/non-shockable
rather than being dropped) caught by code review before the real run and
fixed prior to any measurement below.

**Window duration + labeling rule, chosen empirically:** VFDB's `VFL`
episodes have median duration 2.8 s (73% under 5 s), `VT` episodes median
4.2 s (54% under 5 s) — shorter than the AED-window range itself for a
majority of cases, so full-window containment doesn't require picking a
sub-2s window; it just means many short episodes contribute zero windows
(dropped as ambiguous, not force-labeled). A Part-0 sweep of `window_sec` in
{4,5,6,8} × `min_coverage` in {0.5,0.6,0.8,1.0} against the cached real
annotation timelines showed full containment (`min_coverage=1.0`) at 5.0s
still yields ~8,000 candidate windows combined with a healthy shockable
count — full containment costs comparatively little yield vs. the loosest
config, so the strictest, least label-ambiguous rule was kept. 5.0s sits at
the low end of the task's 5–8s AED-window range (longer windows cost yield
roughly proportionally with no offsetting quality gain in the sweep).

## Class balance + yield (real data, default record set)

773 windows total from all 57 records (22 VFDB + 35 CUDB): **174 shockable /
599 non-shockable (22.5% shockable — the minority class, as expected)**.
Dropped: 269 ambiguous/transition (straddling a rhythm change), 118 noise,
0 unclassifiable-VT (every VT window in this pull got a usable rate
estimate).

**A real, unplanned data-quality finding: CUDB contributed far less than
VFDB.** All 22 VFDB records loaded cleanly and contributed windows (708 of
the 773, ~92%). Of CUDB's 35 records, **28 have genuine NaN samples in their
raw channel-0 signal** (confirmed directly — e.g. `cu02` has 538 NaN samples,
`cu09` has 1,099, both ~0.04–0.86% of the record — this is a real defect in
the source data, not a loader bug, the same class of issue already
documented for a handful of CinC 2016 heart-sound recordings in
`docs/heart_sounds_results.md`) and are skipped by the existing
`np.isfinite` guard; 5 more loaded cleanly but produced zero windows after
labeling (short/fragmented rhythm episodes, mostly `AF`, with no 5s span
meeting full containment). Only **2 of 35 CUDB records (`cu01`, `cu18`)**
actually contributed usable windows (65 of 773, ~8%). The measurement below
is therefore overwhelmingly VFDB-driven — a real limitation of the combined
pull, not a bug, and worth carrying forward if this task is revisited with a
larger CUDB pull or a NaN-interpolation strategy instead of a hard skip.

## Front-end and architecture

Raw waveform (VF/VT is morphology/rhythm, not spectral) — NOT the audio
filterbank. Each 5s window (1250 native samples @ 250 Hz) is FFT-resampled to
`resample_to=500` (effective 100 Hz, Nyquist 50 Hz — comfortably above VF's
~3–9 Hz dominant-frequency band and a common downsampling target in
published rhythm-classification work), then per-window z-scored. Single-lead,
so it reuses `build_akida_model` (ECG-arrhythmia's/CRM's architecture)
**unchanged** — no new model builder, confirmed by
`test_shockable_reuses_build_akida_model_unchanged_end_to_end` converting and
running end-to-end on `make_synthetic_shockable`'s data shape. Same confirmed
Akida v2 layer constraints apply (square kernel/stride/pool, valid
`Conv2D → ReLU → GlobalAveragePooling2D` block ordering) since nothing about
the architecture itself changed.

## Part A — measurement (float model first, then Akida-sim)

5 seeds, `case_level.split_data` (`GroupShuffleSplit` on record id — VFDB/
CUDB records are different patients; the split log confirms zero record
overlap across train/val/test on every seed), class-weighted loss,
balanced-accuracy checkpoint selection over `n_restarts=10` at `epochs=30`
(bumped up from the ECG/MI/heart/CRM default of `n_restarts=3`/`epochs=15`
— an early 1-seed run at the defaults landed a majority-collapsed checkpoint,
balanced acc exactly 0.500 with AUROC already 0.722, i.e. real ranking
signal the default restart budget wasn't reliably escaping; a quick restart/
epoch sweep confirmed more restarts reliably finds a non-collapsed
checkpoint — same "checkpoint sensitivity" pattern already documented for
real MIT-BIH ECG in `docs/ecg_quant_fixes_results.md`, not a new phenomenon).

**Float (AED-standard metrics, sensitivity = shockable recall, specificity =
non-shockable recall):**

| seed | bal. acc | specificity (class 0) | sensitivity (class 1) | AUROC |
|------|----------|------------------------|-------------------------|-------|
| 0 | 0.817 | 0.978 | 0.655 | 0.926 |
| 1 | 0.827 | 0.993 | 0.661 | 0.944 |
| 2 | 0.873 | 0.835 | 0.910 | 0.963 |
| 3 | 0.703 | 0.993 | 0.413 | 0.937 |
| 4 | 0.717 | 0.481 | 0.952 | 0.938 |

**Float, 5-seed mean ± std:** balanced accuracy **0.787 ± 0.066**, AUROC
**0.942 ± 0.012** (tight — the strongest, most consistent signal in this
measurement), specificity **0.856 ± 0.197**, sensitivity **0.718 ± 0.196**.

**Against the AHA goals (sensitivity ≥90% VF, specificity ≥95%): NOT met on
average**, but the picture is nuanced, not flat failure — 2 of 5 seeds
individually clear the sensitivity bar (seed 2: 0.910, seed 4: 0.952) and 2
of 5 clear the specificity bar (seed 1: 0.993, seed 3: 0.993), just never
both at once in the same seed. The AUROC's tightness (0.942 ± 0.012 vs.
balanced accuracy's 0.787 ± 0.066) is the load-bearing observation: the
underlying ranking signal is strong and stable across seeds, but the
ARGMAX decision threshold is not — the same checkpoint-sensitivity pattern
`docs/ecg_quant_fixes_results.md` found for real MIT-BIH ECG (a better-
trained float model can land on a more fragile decision boundary than a
worse one). This matches the task's own calibration note ("expect the float
model to learn strongly; if near chance, re-check Part 0") — 0.942 AUROC is
unambiguously NOT near chance, so the rhythm mapping/window labeling is not
in question here; the gap to the AHA goals is a decision-threshold/class-
imbalance-at-small-N problem, not a signal problem.

**Akida-sim (post quantization + 5-epoch QAT fine-tune):**

| seed | bal. acc | specificity (class 0) | sensitivity (class 1) | float-vs-sim agreement |
|------|----------|------------------------|-------------------------|--------------------------|
| 0 | 0.864 | 0.978 | 0.750 | 0.964 |
| 1 | 0.880 | 0.947 | 0.814 | 0.924 |
| 2 | 0.933 | 1.000 | 0.865 | 0.894 |
| 3 | 0.819 | 0.986 | 0.652 | 0.937 |
| 4 | 0.896 | 0.902 | 0.889 | 0.694 |

**Akida-sim, 5-seed mean ± std:** balanced accuracy **0.878 ± 0.037**,
specificity **0.963 ± 0.035** (clears the AHA ≥95% goal ON AVERAGE),
sensitivity **0.794 ± 0.085**, float-vs-Akida-sim agreement **0.883 ± 0.097**.

**Akida-sim is consistently ABOVE the float mean on every headline metric**
(balanced acc 0.878 vs. 0.787, specificity 0.963 vs. 0.856, sensitivity
0.794 vs. 0.718), with tighter variance too — the QAT fine-tune step
(5 epochs on the quantized model with the same class-weighted loss) appears
to be doing real, additional optimization beyond what the float-restart
selection alone found, not just "quantization happened to not hurt." This is
the opposite direction from ECG-arrhythmia/heart sounds/MI's usual
near-parity story and is the most notable finding of this measurement — worth
watching if this task is revisited, since it suggests the QAT step is
functioning here as a second optimization pass on an already-marginal
decision boundary rather than a pure fidelity check. Float-vs-Akida-sim
agreement (0.883 ± 0.097) is real but lower and more variable than
ECG-arrhythmia's (~0.98) or MI's (~0.92) — consistent with `docs/
ecg_quant_diagnosis.md`'s finding that disagreements cluster at low decision
margin: this task's float decisions are, on average, closer to the margin
than those tasks' were.

**Footprint:** `(500, 1, 1)` uint8 in → `(2,)` logits out, 5 mapped Akida
layers, 8/8 weight/activation quantization — same footprint shape as
ECG-arrhythmia/CRM (same architecture, larger window: 500 vs. 187 timesteps).

## Verdict

The float model genuinely detects shockable rhythm on real VFDB/CUDB data —
AUROC 0.942 ± 0.012 is a strong, stable, well-above-chance signal, confirming
the Part-0 rhythm mapping and window-labeling rule are correct (per the
task's own calibration note, a near-chance AUROC would have meant
re-checking Part 0; this AUROC rules that out). The AHA's specific
sensitivity ≥90% / specificity ≥95% targets are **not met simultaneously by
the float model's argmax decision on average**, though the Akida-sim path
(after QAT) DOES clear specificity on average and comes closer on
sensitivity than float does. This is a checkpoint-sensitivity /
class-imbalance-at-small-N story (599 non-shockable vs. 174 shockable
windows total, split by record into 14 train / 4 val / 6 test cases per
seed), the same pattern already seen for real MIT-BIH ECG in
`docs/ecg_quant_fixes_results.md` — not a broken pipeline, and not a signal
that isn't there. Fix 1 from that doc's ranked list (more/better-targeted
training data, here more CUDB records or a NaN-interpolation strategy
instead of a hard skip) is the most promising lever if this is revisited,
given CUDB contributed only 8% of usable windows here.

## Caveats (carried forward, unchanged)

- **Akida-sim is not a confirmed bit-exact simulator** — BrainChip does not
  publish an explicit bit/cycle-accurate claim the way SynSense does for
  XyloSim (see `docs/akida_ecg_results.md` Part 0). Report "Akida-sim
  agreement," not "verified against silicon."
- **This measurement targets Akida 2.0 (FPGA/software backend), not the
  Pico-class production target** — no hardware-in-the-loop measurement here.
- **VT-rate estimation is a lightweight peak-picker, not a diagnostic-grade
  QRS detector** — validated qualitatively against known-VT/known-normal
  windows in Part 0, not a clinical-accuracy guarantee. It found a usable
  rate for every VT window in this pull (0 unclassifiable-VT drops), but
  that's this dataset's VT episodes being uniformly fast and clean, not
  proof the estimator handles every waveform.
- **CUDB's real contribution to this measurement is small** (~8% of
  windows, from 2 of 35 records) — see the class-balance section above. The
  22.5% shockable / 5-record-per-test-split class balance is closer to a
  VFDB-only measurement than a genuinely combined one.
- **No VF-vs-VT sub-split** — deliberately out of scope per
  `docs/shockable_rhythm_task.md`, noted as a possible follow-up.
- **Single-lead** (VFDB channel 0 only, for shape consistency with CUDB's
  single channel) — VFDB's second channel is unused; a `(leads, time)`
  2-channel variant (mirroring `build_akida_mi_model`'s genuinely-2-D
  approach) is a possible follow-up, not built here.

## Reproduce

```bash
scripts/akida_docker_run.sh pytest -q
scripts/akida_docker_run.sh python scripts/akida_verify.py \
    --modality shockable --real --n-seeds 5 --n-restarts 10 --epochs 30
```

## Next steps (noted, not built here)

- Pull more/all-available CUDB records with a NaN-interpolation (or
  segment-level NaN-skip, not whole-record-skip) strategy instead of a hard
  per-record skip, to get a genuinely balanced VFDB+CUDB contribution.
- Margin-aware training (per `docs/ecg_quant_diagnosis.md`'s ranked-fix
  list) to stabilize the argmax decision boundary given the AUROC signal is
  already strong.
- VF-vs-VT sub-classification (a finer decision than binary shockable/not).
- A `(leads, time)` 2-channel VFDB variant (`build_akida_mi_model`-style),
  since VFDB's second channel is currently unused.
