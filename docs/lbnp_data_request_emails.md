# LBNP data-access request emails (hemorrhage / CRM — Track 2)

Draft outreach to the two gated induced-hypovolemia (LBNP) PPG datasets that have
the right physiology + time-resolved labels for the flagship occult-hemorrhage
signal (see `docs/synthetic_crm_results.md` for the in-hand synthetic Track 1).
Research-first, non-commercial framing (gated clinical data is released far more
readily to academic research than to a venture). Fill any remaining brackets and
confirm current emails before sending.

---

## Recipients

**Oslo (dynamic-LBNP).** Nesaragi et al., *Biocybernetics and Biomedical
Engineering* 43 (2023) 551–567 — "Non-invasive waveform analysis for emergency
triage via simulated hemorrhage: an experimental study using novel dynamic lower
body negative pressure model."
- **To:** Dr. Naimahmed Nesaragi (corresponding author) — `naimahmed.nesaragi@gmail.com`
- **Cc:** Prof. Ilangko Balasingham (senior author, NTNU / Oslo University Hospital
  — confirm current NTNU/UiO email)

**Yale.** Chand, Chiu, Chou, Alian, Shelley, Wu — medRxiv 2025 (10.1101/2025.05.02.25326908)
— "Comparison of feature-based indices derived from photoplethysmogram recorded
from different body locations during lower body negative pressure." Data collected
at Yale-New Haven.
- **To:** Dr. Aymen Alian (led data collection, Yale Anesthesiology) — `aymen.alian@yale.edu`
- **Cc:** Prof. Kirk Shelley (Yale Anesthesiology); Prof. Hau-Tieng Wu (NYU Courant)
  — `hauwu@cims.nyu.edu`

---

## Email 1 — Oslo

**Subject:** Data-access request — dynamic-LBNP PPG dataset (BBE 2023) for neuromorphic hemorrhage-detection research

Dear Dr. Nesaragi,

I read your 2023 *Biocybernetics and Biomedical Engineering* paper on classifying
hypovolemia from non-invasive waveforms under a dynamic LBNP model with great
interest. Your use of a *dynamic* rather than step-wise protocol — so the labels
track fluctuating blood volume over time — is exactly the property I've found
missing in other datasets, where a single whole-case blood-loss value gets applied
to every segment.

I'm an undergraduate Electrical and Computer Engineering researcher at Carnegie
Mellon University, where I work in the NeuroAI Computer Architecture Lab on spiking
neural networks for energy-efficient edge hardware (previously on processing-in-
memory architectures at ETH Zürich's SAFARI group). I'm developing an offline,
ultra-low-power method for occult-hemorrhage detection — running compensatory-
reserve estimation as a compact network on neuromorphic hardware, for austere
pre-hospital settings with no power or network.

The main bottleneck is data with the right physiology: induced central hypovolemia
in conscious subjects with time-resolved labels. Your dynamic-LBNP PPG (and
arterial-waveform) dataset is one of very few that fits. Would you be open to
sharing it for research use? I'm glad to sign a data-use agreement, keep the data
restricted to this project and non-commercial research, cite your work prominently,
and share methods and results back — and I'd welcome a collaboration if that's of
interest.

I'm happy to send a fuller description of the project on request. Thank you very
much for considering it.

With appreciation,
Dilara Caglar
B.S. Electrical & Computer Engineering, Carnegie Mellon University
dcaglar@andrew.cmu.edu · dilaracaglar.com · github.com/dcaglar-28

---

## Email 2 — Yale

**Subject:** Data-access request — Yale LBNP multi-site PPG dataset for neuromorphic hemorrhage-detection research

Dear Dr. Alian,

I read your 2025 study comparing feature-based PPG indices recorded from the ear,
nose, and finger during lower body negative pressure with great interest — the
multi-site comparison and the harmonic phase/amplitude indices are directly
relevant to what I'm working on.

I'm an undergraduate Electrical and Computer Engineering researcher at Carnegie
Mellon University, working in the NeuroAI Computer Architecture Lab on spiking
neural networks for energy-efficient edge hardware (previously on processing-in-
memory at ETH Zürich's SAFARI group). I'm developing an offline, ultra-low-power
method for occult-hemorrhage detection — compensatory-reserve estimation from the
PPG waveform running as a compact network on neuromorphic hardware, for field
settings without power or connectivity.

Correctly time-resolved, induced-hypovolemia PPG from conscious subjects is the
piece I can't obtain openly, and your Yale-New Haven LBNP dataset fits closely.
Would you consider sharing it for research purposes? I'm happy to complete a
data-use agreement, restrict the data to this project and non-commercial research,
cite the work prominently, and share methods and results back — and I'd be glad to
collaborate if you're open to it.

I can send a fuller description of the project and intended use on request. Thank
you very much for your time.

With appreciation,
Dilara Caglar
B.S. Electrical & Computer Engineering, Carnegie Mellon University
dcaglar@andrew.cmu.edu · dilaracaglar.com · github.com/dcaglar-28
