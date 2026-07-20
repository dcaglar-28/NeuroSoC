# Hardware bring-up plan — parallel work + bench bring-up

Playbook for the current stage: software + Akida-simulator validation are done; the
project is at hardware bring-up (Akida Cloud eval + an off-the-shelf Akida board on
order). This covers (A) what to build *now* while waiting — none of it needs the
board or the cloud — and (B) the bench bring-up once hardware arrives. No custom
silicon or custom PCB is on the critical path.

## A. While you wait (needs neither the board nor cloud access)

The compute is de-risked, so the remaining critical-path work is host-side software
and sensors — all independent of Akida.

1. **Order the sensor eval breakouts now** (cheap, ship fast, Akida-independent):
   - **ECG:** ADS1298 breakout/eval (8-ch → 12-lead) — SPI.
   - **PPG / SpO₂:** MAX30101 (or MAX86141) breakout — I²C.
   - **Heart sounds:** an I²S MEMS-mic breakout (e.g. SPH0645 / ICS-43434) + a
     stethoscope-style acoustic coupler.
   - **Capnography:** the true medical module (Microstream) is hard to source for a
     bench build; start with an accessible NDIR CO₂ sensor (Sensirion SCD4x /
     SprintIR) to prototype the *ventilation-present / EtCO₂ / RR* logic, and note
     it is a slower/rougher capnogram than a clinical module — a stand-in for the
     software, not the final sensor.

2. **Build the host-side acquisition + preprocessing layer** on a Raspberry Pi 5
   (or similar SBC): read each breakout over its bus (SPI/I²C/I²S/UART), window it,
   and run the *existing* `eia` preprocessing (encoding / band-power / per-window
   normalization). This is the bulk of the remaining glue and it never touches
   Akida — Akida only consumes the finished tensor. De-risks: the whole sensor →
   host → preprocessed-input chain, before the board even arrives.

3. **Build the rule-based fusion layer** (pure software, no hardware): combine the
   existing calibrated single-signal model outputs into shock-etiology (cardiogenic
   vs. hemorrhagic vs. obstructive) using published clinical physiology. This is the
   differentiated capability the brief leads with, and it's buildable today.

4. **Prep the energy analysis:** assemble the MAC/SOP-vs-measured framework so the
   moment cloud/board access lands you can drop in real numbers — energy is the one
   open thesis item and the reason the cloud FPGA matters.

Sequence: (1) + (2) in parallel (order parts, build acquisition against them as they
arrive), then (3), with (4) staged for hardware.

## B. Bench bring-up (when the board + breakouts arrive)

**Host / board:** a **Raspberry Pi 5** is the sweet spot — it runs Linux for the
MetaTF/akida runtime, has PCIe for an Akida M.2/PCIe module, *and* exposes the GPIO
buses (SPI/I²C/I²S/UART) for the sensor breakouts, so one board hosts both Akida and
the sensors. (Match the Akida module's interface: M.2 → M.2 hat; PCIe card → the Pi 5
PCIe connector or an x86 mini-PC with GPIO sensors bridged separately.)

**Wiring:** each breakout to its bus on the Pi — ECG (ADS1298) on SPI, PPG (MAX30101)
on I²C, mic on I²S, CO₂ module on UART. Shared ground; keep the ECG analog lines short.

**Software bring-up order (one thing at a time — don't integrate everything at once):**
1. Install the akida runtime on the Pi; confirm it **enumerates the Akida board**.
2. Load one converted model (start with ECG) and run it on a **canned test input** →
   confirm the on-board result matches your Akida-simulator result. (Proves the
   board + runtime before any sensor.)
3. Stream **one live sensor** (ECG via ADS1298) → `eia` preprocessing → the board →
   **live inference**. First real end-to-end modality.
4. Add the remaining modalities one at a time (PPG, heart sounds, then capnography's
   rule layer).
5. Wire the **rule-based fusion** over the live model outputs.
6. Package as a repeatable demo (the same story as `notebooks/04_mvp_pitch.ipynb`,
   now on real hardware and live sensors).

## C. What Akida Cloud unblocks (in parallel)

The cloud FPGA is for what the purchasable boards *can't* give you: evaluating the
sub-milliwatt **Akida Pico** / Akida 2.0 (IP-only) and **characterizing energy/power**
on the target-class silicon. That closes the one open thesis claim — the always-on
milliwatt story — which the ~1 W-class AKD1000/AKD1500 dev boards can prove the
*function* of but not the *power* of.

## Honest note

The dev board proves the **system** (sensors → inference → fusion, live); the cloud
proves the **power** (Pico). Neither is the other. A working board demo is not the
low-power claim — keep those two results distinct in the pitch.
