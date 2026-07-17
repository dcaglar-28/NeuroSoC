# Hardware design / interconnect spec — v1 prototype board (Akida)

Engineering starting point for schematic capture (KiCad / Fusion Electronics) of the
v1 board: the always-on biosignal cluster (ECG, PPG, heart sounds) + capnography,
around a BrainChip Akida core. This is a block/interconnect-level plan, not a
schematic — it defines the parts, buses, power, and the items to confirm from
BrainChip's design collateral before layout. Pairs with `docs/hardware_bom.md`.

## System block diagram

```
  ECG AFE (ADS1298)      ─SPI──┐
  PPG (MAX30101/86141)   ─I2C──┤        ┌──────────────────────┐        ┌───────────┐
  MEMS mic (ICS-43434)   ─I2S──┼──────► │  Host (MCU or Linux   │──PCIe  │  Akida    │
  Capno module (Micro-   ─UART─┘        │  SoC): acquire ·      │  or    │  (AKD1000 │
    stream) [+ pump PWR]                │  preprocess/encode ·  │  SPI   │  or module│
                                        │  feed Akida · rules · │◄────── │  ) infer  │
                                        │  fusion               │        └───────────┘
                                        └──────────────────────┘
```

## 1. Compute core — Akida

- **Part choice (confirm bare-chip availability with BrainChip):**
  - *Custom board:* AKD1000 in **FCBGA** (Akida 1.0, purchasable discrete). Verify
    whether AKD1500 is also offered as a bare component vs. board-only.
  - *Carrier board (recommended first):* AKD1000/AKD1500 **M.2 / PCIe module** on a
    carrier — no BGA layout, fastest path.
  - *Committed always-on core (Akida Pico) is IP-only* — not solderable; a later
    silicon-partner / FPGA step, not this board.
- **Host interface:** PCIe (native for the M.2/PCIe modules) **or** SPI for an
  MCU-hosted embedded design. Confirm the AKD1000 embedded host-interface options
  and required lanes/pins from the datasheet — this choice drives the host (below).
- **Power:** multiple rails (core + I/O) with **sequencing** — pull exact rails,
  order, and decoupling from BrainChip's AKD1000 hardware design guide / reference
  schematic. PDN is a real design item (your SI/PI/PDN skills apply).
- **Clock:** reference clock per datasheet. **Memory:** AKD1000 has on-chip SRAM;
  our models are small (footprints ≤ a few mapped layers), so external memory is
  likely unnecessary — confirm against the model sizes in `docs/akida_*_results.md`.

## 2. Host — MCU vs. Linux SoC (two viable paths)

The host acquires every sensor, runs the preprocessing the `eia` software already
does (windowing, band-pass, delta/band-power encoding, per-lead/-window
normalization), feeds Akida the quantized input, reads results, and runs the rules
layer (capnography thresholds) + the fusion logic.

- **Path A — Linux SoC + Akida module (recommended for the prototype).** e.g. an
  i.MX8 / Raspberry Pi CM4-class SoC + Akida **M.2 (PCIe)**. Upside: **MetaTF /
  akida runtime runs on Linux as-is**, matching the software you've validated; PCIe
  data path is turnkey. Downside: higher idle power — fine for a prototype, not the
  always-on target.
- **Path B — MCU + embedded Akida via SPI (for the real low-power device).** e.g.
  **STM32H7** (you know STM32): FPU + DSP for preprocessing, plenty of SPI/I²C/I²S/
  UART, decent RAM. Upside: low power, small. Downside: the Akida host-side runtime
  on a bare-metal/RTOS MCU is more work than on Linux — confirm BrainChip's
  MCU-host support before committing.

Recommendation: **build Path A first** (de-risk sensors + software + Akida
integration with no custom silicon), then move to Path B for the field device.

## 3. Sensor front-ends — per-channel wiring

- **ECG — ADS1298 (SPI).** Analog + digital supplies + reference; electrode
  connector (12-lead for MI; fewer for arrhythmia/shockable); right-leg-drive and
  lead-off detection. SI: short, guarded analog traces; partition analog/digital
  ground.
- **PPG / SpO₂ — MAX30101 / MAX86141 (I²C).** LED drive current budget; shield the
  photodiode from ambient light; place at the skin-contact site (finger/ear).
- **Heart sounds — ICS-43434 I²S MEMS mic** behind an acoustic coupler to the chest
  (or a piezo contact sensor + ADC). Mechanical coupling quality dominates signal
  quality — a mechanical design item as much as electrical.
- **Capnography — OEM sidestream module (Microstream) over UART.** The module does
  the CO₂ measurement; the board provides its power (incl. the **sample pump**,
  higher current), a connector for the cannula/sample line, and the water trap.
  Pump → switched rail (§4).

## 4. Power tree

- Battery → PMIC/fuel gauge → two domains:
  - **Always-on rail:** host + ECG AFE + PPG + mic (+ Akida core if run always-on).
  - **Switched (wake-on-demand) rail:** capnography pump; and the heavier Akida
    module if it isn't kept always-on.
- Akida rail **sequencing** per the design guide. Size the always-on budget against
  the µW–mW target; the pump is the dominant intermittent load.

## 5. Connectors / mechanical

Electrode leads (ECG), PPG optical (integrated or cabled to a fingertip/ear probe),
chest-contact mic, capnography luer + sample line + trap, USB-C (debug/charge),
battery. Handheld form factor → this is where Fusion's ECAD+MCAD integration helps.

## 6. To obtain from BrainChip (before layout)

AKD1000 (and/or AKD1500) **datasheet + pinout**, **hardware design guide /
reference schematic**, power-sequencing spec, and IBIS models — via BrainChip's
eval/partner program (likely under NDA). These resolve the open items below.

## 7. Open decisions / to confirm

- Bare-chip availability: AKD1000 vs AKD1500 as a solderable FCBGA (vs module-only).
- Host interface: PCIe vs SPI → sets host choice (Linux SoC vs MCU).
- Host path A (Linux + M.2) vs B (MCU + embedded Akida) for v1.
- Whether the always-on cluster stays on the (1 W-class) AKD1000 for the prototype,
  with the sub-mW power story deferred to a Pico (IP) spin — a discrete-Akida
  board proves *function*, not the milliwatt claim.

## 8. Suggested prototyping sequence

1. **No custom PCB:** Linux SoC / RPi + Akida M.2 + sensor **eval breakouts** —
   validate the full sensor→host→Akida→inference chain and the software.
2. **Custom carrier PCB:** integrate host MCU/SoC + the four sensor front-ends +
   Akida module; your first board (KiCad/Fusion).
3. **Full custom board:** AKD1000 FCBGA placed directly, once §6/§7 are resolved.
