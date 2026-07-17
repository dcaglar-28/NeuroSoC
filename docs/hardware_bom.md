# Hardware sensor front-ends & Akida board integration (v1)

One-page hardware map for the working-modality cluster plus capnography, and how
it integrates with an Akida-based board. Engineering/pitch reference; not a
committed BOM (final part selection is a procurement step).

## Sensor front-ends

The working modalities collapse to **three sensor front-ends**; capnography adds a
fourth. All are **digital-output** over standard buses.

| Front-end | Modalities served | Sensor / transducer | Representative parts | Bus | Power tier |
|---|---|---|---|---|---|
| **ECG** | arrhythmia · MI · shockable rhythm | electrodes + ECG analog front-end (AFE) | TI **ADS1298** (8-ch → 12-lead, for MI); ADS1292 (2-ch); Maxim **MAX30003** (single-lead, low-power) | SPI | always-on |
| **PPG / pulse-ox** | occult hemorrhage (CRM) · SpO₂ | optical: red+IR LEDs + photodiode | Maxim **MAX30101 / MAX86141**; TI AFE4404; ADI ADPD4000 | I²C / SPI | always-on |
| **PCG (heart sounds)** | cardiac auscultation | contact/MEMS mic behind a stethoscope coupler, or piezo | TDK **ICS-43434** (I²S MEMS); Knowles; or piezo + ADC | I²S | always-on |
| **Capnography (CO₂)** | airway | sidestream **NDIR CO₂ module** + nasal cannula + micro-pump + water trap | Medtronic **Microstream**, Masimo ISA/NomoLine (clinical); Sensirion SCD4x / SprintIR (prototype) | UART | **wake-on-demand** |

12-lead ECG (needed for MI localization) drives the choice of a multi-channel AFE
(ADS1298); single-lead arrhythmia/shockable can run on a 2-ch or single-ch AFE.

## Board topology — the MCU bridges sensors to Akida

**Akida is a digital compute engine — it has no ADC or analog front-end.** No sensor
connects to it directly. A host MCU/application processor acquires every sensor,
does the preprocessing (the same windowing / band-pass / delta-encoding /
band-power / normalization the `eia` software does today), feeds Akida the quantized
input tensor, and reads back the inference.

```
 [ECG AFE      — SPI ]  ┐
 [PPG optical  — I2C ]  │      ┌──────────────────────────┐      ┌──────────────┐
 [MEMS mic     — I2S ]  ├────► │  Host MCU / app processor │────► │    Akida     │
 [CO2 module   — UART]  ┘      │  acquire · window ·       │ PCIe │  quantized   │
                               │  preprocess/encode · feed │  or  │  CNN         │
                               │  Akida · read result      │◄──── │  inference   │
                               └──────────────────────────┘ SPI  └──────────────┘
```

Compatibility rules: (1) every sensor is digital-output over a standard bus (all
parts above are); (2) the MCU has the bus count + horsepower to acquire and
preprocess; (3) Akida connects to the MCU over its host interface (PCIe or SPI).
Akida never sees a raw sensor — only the preprocessed tensor.

*(Akida is event-based and could ingest spike/event input natively, but for
conventional biosignal sensors you sample + preprocess on the MCU. Only true
neuromorphic sensors — event cameras, silicon cochleas — would stream events
directly; not needed here.)*

## Power / always-on tiering

- **Always-on cluster (µW–mW):** ECG, PPG, MEMS mic — low-power, run continuously;
  the intended job of a sub-mW Akida (Pico-class) core.
- **Wake-on-demand:** capnography — the sidestream pump (and consumable
  cannula/trap) draws real power; run intermittently, not 24/7. Fits the tiered
  architecture (always-on biosignal cluster + intermittently-powered modules).

## Akida silicon options (what you can actually build around)

| Option | What it is | Fit |
|---|---|---|
| **AKD1000 — discrete SoC (FCBGA)** | Akida 1.0 as a purchasable chip (also on DigiKey) | design onto a **custom PCB** |
| **AKD1000 — M.2 / PCIe dev board** | the chip on a module | fastest prototype on a carrier + Raspberry Pi / MCU |
| **Akida 2.0 / Pico — IP** | licensable IP for custom silicon (FPGA eval available) | production always-on core; **not a solderable chip** |

The committed always-on part (**Pico**) is **IP-only** — it lives inside custom
silicon or an FPGA eval, not a chip you solder. So a custom *discrete-Akida* board
today means **AKD1000 (Akida 1.0)**; Pico is a later custom-silicon step. Practical
path: prototype on **AKD1000 M.2 + MCU**, then a **custom AKD1000 board**, with
Pico-as-IP reserved for a production silicon spin.
