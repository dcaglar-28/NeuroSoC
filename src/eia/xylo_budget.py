"""Xylo resource-budget checker — one chip vs. many?

Answers a concrete deployment question: given the per-modality SNNs we train
(each: 2 delta input channels, a hidden LIF pool, a small readout), how many
modalities fit on a *single* Xylo chip, and which hardware limit binds first?

The Xylo-Audio 2 (SYNS61201) SNN-core limits (Rockpool "Xylo in numbers"):
    input channels   <= 16
    hidden neurons   <= 1000
    output neurons   <= 8
    input synapses / hidden neuron <= 2   (our delta ON/OFF = 2, exactly fits)
    weights 8-bit

Pure python — no rockpool, no torch. This is design-time arithmetic, not the
bit-precise XyloSim check (that lives in `rockpool_models.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

XYLO_MAX_INPUT_CHANNELS = 16
XYLO_MAX_HIDDEN_NEURONS = 1000
XYLO_MAX_OUTPUT_CHANNELS = 8


@dataclass
class Modality:
    name: str
    n_in: int = 2       # delta ON/OFF channels
    n_hidden: int = 63
    n_out: int = 2


@dataclass
class BudgetReport:
    total_in: int
    total_hidden: int
    total_out: int
    fits: bool
    binding: str          # which limit binds first
    max_modalities: int   # how many of the *given* modalities fit on one chip
    detail: str

    def __str__(self) -> str:
        ok = "FITS" if self.fits else "DOES NOT FIT"
        return (
            f"Xylo one-chip budget: {ok}\n"
            f"  input channels : {self.total_in} / {XYLO_MAX_INPUT_CHANNELS}\n"
            f"  hidden neurons : {self.total_hidden} / {XYLO_MAX_HIDDEN_NEURONS}\n"
            f"  output neurons : {self.total_out} / {XYLO_MAX_OUTPUT_CHANNELS}\n"
            f"  binding limit  : {self.binding}\n"
            f"  {self.detail}"
        )


def fits_one_chip(modalities: list[Modality]) -> BudgetReport:
    """Check whether a set of per-modality sub-networks co-resident on one Xylo
    chip (concatenated inputs, separate hidden pools, separate outputs) fit."""
    tin = sum(m.n_in for m in modalities)
    thid = sum(m.n_hidden for m in modalities)
    tout = sum(m.n_out for m in modalities)

    over = []
    if tin > XYLO_MAX_INPUT_CHANNELS:
        over.append("input channels")
    if thid > XYLO_MAX_HIDDEN_NEURONS:
        over.append("hidden neurons")
    if tout > XYLO_MAX_OUTPUT_CHANNELS:
        over.append("output neurons")

    # Which limit binds first as you add identical modalities?
    if modalities:
        per = modalities[0]
        cap_in = XYLO_MAX_INPUT_CHANNELS // max(per.n_in, 1)
        cap_hid = XYLO_MAX_HIDDEN_NEURONS // max(per.n_hidden, 1)
        cap_out = XYLO_MAX_OUTPUT_CHANNELS // max(per.n_out, 1)
        caps = {"input channels": cap_in, "hidden neurons": cap_hid,
                "output neurons": cap_out}
        binding = min(caps, key=caps.get)
        max_mods = caps[binding]
    else:
        binding, max_mods = "none", 0

    fits = not over
    detail = (
        f"of these {len(modalities)} modalities, "
        f"up to {max_mods} such modalities fit on ONE chip "
        f"(binding: {binding})."
    )
    return BudgetReport(tin, thid, tout, fits, binding, max_mods, detail)


def default_modalities(names) -> list[Modality]:
    """Build default-sized Modality entries (matches build_xylo_snn defaults)."""
    return [Modality(n) for n in names]
