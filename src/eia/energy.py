"""Analytical energy model: event-driven (SNN) vs. dense (conventional) pipelines.

There is no silicon in Phase 0, so we estimate energy the way the architecture
literature does: count the dominant arithmetic operations and multiply by a
per-operation energy figure.

  * A conventional network spends most of its energy on multiply-accumulates
    (MACs), one per synapse per inference, regardless of the input.
  * A spiking network spends energy on synaptic operations (SOPs) that only
    happen when a presynaptic neuron *fires*. Fewer spikes -> fewer SOPs.

Per-op energy figures below are order-of-magnitude values drawn from commonly
cited 45 nm estimates (Horowitz, ISSCC 2014) and neuromorphic accelerator
reports. They are documented constants you can swap out — the point of the
harness is the *methodology* and the relative comparison, not absolute joules.
"""

from __future__ import annotations

from dataclasses import dataclass

# Order-of-magnitude energy per operation, in joules.
# 32-bit MAC ~ 3.1 pJ; a spiking synaptic op (accumulate only, no multiply) is
# roughly an order of magnitude cheaper. Adjust for your target process/silicon.
E_MAC_J: float = 3.1e-12   # energy of one dense multiply-accumulate
E_SOP_J: float = 0.9e-12   # energy of one spiking synaptic operation


@dataclass
class EnergyReport:
    dense_macs: int
    dense_energy_j: float
    snn_sops: int
    snn_energy_j: float

    @property
    def energy_ratio(self) -> float:
        """Dense energy / SNN energy. >1 means the SNN is cheaper."""
        return self.dense_energy_j / self.snn_energy_j if self.snn_energy_j else float("inf")

    def __str__(self) -> str:
        return (
            f"Dense baseline : {self.dense_macs:,} MACs  ->  {self.dense_energy_j*1e6:.3f} uJ\n"
            f"Event-driven   : {self.snn_sops:,} SOPs  ->  {self.snn_energy_j*1e6:.3f} uJ\n"
            f"Energy ratio   : {self.energy_ratio:.1f}x cheaper (event-driven)"
        )


def dense_macs_per_inference(layer_sizes: list[int]) -> int:
    """MACs for one forward pass of a fully-connected net.

    Args:
        layer_sizes: neuron counts per layer, input first, e.g. [187, 128, 64, 5].
    """
    return sum(a * b for a, b in zip(layer_sizes[:-1], layer_sizes[1:]))


def snn_sops_per_inference(
    layer_sizes: list[int], timesteps: int, avg_spike_rate: float
) -> int:
    """SOPs for one forward pass of a spiking net.

    Each layer performs (presynaptic spikes) x (fan-out) accumulates per timestep.
    We approximate presynaptic spikes as `avg_spike_rate` x (neurons in that layer).

    Args:
        layer_sizes: neuron counts per layer, input first.
        timesteps: number of simulation steps per inference.
        avg_spike_rate: mean fraction of neurons firing per step (0..1),
            measured empirically from the network's spike activity.
    """
    sops = 0.0
    for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
        active_pre = a * avg_spike_rate
        sops += active_pre * b * timesteps
    return int(sops)


def compare(
    layer_sizes: list[int], timesteps: int, avg_spike_rate: float
) -> EnergyReport:
    """Build an EnergyReport comparing dense vs event-driven for the same net."""
    macs = dense_macs_per_inference(layer_sizes)
    sops = snn_sops_per_inference(layer_sizes, timesteps, avg_spike_rate)
    return EnergyReport(
        dense_macs=macs,
        dense_energy_j=macs * E_MAC_J,
        snn_sops=sops,
        snn_energy_j=sops * E_SOP_J,
    )
