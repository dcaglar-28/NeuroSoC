"""Models: a spiking classifier (snnTorch) and a conventional baseline.

Both consume a 1-D ECG window and output class logits. The SNN runs over
`timesteps` simulation steps using delta-encoded input spikes and leaky
integrate-and-fire neurons trained with surrogate gradients.

torch/snntorch are imported lazily so the numpy-only parts of the package
(encoding, energy, datasets) import even before the ML stack is installed.
"""

from __future__ import annotations


def _imports():
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate
    return torch, nn, snn, surrogate


def build_snn(window: int = 187, hidden: int = 128, n_classes: int = 2,
              timesteps: int = 50, beta: float = 0.9):
    """Return an instantiated spiking MLP classifier.

    Architecture: (2-channel delta-encoded input, flattened) -> LIF -> LIF -> readout.
    `beta` is the membrane decay; `timesteps` the number of steps we integrate over.
    """
    torch, nn, snn, surrogate = _imports()
    spike_grad = surrogate.fast_sigmoid()

    class SpikingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.layer_sizes = [2 * window, hidden, hidden, n_classes]
            self.fc1 = nn.Linear(2 * window, hidden)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc2 = nn.Linear(hidden, hidden)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc3 = nn.Linear(hidden, n_classes)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

        def forward(self, x):
            """x: (batch, 2, window) input spikes, replayed each timestep.

            Returns (logits, mean_spike_rate) where logits are accumulated output
            membrane potentials and mean_spike_rate is the average hidden-layer
            firing fraction (used by the energy model).
            """
            b = x.shape[0]
            x = x.reshape(b, -1)
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            out_sum = 0.0
            spikes_total = 0.0
            spikes_possible = 0.0
            for _ in range(self.timesteps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                cur3 = self.fc3(spk2)
                _spk3, mem3 = self.lif3(cur3, mem3)
                out_sum = out_sum + mem3
                spikes_total = spikes_total + spk1.sum() + spk2.sum()
                spikes_possible = spikes_possible + spk1.numel() + spk2.numel()
            # Differentiable mean firing rate (fraction of hidden neurons firing
            # per timestep). Returned as a tensor so it can be used as a
            # sparsity regularizer during training; call .item() for reporting.
            spk_rate = spikes_total / spikes_possible
            return out_sum / self.timesteps, spk_rate

    model = SpikingMLP()
    model.timesteps = timesteps
    return model


def build_baseline(window: int = 187, hidden: int = 128, n_classes: int = 2):
    """Conventional (non-spiking) MLP baseline with the same layer sizes."""
    torch, nn, _snn, _sur = _imports()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_sizes = [window, hidden, hidden, n_classes]
            self.net = nn.Sequential(
                nn.Linear(window, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    return MLP()
