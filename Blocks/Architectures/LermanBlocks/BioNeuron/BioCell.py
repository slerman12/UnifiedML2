import torch
from torch.distributions import Bernoulli
from torch.nn import Parameter, Module


class BioCell(Module):
    """"Sparse, Recurrent, And Kinda Differentiable: The Biological Neuron"
            - A differentiable leaky-integrate-and-fire neuron
    Language:
    Spike: binary 1 or 0 whether the neuron outputs non-zero
    Membrane: inner memory state whose magnitude determines a spike, a.k.a. "membrane potential"
    """
    def __init__(self, out_features, neurotransmitters=False, action_threshold=False, leaky=False, sample=False):
        super(BioCell, self).__init__()

        # Pre-designated outputs vs. outputting membrane directly
        self.neurotransmitters = Parameter(torch.Tensor(out_features)) if neurotransmitters \
            else None
        # Spike probability "bias" - the "action potential"
        self.action_threshold = Parameter(torch.Tensor(out_features)) if action_threshold \
            else 0  # Default 0, similar to ReLU
        # Proportion membrane decay per time step
        self.leak = Parameter(torch.Tensor(out_features)) if leaky \
            else None

        # Stochastic spiking
        self.sample = sample

    def forward(self, diff, prev_membrane=0, prev_spike=0):
        # Reset spike after firing
        membrane = prev_membrane * (1 - prev_spike)

        # Leakiness
        if self.leak is not None:
            membrane *= torch.sigmoid(self.leak)

        # Integrate inputs
        membrane += diff

        # Fire probability
        spike_proba = torch.sigmoid(membrane + self.action_threshold)

        # Fire
        spike = Bernoulli(probs=spike_proba).sample() if self.sample \
            else torch.round(spike_proba)

        # Differentiability via "reparam trick"
        spike = spike_proba + (spike - spike_proba).detach()

        # Output membrane or pre-designated outputs
        output = membrane * spike if self.neurotransmitters is None \
            else self.neurotransmitters * spike

        # Return output, membrane, and whether it fired
        return output, membrane, spike

