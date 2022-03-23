from torch import nn

from Blocks.Architectures.LermanBlocks.BioNeuron.BioCell import BioCell


class BioLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 neurotransmitters=False, action_threshold=False, leaky=False, sample=False):
        super().__init__(in_features, out_features, bias)

        self.cell = BioCell(out_features, neurotransmitters, action_threshold, leaky, sample)

    def forward(self, x, prev_membrane=None, prev_spike=None):
        diff = self.forward(x)

        return self.cell(diff,
                         prev_membrane, prev_spike)
