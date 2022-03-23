from torch.nn import Conv2d

from Blocks.Architectures.LermanBlocks.BioNeuron.BioCell import BioCell


class BioConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 neurotransmitters=False, action_threshold=False, leaky=False, sample=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.cell = BioCell(out_channels, neurotransmitters, action_threshold, leaky, sample)

    def forward(self, x, prev_membrane= None, prev_spike=None):
        diff = self.forward(x)

        diff = diff.transpose(-1, -3)  # Channels-last

        y, membrane, spike = self.cell(diff,
                                       prev_membrane, prev_spike)

        y = y.transpose(-1, -3)  # Channels-first

        return y, membrane, spike
