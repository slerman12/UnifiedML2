import math

import torch
from torch import nn
from torch.nn import functional as F

import Utils


class CNN(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, batch_norm=False, last_relu=True,
                 kernel_size=3, stride=2, padding=0, output_dim=None):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = [input_shape]

        self.input_shape = torch.Size(input_shape)
        in_channels = input_shape[0]

        self.trunk = nn.Identity()

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, kernel_size, stride=stride if i == 0 else 1,
                                      padding=padding),
                            nn.BatchNorm2d(self.out_channels) if batch_norm else nn.Identity(),
                            nn.ReLU() if i < depth or last_relu else nn.Identity()) for i in range(depth + 1)],
        )

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(AvgPool(), nn.Linear(out_channels, output_dim))

        self.apply(Utils.weight_init)

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.trunk, self.CNN, self.project)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        hw = self.input_shape[1:] if len(self.input_shape) == 3 else x[0].shape[-2:]
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *hw) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *hw) if context.shape[-1] % math.prod(hw) == 0
             else context.view(*context.shape, 1, 1).expand(*context.shape, *hw)
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.trunk(x)
        x = self.CNN(x)
        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


class AvgPool(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def repr_shape(self, c, h, w):
        return c, 1, 1

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(-3)
