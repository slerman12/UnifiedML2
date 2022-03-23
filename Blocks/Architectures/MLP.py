# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils


class MLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=1024, hidden_dim=512, depth=1, non_linearity=nn.ReLU(inplace=True),
                 dropout=0, binary=False, l2_norm=False, input_shape=None):
        super().__init__()

        if input_shape is not None:
            input_dim = math.prod(input_shape)
        self.output_dim = output_dim

        self.MLP = nn.Sequential(*[nn.Sequential(
            # Optional L2 norm of penultimate
            # (See: https://openreview.net/pdf?id=9xhgmsNVHu)
            # Similarly, Efficient-Zero initializes 2nd-to-last layer as all 0s  TODO
            Utils.L2Norm() if l2_norm and i == depth else nn.Identity(),
            nn.Linear(input_dim if i == 0 else hidden_dim,
                      hidden_dim if i < depth else output_dim),
            non_linearity if i < depth else nn.Sigmoid() if binary else nn.Identity(),
            nn.Dropout(dropout) if i < depth else nn.Identity()
        )
            for i in range(depth + 1)]
        )

        self.apply(Utils.weight_init)

    def repr_shape(self, *_):
        return self.output_dim, *_[1:]

    def forward(self, *x):
        return self.MLP(torch.cat(x, -1))
