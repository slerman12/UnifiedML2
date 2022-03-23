# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from Blocks.Architectures import MLP
from Blocks.Architectures.Vision.CNN import CNN

import Utils


class RN(nn.Module):
    """Relation Network https://arxiv.org/abs/1706.01427"""
    def __init__(self, dim, context_dim=None, inner_depth=3, outer_depth=2, hidden_dim=None,
                 output_dim=None, input_shape=None, mid_nonlinearity=nn.Identity(), dropout=0):
        super().__init__()

        if input_shape is not None:
            dim = input_shape[-3]

        if context_dim is None:
            context_dim = dim

        if hidden_dim is None:
            hidden_dim = dim * 4

        self.output_dim = dim if output_dim is None \
            else output_dim

        self.inner = nn.Sequential(MLP(dim + context_dim, hidden_dim, hidden_dim, inner_depth), nn.Dropout(dropout))
        # self.inner = nn.Sequential(Utils.ChSwap, CNN(dim + context_dim, hidden_dim, inner_depth,
        #                                              kernel_size=1, stride=1, last_relu=False), Utils.ChSwap,
        #                            nn.Dropout(dropout))
        self.mid_nonlinearity = mid_nonlinearity
        self.outer = MLP(hidden_dim, self.output_dim, hidden_dim, outer_depth)

    def repr_shape(self, c, h, w):
        return self.output_dim, 1, 1

    def forward(self, x, context=None):
        x = x.flatten(1, -2)

        if context is None:
            context = x

        context = context.flatten(1, -2)

        x = x.unsqueeze(1).expand(-1, context.shape[1], -1, -1)
        context = context.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        pair = torch.cat([x, context], -1)

        relations = self.inner(pair)

        mid = self.mid_nonlinearity(relations.sum(1).sum(1))

        out = self.outer(mid)

        return out
