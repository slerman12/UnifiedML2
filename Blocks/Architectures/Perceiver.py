# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import torch

from torch import nn
from torch.nn import init

from Blocks.Architectures.MultiHeadAttention import CrossAttentionBlock


class Perceiver(nn.Module):
    def __init__(self, dim, heads, tokens=32, token_dim=None, v_dim=None, depths=None, recursions=None, fix_token=True):
        super().__init__()

        token_dim = dim if token_dim is None else token_dim
        v_dim = dim if v_dim is None else v_dim

        depths = [3] if depths is None else depths
        recursions = [1 for _ in depths] if recursions is None else recursions

        assert len(depths) == len(recursions), 'Recursion must be specified for each depth'
        assert token_dim == v_dim or recursions[0] == 1, 'First depth cannot be recursive if token_dim ≠ value_dim'

        self.tokens = torch.randn(tokens, token_dim)

        if not fix_token:
            self.tokens = nn.Parameter(self.tokens)
            init.kaiming_uniform_(self.tokens, a=math.sqrt(5))

        self.reattn = nn.ModuleList(sum([[CrossAttentionBlock(token_dim if i == 0 else v_dim,
                                                              heads, dim, v_dim)] * recurs
                                         for i, recurs in enumerate(recursions)], []))
        self.attn = nn.ModuleList(sum([[nn.Sequential(*[CrossAttentionBlock(v_dim, heads)
                                                        for _ in range(inner_depth - 1)])] * recurs
                                       for recurs, inner_depth in zip(recursions, depths)], []))

    def forward(self, x):
        out = self.tokens.to(x.device)
        for reattn, attn in zip(self.reattn, self.attn):
            out = attn(reattn(out, x))
        return out
