# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Blocks.Architectures import ViT
from Blocks.Architectures.Perceiver import Perceiver


class ViPer(ViT):
    """Vision Perceiver, a patch-based adaptation of Perceiver for images"""
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, tokens=100,
                 token_dim=32, depth=3, pool_type='cls', output_dim=None):
        self.tokens = tokens

        super().__init__(input_shape, patch_size, out_channels, heads, depth, pool_type, True, output_dim)

        self.P = Perceiver(out_channels, heads, tokens, token_dim, depth=depth, relu=True)
        self.attn = self.P

    def repr_shape(self, c, h, w):
        return self.out_channels, self.tokens, 1