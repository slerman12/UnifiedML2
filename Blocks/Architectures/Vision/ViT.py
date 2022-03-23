# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from Blocks.Architectures.MultiHeadAttention import SelfAttentionBlock
from Blocks.Architectures.Vision.CNN import AvgPool


class ViT(nn.Module):
    def __init__(self, input_shape, patch_size=4, out_channels=32,
                 emb_dropout=0.1, qk_dim=None, v_dim=None, hidden_dim=None, heads=8, depth=3, dropout=0.1,
                 pool_type='cls', rela=False, output_dim=None):
        super().__init__()

        self.input_shape = input_shape
        in_channels = input_shape[0]
        self.out_channels = out_channels
        image_size = input_shape[1]
        self.patch_size = patch_size
        self.output_dim = output_dim

        assert input_shape[1] == input_shape[2], 'Compatible with square images only'
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        assert pool_type in {'cls', 'mean'}, 'Pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, out_channels),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, out_channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        self.emb_dropout = nn.Dropout(emb_dropout)

        _, self.h, self.w = self.repr_shape(*input_shape)

        self.attn = nn.Sequential(*[SelfAttentionBlock(out_channels, heads, out_channels, qk_dim, v_dim, hidden_dim,
                                                       dropout=dropout, rela=rela) for _ in range(depth)])

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(CLSPool() if pool_type == 'cls' else AvgPool(),
                               nn.Linear(out_channels, output_dim))

    def repr_shape(self, c, h, w):
        return (self.out_channels, 1, (h // self.patch_size) * (w // self.patch_size) + 1) if self.output_dim is None \
            else (self.output_dim, 1, 1)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *self.input_shape[1:]) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if context.shape[-1]
                                                                                  % math.prod(self.input_shape[1:]) == 0
            else context.view(*context.shape, 1, 1).expand(*context.shape, *self.input_shape[1:])
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.emb_dropout(x)

        x = self.attn(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)  # Channels 1st

        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


class CLSPool(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def repr_shape(self, c, h, w):
        return c, 1, 1

    def forward(self, x):
        return x.flatten(-2)[..., 0]

