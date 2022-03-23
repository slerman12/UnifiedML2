# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
from functools import partial

from einops import rearrange
from opt_einsum_torch import EinsumPlanner

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from Blocks.Architectures.MLP import MLP

import Utils


class CrossAttention(nn.Module):
    def __init__(self, dim=32, heads=None, s_dim=None, qk_dim=None, v_dim=None, talk_h=False, rela=False):
        super().__init__()

        self.dim = dim

        s_dim = dim if s_dim is None else s_dim
        qk_dim = dim if qk_dim is None else qk_dim
        v_dim = dim if v_dim is None else v_dim

        heads = math.gcd(8, v_dim) if heads is None \
            else heads

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.heads = heads

        assert v_dim % heads == 0, f'value dim={dim} is not divisible by heads={heads}'

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_kv = nn.Linear(s_dim, qk_dim + v_dim, bias=False)

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChSwap) if talk_h else nn.Identity()

        self.relu = nn.ReLU(inplace=True) if rela else None

    def forward(self, x, s=None):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim, f'input dim ≠ pre-specified {shape[-1]}≠{self.dim}'

        if s is None:
            s = x

        tokens = len(x.shape) == 2  # Tokens distinguished by having axes=2
        if not tokens:
            x = x.flatten(1, -2)
        s = s.flatten(1, -2)

        q = x if tokens else self.to_q(x)
        k, v = self.to_kv(s).tensor_split([self.qk_dim], dim=-1)

        multi_head_tokens = q.shape[-1] == k.shape[-1] and tokens

        assert q.shape[-1] == k.shape[-1] / self.heads or not tokens, \
            f'Tokens, keys cannot be broadcast {q.shape[-1]}, {k.shape[-1]}'

        if multi_head_tokens or not tokens:
            pattern = 'n (h d) -> h n d' if tokens \
                else 'b n (h d) -> b h n d'
            q = rearrange(q, pattern, h=self.heads)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (k, v))

        # Memory limit toggle, e.g., =0.5
        mem_limit = False
        einsum = EinsumPlanner(q.device, cuda_mem_limit=mem_limit).einsum if 0 < mem_limit < 1 \
            else torch.einsum

        scale = q.shape[-1] ** -0.5
        q = q * scale

        pattern = 'h i d, b h j d -> b h i j' if multi_head_tokens \
            else 'i d, b h j d -> b h i j' if tokens \
            else 'b h i d, b h j d -> b h i j'

        # Memory efficient toggle
        mem_efficient = False
        if mem_efficient:
            attn, weights = mem_efficient_attend(q, k, v, pattern=pattern)
        else:
            self.weights = einsum(pattern, q, k)
            # self.dots = self.dots - self.dots.amax(dim=-1, keepdim=True).detach()

            weights = self.weights.softmax(dim=-1) if self.relu is None \
                else self.relu(self.weights)

            if 0 < mem_limit < 1:
                weights = weights.to(q.device)

            # "Talking heads"
            weights = self.talk_h(weights)

            # attn = torch.einsum('b h i j, b h j d -> b h i d', weights, v)
            attn = torch.matmul(weights, v)

        out = rearrange(attn, 'b h n d -> b n (h d)')

        # Restores original leading dims
        if not tokens:
            out = out.view(*shape[:-1], -1)

        if 0 < mem_limit < 1:
            out = out.to(q.device)

        return out


class ReLA(CrossAttention):
    """ReLA: Rectified linear attention"""
    def __init__(self, dim=32, heads=None, s_dim=None, qk_dim=None, v_dim=None):
        super().__init__(dim, heads, s_dim, qk_dim, v_dim, False, True)


# Memory-efficient attention https://arxiv.org/abs/2112.05682
# https://github.com/lucidrains/memory-efficient-attention-pytorch
def mem_efficient_attend(q, k, v, q_bucket_size=512, k_bucket_size=1024, eps=1e-8,
                         pattern='b h i d, b h j d -> b h i j'):
    def chunk(q, k, v):
        weight = torch.einsum(pattern, q, k)

        weight_max = weight.amax(dim=-1, keepdim=True).detach()
        weight = weight - weight_max

        exp_weight = weight.exp()
        weighted_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, v)

        return exp_weight.sum(dim=-1), weighted_value, rearrange(weight_max, '... 1 -> ...')

    chunk = partial(checkpoint, chunk)

    # Chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)

    # Loop through all chunks and accumulate

    out = []
    weights = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = \
                chunk(q_chunk, k_chunk, v_chunk)

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim=-1)

        weighted_values = torch.stack(weighted_values, dim=-1)
        exp_weights = torch.stack(exp_weights, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim=-1)
        all_weights = exp_weights.sum(dim=-1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)
        weights.append(exp_weights)

    out = torch.cat(out, dim=-2)
    weights = torch.cat(weights, dim=-3)

    return out, weights


# A minimalist implementation using only Pytorch natives
class CrossAttend(nn.Module):
    def __init__(self, dim=32, heads=8, s_dim=None, v_dim=None, *_):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, kdim=s_dim, vdim=v_dim, batch_first=True)

    def forward(self, x, s):
        # Conserves shape
        mid_shape = x.shape[1:-1]

        x = x.flatten(1, -2)
        s = s.flatten(1, -2)

        attn, self.weights = self.attn(x, s, s)

        # Restores original shape
        return attn.view(-1, *mid_shape, attn.shape[-1])


class SelfAttention(CrossAttention):
    def forward(self, x, *_):
        return super().forward(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=32, heads=1, s_dim=None, qk_dim=None, v_dim=None, hidden_dim=None, dropout=0,
                 talk_h=False, rela=False):
        super().__init__()

        v_dim = dim if v_dim is None else v_dim
        hidden_dim = v_dim * 4 if hidden_dim is None else hidden_dim

        self.heads = math.gcd(8, v_dim) if heads is None else heads

        self.v_dim = v_dim

        self.attn = CrossAttention(dim, self.heads, s_dim, qk_dim, v_dim, talk_h, rela)
        self.LN_ReLA = nn.LayerNorm(v_dim) if rela \
            else nn.Identity()
        self.project = nn.Identity() if heads == 1 \
            else nn.Sequential(nn.Linear(v_dim, dim), nn.Dropout(dropout))
        self.mlp = nn.Sequential(MLP(dim, dim, hidden_dim, 1, nn.GELU(), dropout), nn.Dropout(dropout))

        self.LN_pre = nn.LayerNorm(dim)
        self.LN_mid = nn.LayerNorm(dim)

    def repr_shape(self, c, h, w):
        return self.v_dim, h, w  # Assumes channels last

    def forward(self, x, context=None):
        pre_norm = self.LN_pre(x)

        if context is None:
            context = pre_norm

        attn = self.project(self.LN_ReLA(self.attn(pre_norm, context))) + x
        out = self.mlp(self.LN_mid(attn)) + attn

        return out


class SelfAttentionBlock(CrossAttentionBlock):
    def forward(self, x, *_):
        return super().forward(x)


class AttentionPool(nn.Module):
    def __init__(self, channels_in=32, heads=None, output_dim=None, depth=1, recursions=0, input_shape=None):
        super().__init__()

        self.input_shape = input_shape

        if input_shape is not None:
            channels_in = input_shape[-3]

        if output_dim is None:
            output_dim = channels_in

        if heads is None:
            heads = math.gcd(output_dim, 8)  # Approx 8

        self.pool = nn.Sequential(Utils.ChSwap,
                                  # Alternatively could also recurse
                                  *([SelfAttentionBlock(channels_in, heads)] * recursions),
                                  # "Transformer"
                                  *[SelfAttentionBlock(dim=channels_in if i == 0 else output_dim, heads=heads,
                                                       v_dim=output_dim) for i in range(depth)],
                                  Utils.ChSwap,
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten(-3))

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.pool)

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

        x = self.pool(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
