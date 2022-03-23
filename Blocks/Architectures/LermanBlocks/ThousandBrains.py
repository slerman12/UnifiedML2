# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn import functional as F

from torch.optim import Adam


# Given N1 x ... x Nn x V1 x ... x Vm grid,
# B x M1 x ... x Mn x n window,
# B x M1 x ... x Mn source,
# writes source to grid at window
def write(grid, window, src):
    value_shape = grid.shape[len(window.shape) - 2:]
    index = window_to_index(window, grid)
    grid.view(-1, *value_shape).scatter_(0, index.long(), src.view(-1, *value_shape))
    # grid.view(-1, *value_shape)[index.long()] = src.view(-1, *value_shape)


# Given N1 x ... x Nn x V1 x ... x Vm grid,
# B x M1 x ... x Mn x n window,
# returns B x M1 x ... x Mn x V1 x ... x Vm memory
def lookup(grid, window):
    value_shape = grid.shape[len(window.shape) - 2:]

    index = window_to_index(window, grid)

    out = grid.view(-1, *value_shape).gather(0, index.long())
    # out = grid.view(-1, *value_shape)[index]

    return out.view(*window.shape[:-1], *value_shape)


# Given B x M1 x ... x Mn x n window,
# return B x M1 * ... * Mn index
def window_to_index(window, grid):
    axes = window.shape[-1]
    grid_size = grid.shape[0]
    value_shape = grid.shape[len(window.shape) - 2:]

    index_lead = torch.matmul(window, grid_size ** reversed(torch.arange(axes, dtype=window.dtype)))
    index = index_lead.view(-1, *[1 for _ in value_shape]).expand(-1, *value_shape)

    return index


# Given B x n coord,
# returns B x M1 x ... x Mn x n window
def coord_to_window(coord, window_size):
    axes = coord.shape[1]
    range = torch.arange(window_size) - window_size // 2
    mesh = torch.stack(torch.meshgrid(*([range] * axes)), -1)
    window = coord.view(-1, *([1] * axes), axes).expand(-1, *([window_size] * axes), -1) + mesh
    return window


def northity(input, radius=0.5, axis_dim=None, one_hot=False):
    if axis_dim is None:
        axis_dim = input.shape[:-1]

    needle = input.view(*input.shape[:-1], -1, axis_dim)

    north = F.one_hot(torch.tensor(axis_dim - 1), axis_dim) if one_hot \
        else torch.ones(axis_dim)

    return F.cosine_similarity(needle, north, -1) * radius


class Northity(torch.nn.Module):
    def __init__(self, radius=0.5, axis_dim=None, one_hot=False):
        super().__init__()

        self.radius = radius
        self.axis_dim = axis_dim
        self.one_hot = one_hot

    def forward(self, input):
        return northity(input, self.radius, self.axis_dim, self.one_hot)


def compass_wheel_localize(input, num_degrees=10, axis_dim=None, one_hot=False):
    nrt = northity(input, num_degrees / 2, axis_dim, one_hot) + num_degrees / 2
    quant = torch.round(nrt)
    return nrt - (nrt - quant).detach()


class CompassWheelLocalize(torch.nn.Module):
    def __init__(self, num_degrees=10, axis_dim=None, one_hot=False):
        super().__init__()

        self.num_degrees = num_degrees
        self.axis_dim = axis_dim
        self.one_hot = one_hot

    def forward(self, input):
        return compass_wheel_localize(input, self.num_degrees, self.axis_dim, self.one_hot)


axes = 2
compass_axis_dim = 2
north_one_hot = False
grid_size = 3
value_shape = torch.Size([])
window_size = 3
batch_dim = 3

G = torch.randn(*([grid_size] * axes), *value_shape)  # (N x ...) n times x V1 x ...
print("grid:")
print(G)
# Sample, and to make more stable temporally, can traverse as velocity
P = torch.rand((batch_dim, compass_axis_dim * axes)) - 0.5  # B x 2 * n
print("pos:")
print(P)
P = compass_wheel_localize(P, grid_size - 1, compass_axis_dim, north_one_hot)
print("quant:")
print(P.shape)
print(P)
W = coord_to_window(P, window_size) % grid_size  # B x (M x ...) n times x n

param = torch.nn.Parameter(torch.randn((batch_dim, *([window_size] * axes), *value_shape)))
optim = Adam([param], lr=0.8)

t = time.time()
mem = lookup(G, W)  # B x (M x ...) n times x V1 x ...
print('read time', time.time() - t)

print('mem:')
print(mem.shape)
print(mem)

# Is this too expensive compared to just using mem.requires_grad = True ?
param.data.copy_(mem.data)

loss = param.mean()

t - time.time()
loss.backward()
optim.step()
print('back time', time.time() - t)

clone_G = G.clone()

t - time.time()
write(G, W, param)
print('write time', time.time() - t)

# assert (G == clone_G).all()

# Scatter/indexing slows with tensor size?
for size in [100, 100000, 100000000]:
    src = torch.arange(100)[None, :]
    index = torch.ones_like(src)
    x = torch.zeros(size, dtype=src.dtype)
    t = time.time()
    x.view(-1, 100).scatter_(1, index, src)
    print('time', time.time() - t)
