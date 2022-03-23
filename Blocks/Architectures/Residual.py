# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn

import Utils


class Residual(nn.Module):
    def __init__(self, model, down_sample=None):
        super().__init__()
        self.model = model
        self.down_sample = down_sample

    def forward(self, x):
        y = self.model(x)
        if self.down_sample is not None:
            x = self.down_sample(x)
        return y + x

    def repr_shape(self, channels, height, width):
        return Utils.cnn_feature_shape(channels, height, width, self.model)


