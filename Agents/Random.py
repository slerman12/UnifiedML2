# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import time

import torch

import Utils


class RandomAgent(torch.nn.Module):
    """Random Agent"""
    def __init__(self,
                 obs_shape, action_shape, trunk_dim, hidden_dim, recipes,  # Architecture
                 lr, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, log  # On-boarding
                 ):
        super().__init__()

        self.device = device
        self.birthday = time.time()
        self.step = self.episode = 0

        action_dim = math.prod(obs_shape) if generate else action_shape[-1]

        self.actor = Utils.Rand(action_dim, uniform=True)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.actor):
            action = self.actor(obs) * 2 - 1  # [-1, 1]

            if self.discrete:
                action = torch.argmax(action, -1)

            if self.training:
                self.step += 1

            return action

    # "Dream"
    def learn(self, replay=None):
        return
