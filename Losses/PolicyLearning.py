# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils


def deepPolicyGradient(actor, critic, obs, step, num_actions=1, reward=0, discount=1,
                       one_hot=False, priority_temp=0, logs=None):
    Pi = actor(obs, step)

    actions = Pi.rsample(num_actions)
    if actor.discrete and one_hot:
        actions = Utils.rone_hot(actions)

    Q = critic(obs, actions)

    q = torch.min(Q.Qs, 0)[0]

    q = reward + q * discount

    # Re-prioritize based on certainty e.g., https://arxiv.org/pdf/2007.04938.pdf
    q *= torch.sigmoid(-Q.stddev * priority_temp) + 0.5

    policy_loss = -q.mean()

    if logs is not None:
        assert isinstance(logs, dict)
        logs['policy_loss'] = policy_loss.item()
        logs['DPG_q_stddev'] = Q.stddev.mean().item()
        logs['Pi_prob'] = Pi.log_prob(actions).exp().mean().item()
        logs['DPG_q_mean'] = q.mean().item()

    return policy_loss
