# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import Utils


def ensembleQLearning(critic, actor, obs, action, reward, discount, next_obs, step,
                      num_actions=1, priority_temp=0, logs=None):
    # Non-NaN next_obs
    has_future = ~torch.isnan(next_obs.flatten(1).sum(1))
    next_obs = next_obs[has_future]

    # Compute Bellman target
    with torch.no_grad():
        # Get actions for next_obs

        if critic.discrete:
            # All actions
            next_actions_log_probs = 0
            next_actions = None
        else:
            if actor.discrete:
                # One-hots
                action = Utils.one_hot(action, critic.action_dim)
                next_actions = torch.eye(critic.action_dim, device=obs.device).expand(next_obs.shape[0], -1, -1)
                next_actions_log_probs = 0
            else:
                if has_future.any():
                    # Sample actions
                    next_Pi = actor(next_obs, step)
                    next_actions = next_Pi.rsample(num_actions)
                    next_actions_log_probs = next_Pi.log_prob(next_actions).sum(-1, keepdim=True).flatten(1)

        target_q = reward

        if has_future.any():
            next_Q = critic.ema(next_obs, next_actions)

            next_q = torch.min(next_Q.Qs, 0)[0]

            next_v = torch.zeros_like(discount)
            next_q_logits = next_q - next_q.max(-1, keepdim=True)[0]
            next_probs = torch.softmax(next_q_logits + next_actions_log_probs, -1)
            next_v[has_future] = torch.sum(next_q * next_probs, -1, keepdim=True)

            target_q += discount * next_v

    Q = critic(obs, action)

    # Temporal difference (TD) error (via MSE, but could also use Huber)
    td_error = F.mse_loss(Q.Qs, target_q.expand_as(Q.Qs), reduction='none')

    # Re-prioritize based on certainty e.g., https://arxiv.org/pdf/2007.04938.pdf
    re_prioritized_td_error = td_error * torch.sigmoid(-Q.stddev * priority_temp) + 0.5

    q_loss = re_prioritized_td_error.mean()

    if logs is not None:
        assert isinstance(logs, dict)
        logs['q_mean'] = Q.mean.mean().item()
        logs['q_stddev'] = Q.stddev.mean().item()
        logs.update({f'q{i}': q.median().item() for i, q in enumerate(Q.Qs)})
        logs['target_q'] = target_q.mean().item()
        logs['temporal_difference_error'] = td_error.mean().item()
        logs['q_loss'] = q_loss.item()

    return q_loss
