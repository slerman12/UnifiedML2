# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Agents import DQNAgent

from Losses import QLearning, PolicyLearning


class MetaDQNAgent(DQNAgent):
    """Meta Deep Q Network
    Generalized to continuous action spaces, classification, and generative modeling
    Passes labels into actor during classification learning, but not acting/evaluation"""
    def __init__(self, *_, discrete=False,
                 meta=False):  # MetaDQN
        super().__init__(*_, discrete=discrete)

        self.meta = meta

        class IgnoreLabel(torch.nn.Module):
            def __init__(self, trunk):
                super().__init__()
                self.trunk = trunk

            def forward(self, obs):
                return super().forward(obs[0] if isinstance(obs, tuple) else obs)

        if not discrete:
            self.actor.trunk = IgnoreLabel(self.actor.trunk)

        # Birth

    # "Dream"
    def learn(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *traj, step = Utils.to_torch(
            batch, self.device)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            action = obs.flatten(-3) / 127.5 - 1
            reward[:] = 1
            next_obs[:] = label[:] = float('nan')

        # "Envision" / "Perceive"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # "Journal teachings"

        logs = {'time': time.time() - self.birthday,
                'step': self.step, 'episode': self.episode} if self.log \
            else None

        instruction = ~torch.isnan(label)

        # "Acquire Wisdom"

        # Classification
        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # Inference
            x = (obs[instruction], label[instruction]) if self.meta \
                else obs[instruction]

            y_predicted = self.actor(x, self.step).mean[:, 0]

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = mistake.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, retain_graph=True)

                if self.log:
                    correct = (torch.argmax(y_predicted, -1) == label[instruction]).float()

                    logs.update({'supervised_loss': supervised_loss.item(),
                                 'accuracy': correct.mean().item()})

            # (Auxiliary) reinforcement
            if self.RL:
                half = len(instruction) // 2
                mistake[:half] = cross_entropy(y_predicted[:half].uniform_(-1, 1),
                                               label[instruction][:half].long(), reduction='none')
                action[instruction] = y_predicted.detach()
                reward[instruction] = -mistake[:, None].detach()  # reward = -error
                next_obs[instruction] = float('nan')

        # Reinforcement learning / generative modeling
        if self.RL or self.generate:
            # "Imagine"

            # Generative modeling
            if self.generate:
                half = len(obs) // 2
                generated_image = self.actor(obs[:half], self.step).mean[:, 0]

                action[:half], reward[:half] = generated_image, 0  # Discriminate
                next_obs[:] = float('nan')  # Can delete for Cuda > 11

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.critic)

        # Update encoder
        if not self.generate:
            Utils.optimize(None,  # Using gradients from previous losses
                           self.encoder)

        if self.generate or self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, self.num_actions, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor)

        return logs
