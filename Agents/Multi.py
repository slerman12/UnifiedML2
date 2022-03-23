# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
import math

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsembleGaussianActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Fusion import SimpleFusion

from Losses import QLearning, PolicyLearning


class MultiAgent(torch.nn.Module):
    """Multi-Modal Deep Q Network
    Generalized to continuous action spaces, multi-modal classification, and generative modeling"""
    def __init__(self,
                 obs_spec, action_shape, trunk_dim, data_norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 visual=True, auditory=False, proprioceptive=False, temporal=False,  # Modal
                 ):
        super().__init__()

        self.discrete = discrete and not generate  # Continuous supported!
        self.supervise = supervise  # And classification...
        self.RL = RL
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps
        self.ema = ema

        self.action_dim = math.prod(obs_spec.image.shape) if generate else action_shape[-1]

        # Senses

        if generate:
            self.sensory_encoder, repr_shape = Utils.Rand(trunk_dim), (trunk_dim,)
        else:
            vision = CNNEncoder(obs_spec.image.shape, data_norm=data_norm, recipe=recipes.visual, parallel=parallel,
                                lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                ema_decay=ema_decay * ema) if visual else None  # 2D Space

            proprioception = None  # 1D Space

            pulse = None  # Time

            self.sensory_encoder = SimpleFusion(vision, proprioception, pulse)

            repr_shape = self.sensory_encoder.repr_shape

        # Continuous actions
        self.actor = None if self.discrete \
            else EnsembleGaussianActor(repr_shape, trunk_dim, 1024, self.action_dim, recipes.actor, 1,
                                       stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                       lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                       ema_decay=ema_decay * ema)

        self.critic = EnsembleQCritic(repr_shape, trunk_dim, 1024, self.action_dim, recipes.critic,
                                      discrete=self.discrete, ignore_obs=generate, 
                                      lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay, 
                                      ema_decay=ema_decay)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Image augmentation
        self.aug = Utils.init(recipes, 'aug',
                              __default=IntensityAug(0.05) if discrete else RandomShiftsAug(pad=4))

        # Birth

    def act(self, obs):
        obs = Utils.to_torch(obs, self.device)

        # EMA shadows
        sensory_encoder = self.sensory_encoder.ema if self.ema and not self.generate else self.sensory_encoder
        actor = self.actor.ema if self.ema and not self.discrete else self.actor
        critic = self.critic.ema if self.ema else self.critic

        with torch.no_grad(), Utils.act_mode(sensory_encoder, actor, critic):
            # "Sense"

            obs = sensory_encoder(obs)

            # "Interact With World"

            actions = None if self.discrete \
                else actor(obs, self.step).sample(self.num_actions) if self.training \
                else actor(obs, self.step).mean

            # DQN action selector is based on critic
            Pi = self.action_selector(critic(obs, actions), self.step)

            action = Pi.sample() if self.training \
                else Pi.best

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and not self.generate:
                    action = torch.randint(self.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def learn(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *traj, step = Utils.to_torch(
            batch, self.device)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            action, reward[:] = obs.flatten(-3) / 127.5 - 1, 1
            next_obs[:] = label[:] = float('nan')

        # "Envision" / "Perceive"

        # Augment and encode
        obs = self.aug(obs)
        obs = self.sensory_encoder(obs)

        # Augment and encode future
        if replay.nstep > 0 and not self.generate:
            with torch.no_grad():
                next_obs = self.aug(next_obs)
                next_obs = self.sensory_encoder(next_obs)

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
            y_predicted = self.actor(obs[instruction], self.step).mean

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = mistake.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, epoch=replay.epoch, retain_graph=True)

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
                generated_image = self.actor(obs[:half], self.step).mean

                action[:half], reward[:half] = generated_image, 0  # Discriminate

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.critic, epoch=replay.epoch)

        # Update sensory encoder
        if not self.generate:
            Utils.optimize(None,  # Using gradients from previous losses
                           self.sensory_encoder, epoch=replay.epoch)

        if self.generate or self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, self.num_actions, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor, epoch=replay.epoch)

        return logs
