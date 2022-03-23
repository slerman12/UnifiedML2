# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

from Datasets.Suites import DMC, Atari, Classify


class Environment:
    def __init__(self, task_name, frame_stack, action_repeat, episode_max_frames, episode_truncate_resume_frames,
                 seed=0, train=True, suite="DMC", offline=False, generate=False, batch_size=1, num_workers=1):
        self.suite = suite
        self.offline = (offline or generate) and train
        self.generate = generate

        self.env = self.raw_env.make(task_name, frame_stack, action_repeat, episode_max_frames,
                                     episode_truncate_resume_frames, offline, train, seed, batch_size, num_workers)

        self.env.reset()

        self.episode_done = self.episode_step = self.last_episode_len = self.episode_reward = 0
        self.daybreak = None

    @property
    def raw_env(self):
        if self.suite.lower() == "dmc":
            return DMC
        elif self.suite.lower() == "atari":
            return Atari
        elif self.suite.lower() == 'classify':
            return Classify

    def __getattr__(self, item):
        return getattr(self.env, item)

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = []
        video_image = []

        exp = self.exp

        self.episode_done = agent.training and self.offline

        step = 0
        while not self.episode_done and step < steps:
            # Act
            if not self.offline:
                action = agent.act(exp.observation)

                exp = self.env.step(None if self.generate else action.cpu().numpy())

                exp.step = agent.step
                experiences.append(exp)

                if vlog or self.generate:
                    frame = action[:24].view(-1, *exp.observation.shape[1:]) if self.generate \
                        else self.env.physics.render(height=256, width=256, camera_id=0) \
                        if hasattr(self.env, 'physics') else self.env.render()
                    video_image.append(frame)

                # Tally reward, done
                self.episode_reward += exp.reward.mean()
                self.episode_done = exp.last()

            step += 1

        self.episode_step += step

        if agent.training and self.offline:
            agent.step += 1

        if self.episode_done:
            if agent.training:
                agent.episode += 1
            self.env.reset()

            self.last_episode_len = self.episode_step

        # Log stats
        sundown = time.time()
        frames = self.episode_step * self.action_repeat

        logs = {'time': sundown - agent.birthday,
                'step': agent.step,
                'frame': agent.step * self.action_repeat,
                'episode': agent.episode,
                'accuracy'if self.suite == 'classify' else 'reward':
                    self.episode_reward / max(1, self.episode_step * self.suite == 'classify'),
                'fps': frames / (sundown - self.daybreak)} if not self.offline else None

        if self.episode_done:
            self.episode_step = self.episode_reward = 0
            self.daybreak = sundown

        return experiences, logs, video_image
