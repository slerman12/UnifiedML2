# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
import random
import glob
import shutil
import atexit
import warnings
from pathlib import Path
import datetime
import io
import traceback

from omegaconf import OmegaConf

import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset

from torchvision.transforms import transforms


class ExperienceReplay:
    def __init__(self, batch_size, num_workers, capacity, action_spec, suite, task, offline, generate, save, load, path,
                 obs_spec=None, steps_per_epoch=None, nstep=0, discount=1, transform=None):
        # Path and loading

        path = path.replace("Agents.", "")
        exists = glob.glob(path + '*/')

        offline = offline or generate

        if load or offline:
            if suite == 'classify':
                standard = f'./Datasets/ReplayBuffer/Classify/{task}_Buffer'
                if len(exists) == 0:
                    exists = [standard + '/']
                    print('All data loaded. Training of classifier underway.')
                else:
                    if path != standard:
                        warnings.warn(f'Loading a saved replay of a classify task from a previous online session,'
                                      f'which may not be intentional. '
                                      f'For the standard offline dataset, set replay.path="{standard}" '  # If exists
                                      f'or delete the saved buffer in {path}.')
            assert len(exists) > 0, f'No existing replay buffer found in path: {path}'
            self.path = Path(sorted(exists)[-1])
            save = offline or save
        else:
            self.path = Path(path + '_' + str(datetime.datetime.now()))
            self.path.mkdir(exist_ok=True, parents=True)

        if not save:
            # Delete replay on terminate
            atexit.register(lambda p: (shutil.rmtree(p), print('Deleting replay')), self.path)

        # Data specs

        if obs_spec is None:
            obs_spec = {'name': 'obs', 'shape': (1,), 'dtype': 'float32'},

        self.specs = (obs_spec, action_spec,
                      {'name': 'reward', 'shape': (1,), 'dtype': 'float32'},
                      {'name': 'discount', 'shape': (1,), 'dtype': 'float32'},
                      {'name': 'label', 'shape': (1,), 'dtype': 'float32'},
                      {'name': 'step', 'shape': (1,), 'dtype': 'float32'},)

        # Episode traces

        self.episode = {spec['name']: [] for spec in self.specs}
        self.episode_len = 0
        self.episodes_stored = len(list(self.path.glob('*.npz')))
        self.save = save

        # Data transform

        if transform is not None:
            if isinstance(transform, str):
                transform = OmegaConf.create(transform)
            if 'RandomCrop' in transform and 'size' not in transform['RandomCrop']:
                transform['RandomCrop']['size'] = obs_spec['shape'][-2:]
            # Can pass in a dict of torchvision transform names and args
            transform = transforms.Compose([getattr(transforms, t)(**transform[t]) for t in transform])

        # Parallelized experience loading

        self.nstep = nstep

        self.experiences = Experiences(offline)(path=self.path,
                                                capacity=np.inf if save else capacity // max(1, num_workers),
                                                num_workers=min(num_workers, os.cpu_count()),
                                                fetch_per=1000,
                                                save=save,
                                                nstep=nstep,
                                                discount=discount,
                                                transform=transform)

        # Batch loading

        self.epoch = self.step = 0
        self.steps_per_epoch = steps_per_epoch

        self.batches = torch.utils.data.DataLoader(dataset=self.experiences,
                                                   batch_size=batch_size,
                                                   shuffle=offline,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn)
        # Replay

        self._replay = None

    # Returns a batch of experiences
    def sample(self):
        return next(self)  # Can iterate via next

    # Allows iteration
    def __next__(self):
        self.step += 1
        if self.steps_per_epoch and self.step % self.steps_per_epoch == 0:
            self.epoch += 1
        try:
            return next(self.replay)
        except StopIteration:
            if self.steps_per_epoch is None:
                self.epoch += 1
            self._replay = None
            return next(self)

    # Allows iteration
    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)
        return self._replay

    # Tracks single episode in memory buffer
    def add(self, experiences=None, store=False):
        if experiences is None:
            experiences = []

        # An "episode" of experiences
        assert isinstance(experiences, (list, tuple))

        for exp in experiences:
            for spec in self.specs:
                # Make sure everything is a numpy batch
                if np.isscalar(exp[spec['name']]) or exp[spec['name']] is None:
                    exp[spec['name']] = np.full((1,) + tuple(spec['shape']), exp[spec['name']], spec['dtype'])
                if len(exp[spec['name']].shape) == len(spec['shape']):
                    exp[spec['name']] = np.expand_dims(exp[spec['name']], 0)

                # Validate consistency
                assert spec['shape'] == exp[spec['name']].shape[1:], \
                    f'Unexpected {spec["name"]} shape: {exp[spec["name"]].shape} vs {spec["shape"]}'
                assert spec['dtype'] == exp[spec['name']].dtype.name, \
                    f'Unexpected {spec["name"]} dtype: {exp[spec["name"]].dtype.name} vs. {spec["dtype"]}'

                # Adds the experiences
                self.episode[spec['name']].append(exp[spec['name']])

        self.episode_len += len(experiences)

        if store:
            self.store_episode()  # Stores them in file system

    # Stores episode (to file in system)
    def store_episode(self):
        if self.episode_len == 0:
            return

        for spec in self.specs:
            # Concatenate into one big episode batch
            self.episode[spec['name']] = np.concatenate(self.episode[spec['name']], axis=0)

        self.episode_len = self.episode['observation'].shape[0]

        # Expands 'step' since it has no batch length in classification
        if self.episode['step'].shape[0] == 1:
            self.episode['step'] = np.repeat(self.episode['step'], self.episode_len, axis=0)

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        num_episodes = len(self)
        episode_name = f'{timestamp}_{num_episodes}_{self.episode_len}.npz'

        # Save episode
        save_path = self.path / episode_name
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **self.episode)
            buffer.seek(0)
            with save_path.open('wb') as f:
                f.write(buffer.read())

        self.episode = {spec['name']: [] for spec in self.specs}
        self.episode_len = 0
        self.episodes_stored += 1

    def __len__(self):
        return self.episodes_stored


# How to initialize each worker
def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


# Multi-cpu workers iteratively and efficiently build batches of experience in parallel (from files)
def Experiences(offline):
    class _Experiences(Dataset if offline else IterableDataset):
        def __init__(self, path, capacity, num_workers, fetch_per, save=False, nstep=0, discount=1, transform=None):

            # Dataset construction via parallel workers

            self.path = path

            self.episode_names = []
            self.episodes = dict()

            self.num_experiences_loaded = 0
            self.capacity = capacity
            self.index = []

            self.num_workers = max(1, num_workers)

            self.fetch_per = fetch_per
            self.samples_since_last_fetch = fetch_per

            self.save = save

            # Load in existing data
            list(map(self.load_episode, sorted(path.glob('*.npz'))))

            self.nstep = nstep
            self.discount = discount

            self.transform = transform

        def load_episode(self, episode_name):
            try:
                with episode_name.open('rb') as episode_file:
                    episode = np.load(episode_file)
                    episode = {key: episode[key] for key in episode.keys()}
            except:
                return False

            episode_len = next(iter(episode.values())).shape[0] - 1

            while episode_len + self.num_experiences_loaded > self.capacity:
                early_episode_name = self.episode_names.pop(0)
                early_episode = self.episodes.pop(early_episode_name)
                early_episode_len = next(iter(early_episode.values())).shape[0] - 1
                self.num_experiences_loaded -= early_episode_len
                if offline:
                    self.index = self.index[early_episode_len:]
                else:
                    # Deletes early episode file
                    early_episode_name.unlink(missing_ok=True)
            self.episode_names.append(episode_name)
            self.episode_names.sort()
            self.episodes[episode_name] = episode
            self.num_experiences_loaded += episode_len
            if offline:
                self.index += list(enumerate([episode_name] * episode_len))

            if not self.save:
                episode_name.unlink(missing_ok=True)  # Deletes file

            return True

        # Populates workers with up-to-date data
        def worker_fetch_episodes(self):
            if self.samples_since_last_fetch < self.fetch_per:
                return

            self.samples_since_last_fetch = 0

            try:
                worker = torch.utils.data.get_worker_info().id
            except:
                worker = 0

            episode_names = sorted(self.path.glob('*.npz'), reverse=True)  # Episodes
            num_fetched = 0
            # Find new episodes
            for episode_name in episode_names:
                episode_idx, episode_len = [int(x) for x in episode_name.stem.split('_')[1:]]
                if episode_idx % self.num_workers != worker:  # Each worker stores their own dedicated data
                    continue
                if episode_name in self.episodes.keys():  # Don't store redundantly
                    break
                # if num_fetched + episode_len > self.capacity:  # Don't overfill  (This is already accounted for)
                #     break
                num_fetched += episode_len
                if not self.load_episode(episode_name):
                    break  # Resolve conflicts

        def sample(self, episode_names, metrics=None):
            episode_name = random.choice(episode_names)  # Uniform sampling of experiences
            return episode_name

        # N-step cumulative discounted rewards
        def process(self, episode, idx=None):
            episode_len = len(episode['observation'])
            limit = episode_len - (self.nstep or 1)
            idx = np.random.randint(limit) if idx is None else idx % limit

            # Transition
            obs = episode['observation'][idx]
            action = episode['action'][idx + 1]
            next_obs = episode['observation'][idx + self.nstep]
            reward = np.full_like(episode['reward'][idx + 1], np.NaN)
            discount = np.ones_like(episode['discount'][idx + 1])
            label = episode['label'][idx].squeeze()
            step = episode['step'][idx]

            # Trajectory
            if self.nstep > 0:
                traj_o = episode['observation'][idx:idx + self.nstep + 1]
                traj_a = episode['action'][idx + 1:idx + self.nstep + 1]  # 1 len smaller than traj_o
                traj_r = episode['reward'][idx + 1:idx + self.nstep + 1]  # 1 len smaller than traj_o
                traj_l = episode['label'][idx:idx + self.nstep + 1]
            else:
                traj_o = traj_a = traj_r = traj_l = np.NaN

            # Compute cumulative discounted reward
            for i in range(1, self.nstep + 1):
                if episode['reward'][idx + i] != np.NaN:
                    step_reward = episode['reward'][idx + i]
                    if np.isnan(reward):
                        reward = np.zeros(1)
                    reward += discount * step_reward
                    discount *= episode['discount'][idx + i] * self.discount

            # Transform
            if self.transform is not None:
                obs = self.transform(torch.as_tensor(obs).div(255)) * 255

            return obs, action, reward, discount, next_obs, label, traj_o, traj_a, traj_r, traj_l, step

        def fetch_sample_process(self, idx=None):
            try:
                if not offline:
                    self.worker_fetch_episodes()  # Populate workers with up-to-date data
            except:
                traceback.print_exc()

            self.samples_since_last_fetch += 1

            # Sample or index an episode
            if idx is None:
                episode_name = self.sample(self.episode_names)
            else:
                idx, episode_name = self.index[idx]

            episode = self.episodes[episode_name]

            return self.process(episode, idx)  # Process episode into a compact experience

        def __iter__(self):
            # Keep fetching, sampling, and building batches
            while True:
                yield self.fetch_sample_process()  # Yields a single experience

        def __getitem__(self, idx):
            return self.fetch_sample_process(idx)  # Get single experience by index

        def __len__(self):
            return self.num_experiences_loaded

    return _Experiences
