# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import hydra
from hydra.utils import instantiate, call

import Utils

import torch
torch.backends.cudnn.benchmark = True


# Hydra conveniently and cleanly manages sys args
# Hyper-param cfg files located in ./Hyperparams

@hydra.main(config_path='Hyperparams', config_name='cfg')
def main(args):
    # Set seeds
    Utils.set_seeds(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train, test environments
    env = instantiate(args.environment)
    generalize = instantiate(args.environment, train=False, seed=args.seed + 1234)

    for arg in ('obs_shape', 'action_shape', 'discrete', 'obs_spec', 'action_spec', 'evaluate_episodes', 'data_norm'):
        if hasattr(generalize, arg):
            setattr(args, arg, getattr(generalize, arg))

    # Agent
    agent = Utils.load(args.save_path, args.device) if args.load \
        else instantiate(args.agent).to(args.device)

    args.train_steps += agent.step

    # Experience replay
    replay = instantiate(args.replay)

    # Loggers
    logger = instantiate(args.logger)

    vlogger = instantiate(args.vlogger)

    # Start
    converged = training = False
    while True:
        # Evaluate
        if args.evaluate_per_steps and agent.step % args.evaluate_per_steps == 0:

            for ep in range(args.evaluate_episodes):
                _, logs, vlogs = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                    vlog=args.log_video)

                logger.log(logs, 'Eval')

            logger.dump_logs('Eval')

            if args.log_video:
                vlogger.dump_vlogs(vlogs, f'{agent.step}')

        if args.plot_per_steps and agent.step % args.plot_per_steps == 0:
            call(args.plotting)

        if converged:
            break

        # Rollout
        experiences, logs, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True

        replay.add(experiences)

        if env.episode_done:
            if args.log_per_episodes and agent.episode % args.log_per_episodes == 0:
                logger.log(logs, 'Train' if training else 'Seed', dump=True)

            if env.last_episode_len > args.nstep:
                replay.add(store=True)  # Only store full episodes

        converged = agent.step >= args.train_steps
        training = (training or agent.step > args.seed_steps or env.offline) and len(replay) >= args.num_workers

        # Train agent
        if training and args.learn_per_steps and agent.step % args.learn_per_steps == 0 or converged:

            for _ in range(args.learn_steps_after if converged else 1):  # Additional updates after all rollouts
                logs = agent.train().learn(replay)  # Trains the agent
                if args.log_per_episodes and agent.episode % args.log_per_episodes == 0:
                    logger.log(logs, 'Train')

        if training and args.save_per_steps and agent.step % args.save_per_steps == 0 or (converged and args.save):
            Utils.save(args.save_path, agent)

        if training and args.load_per_steps and agent.step % args.load_per_steps == 0:
            agent = Utils.load(args.save_path, args.device)


if __name__ == '__main__':
    main()
