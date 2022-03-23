# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import argparse
import json
import subprocess

# Common sweeps
from Hyperparams.task.atari.generate_atari import atari_tasks
from Hyperparams.task.dmc.generate_dmc import easy, medium, hard
agents = [
    # 'SPR',
    # 'DQN',
    # 'DrQV2',
    # 'DQNDPG',
    # 'DynoSOAR',
    # 'Ascend', 'AC2'
    'Random'
          ]
seeds = [1]
experiment = 'Random 100 Episodes (Original Settings)'

common_params = f'train_steps=0 evaluate_episodes=100 experiment={experiment}'
classify_RL = 'false'
classify_supervise = 'true'

common_sweeps = {'atari': [f'task=atari/{task.lower()} Agent=Agents.{agent}Agent seed={seed} {common_params}' for task in atari_tasks for agent in agents for seed in seeds],
                 'dmc': [f'task=dmc/{task.lower()} Agent=Agents.{agent}Agent seed={seed} {common_params}' for task in easy + medium + hard for agent in agents for seed in seeds],
                 'classify': [f'task=classify/{task.lower()} Agent=Agents.{agent}Agent RL={classify_RL} supervise={classify_supervise} seed={seed}' for task in ['mnist', 'cifar10', 'tinyimagenet'] for agent in agents for seed in seeds]}
common_sweeps.update({'all': sum(common_sweeps.values(), [])})
common_sweeps.update({'rl': common_sweeps['dmc'] + common_sweeps['atari']})
common_sweeps.update({'continuous': common_sweeps['dmc'] + common_sweeps['classify']})

sweep = {'ViRP':
             [f'recipes.Encoder.Eyes=Blocks.Architectures.LermanBlocks.ViRP '
                  f'experiment="ViRP-3" ema=true weight_decay=0.01 task=classify/{task.lower()} RL=False'
                  for task in ['cifar10']]
             + []
                 # [f'recipes.Encoder.Eyes=Blocks.Architectures.ViT '
                 #  f'experiment="ViT-3" ema=true weight_decay=0.01 task=classify/{task.lower()} RL=False'
                 #  for task in ['cifar10']]
         }
common_sweeps.update(sweep)
# +recipes.encoder.eyes.experiment='plus_in_concat_plus_mid'

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="job",
                    help='job name')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='uses CPUs only, not GPUs')
parser.add_argument('--lab', action='store_true', default=False,
                    help='uses csxu')
parser.add_argument('--K80', action='store_true', default=False,
                    help='uses K80 GPU')
parser.add_argument('--V100', action='store_true', default=False,
                    help='uses V100 GPU')
parser.add_argument('--A100', action='store_true', default=False,
                    help='uses A100 GPU')
parser.add_argument('--ANY_BIG', action='store_true', default=False,
                    help='uses K80, V100, or A100 GPU')
parser.add_argument('--ANY_BIGish', action='store_true', default=False,
                    help='uses K80 or V100 GPU, no A100')
parser.add_argument('--num-cpus', type=int, default=5,
                    help='how many CPUs to use')
parser.add_argument('--mem', type=int, default=25,
                    help='memory to request')
parser.add_argument('--file', type=str, default="Run.py",
                    help='file to run')
parser.add_argument('--params', type=str, default="",
                    help='params to pass into file')
parser.add_argument('--sweep_name', type=str, default="",
                    help='a common sweep to run')
args = parser.parse_args()

if len(args.sweep_name) > 0 and args.sweep_name in common_sweeps:
    args.params = common_sweeps[args.sweep_name]
elif args.params[0] == '[':
    args.params = json.loads(args.params)
else:
    args.params = [args.params]

# Sweep
for i, param in enumerate(args.params):
    K80 = args.K80
    # if len(args.sweep_name) > 0 and args.sweep_name in common_sweeps:
    #     K80 = True if 'task=dmc/' in param.lower() else args.K80

    slurm_script = f"""#!/bin/bash
#SBATCH {"-c {}".format(args.num_cpus) if args.cpu else "-p gpu -c {}".format(args.num_cpus)}
{"" if args.cpu else "#SBATCH --gres=gpu"}
{"#SBATCH -p csxu -A cxu22_lab" if args.cpu and args.lab else "#SBATCH -p csxu -A cxu22_lab --gres=gpu" if args.lab else ""}
#SBATCH -t {"15-00:00:00" if args.lab else "5-00:00:00"} -o ./{args.name}_{i}.log -J {args.name}_{i}
#SBATCH --mem={args.mem}gb 
{"#SBATCH -C K80" if K80 else "#SBATCH -C V100" if args.V100 else "#SBATCH -C A100" if args.A100 else "#SBATCH -C K80|V100|A100" if args.ANY_BIG else "#SBATCH -C K80|V100" if args.ANY_BIGish else ""}
source /scratch/slerman/miniconda/bin/activate agi
python3 {args.file} {param}
"""

    # Write script
    with open("sbatch_script", "w") as file:
        file.write(slurm_script)

    # Launch script (with error checking / re-launching)
    success = "error"
    while "error" in success:
        try:
            success = str(subprocess.check_output(['sbatch {}'.format("sbatch_script")], shell=True))
            print(success[2:][:-3])
            if "error" in success:
                print("Errored... trying again")
        except:
            success = "error"
            if "error" in success:
                print("Errored... trying again")
    print("Success!")
