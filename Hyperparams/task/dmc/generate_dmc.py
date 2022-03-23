import glob
import os

easy = ['cup_catch', 'pendulum_swingup', 'reacher_easy', 'cartpole_swingup', 'cartpole_balance_sparse', 'finger_turn_easy', 'walker_stand', 'cartpole_balance', 'hopper_stand', 'finger_spin', 'walker_walk']
medium = ['cheetah_run', 'reach_duplo', 'quadruped_walk', 'walker_run', 'reacher_easy', 'reacher_hard', 'finger_turn_easy', 'finger_turn_hard', 'quadruped_run', 'acrobot_swingup', 'hopper_hop', 'cartpole_swingup_sparse']
hard = ['humanoid_stand', 'reacher_hard', 'humanoid_walk', 'humanoid_run', 'finger_turn_hard']

if __name__ == '__main__':
    files = glob.glob(os.getcwd() + "/*")

    # Prints tasks by difficulty
    # print([f.split('.')[-2].split('/')[-1] for f in files if 'hard' in open(f, 'r').read() and 'generate' not in f])

    out = ""
    for task in easy + medium + hard:
        f = open(f"./{task.lower()}.yaml", "w")
        f.write(r"""defaults:
      - {}
      - _self_
    
suite: dmc
action_repeat: 2
frame_stack: 3
nstep: 3
task_name: {}
    
hydra:
    job:
        env_set:
          # Environment variables for MuJoCo
          MKL_SERVICE_FORCE_INTEL: '1'
          MUJOCO_GL: 'egl'""".format('500K' if task in easy else '1M500K' if task in medium else '15M', task))
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)