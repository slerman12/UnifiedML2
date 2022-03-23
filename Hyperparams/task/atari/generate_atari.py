atari_tasks = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]

if __name__ == '__main__':
    out = ""
    for task in atari_tasks:
        f = open(f"./{task.lower()}.yaml", "w")
        f.write(r"""defaults:
      - 100K
      - _self_
    
suite: atari
action_repeat: 4
nstep: 10
frame_stack: 3
task_name: {}""".format(task))
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
