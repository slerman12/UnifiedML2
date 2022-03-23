# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import csv
import datetime
import re
from pathlib import Path
from termcolor import colored


def shorthand(log_name):
    return ''.join([s[0].upper() for s in re.split('_|[ ]', log_name)] if len(log_name) > 3 else log_name.upper())


def format(log, log_name):
    l = shorthand(log_name)

    if 'time' in log_name.lower():
        log = str(datetime.timedelta(seconds=int(log)))
        return f'{l}: {log}'
    elif float(log).is_integer():
        log = int(log)
        return f'{l}: {log}'
    else:
        return f'{l}: {log:.04f}'


class Logger:
    def __init__(self, task, seed, path='.', wandb=False):

        self.path = path.replace('Agents.', '')
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.task = task
        self.seed = seed

        self.logs = {}
        self.counts = {}

        self.wandb = wandb

    def log(self, log=None, name="Logs", dump=False):
        if log is not None:

            if name not in self.logs:
                self.logs[name] = {}
                self.counts[name] = {}

            logs = self.logs[name]
            counts = self.counts[name]

            for k, l in log.items():  # TODO Aggregate per step, median
                if k in logs:
                    logs[k] += l
                    counts[k] += 1
                else:
                    logs[k] = l
                    counts[k] = 1

        if dump:
            self.dump_logs(name)

    def dump_logs(self, name=None):
        if name is None:
            for n in self.logs:
                for log_name in self.logs[n]:
                    self.logs[n][log_name] /= self.counts[n][log_name]
                self._dump_logs(self.logs[n], name=n)
                del self.logs[n]
                del self.counts[n]
        else:
            if name not in self.logs:
                return
            for log_name in self.logs[name]:
                self.logs[name][log_name] /= self.counts[name][log_name]
            self._dump_logs(self.logs[name], name=name)
            self.logs[name] = {}
            del self.logs[name]
            del self.counts[name]

    def _dump_logs(self, logs, name):
        self.dump_to_console(logs, name=name)
        self.dump_to_csv(logs, name=name)
        if self.wandb:
            self.log_wandb(logs, name=name)

    def dump_to_console(self, logs, name):
        name = colored(name, 'yellow' if name.lower() == 'train' else 'green' if name.lower() == 'eval' else None,
                       attrs=['dark'] if name.lower() == 'seed' else None)
        pieces = [f'| {name: <14}']
        for log_name, log in logs.items():
            pieces.append(format(log, log_name))
        print(' | '.join(pieces))

    def remove_old_entries(self, logs, file_name):
        rows = []
        with file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= logs['step']:
                    break
                rows.append(row)
        with file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=logs.keys(),
                                    extrasaction='ignore',
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def dump_to_csv(self, logs, name):
        logs = dict(logs)

        assert 'step' in logs

        file_name = Path(self.path) / f'{self.task}_{self.seed}_{name}.csv'

        write_header = True
        if file_name.exists():
            write_header = False
            self.remove_old_entries(logs, file_name)

        file = file_name.open('a')
        writer = csv.DictWriter(file,
                                fieldnames=logs.keys(),
                                restval=0.0)
        if write_header:
            writer.writeheader()

        writer.writerow(logs)
        file.flush()

    # TODO
    # def log_tensorboard(self, logs, name):
    #     if self.tensorboard_writer is None:
    #         self.tensorboard_writer = SummaryWriter(self.path + f'/{self.task}_{self.seed}_{name}_TensorBoard.csv')
    #
    #     for key in logs:
    #         if key != 'step' and key != 'episode':
    #             self.tensorboard_writer.add_scalar(f'{key}', logs[key], logs['step'])

    def log_wandb(self, logs, name):
        if self.wandb != 'initialized':
            import wandb
            wandb.init(project=str(datetime.datetime.now()),
                       name=self.path.replace('/', '_') + f'_{self.task}_{self.seed}')
            self.wandb = 'initialized'
        logs.update({'name': name})
        wandb.log(logs)
