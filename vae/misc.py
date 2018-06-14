import math
from collections import UserList, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


def reject_outliers(data, m=2):
    data = np.array(data)
    return data[abs(data - np.mean(data)) <= m * np.std(data)]


class KLAnnealer:
    def __init__(self, **kwargs):
        self.i_start = kwargs['i_start']
        self.w_max = kwargs['w_max']
        self.w_start = kwargs['w_start']
        self.n_epoch = kwargs['n_epoch']

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer, **kwargs):
        self.n_period = kwargs['n_period']
        self.n_mult = kwargs['n_mult']
        self.lr_min = kwargs['lr_min']

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_min + (base_lr - self.lr_min) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end


class Logger(UserList):
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return Logger(self.data[key])
        else:
            ldata = self.sdata[key]
            if isinstance(ldata[0], dict):
                return Logger(ldata)
            else:
                return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)


class LogPlotter:
    def __init__(self, log):
        self.log = log

    def line(self, ax, name):
        if isinstance(self.log[0][name], dict):
            for k in self.log[0][name]:
                ax.plot(self.log[name][k], label=k)
            ax.legend()
        else:
            ax.plot(self.log[name])

        ax.set_ylabel('value')
        ax.set_xlabel('epoch')
        ax.set_title(name)

    def grid(self, names, size=7):
        _, axs = plt.subplots(nrows=len(names) // 2, ncols=2,
                              figsize=(size * 2, size * (len(names) // 2)))

        for ax, name in zip(axs.flatten(), names):
            self.line(ax, name)


class SamplePlotter:
    def __init__(self, corpus, model, sample_params=None):
        self.corpus = corpus
        self.model = model
        if sample_params is None:
            sample_params = {}
        self.sample_params = sample_params

    def plot_one(self, size=8):
        _, _, x, a = self.model.sample_sentence(1, **self.sample_params)
        print(f"sent: '{self.corpus.reverse(x)[0]}'")
        vocab = self.corpus.vocab('x')
        sent = [vocab.itos[i] for i in x[0]]
        _, ax = plt.subplots(figsize=(size, size))
        a = a[0].t().cpu().detach().numpy()[::-1, :]
        ax.imshow(a)
        ax.set_xticks(np.arange(len(sent)))
        ax.set_yticks(np.arange(len(sent)))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_xticklabels(sent)
        ax.set_yticklabels(sent[::-1])
        ax.set_xlabel('generated sequence')
        ax.set_ylabel('context')
