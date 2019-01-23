import random
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import UserList, defaultdict
from rdkit import rdBase

# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
def set_torch_seed_to_all_gens(worker_id):
    seed = torch.initial_seed() % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)

class SS:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'


class CharVocab:
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SS):
        if (ss.bos in chars) or (ss.eos in chars) or (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SS in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    elif isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    else:
        return n_jobs.map

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
        for k in sorted(step_dict.keys()):
            self.sdata[k].append(step_dict[k])

    def save(self, path):
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)


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

def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')


def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')
