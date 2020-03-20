import numpy as np
from tqdm.auto import tqdm
import moses
from moses import CharVocab


class NGram:
    def __init__(self, max_context_len=3, verbose=False):
        self.max_context_len = max_context_len
        self._dict = dict()
        self.vocab = None
        self.default_probs = None
        self.zero_probs = None
        self.verbose = verbose

    def fit(self, data):
        self.vocab = CharVocab.from_data(data)
        self.default_probs = np.hstack([np.ones(len(self.vocab)-4),
                                        np.array([0., 1., 0., 0.])])
        self.zero_probs = np.zeros(len(self.vocab))
        if self.verbose:
            print('fitting...')
            data = tqdm(data, total=len(data))
        for line in data:
            t_line = tuple(self.vocab.string2ids(line, True, True))
            for i in range(len(t_line)):
                for shift in range(self.max_context_len):
                    if i + shift + 1 >= len(t_line):
                        break
                    context = t_line[i:i+shift+1]
                    cid = t_line[i+shift+1]
                    probs = self._dict.get(context, self.zero_probs.copy())
                    probs[cid] += 1.
                    self._dict[context] = probs

    def fit_update(self, data):
        if self.verbose:
            print('fitting...')
            data = tqdm(data, total=len(data))
        for line in data:
            t_line = tuple(self.vocab.string2ids(line, True, True))
            for i in range(len(t_line)):
                for shift in range(self.max_context_len):
                    if i + shift + 1 >= len(t_line):
                        break
                    context = t_line[i:i+shift+1]
                    cid = t_line[i+shift+1]
                    probs = self._dict.get(context, self.zero_probs.copy())
                    probs[cid] += 1.
                    self._dict[context] = probs

    def generate_one(self, l_smooth=1., context_len=None, max_len=100):
        if self.vocab is None:
            raise RuntimeError('Error: Fit the model before generating')

        if context_len is None:
            context_len = self.max_context_len
        elif context_len <= 0 or context_len > self.max_context_len:
            context_len = self.max_context_len

        res = [self.vocab.bos]

        while res[-1] != self.vocab.eos and len(res) < max_len:
            begin_index = max(len(res)-context_len, 0)
            context = tuple(res[begin_index:])
            while context not in self._dict:
                context = context[1:]
            probs = self._dict[context]
            probs += self.default_probs*l_smooth
            normed = probs / probs.sum()
            next_symbol = np.random.choice(len(self.vocab), p=normed)
            res.append(next_symbol)

        return self.vocab.ids2string(res)

    def nll(self, smiles, l_smooth=1., context_len=None):
        if self.vocab is None:
            raise RuntimeError('Error: model is not trained')

        if context_len is None:
            context_len = self.max_context_len
        elif context_len <= 0 or context_len > self.max_context_len:
            context_len = self.max_context_len

        tokens = tuple(self.vocab.string2ids(smiles, True, True))

        likelihood = 0.
        for i in range(1, len(tokens)):
            begin_index = max(i-context_len, 0)
            context = tokens[begin_index:i]
            while context not in self._dict:
                context = context[1:]

            probs = self._dict[context] + self.default_probs
            normed = probs / probs.sum()
            prob = normed[tokens[i]]
            if prob == 0.:
                return np.inf
            likelihood -= np.log(prob)

        return likelihood

    def generate(self, n, l_smooth=1., context_len=None,
                 max_len=100, verbose=False):
        generator = (self.generate_one(l_smooth,
                                       context_len,
                                       max_len) for i in range(n))
        if verbose:
            print('generating...')
            generator = tqdm(generator, total=n)
        return list(generator)


def reproduce(seed, samples_path=None, metrics_path=None,
              n_jobs=1, device='cpu', verbose=False,
              samples=30000):
    data = moses.get_dataset('train')
    model = NGram(5)
    model.fit(data)
    np.random.seed(seed)
    smiles = model.generate(samples, verbose=verbose)
    metrics = moses.get_all_metrics(smiles, n_jobs=n_jobs, device=device)

    if samples_path is not None:
        with open(samples_path, 'w') as out:
            out.write('SMILES\n')
            for s in smiles:
                out.write(s+'\n')

    if metrics_path is not None:
        with open(metrics_path, 'w') as out:
            for key, value in metrics.items():
                out.write("%s,%f\n" % (key, value))

    return samples, metrics
