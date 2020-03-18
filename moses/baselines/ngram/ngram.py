import numpy as np
import moses
from moses import CharVocab
from tqdm import tqdm


class Model:
    def __init__(self, dataset, context_len=3, l_smooth=0.1):
        self.vocab = CharVocab.from_data(dataset)
        self.n = context_len
        self.n_tokens = len(self.vocab)
        self.l_smooth = l_smooth
        self._dict = dict()
        self.default_probs = np.hstack([np.ones(self.n_tokens-4),
                                        np.array([0., 1., 0., 0.])]) * l_smooth
        self.fit(dataset)
    
    def fit(self, data):
        with tqdm(total=len(data)) as pbar:
            for i, line in enumerate(data):
                if i % 100 == 0:
                    pbar.update(100)
                t_line = tuple(self.vocab.string2ids(line, True, True))
                for shift in range(self.n):
                    for i in range(len(t_line)):
                        if i + shift + 1 >= len(t_line):
                            break
                        k = t_line[i:i+shift+1]
                        v = t_line[i+shift+1]
                        probs = self._dict.get(k, self.default_probs.copy())
                        probs[v] += 1.
                        self._dict[k] = probs
    
    def generate_one(self):
        res = [self.vocab.c2i[self.vocab.ss.bos]]

        while res[-1] != self.vocab.c2i[self.vocab.ss.eos]:
            context = tuple(res[max(len(res)-self.n, 0):])
            while context not in self._dict:
                context = context[1:]
            probs = self._dict[context]
            normed = probs / probs.sum()
            next_symbol = np.random.choice(self.n_tokens, p=normed)
            res.append(next_symbol)
        
        return self.vocab.ids2string(res)
    
    def likelihood(self, smiles):
        pass