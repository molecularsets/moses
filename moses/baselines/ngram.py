import numpy as np
import moses
from moses import CharVocab
from tqdm.auto import tqdm


class NGram:
    def __init__(self, max_context_len=3):
        self.max_context_len = max_context_len
        self._dict = dict()
        self.vocab = None
        self.default_probs = None

    def fit(self, data):
        self.vocab = CharVocab.from_data(data)
        self.default_probs = np.hstack([np.ones(len(self.vocab)-4),
                                        np.array([0., 1., 0., 0.])])
        self.zero_probs = np.zeros(len(self.vocab))
        print('fitting...')
        for line in tqdm(data, total=len(data)):
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
        print('fitting...')
        for line in tqdm(data, total=len(data)):
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
            raise Exception('Error: model is not trained')
        
        if context_len == None or context_len <= 0 or context_len > self.max_context_len:
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
            raise Exception('Error: model is not trained')
            
        if context_len == None or context_len <= 0 or context_len > self.max_context_len:
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
    
    def generate(self, n, l_smooth=1., context_len=None, max_len=100):
        generator = (self.generate_one(l_smooth=1., context_len=None, max_len=100) for i in range(n))
        print('generating...')
        return list(tqdm(generator, total=n))

def reproduce():
    data = moses.get_dataset('train')
    model = NGram()
    model.fit(data)
    
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        smiles = model.generate(30000)
        with open('../../data/samples/n_gram/n_gram_%d.csv' % (seed+3), 'w') as out:
            out.write('smiles\n')
            out.write('\n'.join(smiles))
