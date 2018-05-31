import abc

from torchtext import data, datasets
from torchtext.vocab import GloVe

__all__ = ['SSTCorpus', 'WikiText2Corpus', 'IMDBCorpus']


class Corpus(abc.ABC):
    """
    Handle both preprocessing and batching, also store vocabs doing all
    deterministically.
    """

    @abc.abstractmethod
    def size(self, split):
        """Gets data size by split name or None if infinity"""
        pass

    @abc.abstractmethod
    def vocab(self, name):
        """Gets vocab instance by var name"""
        pass

    @abc.abstractmethod
    def batcher(self, split, mode, n_batch=None, device=None):
        """Make batcher generator, train produces inf batches"""
        pass

    @abc.abstractmethod
    def reverse(self, example, name):
        """Reverse given example to human-readable form"""
        pass


class SSTCorpus(Corpus):
    def __init__(self, **kwargs):
        self.n_batch = kwargs['n_batch']
        self.n_len = kwargs['n_len']
        self.n_vocab = kwargs['n_vocab']
        self.d_emb = kwargs['d_emb']
        self.device = kwargs['device']

        self._load_data()

    def size(self, split):
        return len(getattr(self, split))

    def vocab(self, name):
        vocab = getattr(self, name).vocab
        # vocab.vectors = vocab.vectors.to(self.device)
        return vocab

    def batcher(self, split, mode, n_batch=None, device=None):
        n_batch = n_batch or self.n_batch
        device = device or self.device
        b_iter = data.BucketIterator(
            self._choose_split(split),
            n_batch,
            train=(split == 'train'),
            device=device  # also explicitly due to torchtext bug
        )

        if mode == 'unlabeled':
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    yield batch.text.to(device)
        elif mode == 'labeled':
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    yield batch.text.to(device), batch.label.to(device)
        else:
            raise ValueError(
                "Invalid mode, should be one of the ('unlabeled', 'labeled')"
            )

    def reverse(self, example, name='x'):
        # sent = ' '.join(corpus.vocab('x').itos[w] \
        #                 for w in model.sample_sentence(device=device) \
        #                 if w > 3)
        if name == 'x':
            return self.x.reverse(example)
        elif name == 'y':
            return self.y.vocab.itos[example]
        else:
            raise ValueError(
                "Invalid name, should be one of the ('x', 'y')"
            )

    def _load_data(self):
        self.x = data.ReversibleField(
            init_token='<bos>',
            eos_token='<eos>',
            fix_length=self.n_len,
            lower=True,
            # commented to enable reversiblity
            # tokenize='spacy',
            pad_token=' <pad> ',
            unk_token=' <unk> ',
            batch_first=True
        )
        self.y = data.Field(
            sequential=False,
            unk_token=None,
            batch_first=True
        )

        def filter_pred(e):
            return len(e.text) <= self.n_len and e.label != 'neutral'

        self.train, self.val, self.test = datasets.SST.splits(
            self.x,
            self.y,
            fine_grained=False,
            train_subtrees=False,
            filter_pred=filter_pred
        )

        self.x.build_vocab(self.train,
                           max_size=self.n_vocab - 4,
                           vectors=GloVe('6B', dim=self.d_emb))
        self.y.build_vocab(self.train)

        self.x.vocab.itos[0] = ' <unk> '
        self.x.vocab.itos[1] = ' <pad> '
        self.x.vocab.stoi['<unk>'] = 0
        self.x.vocab.stoi['<pad>'] = 1
        # self.x.vocab.stoi.pop(' <unk> ')
        # self.x.vocab.stoi.pop(' <pad> ')


    def _choose_split(self, split):
        # if split == 'train':
        #     return self.train
        # elif split == 'val':
        #     return self.val
        # elif split == 'test':
        #     return self.test
        # else:
        #     raise ValueError(
        #         "Invalid split, should be one of the ('train', 'val', \
        # 'test')"
        #     )
        return getattr(self, split)


class IMDBCorpus(Corpus):
    def __init__(self, **kwargs):
        self.n_batch = kwargs['n_batch']
        self.n_len = kwargs['n_len']
        self.n_vocab = kwargs['n_vocab']
        self.d_emb = kwargs['d_emb']
        self.device = kwargs['device']

        self._load_data()

    def size(self, split):
        return len(getattr(self, split))

    def vocab(self, name):
        vocab = getattr(self, name).vocab
        # vocab.vectors = vocab.vectors.to(self.device)
        return vocab

    def batcher(self, split, mode, n_batch=None, device=None):
        n_batch = n_batch or self.n_batch
        device = device or self.device
        b_iter = data.BucketIterator(
            self._choose_split(split),
            n_batch,
            train=(split == 'train'),
            device=device  # also explicitly due to torchtext bug
        )

        if mode == 'unlabeled':
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    yield batch.text.to(device)
        elif mode == 'labeled':
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    yield batch.text.to(device), batch.label.to(device)
        else:
            raise ValueError(
                "Invalid mode, should be one of the ('unlabeled', 'labeled')"
            )

    def reverse(self, example, name='x'):
        # sent = ' '.join(corpus.vocab('x').itos[w] \
        #                 for w in model.sample_sentence(device=device) \
        #                 if w > 3)
        if name == 'x':
            return self.x.reverse(example)
        elif name == 'y':
            return self.y.vocab.itos[example]
        else:
            raise ValueError(
                "Invalid name, should be one of the ('x', 'y')"
            )

    def _load_data(self):
        self.x = data.ReversibleField(
            init_token='<bos>',
            eos_token='<eos>',
            fix_length=self.n_len,
            lower=True,
            # commented to enable reversiblity
            # tokenize='spacy',
            pad_token=' <pad> ',
            unk_token=' <unk> ',
            batch_first=True
        )
        self.y = data.Field(
            sequential=False,
            unk_token=None,
            batch_first=True
        )

        # def filter_pred(e):
        #     return len(e.text) <= self.n_len and e.label != 'neutral'

        self.train, self.val = datasets.IMDB.splits(
            self.x,
            self.y
        )
        self.test = self.val

        self.x.build_vocab(self.train,
                           max_size=self.n_vocab - 4,
                           vectors=GloVe('6B', dim=self.d_emb))
        self.y.build_vocab(self.train)

    def _choose_split(self, split):
        # if split == 'train':
        #     return self.train
        # elif split == 'val':
        #     return self.val
        # elif split == 'test':
        #     return self.test
        # else:
        #     raise ValueError(
        #         "Invalid split, should be one of the ('train', 'val', \
        # 'test')"
        #     )
        return getattr(self, split)


class WikiText2Corpus(Corpus):
    def __init__(self, **kwargs):
        self.n_batch = kwargs['n_batch']
        self.n_len = kwargs['n_len']
        self.n_vocab = kwargs['n_vocab']
        self.d_emb = kwargs['d_emb']
        self.device = kwargs['device']

        self._load_data()

    def size(self, split):
        return len(self._get_bppt(split, 1, 'cpu'))

    def vocab(self, name='x'):
        return getattr(self, name).vocab

    def batcher(self, split, mode='unlabeled', n_batch=None, device=None):
        n_batch = n_batch or self.n_batch
        device = device or self.device
        b_iter = self._get_bppt(split, n_batch, device)

        if mode == 'unlabeled':
            for batch in b_iter:
                if batch.batch_size == n_batch \
                        and batch.text.shape[0] == self.n_len:
                    yield batch.text.to(device).t().contiguous()
        else:
            raise ValueError(
                "Invalid mode, should be one of the ('unlabeled',)"
            )

    def reverse(self, example, name='x'):
        if name == 'x':
            return self.x.reverse(example)
        else:
            raise ValueError(
                "Invalid name, should be one of the ('x',)"
            )

    def _load_data(self):
        self.x = data.ReversibleField(
            init_token='<bos>',
            eos_token='<eos>',
            fix_length=self.n_len,
            lower=True,
            # commented to enable reversiblity
            # tokenize='spacy',
            pad_token=' <pad> ',
            unk_token=' <unk> ',
            batch_first=True
        )
        self.train, self.val, self.test = datasets.WikiText2.splits(
            self.x
        )
        self.x.build_vocab(self.train,
                           max_size=self.n_vocab - 4,
                           vectors=GloVe('6B', dim=self.d_emb))

    def _get_bppt(self, split, n_batch, device):
        return data.BPTTIterator(
            getattr(self, split),
            n_batch,
            self.n_len,
            train=(split == 'train'),
            device=device  # also explicitly due to torchtext bug
        )
