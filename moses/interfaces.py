from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from moses.utils import set_torch_seed_to_all_gens


class MosesTrainer(ABC):
    @property
    def n_workers(self):
        n_workers = self.config.n_workers
        return n_workers if n_workers != 1 else 0

    def get_collate_device(self, model):
        n_workers = self.n_workers
        return 'cpu' if n_workers > 0 else model.device

    def get_dataloader(self, model, data, collate_fn=None, shuffle=True):
        if collate_fn is None:
            collate_fn = self.get_collate_fn(model)
        return DataLoader(data, batch_size=self.config.n_batch,
                          shuffle=shuffle,
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.n_workers > 0 else None)

    def get_collate_fn(self, model):
        return None

    @abstractmethod
    def get_vocabulary(self, data):
        pass

    @abstractmethod
    def fit(self, model, train_data, val_data=None):
        pass
