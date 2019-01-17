from abc import ABC, abstractmethod


class MosesTrainer(ABC):
    @abstractmethod
    def get_vocabulary(self, data, **args):
        pass

    @abstractmethod
    def get_dataloader(self, model, data, **args):
        pass

    @abstractmethod
    def fit(self, model, train_data, val_data=None):
        pass