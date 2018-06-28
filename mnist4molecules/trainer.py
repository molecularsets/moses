import abc


class Trainer(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def fit(self, model, data):
        pass
