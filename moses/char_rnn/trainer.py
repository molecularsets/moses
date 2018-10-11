import torch.nn as nn
import torch.optim as optim
import tqdm


class CharRNNTrainer:

    def __init__(self, config):
        self.config = config

    def fit(self, model, data):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        if isinstance(data, tuple):
            train_dataloader = data[0]
            val_dataloader = data[1]
        else:
            train_dataloader = data
            val_dataloader = None

        num_epochs = self.config.num_epochs

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)

        for epoch in range(num_epochs):
            model.train()
            train_dataloader = tqdm.tqdm(train_dataloader)
            train_dataloader.set_description('Train (epoch #{})'.format(epoch))

            self._pass_data(model, train_dataloader, criterion, optimizer)

            if val_dataloader is not None:
                val_dataloader = tqdm.tqdm(val_dataloader)
                val_dataloader.set_description('Validation (epoch #{})'.format(epoch))

                self._pass_data(model, val_dataloader, criterion)

    def _pass_data(self, model, dataloader, criterion, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        running_loss = 0

        for i, (prevs, nexts, lens) in enumerate(dataloader):

            outputs, _, _ = model(prevs, lens)

            loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            postfix = {'loss': loss.item()}
            dataloader.set_postfix(postfix)

            running_loss += loss * len(prevs)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return running_loss.item() / len(dataloader)
