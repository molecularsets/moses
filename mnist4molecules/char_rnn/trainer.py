import os

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class CharRNNTrainer:
    def __init__(self, config):
        self.config = config

    def fit(self, model, data):

        if isinstance(data, tuple):
            train_dataloader = data[0]
            val_dataloader = data[1]
        else:
            train_dataloader = data
            val_dataloader = None

        num_epochs = self.config.num_epochs
        get_params = lambda: (p for p in model.parameters() if p.requires_grad)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            train_dataloader = tqdm.tqdm(train_dataloader)
            train_dataloader.set_description('Train (epoch #{})'.format(epoch))

            self.pass_data(model, train_dataloader, criterion, optimizer)

            if val_dataloader is not None:
                model.eval()

                val_dataloader = tqdm.tqdm(val_dataloader)
                val_dataloader.set_description('Validation (epoch #{})'.format(epoch))

                val_loss = self.pass_data(model, val_dataloader, criterion)

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(),
                               os.path.join(self.config.model_save, "model-iter" + str(epoch) + ".pt"))  # TODO
            else:
                torch.save(model.state_dict(),
                           os.path.join(self.config.model_save, "model-iter" + str(epoch) + ".pt"))  # TODO

    def pass_data(self, model, dataloader, criterion, optimizer=None):
        running_loss = 0

        for i, data in enumerate(dataloader):
            model.zero_grad()

            inputs = [t.to(model.device) for t in data[0]]  # TODO
            targets = [t.to(model.device) for t in data[1]]  # TODO

            outputs, _ = model(inputs)

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)

            loss = criterion(outputs, targets)

            postfix = {'loss': loss.item()}
            dataloader.set_postfix(postfix)

            running_loss += loss * len(inputs)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        return running_loss / len(dataloader)
