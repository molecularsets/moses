import torch.nn as nn
import torch.optim as optim
import tqdm
from moses.utils import Logger
import torch


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
        device = torch.device(self.config.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.config.step_size, self.config.gamma)
        elog = Logger()
        for epoch in range(num_epochs):
            scheduler.step()
            model.train()
            train_dataloader = tqdm.tqdm(train_dataloader)
            train_dataloader.set_description('Train (epoch #{})'.format(epoch))

            loss = self._pass_data(model, train_dataloader, criterion, optimizer)
            elog.append({'loss': loss})
            if val_dataloader is not None:
                val_dataloader = tqdm.tqdm(val_dataloader)
                val_dataloader.set_description('Validation (epoch #{})'.format(epoch))

                self._pass_data(model, val_dataloader, criterion)

            if epoch % self.config.save_frequency == 0:
                model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model.to(device)
            elog.save(self.config.log_file)
        torch.save(model.state_dict(), config.model_save)


    def _pass_data(self, model, dataloader, criterion, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        running_loss = 0
        total = 0

        for i, (prevs, nexts, lens) in enumerate(dataloader):

            outputs, _, _ = model(prevs, lens)

            loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            postfix = {'loss': loss.item()}
            dataloader.set_postfix(postfix)

            running_loss += loss.item() * len(prevs)
            total += len(prevs)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return running_loss / total
