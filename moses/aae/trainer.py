import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab

__all__ = ['AAETrainer']


class AAETrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config

    def _pretrain_epoch(self, model, tqdm_data, criterion, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix = {'loss': 0}

        for i, (encoder_inputs, decoder_inputs, decoder_targets) in enumerate(tqdm_data):
            latent_codes = model.encoder_forward(*encoder_inputs)
            decoder_outputs, decoder_output_lengths, _ = model.decoder_forward(
                *decoder_inputs, latent_codes, is_latent_states=True)

            decoder_outputs = torch.cat([t[:l] for t, l in zip(decoder_outputs, decoder_output_lengths)], dim=0)
            decoder_targets = torch.cat([t[:l] for t, l in zip(*decoder_targets)], dim=0)

            loss = criterion(decoder_outputs, decoder_targets)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] += (loss.item() - postfix['loss']) / (i + 1)

            tqdm_data.set_postfix(postfix)

    def _pretrain(self, model, train_loader, val_loader=None):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(model.encoder.parameters()) +
                                     list(model.decoder.parameters()), lr=self.config.lr)

        for epoch in range(self.config.pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc='Pretraining (epoch #{})'.format(epoch))
            self._pretrain_epoch(model, tqdm_data, criterion, optimizer)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                self._pretrain_epoch(model, tqdm_data, criterion)

    def _train_epoch(self, model, tqdm_data, criterions, optimizers=None):
        if optimizers is None:
            model.eval()
        else:
            model.train()

        postfix = OrderedDict()
        postfix['autoencoder_loss'] = 0
        postfix['generator_loss'] = 0
        postfix['discriminator_loss'] = 0

        for i, (encoder_inputs, decoder_inputs, decoder_targets) in enumerate(tqdm_data):
            latent_codes = model.encoder_forward(*encoder_inputs)
            decoder_outputs, decoder_output_lengths, _ = model.decoder_forward(
                *decoder_inputs, latent_codes, is_latent_states=True)
            discriminator_outputs = model.discriminator_forward(latent_codes)

            decoder_outputs = torch.cat([t[:l] for t, l in zip(decoder_outputs, decoder_output_lengths)], dim=0)
            decoder_targets = torch.cat([t[:l] for t, l in zip(*decoder_targets)], dim=0)

            autoencoder_loss = criterions['autoencoder'](decoder_outputs, decoder_targets)
            generator_loss = criterions['generator'](discriminator_outputs)

            if i % 2 == 0:
                discriminator_inputs = model.sample_latent(latent_codes.shape[0])
                discriminator_outputs = model.discriminator(discriminator_inputs)
                discriminator_targets = torch.ones(latent_codes.shape[0], 1, device=model.device)
            else:
                discriminator_targets = torch.zeros(latent_codes.shape[0], 1, device=model.device)

            discriminator_loss = criterions['discriminator'](discriminator_outputs, discriminator_targets)

            if optimizers is not None:
                optimizers['autoencoder'].zero_grad()
                autoencoder_loss.backward(retain_graph=True)
                optimizers['autoencoder'].step()

                optimizers['generator'].zero_grad()
                generator_loss.backward(retain_graph=True)
                optimizers['generator'].step()

                optimizers['discriminator'].zero_grad()
                discriminator_loss.backward()
                optimizers['discriminator'].step()

            postfix['autoencoder_loss'] += (autoencoder_loss.item() - postfix['autoencoder_loss']) / (i + 1)
            postfix['generator_loss'] += (generator_loss.item() - postfix['generator_loss']) / (i + 1)
            postfix['discriminator_loss'] += (discriminator_loss.item() - postfix['discriminator_loss']) / (i + 1)

            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizers is None else 'Train'
        for field, value in postfix.items():
            self.log_file.write(field+' = '+str(value)+'\n')
        self.log_file.write('===\n')
        self.log_file.flush()

    def _train(self, model, train_loader, val_loader=None):
        criterions = {'autoencoder': nn.CrossEntropyLoss(),
                      'generator': lambda t: -torch.mean(F.logsigmoid(t)),
                      'discriminator': nn.BCEWithLogitsLoss()}

        optimizers = {'autoencoder': torch.optim.Adam(list(model.encoder.parameters()) +
                                                      list(model.decoder.parameters()), lr=self.config.lr),
                      'generator': torch.optim.Adam(model.encoder.parameters(), lr=self.config.lr),
                      'discriminator': torch.optim.Adam(model.discriminator.parameters(), lr=self.config.lr)}
        schedulers = {
            k: torch.optim.lr_scheduler.StepLR(v, self.config.step_size, self.config.gamma)
            for k, v in optimizers.items()
        }
        device = torch.device(self.config.device)

        for epoch in range(self.config.train_epochs):
            tqdm_data = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch))

            for scheduler in schedulers.values():
                scheduler.step()

            self._train_epoch(model, tqdm_data, criterions, optimizers)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                self._train_epoch(model, tqdm_data, criterions)

            if epoch % self.config.save_frequency == 0:
                model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model.to(device)

    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def get_dataloader(self, model, data, shuffle=True):
        def collate(data):
            data.sort(key=lambda x: len(x), reverse=True)

            tensors = [model.string2tensor(string) for string in data]
            lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long, device=model.device)

            encoder_inputs = pad_sequence(tensors, batch_first=True, padding_value=model.vocabulary.pad)
            encoder_input_lengths = lengths - 2

            decoder_inputs = pad_sequence([t[:-1] for t in tensors], batch_first=True,
                                          padding_value=model.vocabulary.pad)
            decoder_input_lengths = lengths - 1

            decoder_targets = pad_sequence([t[1:] for t in tensors], batch_first=True,
                                           padding_value=model.vocabulary.pad)
            decoder_target_lengths = lengths - 1

            return (encoder_inputs, encoder_input_lengths), \
                   (decoder_inputs, decoder_input_lengths), \
                   (decoder_targets, decoder_target_lengths)

        return DataLoader(data, batch_size=self.config.n_batch, shuffle=shuffle, collate_fn=collate)

    def fit(self, model, train_data, val_data=None):
        self.log_file = open(self.config.log_file, 'w')
        self.log_file.write(str(self.config)+'\n')
        self.log_file.write(str(model)+'\n')

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)

        self._pretrain(model, train_loader, val_loader)
        self._train(model, train_loader, val_loader)
        self.log_file.close()
        return model
