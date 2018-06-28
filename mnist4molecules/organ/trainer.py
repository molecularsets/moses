import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class PolicyGradientLoss(nn.Module):
    def forward(self, outputs, targets, rewards, lengths):
        log_probs = F.log_softmax(outputs, dim=2)
        items = torch.gather(log_probs, 2, targets.unsqueeze(2)) * rewards.unsqueeze(2)
        loss = -sum([t[:l].sum() for t, l in zip(items, lengths)]) / lengths.sum().float()

        return loss


class ORGANTrainer:
    def __init__(self, config):
        self.config = config

    def _pretrain_generator_epoch(self, model, tqdm_data, criterion, optimizer=None):
        if optimizer is None:
            model.generator.eval()
        else:
            model.generator.train()
    
        postfix = {'loss': 0}

        for i, (prevs, nexts, lens) in enumerate(tqdm_data):
            outputs, _, _ = model.generator_forward(prevs, lens)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] = (postfix['loss'] * i + loss.item()) / (i + 1)

            tqdm_data.set_postfix(postfix)

    def _pretrain_generator(self, model, train_data, val_data=None):
        def collate(data):
            data.sort(key=lambda x: len(x), reverse=True)
            tensors = [model.string2tensor(s, add_bos=True, add_eos=True) for s in data]

            prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=model.vocabulary.pad)
            nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=model.vocabulary.pad)
            lens = torch.tensor([len(t) - 1 for t in tensors], dtype=torch.long, device= model.device)

            return prevs, nexts, lens

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate)
        if val_data is not None:
            val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate)

        criterion = nn.CrossEntropyLoss(ignore_index=model.vocabulary.pad)
        optimizer = torch.optim.Adam(model.generator.parameters(), lr=self.config.lr)

        for epoch in range(self.config.generator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc='Generator training (epoch #{})'.format(epoch))
            self._pretrain_generator_epoch(model, tqdm_data, criterion, optimizer)

            if val_data is not None:
                tqdm_data = tqdm(val_loader, desc='Generator validation (epoch #{})'.format(epoch))
                self._pretrain_generator_epoch(model, tqdm_data, criterion)

    def _pretrain_discriminator_epoch(self, model, tqdm_data, criterion, optimizer=None):
        model.generator.eval()
        if optimizer is None:
            model.discriminator.eval()
        else:
            model.discriminator.train()

        postfix = {'loss': 0}

        i = 0
        for inputs_from_data, targets_from_data in tqdm_data:
            for is_sampling in [False, True]:
                i += 1

                if is_sampling:
                    inputs, _ = model.sample_tensor(len(inputs), self.config.max_length)
                    targets = torch.zeros(len(inputs), 1, device=model.device)
                else:
                    inputs = inputs_from_data
                    targets = targets_from_data
                
                outputs = model.discriminator_forward(inputs)
                loss = criterion(outputs, targets)

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                postfix['loss'] = (postfix['loss'] * (i - 1) + loss.item()) / i

                tqdm_data.set_postfix(postfix)
                
    def _pretrain_discriminator(self, model, train_data, val_data=None):
        def collate(data):
            data.sort(key=lambda x: len(x), reverse=True)
            tensors = [model.string2tensor(s, add_bos=True, add_eos=True) for s in data]

            inputs = pad_sequence(tensors, batch_first=True, padding_value=model.vocabulary.pad)
            targets = torch.ones(inputs.shape[0], 1, device=model.device)

            return inputs, targets

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate)
        if val_data is not None:
            val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=self.config.lr)

        for epoch in range(self.config.discriminator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc='Discriminator training (epoch #{})'.format(epoch))
            self._pretrain_discriminator_epoch(model, tqdm_data, criterion, optimizer)

            if val_data is not None:
                tqdm_data = tqdm(val_loader, desc='Discriminator validation (epoch #{})'.format(epoch))
                self._pretrain_discriminator_epoch(model, tqdm_data, criterion)

    def _train_policy_gradient(self, model, train_data):
        generator_criterion = PolicyGradientLoss()
        discriminator_criterion = nn.BCEWithLogitsLoss()

        generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=self.config.lr)
        discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=self.config.lr)

        pg_iters = tqdm(range(self.config.pg_iters), desc='Policy gradient training')

        postfix= {'generator_loss': 0,
                  'discriminator_loss': 0,
                  'reward': 0}

        for i in pg_iters:
            model.eval()
            sequences, rewards, lengths = model.rollout(self.config.batch_size, self.config.rollouts, self.config.max_length)
            model.train()

            lengths, indices = torch.sort(lengths, descending=True)
            sequences = sequences[indices, ...]
            rewards = rewards[indices, ...]

            generator_outputs, lengths, _ = model.generator_forward(sequences[:, :-1], lengths - 1)
            generator_loss = generator_criterion(generator_outputs, sequences[:, 1:], rewards, lengths)

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()


            if i % 2 == 0:
                model.generator.eval()
                discriminator_inputs, _ = model.sample_tensor(self.config.batch_size, self.config.max_length)
                discriminator_targets = torch.zeros(self.config.batch_size, 1, device=model.device)
            else:
                samples = random.sample(train_data, self.config.batch_size)
                samples.sort(key=lambda x: len(x), reverse=True)
                tensors = [model.string2tensor(s, add_bos=True, add_eos=True) for s in samples]
                discriminator_inputs = pad_sequence(tensors, batch_first=True, padding_value=model.vocabulary.pad)
                discriminator_targets = torch.ones(self.config.batch_size, 1, device=model.device)

            discriminator_outputs = model.discriminator_forward(discriminator_inputs)
            discriminator_loss = discriminator_criterion(discriminator_outputs, discriminator_targets)
            
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            postfix['generator_loss'] = postfix['generator_loss'] * 0.8 + generator_loss.item() * 0.2
            postfix['discriminator_loss'] = postfix['discriminator_loss'] * 0.8 + discriminator_loss.item() * 0.2
            postfix['reward'] = postfix['reward'] * 0.8 + torch.cat([t[:l] for t, l in zip(rewards, lengths)]).mean().item() * 0.2

            pg_iters.set_postfix(postfix)
        
    def fit(self, model, train_data, val_data=None):
        self._pretrain_generator(model, train_data, val_data)
        self._pretrain_discriminator(model, train_data, val_data)
        self._train_policy_gradient(model, train_data)
