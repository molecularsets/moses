import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from moses.utils import CharVocab, set_torch_seed_to_all_gens


class PolicyGradientLoss(nn.Module):
    def forward(self, outputs, targets, rewards, lengths):
        log_probs = F.log_softmax(outputs, dim=2)
        items = torch.gather(log_probs, 2, targets.unsqueeze(2)) * rewards.unsqueeze(2)
        loss = -sum([t[:l].sum() for t, l in zip(items, lengths)]) / lengths.sum().float()
        return loss


class ORGANTrainer:
    def __init__(self, config):
        self.config = config

    @property
    def n_workers(self):
        n_workers = self.config.n_workers
        return n_workers if n_workers != 1 else 0

    def get_collate_device(self, model):
        n_workers = self.n_workers
        return 'cpu' if n_workers > 0 else model.device

    def generator_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device) for string in data]

            pad = model.vocabulary.pad
            prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=pad)
            nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=pad)
            lens = torch.tensor([len(t) - 1 for t in tensors], dtype=torch.long, device=device)
            return prevs, nexts, lens

        return collate

    def get_dataloader(self, model, data, collate_fn, shuffle=True):
        return DataLoader(data, batch_size=self.config.n_batch, shuffle=shuffle,
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens if self.n_workers > 0 else None)

    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def _pretrain_generator_epoch(self, model, tqdm_data, criterion, optimizer=None):
        model.discriminator.eval()
        if optimizer is None:
            model.generator.eval()
        else:
            model.generator.train()

        postfix = OrderedDict()
        postfix['loss'] = 0
        postfix['running_loss'] = 0

        for i, batch in enumerate(tqdm_data):
            (prevs, nexts, lens) = (data.to(model.device) for data in batch)
            outputs, _, _ = model.generator_forward(prevs, lens)

            loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] = loss.item()
            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval generator' if optimizer is None else 'Train generator'
        for field, value in postfix.items():
            self.log_file.write(field+' = '+str(value)+'\n')
        self.log_file.write('===\n')
        self.log_file.flush()

    def _pretrain_generator(self, model, train_loader, val_loader=None):
        device = model.device
        generator = model.generator
        criterion = nn.CrossEntropyLoss(ignore_index=model.vocabulary.pad)
        optimizer = torch.optim.Adam(generator.parameters(), lr=self.config.lr)

        generator.zero_grad()
        for epoch in range(self.config.generator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc='Generator training (epoch #{})'.format(epoch))
            self._pretrain_generator_epoch(model, tqdm_data, criterion, optimizer)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Generator validation (epoch #{})'.format(epoch))
                self._pretrain_generator_epoch(model, tqdm_data, criterion)

            if epoch % self.config.save_frequency == 0:
                generator = generator.to('cpu')
                torch.save(generator.state_dict(), self.config.model_save[:-3]+'_generator_{0:03d}.pt'.format(epoch))
                generator = generator.to(device)

    def _pretrain_discriminator_epoch(self, model, tqdm_data, criterion, optimizer=None):
        model.generator.eval()
        if optimizer is None:
            model.discriminator.eval()
        else:
            model.discriminator.train()

        postfix = OrderedDict()
        postfix['loss'] = 0
        postfix['running_loss'] = 0

        for i, inputs_from_data in enumerate(tqdm_data):
            inputs_from_data = inputs_from_data.to(model.device)
            inputs_from_model, _ = model.sample_tensor(self.config.n_batch, self.config.max_length)

            targets = torch.zeros(self.config.n_batch, 1, device=model.device)
            outputs = model.discriminator_forward(inputs_from_model)
            loss = criterion(outputs, targets) / 2

            targets = torch.ones(inputs_from_data.shape[0], 1, device=model.device)
            outputs = model.discriminator_forward(inputs_from_data)
            loss += criterion(outputs, targets) / 2

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] = loss.item()
            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval discriminator' if optimizer is None else 'Train discriminator'
        for field, value in postfix.items():
            self.log_file.write(field+' = '+str(value)+'\n')
        self.log_file.write('===\n')
        self.log_file.flush()

    def discriminator_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device) for string in data]
            inputs = pad_sequence(tensors, batch_first=True, padding_value=model.vocabulary.pad)
            return inputs

        return collate

    def _pretrain_discriminator(self, model, train_loader, val_loader=None):
        device = model.device
        discriminator = model.discriminator
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.config.lr)

        discriminator.zero_grad()
        for epoch in range(self.config.discriminator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc='Discriminator training (epoch #{})'.format(epoch))
            self._pretrain_discriminator_epoch(model, tqdm_data, criterion, optimizer)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Discriminator validation (epoch #{})'.format(epoch))
                self._pretrain_discriminator_epoch(model, tqdm_data, criterion)

            if epoch % self.config.save_frequency == 0:
                discriminator = discriminator.to('cpu')
                torch.save(discriminator.state_dict(), self.config.model_save[:-3]+'_discriminator_{0:03d}.pt'.format(epoch))
                discriminator = discriminator.to(device)

    def _policy_gradient_iter(self, model, train_loader, criterion, optimizer, iter_):
        smooth = self.config.pg_smooth_const if iter_ > 0 else 1

        # Generator
        postfix = OrderedDict()
        postfix['generator_loss'] = 0
        postfix['smoothed_reward'] = 0
        gen_tqdm = tqdm(range(self.config.generator_updates),
                        desc='PG generator training (iter #{})'.format(iter_))
        for _ in gen_tqdm:
            model.eval()
            sequences, rewards, lengths = model.rollout(
                self.config.n_batch, self.config.rollouts, self.ref_smiles, self.ref_mols, self.config.max_length)
            model.train()

            lengths, indices = torch.sort(lengths, descending=True)
            sequences = sequences[indices, ...]
            rewards = rewards[indices, ...]

            generator_outputs, lengths, _ = model.generator_forward(sequences[:, :-1], lengths - 1)
            generator_loss = criterion['generator'](generator_outputs, sequences[:, 1:], rewards, lengths)

            optimizer['generator'].zero_grad()
            generator_loss.backward()
            nn.utils.clip_grad_value_(model.generator.parameters(), self.config.clip_grad)
            optimizer['generator'].step()

            postfix['generator_loss'] += (generator_loss.item() - postfix['generator_loss']) * smooth
            mean_episode_reward = torch.cat([t[:l] for t, l in zip(rewards, lengths)]).mean().item()
            postfix['smoothed_reward'] += (mean_episode_reward - postfix['smoothed_reward']) * smooth
            gen_tqdm.set_postfix(postfix)

        postfix['mode'] = 'PG generator (iter #{})'.format(iter_)
        for field, value in postfix.items():
            self.log_file.write(field+' = '+str(value)+'\n')
        self.log_file.write('===\n')
        self.log_file.flush()

        # Discriminator
        postfix = { 'discriminator_loss' : 0 }
        discrim_tqdm = tqdm(range(self.config.discriminator_updates),
                            desc='PG discrim-r training (iter #{})'.format(iter_))
        for _ in discrim_tqdm:
            model.generator.eval()
            n_batches = (len(train_loader) + self.config.n_batch - 1) // self.config.n_batch
            sampled_batches = [model.sample_tensor(self.config.n_batch, self.config.max_length)[0]
                               for _ in range(n_batches)]

            for _ in range(self.config.discriminator_epochs):
                random.shuffle(sampled_batches)

                for inputs_from_model, inputs_from_data in zip(sampled_batches, train_loader):
                    inputs_from_data = inputs_from_data.to(model.device)

                    discrim_targets = torch.zeros(self.config.n_batch, 1, device=model.device)
                    discrim_outputs = model.discriminator_forward(inputs_from_model)
                    discrim_loss = criterion['discriminator'](discrim_outputs, discrim_targets) / 2

                    discrim_targets = torch.ones(self.config.n_batch, 1, device=model.device)
                    discrim_outputs = model.discriminator_forward(inputs_from_data)
                    discrim_loss += criterion['discriminator'](discrim_outputs, discrim_targets) / 2

                    optimizer['discriminator'].zero_grad()
                    discrim_loss.backward()
                    optimizer['discriminator'].step()

                    postfix['discriminator_loss'] += (discrim_loss.item() - postfix['discriminator_loss']) * smooth

            discrim_tqdm.set_postfix(postfix)

        postfix['mode'] = 'PG discriminator (iter #{})'.format(iter_)
        for field, value in postfix.items():
            self.log_file.write(field+' = '+str(value)+'\n')
        self.log_file.write('===\n')
        self.log_file.flush()

    def _train_policy_gradient(self, model, train_loader):
        device = model.device

        criterion = {
            'generator' : PolicyGradientLoss(),
            'discriminator' : nn.BCEWithLogitsLoss(),
        }

        optimizer = {
            'generator' : torch.optim.Adam(model.generator.parameters(), lr=self.config.lr),
            'discriminator' : torch.optim.Adam(model.discriminator.parameters(), lr=self.config.lr),
        }

        model.zero_grad()
        for iter_ in range(self.config.pg_iters):
            self._policy_gradient_iter(model, train_loader, criterion, optimizer, iter_)

            if iter_ % self.config.save_frequency == 0:
                model = model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(iter_))
                model = model.to(device)

    def fit(self, model, train_data, val_data=None):
        self.log_file = open(self.config.log_file, 'w')
        self.log_file.write(str(self.config)+'\n')
        self.log_file.write(str(model)+'\n')

        # Generator
        gen_collate_fn = self.generator_collate_fn(model)
        gen_train_loader = self.get_dataloader(model, train_data, gen_collate_fn, shuffle=True)
        gen_val_loader = None if val_data is None else self.get_dataloader(model, val_data, gen_collate_fn, shuffle=False)
        self._pretrain_generator(model, gen_train_loader, gen_val_loader)

        # Discriminator
        dsc_collate_fn = self.discriminator_collate_fn(model)
        dsc_train_loader = self.get_dataloader(model, train_data, dsc_collate_fn, shuffle=True)
        dsc_val_loader = None if val_data is None else self.get_dataloader(model, val_data, dsc_collate_fn, shuffle=False)
        self._pretrain_discriminator(model, dsc_train_loader, dsc_val_loader)

        # Policy gradient
        self.ref_smiles, self.ref_mols = None, None
        if model.metrics_reward is not None:
            self.ref_smiles, self.ref_mols = model.metrics_reward.get_reference_data(train_data)

        pg_train_loader = dsc_train_loader
        self._train_policy_gradient(model, pg_train_loader)

        del self.ref_smiles
        del self.ref_mols

        self.log_file.close()
        return model
