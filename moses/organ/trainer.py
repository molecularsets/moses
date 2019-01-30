import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from moses.utils import SmilesDataset
from moses.utils import Logger

import random
from random import shuffle
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from __future__ import division


# check smiles validity
def isValid(smile_str):
    try:
        m = Chem.MolFromSmiles(smile_str)
        if m:
            return 1 
        else:
            return 0
    except:
        return 0

# convert smiles to maccs fingerprints
def smi_to_maccs(smi):
    MACCS_SIZE = 167
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    else:
        return np.zeros(MACCS_SIZE, dtype=int)

# calculate tanimoto similarity
def Tanimoto(l1,l2):
    a = sum(l1)
    b = sum(l2)
    c = sum([l1[i]&l2[i] for i in range(len(l1))])
    return c/(a+b-c)


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
        model.discriminator.eval()
        if optimizer is None:
            model.generator.eval()
        else:
            model.generator.train()

        postfix = {'loss': 0}

        for i, t in enumerate(tqdm_data):

            # condition
            if config.conditional:
                batch = t[0]
                c = t[1].to(model.device)
            else:
                batch = t
                c = None

            (prevs, nexts, lens) = [x.to(model.device) for x in batch]
            outputs, _, _ = model.generator_forward(prevs, lens, c, config.conditional)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] = (postfix['loss'] * i + loss.item()) / (i + 1)

            tqdm_data.set_postfix(postfix)
        return postfix['loss']

    def _pretrain_generator(self, model, train_data, fps_data, log, val_data=None):
        def collate(tensors):
            tensors.sort(key=lambda x: len(x), reverse=True)
            prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=model.vocabulary.pad)
            nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=model.vocabulary.pad)
            lens = torch.tensor([len(t) - 1 for t in tensors], dtype=torch.long)
            return prevs, nexts, lens

        num_workers = self.config.n_jobs
        if num_workers == 1:
            num_workers = 0

        train_loader = DataLoader(SmilesDataset(train_data, transform=model.string2tensor),
                                  batch_size=self.config.n_batch,
                                  num_workers=num_workers,
                                  shuffle=False,
                                  collate_fn=collate, drop_last=True)

        # condition data
        if config.conditional:
            fps_loader = DataLoader(fps_data, batch_size=self.config.n_batch, shuffle=False, drop_last=True)
            train_loader = zip(train_loader, fps_loader)
            shuffle(train_loader)

        if val_data is not None:
            val_loader = DataLoader(SmilesDataset(val_data, transform=model.string2tensor),
                                    batch_size=self.config.n_batch,
                                    transform=model.string2tensor,
                                    num_workers=num_workers,
                                    shuffle=False,
                                    collate_fn=collate)

        criterion = nn.CrossEntropyLoss(ignore_index=model.vocabulary.pad)
        optimizer = torch.optim.Adam(model.generator.parameters(), lr=self.config.lr)

        for epoch in range(self.config.generator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc='Generator training (epoch #{})'.format(epoch))
            loss = self._pretrain_generator_epoch(model, tqdm_data, criterion, optimizer)
            log.append({'generator_loss': loss})
            log.save(self.config.log_file)
            if val_data is not None:
                tqdm_data = tqdm(val_loader, desc='Generator validation (epoch #{})'.format(epoch))
                self._pretrain_generator_epoch(model, tqdm_data, criterion)

            if epoch % self.config.save_frequency == 0:
                model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_g{0:03d}.pt'.format(epoch))
                model.to(self.config.device)

    def _pretrain_discriminator_epoch(self, model, tqdm_data, criterion, optimizer=None):
        model.generator.eval()
        if optimizer is None:
            model.discriminator.eval()
        else:
            model.discriminator.train()

        postfix = {'loss': 0}

        for i, t in enumerate(tqdm_data):

            # condition
            if config.conditional:
                inputs_from_data = t[0]
                c = t[1].to(model.device)
            else:
                inputs_from_data = t
                c = None

            inputs_from_data = inputs_from_data.to(model.device)
            inputs_from_model, _ = model.sample_tensor(c, config.conditional, self.config.n_batch, self.config.max_length)
            targets = torch.zeros(self.config.n_batch, 1, device=model.device)
            outputs = model.discriminator_forward(inputs_from_model, c, config.conditional)
            loss = criterion(outputs, targets) / 2
            targets = torch.ones(inputs_from_data.shape[0], 1, device=model.device)
            outputs = model.discriminator_forward(inputs_from_data, c, config.conditional)
            loss += criterion(outputs, targets) / 2

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] = (postfix['loss'] * i + loss.item()) / (i + 1)

            tqdm_data.set_postfix(postfix)
        return postfix['loss']

    def _pretrain_discriminator(self, model, train_data, fps_data, log, val_data=None):
        def collate(data):
            data.sort(key=lambda x: len(x), reverse=True)
            tensors = data
            inputs = pad_sequence(tensors, batch_first=True, padding_value=model.vocabulary.pad)

            return inputs

        train_loader = DataLoader(SmilesDataset(train_data, transform=model.string2tensor),
                                  batch_size=self.config.n_batch, shuffle=False, collate_fn=collate, drop_last=True)
        
        # condition data
        if config.conditional:
            fps_loader = DataLoader(fps_data, batch_size=self.config.n_batch, shuffle=False, drop_last=True)
            train_loader = zip(train_loader, fps_loader)
            shuffle(train_loader)

        if val_data is not None:
            val_loader = DataLoader(SmilesDataset(val_data, transform=model.string2tensor),
                                    batch_size=self.config.n_batch, shuffle=False, collate_fn=collate)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=self.config.lr)

        for epoch in range(self.config.discriminator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc='Discriminator training (epoch #{})'.format(epoch))
            loss = self._pretrain_discriminator_epoch(model, tqdm_data, criterion, optimizer)
            log.append({'discriminator_loss': loss})
            log.save(self.config.log_file)
            if val_data is not None:
                tqdm_data = tqdm(val_loader, desc='Discriminator validation (epoch #{})'.format(epoch))
                self._pretrain_discriminator_epoch(model, tqdm_data, criterion)
                
            if epoch % self.config.save_frequency == 0:
                model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_d{0:03d}.pt'.format(epoch))
                model.to(self.config.device)


    def _train_policy_gradient(self, model, train_data, fps_data, log):
        def collate(data):
            data.sort(key=lambda x: len(x), reverse=True)
            tensors = data
            inputs = pad_sequence(tensors, batch_first=True, padding_value=model.vocabulary.pad)

            return inputs

        generator_criterion = PolicyGradientLoss()
        discriminator_criterion = nn.BCEWithLogitsLoss()

        generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=self.config.lr)
        discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=self.config.lr)

        train_loader = DataLoader(SmilesDataset(train_data, transform=model.string2tensor),
                                  batch_size=self.config.n_batch, shuffle=False, collate_fn=collate, drop_last=True)
        
        # condition data
        if config.conditional:
            fps_loader = DataLoader(fps_data, batch_size=self.config.n_batch, shuffle=False, drop_last=True)
            data_loader = zip(train_loader, fps_loader)
        
        pg_iters = tqdm(range(self.config.pg_iters), desc='Policy gradient training')

        postfix = {}
        smooth = 0.1

        for i in pg_iters:

            # condition
            c = []
            if config.conditional:
                c = fps_data[i].unsqueeze(0).to(model.device)

            for _ in range(self.config.generator_updates):
                model.eval()
                sequences, rewards, lengths = model.rollout(c, config.conditional, 
                    self.config.n_batch, self.config.rollouts, self.config.max_length)
                model.train()

                lengths, indices = torch.sort(lengths, descending=True)
                sequences = sequences[indices, ...]
                rewards = rewards[indices, ...]

                # condition
                c = []
                if config.conditional:
                    c = fps_data[i].unsqueeze(0).repeat(self.config.n_batch, 1).to(model.device)

                generator_outputs, lengths, _ = model.generator_forward(sequences[:, :-1], lengths - 1, c, config.conditional)
                generator_loss = generator_criterion(generator_outputs, sequences[:, 1:], rewards, lengths)

                generator_optimizer.zero_grad()
                generator_loss.backward()
                nn.utils.clip_grad_value_(model.generator.parameters(), 5)
                generator_optimizer.step()

                if i == 0:
                    postfix['generator_loss'] = generator_loss.item()
                    postfix['reward'] = torch.cat([t[:l] for t, l in zip(rewards, lengths)]).mean().item()
                else:
                    postfix['generator_loss'] = postfix['generator_loss'] * \
                        (1 - smooth) + generator_loss.item() * smooth
                    postfix['reward'] = postfix['reward'] * \
                        (1 - smooth) + torch.cat([t[:l] for t, l in zip(rewards, lengths)]).mean().item() * smooth

            for _ in range(self.config.discriminator_updates):
                model.generator.eval()
                n_batches = (len(train_loader) + self.config.n_batch - 1) // self.config.n_batch
                sampled_batches = [model.sample_tensor(c, config.conditional, self.config.n_batch, self.config.max_length)[0]
                                   for _ in range(n_batches)]

                for _ in range(self.config.discriminator_epochs):
                    random.shuffle(sampled_batches)

                    for inputs_from_model, inputs_from_data in zip(sampled_batches, train_loader):
                        inputs_from_data = inputs_from_data.to(model.device)
                        discriminator_targets = torch.zeros(self.config.n_batch, 1, device=model.device)
                        discriminator_outputs = model.discriminator_forward(inputs_from_model, c, config.conditional)
                        discriminator_loss = discriminator_criterion(discriminator_outputs, discriminator_targets) / 2

                        discriminator_targets = torch.ones(self.config.n_batch, 1, device=model.device)
                        discriminator_outputs = model.discriminator_forward(inputs_from_data, c, config.conditional)
                        discriminator_loss += discriminator_criterion(discriminator_outputs, discriminator_targets) / 2

                        discriminator_optimizer.zero_grad()
                        discriminator_loss.backward()
                        discriminator_optimizer.step()

                        if i == 0:
                            postfix['discriminator_loss'] = discriminator_loss.item()
                        else:
                            postfix['discriminator_loss'] = postfix['discriminator_loss'] * \
                                (1 - smooth) + discriminator_loss.item() * smooth

            pg_iters.set_postfix(postfix)
            log.append({'discriminator_loss': postfix['discriminator_loss'],
                        'generator_loss': postfix['generator_loss']})
            log.save(self.config.log_file)
            
            if i % self.config.save_frequency*100 == 0:
                model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_pg{0:03d}.pt'.format(i))
                model.to(self.config.device)



    def fit(self, model, train_data, fps_data, val_data=None):
        log = Logger()
        self._pretrain_generator(model, train_data, fps_data, log, val_data)
        self._pretrain_discriminator(model, train_data, fps_data, log, val_data)
        self._train_policy_gradient(model, train_data, fps_data, log)
