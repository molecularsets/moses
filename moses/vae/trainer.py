import torch
import torch.optim as optim
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from moses.utils import OneHotVocab, Logger, CircularBuffer, set_torch_seed_to_all_gens
from moses.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer


class VAETrainer:
    def __init__(self, config):
        self.config = config

    def get_vocabulary(self, data):
        return OneHotVocab.from_data(data)

    def get_dataloader(self, model, data, shuffle=True):
        n_workers = self.config.n_workers
        if n_workers == 1:
            n_workers = 0
        device = 'cpu' if n_workers > 0 else model.device

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device) for string in data]

            return tensors

        return DataLoader(data, batch_size=self.config.n_batch, shuffle=shuffle,
                          num_workers=n_workers, collate_fn=collate,
                          worker_init_fn=set_torch_seed_to_all_gens if n_workers > 0 else None)

    def _train_epoch(self, model, epoch, tqdm_data, kl_weight, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfixes_iter = []
        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        for i, input_batch in enumerate(tqdm_data):
            input_batch = tuple(data.to(model.device) for data in input_batch)

            # Forward
            kl_loss, recon_loss = model(input_batch)
            loss = kl_weight * kl_loss + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model), self.config.clip_grad)
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else None
            if optimizer is not None:
                postfixes_iter.append({
                    'epoch' : epoch,
                    'kl_weight' : kl_weight,
                    'lr' : lr,
                    'kl_loss' : kl_loss.item(),
                    'recon_loss' : recon_loss.item(),
                    'loss' : loss.item(),
                })

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f})',
                       f'klw={kl_weight:.5f} lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))
            #tqdm_data.refresh()

        postfix_epoch = {
            'epoch' : epoch,
            'kl_weight' : kl_weight,
            'lr' : lr,
            'kl_loss' : kl_loss_value,
            'recon_loss' : recon_loss_value,
            'loss' : loss_value,
        }

        postfix_epoch['mode'] = 'Eval' if optimizer is None else 'Train'
        return postfixes_iter, postfix_epoch

    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)

    def _train(self, model, train_loader, val_loader=None, loggers=None):
        device = model.device
        n_epoch = self._n_epoch()

        optimizer = optim.Adam(self.get_optim_params(model), lr=self.config.lr_start)
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer, self.config)

        model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)

            tqdm_data = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch))
            postfixes_iter, postfix_epoch = self._train_epoch(model, epoch, tqdm_data, kl_weight, optimizer)
            if loggers['iter'] is not None:
                for postfix in postfixes_iter:
                    loggers['iter'].append(postfix)
                loggers['iter'].save(self.config.log_iter_file)
            if loggers['epoch'] is not None:
                loggers['epoch'].append(postfix_epoch)
                loggers['epoch'].save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                _, postfix_epoch = self._train_epoch(model, epoch, tqdm_data, kl_weight)
                if loggers['epoch'] is not None:
                    loggers['epoch'].append(postfix_epoch)
                    loggers['epoch'].save(self.config.log_file)

            if (self.config.model_save is not None) and (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model = model.to(device)

            # Epoch end
            lr_annealer.step()

    def fit(self, model, train_data, val_data=None):
        loggers = dict()
        loggers['epoch'] = Logger() if self.config.log_file is not None else None
        loggers['iter'] = Logger() if self.config.log_iter_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)

        self._train(model, train_loader, val_loader, loggers)
        return model

    def _n_epoch(self):
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        )
