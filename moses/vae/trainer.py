
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.nn.utils import clip_grad_norm_

from moses.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer, \
    Logger


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



class VAETrainer:
    def __init__(self, config):
        self.config = config

    def fit(self, model, data, conditional):
        def get_params():
            return (p for p in model.vae.parameters() if p.requires_grad)

        model.train()

        n_epoch = self._n_epoch()
        kl_annealer = KLAnnealer(n_epoch, self.config)

        optimizer = optim.Adam(get_params(), lr=self.config.lr_start)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer, self.config)

        n_last = self.config.n_last
        elog, ilog = Logger(), Logger()

        for epoch in range(n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)

            # Iters
            T = tqdm.tqdm(data)

            for i, t in enumerate(T):

                # x: smiles c: conditions (fingerprints)
                if conditional:
                    x = t[0]
                    c = t[1]
                else:
                    x = t
                    c = None

                # Forward
                kl_loss, recon_loss = model(x, c)
                loss = kl_weight * kl_loss + recon_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(get_params(), self.config.grad_clipping)
                optimizer.step()

                # Log
                lr = optimizer.param_groups[0]['lr']
                ilog.append({
                    'epoch': epoch,
                    'kl_loss': kl_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'loss': loss.item(),
                    'kl_weight': kl_weight,
                    'lr': lr
                })

                # Update T
                kl_loss_value = np.mean(ilog['kl_loss'][-n_last:])
                recon_loss_value = np.mean(ilog['recon_loss'][-n_last:])
                loss_value = np.mean(ilog['loss'][-n_last:])
                postfix = [f'loss={loss_value:.5f}',
                           f'(kl={kl_loss_value:.5f}',
                           f'recon={recon_loss_value:.5f})',
                           f'klw={kl_weight:.5f} lr={lr:.5f}']
                T.set_postfix_str(' '.join(postfix))
                T.set_description(f'Train (epoch #{epoch})')
                T.refresh()

            # Log
            elog.append({
                **{k: v for k, v in ilog[-1].items() if 'loss' not in k},
                'kl_loss': kl_loss_value,
                'recon_loss': recon_loss_value,
                'loss': loss_value
            })

            # Save model at each epoch
            torch.save(model.state_dict(), self.config.model_save)

            # Epoch end
            lr_annealer.step()

        return elog, ilog

    def _n_epoch(self):
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        )
