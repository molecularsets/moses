from collections import Counter
import os
import sys
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from rdkit import Chem

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger
from .model import LatentMolsDataset
from .model import load_model
from .model import Sampler


class LatentGANTrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config
        self.latent_size = self.config.latent_vector_dim

    def _train_epoch(self, model, tqdm_data,
                     optimizer_disc=None, optimizer_gen=None):
        if optimizer_disc is None:
            model.eval()
            optimizer_gen = None
        else:
            model.train()
        self.Sampler = Sampler(generator=self.generator)

        postfix = {'generator_loss': 0,
                   'discriminator_loss': 0}
        disc_loss_batch = []
        g_loss_batch = []

        for i, real_mols in enumerate(tqdm_data):

            real_mols = real_mols.type(model.Tensor)
            if optimizer_disc is not None:
                optimizer_disc.zero_grad()
            fake_mols = self.Sampler.sample(real_mols.shape[0])

            real_validity = self.discriminator(real_mols)
            fake_validity = self.discriminator(fake_mols)
            # Gradient penalty
            gradient_penalty = model.compute_gradient_penalty(
                real_mols.data, fake_mols.data, self.discriminator)

            d_loss = -torch.mean(real_validity) \
                + torch.mean(fake_validity) \
                + self.config.gp * gradient_penalty

            disc_loss_batch.append(d_loss.item())

            if optimizer_disc is not None:

                d_loss.backward()
                optimizer_disc.step()

                # Train the generator every n_critic steps
                if i % self.config.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_gen.zero_grad()
                    # Generate a batch of mols
                    fake_mols = self.Sampler.sample(real_mols.shape[0])

                    # Loss measures generator's ability to
                    # fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_mols)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_gen.step()

                    g_loss_batch.append(g_loss.item())
                    postfix['generator_loss'] = np.mean(g_loss_batch)

            postfix['discriminator_loss'] = np.mean(disc_loss_batch)
            tqdm_data.set_postfix(postfix)
        postfix['mode'] = 'Eval' if optimizer_disc is None else 'Train'
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):

        device = model.device
        optimizer_disc = optim.Adam(self.discriminator.parameters(),
                                    lr=self.config.lr,
                                    betas=(self.config.b1, self.config.b2))
        optimizer_gen = optim.Adam(self.generator.parameters(),
                                   lr=self.config.lr,
                                   betas=(self.config.b1, self.config.b2))
        scheduler_disc = optim.lr_scheduler.StepLR(optimizer_disc,
                                                   self.config.step_size,
                                                   self.config.gamma)
        scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen,
                                                  self.config.step_size,
                                                  self.config.gamma)
        sys.stdout.flush()

        for epoch in range(self.config.train_epochs):
            scheduler_disc.step()
            scheduler_gen.step()

            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))

            postfix = self._train_epoch(model, tqdm_data,
                                        optimizer_disc, optimizer_gen)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            sys.stdout.flush()
            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch)
                )
                model = model.to(device)

    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            tensors = torch.tensor([t for t in data],
                                   dtype=torch.float64, device=device)
            return tensors

        return collate

    def _get_dataset_info(self, data, name=None):
        df = pd.DataFrame(data)
        maxlen = df.iloc[:, 0].map(len).max()
        ctr = Counter(''.join(df.unstack().values))
        charset = ''
        for c in list(ctr):
            charset += c
        return {"maxlen": maxlen, "charset": charset, "name": name}

    def fit(self,
            model,
            train_data,
            val_data=None):
        from ddc_pub import ddc_v3 as ddc
        self.generator = model.Generator
        self.discriminator = model.Discriminator
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.discriminator.cuda()
            self.generator.cuda()

        logger = Logger() if self.config.log_file is not None else None

        if self.config.heteroencoder_version == 'new':
            # Train the heteroencoder first
            print("Training heteroencoder.")
            currentDirectory = os.getcwd()
            path = '{}/moses/latentgan/heteroencoder_models/new_model' \
                .format(currentDirectory)
            encoder_checkpoint_path = \
                '{}/moses/latentgan/heteroencoder_models/checkpoints/' \
                .format(currentDirectory)
            # Convert all SMILES to binary RDKit mols to be
            #  compatible with the heteroencoder
            heteroencoder_mols = [Chem.rdchem.Mol
                                      .ToBinary(Chem.MolFromSmiles(smiles))
                                  for smiles in train_data]
            # Dataset information
            dataset_info = self._get_dataset_info(
                train_data, name="heteroencoder_train_data")
            # Initialize heteroencoder with default parameters
            heteroencoder_model = ddc.DDC(x=np.array(heteroencoder_mols),
                                          y=np.array(heteroencoder_mols),
                                          dataset_info=dataset_info,
                                          scaling=False,
                                          noise_std=self.config.
                                          heteroencoder_noise_std,
                                          lstm_dim=self.config.
                                          heteroencoder_layer_dim,
                                          dec_layers=self.config.
                                          heteroencoder_dec_layers,
                                          td_dense_dim=0,
                                          batch_size=self.config.
                                          heteroencoder_batch_size,
                                          codelayer_dim=self.latent_size)
            # Train heteroencoder
            heteroencoder_model.fit(epochs=self.config.heteroencoder_epochs,
                                    lr=self.config.heteroencoder_lr,
                                    model_name="new_model",
                                    mini_epochs=self.config.
                                    heteroencoder_mini_epochs,
                                    patience=self.config.
                                    heteroencoder_patience,
                                    save_period=self.config.
                                    heteroencoder_save_period,
                                    checkpoint_dir=encoder_checkpoint_path,
                                    gpus=1,
                                    use_multiprocessing=False,
                                    workers=1,
                                    lr_decay=self.config.
                                    heteroencoder_lr_decay,
                                    sch_epoch_to_start=self.config.
                                    heteroencoder_lr_decay_start)

            heteroencoder_model.save(path)

        heteroencoder = load_model(
            model_version=self.config.heteroencoder_version)
        print("Training GAN.")
        mols_in = [Chem.rdchem.Mol.ToBinary(
            Chem.MolFromSmiles(smiles)) for smiles in train_data]
        latent_train = heteroencoder.transform(
            heteroencoder.vectorize(mols_in))
        # Now encode the GAN training set to latent vectors

        latent_train = latent_train.reshape(latent_train.shape[0],
                                            self.latent_size)

        if val_data is not None:
            mols_val = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles))
                        for smiles in val_data]
            latent_val = heteroencoder.transform(
                heteroencoder.vectorize(mols_val))
            latent_val = latent_val.reshape(latent_val.shape[0],
                                            self.latent_size)

        train_loader = self.get_dataloader(model,
                                           LatentMolsDataset(latent_train),
                                           shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, LatentMolsDataset(latent_val), shuffle=False
        )

        self._train(model, train_loader, val_loader, logger)
        return model
