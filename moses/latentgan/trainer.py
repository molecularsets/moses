import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger
from .model import LatentMolsDataset
from .model import Generator
from .model import Discriminator
from .model import Sampler

class LatentGANTrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config
        self.generator = Generator()
        self.discriminator = Discriminator()

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.discriminator.cuda()
            self.generator.cuda()

    def _train_epoch(self, model, tqdm_data, optimizer_disc=None, optimizer_gen=None):
        if optimizer_disc is None:
            model.eval()
            optimizer_gen=None
        else:
            model.train()
        self.Sampler = Sampler(generator=self.generator)

        postfix = {'generator_loss': 0,
                   'discriminator_loss': 0}
        disc_loss_batch = []
        g_loss_batch = []


        for i, real_mols in enumerate(tqdm_data):

            real_mols=real_mols.type(model.Tensor)
            if optimizer_disc is not None:
                optimizer_disc.zero_grad()
            fake_mols = self.Sampler.sample(real_mols.shape[0])

            real_validity = self.discriminator(real_mols)
            fake_validity = self.discriminator(fake_mols)
            # Gradient penalty
            gradient_penalty = model.compute_gradient_penalty(real_mols.data, fake_mols.data, self.discriminator)


            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.config.gp * gradient_penalty

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

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_mols)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_gen.step()


                    g_loss_batch.append(g_loss.item())
                    postfix['generator_loss'] = np.mean(g_loss_batch)

                    #batches_done += self.n_critic
            postfix['discriminator_loss'] = np.mean(disc_loss_batch)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer_disc is None else 'Train'
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):


        device = model.device
        optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=self.config.lr,betas=(self.config.b1, self.config.b2))
        optimizer_gen = optim.Adam(self.generator.parameters(), lr=self.config.lr,betas=(self.config.b1, self.config.b2))
        scheduler_disc = optim.lr_scheduler.StepLR(optimizer_disc,
                                              self.config.step_size,
                                              self.config.gamma)
        scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen,
                                                   self.config.step_size,
                                                   self.config.gamma)

        for epoch in range(self.config.train_epochs):
            scheduler_disc.step()
            scheduler_gen.step()

            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))

            postfix = self._train_epoch(model, tqdm_data, optimizer_disc, optimizer_gen)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model,  tqdm_data)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch)
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


    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        latent_train = np.array(model.encode_smiles(train_data, encoder=self.config.heteroencoder_version))
        latent_train = latent_train.reshape(latent_train.shape[0], 512)


        if val_data is not None:
            latent_val = np.array(model.encode_smiles(val_data))
            latent_val = latent_val.reshape(latent_val.shape[0], 512)

        train_loader = self.get_dataloader(model, LatentMolsDataset(latent_train), shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, LatentMolsDataset(latent_val), shuffle=False
        )

        self._train(model, train_loader, val_loader, logger)
        return model