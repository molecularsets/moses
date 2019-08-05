import pickle
from models.Discriminator import Discriminator
from models.Generator import Generator
from datasets.LatentMolsDataset import LatentMolsDataset
from src.Sampler import Sampler
from decode import decode
import os
import torch
import torch.autograd as autograd
import numpy as np
import json
import time
import sys


class TrainModelRunner:
    # Loss weight for gradient penalty
    lambda_gp = 10

    def __init__(self, input_data_path, output_model_folder, decode_mols_save_path='', n_epochs=200, starting_epoch=1,
                 batch_size=64, lr=0.0002, b1=0.5, b2=0.999,  n_critic=5,
                 save_interval=100, sample_after_training=100, message=""):
        self.message = message

        # init params
        self.input_data_path = input_data_path
        self.output_model_folder = output_model_folder
        self.n_epochs = n_epochs
        self.starting_epoch = starting_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_critic = n_critic
        self.save_interval = save_interval
        self.sample_after_training = sample_after_training
        self.decode_mols_save_path = decode_mols_save_path

        # initialize dataloader
        json_smiles = open(self.input_data_path, "r")
        latent_space_mols = np.array(json.load(json_smiles))
        latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)

        self.dataloader = torch.utils.data.DataLoader(LatentMolsDataset(latent_space_mols), shuffle=True,
                                                      batch_size=self.batch_size, drop_last=True)

        # load discriminator
        discriminator_name = 'discriminator.txt' if self.starting_epoch == 1 else str(
            self.starting_epoch - 1) + '_discriminator.txt'
        discriminator_path = os.path.join(output_model_folder, discriminator_name)
        self.D = Discriminator.load(discriminator_path)

        # load generator
        generator_name = 'generator.txt' if self.starting_epoch == 1 else str(
            self.starting_epoch - 1) + '_generator.txt'
        generator_path = os.path.join(output_model_folder, generator_name)
        self.G = Generator.load(generator_path)

        # initialize sampler
        self.Sampler = Sampler(self.G)

        # initialize optimizer
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Tensor
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.G.cuda()
            self.D.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def run(self):

        print("Run began.")
        print("Message: %s" % self.message)
        sys.stdout.flush()

        batches_done = 0
        disc_loss_log = []
        g_loss_log = []

        for epoch in range(self.starting_epoch, self.n_epochs + self.starting_epoch):
            disc_loss_per_batch = []
            g_loss_log_per_batch = []
            for i, real_mols in enumerate(self.dataloader):

                # Configure input
                real_mols = real_mols.type(self.Tensor)
                # real_mols = np.squeeze(real_mols, axis=1)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Generate a batch of mols from noise
                fake_mols = self.Sampler.sample(real_mols.shape[0])

                # Real mols
                real_validity = self.D(real_mols)
                # Fake mols
                fake_validity = self.D(fake_mols)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(real_mols.data, fake_mols.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
                disc_loss_per_batch.append(d_loss.item())

                d_loss.backward()
                self.optimizer_D.step()
                self.optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of mols
                    fake_mols = self.Sampler.sample(real_mols.shape[0])
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.D(fake_mols)
                    g_loss = -torch.mean(fake_validity)
                    g_loss_log_per_batch.append(g_loss.item())

                    g_loss.backward()
                    self.optimizer_G.step()

                    batches_done += self.n_critic

                # If last batch in the set
                if i == len(self.dataloader) - 1:
                    if epoch % self.save_interval == 0:
                        generator_save_path = os.path.join(self.output_model_folder,
                                                           str(epoch) + '_generator.txt')
                        discriminator_save_path = os.path.join(self.output_model_folder,
                                                               str(epoch) + '_discriminator.txt')
                        self.G.save(generator_save_path)
                        self.D.save(discriminator_save_path)

                    disc_loss_log.append([time.time(), epoch, np.mean(disc_loss_per_batch)])
                    g_loss_log.append([time.time(), epoch, np.mean(g_loss_log_per_batch)])

                    # Print and log
                    print(
                        "[Epoch %d/%d]  [Disc loss: %f] [Gen loss: %f] "
                        % (epoch, self.n_epochs + self.starting_epoch, disc_loss_log[-1][2], g_loss_log[-1][2])
                    )
                    sys.stdout.flush()

        # log the losses
        with open(os.path.join(self.output_model_folder, 'disc_loss.json'), 'w') as json_file:
            json.dump(disc_loss_log, json_file)
        with open(os.path.join(self.output_model_folder, 'gen_loss.json'), 'w') as json_file:
            json.dump(g_loss_log, json_file)

        # Sampling after training
        if self.sample_after_training > 0:
            # sampling mode
            torch.no_grad()
            self.G.eval()

            S = Sampler(generator=self.G)
            latent = S.sample(self.sample_after_training)
            latent = latent.detach().cpu().numpy().tolist()

            sampled_mols_save_path = os.path.join(self.output_model_folder, 'sampled.json')
            with open(sampled_mols_save_path, 'w') as json_file:
                json.dump(latent, json_file)

            # decoding sampled mols
            decode(sampled_mols_save_path, self.decode_mols_save_path)

        return 0

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        fake = self.Tensor(real_samples.shape[0], 1).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
