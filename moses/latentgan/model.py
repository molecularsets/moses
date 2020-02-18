import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.autograd as autograd
from rdkit import Chem


class LatentGAN(nn.Module):
    def __init__(self, vocabulary, config):
        super(LatentGAN, self).__init__()
        self.vocabulary = vocabulary
        self.Generator = Generator(
            data_shape=(1, config.latent_vector_dim))
        self.model_version = config.heteroencoder_version
        self.Discriminator = Discriminator(
            data_shape=(1, config.latent_vector_dim))
        self.sample_decoder = None
        self.model_loaded = False
        self.new_batch_size = 256
        # init params
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.Discriminator.cuda()
            self.Generator.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, n_batch):
        out = self.sample(n_batch)
        return out

    def encode_smiles(self, smiles_in, encoder=None):

        model = load_model(model_version=encoder)

        # MUST convert SMILES to binary mols for the model to accept them
        # (it re-converts them to SMILES internally)
        mols_in = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles))
                   for smiles in smiles_in]
        latent = model.transform(model.vectorize(mols_in))

        return latent.tolist()

    def compute_gradient_penalty(self, real_samples,
                                 fake_samples, discriminator):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples +
                        ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
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

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(self, n_batch, max_length=100):
        if not self.model_loaded:
            # Checking for first batch of model to only load model once
            print('Heteroencoder for Sampling Loaded')
            self.sample_decoder = load_model(model_version=self.model_version)
            # load generator

            self.Gen = self.Generator
            self.Gen.eval()

            self.D = self.Discriminator
            torch.no_grad()
            cuda = True if torch.cuda.is_available() else False
            if cuda:
                self.Gen.cuda()
                self.D.cuda()
            self.S = Sampler(generator=self.Gen)
            self.model_loaded = True

            if n_batch <= 256:
                print('Batch size of {} detected. Decoding '
                      'performs poorly when Batch size != 256. \
                 Setting batch size to 256'.format(n_batch))
        # Sampling performs very poorly on default sampling batch parameters.
        #  This takes care of the default scenario.
        if n_batch == 32:
            n_batch = 256

        latent = self.S.sample(n_batch)
        latent = latent.detach().cpu().numpy()

        if self.new_batch_size != n_batch:
            # The batch decoder creates a new instance of the decoder
            # every time a new batch size is given, e.g. for the
            # final batch of the generation.
            self.new_batch_size = n_batch
            self.sample_decoder.batch_input_length = self.new_batch_size
        lat = latent

        sys.stdout.flush()

        smi, _ = self.sample_decoder.predict_batch(lat, temp=0)
        return smi


def load_model(model_version=None):
    from ddc_pub import ddc_v3 as ddc

    # Import model
    currentDirectory = os.getcwd()

    if model_version == 'chembl':
        model_name = 'chembl_pretrained'
    elif model_version == 'moses':
        model_name = 'moses_pretrained'
    elif model_version == 'new':
        model_name = 'new_model'
    else:
        print('No predefined model of that name found. '
              'using the default pre-trained MOSES heteroencoder')
        model_name = 'moses_pretrained'

    path = '{}/moses/latentgan/heteroencoder_models/{}' \
        .format(currentDirectory, model_name)
    print("Loading heteroencoder model titled {}".format(model_version))
    print("Path to model file: {}".format(path))
    model = ddc.DDC(model_name=path)
    sys.stdout.flush()

    return model


class LatentMolsDataset(data.Dataset):
    def __init__(self, latent_space_mols):
        self.data = latent_space_mols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Discriminator(nn.Module):
    def __init__(self, data_shape=(1, 512)):
        super(Discriminator, self).__init__()
        self.data_shape = data_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.data_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, mol):
        validity = self.model(mol)
        return validity


class Generator(nn.Module):
    def __init__(self, data_shape=(1, 512), latent_dim=None):
        super(Generator, self).__init__()
        self.data_shape = data_shape

        # latent dim of the generator is one of the hyperparams.
        # by default it is set to the prod of data_shapes
        self.latent_dim = int(np.prod(self.data_shape)) \
            if latent_dim is None else latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.data_shape))),
            # nn.Tanh() # expecting latent vectors to be not normalized
        )

    def forward(self, z):
        out = self.model(z)
        return out


class Sampler(object):
    """
    Sampling the mols the generator.
    All scripts should use this class for sampling.
    """

    def __init__(self, generator: Generator):
        self.G = generator

    def sample(self, n):
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1,
                                                     (n, self.G.latent_dim)))
        # Generate a batch of mols
        return self.G(z)
