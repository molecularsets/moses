import torch.nn as nn
import numpy as np
import torch
from ddc_pub import ddc_v3 as ddc
import os
from rdkit import Chem
import sys
from torch.utils import data
import torch.autograd as autograd


class LatentGAN(nn.Module):
    def __init__(self,vocabulary, config):
        super(LatentGAN, self).__init__()
        self.vocabulary = vocabulary
        self.generator = Generator()
        self.model_version=config.heteroencoder_version
        self.discriminator = Discriminator()
        self.Sampler = Sampler(generator=Generator())
        self.sample = sample
        # init params
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.discriminator.cuda()
            self.generator.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, n_batch):
        out = sample(n_batch)
        return out

    def encode_smiles(self, smiles_in, encoder=None):

        model = load_model(model_version=encoder)

        # Input SMILES
        #smiles_in = np.array(smiles)

        # MUST convert SMILES to binary mols for the model to accept them (it re-converts them to SMILES internally)
        mols_in = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles)) for smiles in smiles_in]
        latent = model.transform(model.vectorize(mols_in))

        return latent.tolist()

    def compute_gradient_penalty(self, real_samples, fake_samples, discriminator):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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



def load_model(model_version=None):
    # Import model
    currentDirectory = os.getcwd()
    DEFAULT_MODEL_VERSION = 'new_chembl_model'

    if model_version == 'chembl':
        model_name = 'new_chembl_model'

    elif model_version == 'moses':
        model_name = '16888509--1000--0.0927--0.0000010'
    else:
        model_name = DEFAULT_MODEL_VERSION
    #path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_name)
    path = '{}/moses/latentgan/heteroencoder_models/{}'.format(currentDirectory,model_name)
    print("Loading heteroencoder model titled {}".format(model_version))
    print("Path to model file: {}".format(path))
    model = ddc.DDC(model_name=path)

    return model


def sample(self, n_batch):

    sys.stdout.flush()
    #torch.no_grad()
    model = load_model()
    # load generator
    G = self.generator
    #G.eval()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        G.cuda()
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    S = Sampler(generator=G)
    print('Sampling model')
    sys.stdout.flush()
    latent = S.sample(n_batch)

    latent = latent.detach().cpu().numpy().tolist()
    smiles = []
    batch_size = 256  # decoding batch size
    n = len(latent)

    for indx in range(0, n // batch_size):
        lat = np.array(latent[(indx) * batch_size:(indx + 1) * batch_size])
        smi, _ = model.predict_batch(lat, temp=0)
        smiles.append(smi)
    return smiles


class LatentMolsDataset(data.Dataset):
    def __init__(self, latent_space_mols):
        self.data = latent_space_mols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Discriminator(nn.Module):
    def __init__(self, data_shape=(1,512)):
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
    def __init__(self, data_shape=(1,512), latent_dim=None):
        super(Generator, self).__init__()
        self.data_shape = data_shape

        # latent dim of the generator is one of the hyperparams.
        # by default it is set to the prod of data_shapes
        self.latent_dim = int(np.prod(self.data_shape)) if latent_dim is None else latent_dim

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
        cuda = True if torch.cuda.is_available() else False
        self.set_generator(generator,cuda)

    def set_generator(self, generator,cuda):
        self.G = generator.cuda() if cuda else generator

    def sample(self, n):
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (n, self.G.latent_dim)))
        # Generate a batch of mols
        return self.G(z)




