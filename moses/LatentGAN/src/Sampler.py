from models.Generator import Generator
import numpy as np
import torch


class Sampler(object):
    """
    Sampling the mols the generator.
    All scripts should use this class for sampling.
    """

    def __init__(self, generator: Generator):
        self.set_generator(generator)

    def set_generator(self, generator):
        self.G = generator

    def sample(self, n):
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (n, self.G.latent_dim)))
        # Generate a batch of mols
        return self.G(z)
