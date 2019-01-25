# Variational Autoencoder

![VAE_AAE](../../images/VAE_AAE.png)

Variational autoencoder (VAE) [1, 2, 3] is a framework for training two neural networks—an encoder and a decoder—to learn a mapping from high-dimensional data representation into a lower-dimensional space and back. The lower-dimensional space is called the latent space, which is often a continuous vector space with normally distributed latent representation. VAE parameters are optimized to encode and decode data by minimizing the reconstruction loss while also minimizing a KL-divergence term arising from the variational approximation that can loosely be interpreted as a regularization term.  Since molecules are discrete objects, properly trained VAE defines an invertible continuous representation of a molecule.

We combine aspects from both implementations in MOSES. Utilizing a bidirectional77 Gated Recurrent Unit (GRU) with a linear output layer as an encoder. The decoder is a 3-layer GRU RNN of 512 hidden dimensions with intermediate dropout layers with dropout probability 0.2. Training is done with a batch size of 128, utilizing a gradient clipping of 50, KL-term weight of 1, and optimized with Adam with a learning rate of 0.0003 for 50 epochs.


## Links

[1] [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572)

[2] [The cornucopia of meaningful leads: Applying deep adversarial autoencoders for new molecule development in oncology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5355231/)

[3] [Application of generative autoencoder in de novo molecular design](https://arxiv.org/abs/1711.07839)
