# Junction Tree Variational Autoencoder

Junction Tree VAE (JT-VAE) [1] is one of the first deep generative models that explicitly made use of a graph representation of molecules. The JT-VAE generates molecules in two phases by exploiting valid subgraphs as components. In the first phase, it generates a tree-structured object (a junction tree) whose role is to represent the scaffold of subgraph components and their coarse relative arrangements. The components are valid chemical substructures automatically extracted from the training set using tree decomposition and are then used as building blocks. In the second phase, the subgraphs (nodes of the tree) are assembled together into a coherent molecular graph. 

Training is done with a batch size of 40, with the Adam optimizer utilizing a learning rate of 0.001 for 5 epochs. Hyperparameters are based on the original paper: hidden layer dimension 450, a latent space of dimension 56 and the message passing graph of depth 3. The KL term was taken into consideration starting from the second epoch, allowing one epoch for just training the VAE part. The KL term weight was 0.0005. 

## Links

[1] [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364) 

