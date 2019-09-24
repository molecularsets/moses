LatentGAN
=========
[LatentGAN](../../images/LatentGAN.png)
LatentGAN [1] with heteroencoder trained on ChEMBL 25 [2], which encodes SMILES strings into latent vector representations of size 512. A Wasserstein Generative Adversarial network with Gradient Penalty [3] is then trained to generate latent vectors resembling that of the training set, which are then decoded using the heteroencoder. This model uses the heteroencoder available as a package from [4], which is to be available soon. The heteroencoder further requires [This package](https://github.com/EBjerrum/molvecgen) to run properly on the ChEMBL model. 


## Links

[1] [A De Novo Molecular Generation Method Using Latent Vector Based Generative Adversarial Network](https://chemrxiv.org/articles/A_De_Novo_Molecular_Generation_Method_Using_Latent_Vector_Based_Generative_Adversarial_Network/8299544)

[2] [ChEMBL](https://www.ebi.ac.uk/chembl/)

[3] [Improved training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

[4] [Deep-Drug-Coder](https://github.com/pcko1/Deep-Drug-Coder)

