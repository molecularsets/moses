import json, argparse, sys
from decode import decode
import torch
from models.Generator import Generator
from src.Sampler import Sampler


def sample(generator_path, output_sampled_latent_file, number_samples=50000, message='samling the generator',
           decode_sampled=False, output_decoded_smiles_file=''):
    print(message)
    sys.stdout.flush()
    torch.no_grad()

    # load generator
    G = Generator.load(generator_path)
    G.eval()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        G.cuda()
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    S = Sampler(generator=G)
    print('Sampling model')
    sys.stdout.flush()
    latent = S.sample(number_samples)

    latent = latent.detach().cpu().numpy().tolist()

    with open(output_sampled_latent_file, 'w') as json_file:
        # array_fake_mols = fake_mols.data
        json.dump(latent, json_file)

    print('Sampling finished')
    sys.stdout.flush()
    del latent, json_file, G, S

    # decoding sampled mols
    if decode_sampled:
        print('Decoding sampled mols')
        sys.stdout.flush()
        decode(output_sampled_latent_file, output_decoded_smiles_file, message='Decoding mol. Call from sample script.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--generator-path", "-l", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output-sampled-latent_file", "-olf", help="The path to a sampled latents file.", type=str,
                        required=True)
    parser.add_argument("--number-samples", "-n", help="The amount of smiles to sample", type=int)
    parser.add_argument("--message", "-m", help="Message printed before training.", type=str)
    parser.add_argument("--decode-sampled", "-d", help="If the decoding should be done", type=bool)
    parser.add_argument("--output-decoded-smiles-file", "-odsf", help="The location of the sampled smiles.", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    if args['decode_sampled'] == True:
        if 'output_decoded_smiles_file' not in args:
            raise Exception('Argument \'output_decoded_smiles_file\' should be set if \'decode\'=True')
    sample(**args)
