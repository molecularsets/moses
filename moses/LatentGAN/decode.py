from rdkit import Chem
import json
import argparse
import os, sys
import numpy as np
from autoencoder import autoencoder


def decode(latent_mols_file, output_smiles_file_path=None, message=''):
    print("BEGIN")
    print("Message: %s " % message)
    sys.stdout.flush()
    model = autoencoder.load_model()

    if output_smiles_file_path is None:
        output_smiles_file_path = os.path.join(os.path.dirname(latent_mols_file), 'decoded_smiles.smi')

    with open(latent_mols_file, 'r') as f:
        latent = json.load(f)

    invalids = 0
    batch_size = 128    # decoding batch size
    n = len(latent)

    with open(output_smiles_file_path, 'w') as smiles_file:
        for indx in range(0, n // batch_size):
            lat = np.array(latent[(indx) * 128:(indx + 1) * 128])
            if indx % 10 == 0:
                print("[%d/%d] [Invalids: %s]" % (indx, n // batch_size + 1, invalids))
                sys.stdout.flush()
                smiles_file.flush()
            # obs = np.squeeze(lat, 1)
            smiles, _ = model.predict_batch(lat, temp=0)

            for mol in smiles:
                mol = Chem.MolFromSmiles(mol)
                if mol:
                    smile_string = Chem.MolToSmiles(mol)
                    smiles_file.write(smile_string + '\n')
                else:
                    invalids += 1

    print("Total: [%d] Fraction Valid: [0.%d]" % (n, (n - invalids) / n * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--latent_mols_file", "-l", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output_smiles_file_path", "-o", help="Prefix to the folder to save output smiles.", type=str)
    parser.add_argument("--message", "-m", help="Message printed before training.", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    decode(**args)
