import numpy as np
import multiprocessing as mp
from moses.utils import CharVocab
from functools import partial


def compute_vocab(smiles_list):
    """
        simple function that can be used to create a vocabulary for an arbitrary set of smiles strings

        smiles_list: list of smiles strings
    """
    # extract all unique characters in smiles_list 
    char_set = set.union(*[set(x) for x in smiles_list])
    # create the vocab
    vocab = CharVocab(char_set) 

    return vocab


def compute_string_to_int(smiles_list, vocab, n_jobs=mp.cpu_count(), add_bos=False, add_eos=False):
     
    from tqdm import tqdm

    """
        simple function that is used to extract a set of integer representations of a smiles dataset given
            the provided vocab. can compute in parallel by using n_jobs > 1.
        smiles_list: list of smiles strings
        n_jobs: number of processes to use for parallel computation
        add_bos: add the begin of string integer 
        add_eos: add the end of string integer
    """
    string2ids = partial(vocab.string2ids, add_bos=add_bos, add_eos=add_eos)
    with mp.Pool(n_jobs) as pool:
        result = list(tqdm(pool.imap_unordered(string2ids, smiles_list), total=len(smiles_list)))
        data = np.asarray([np.asarray(x, dtype=int) for x in result])

        return data


def main():
    import os
    import torch
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--smiles-path", help="path to csv of smiles strings")
    parser.add_argument("--add-bos", help="add the begin of string character to smiles data", action="store_true")
    parser.add_argument("--add-eos", help="add the end of string character to smiles data", action="store_true")
    parser.add_argument("--n-jobs", type=int, help="number of processes to use for parallel computations")
    parser.add_argument("--test-size", type=float, default=0.2, help="if specified, saves the data into a seperate train/val/test split, where" \
                                            "test set will be test-size %% of the full data, val is then selected from remaining train data" \
                                            "using val-size %% of the train data")
    parser.add_argument("--val-size", type=float, default=0.1, help="%% of the training data to hold out as validation or dev set")
    parser.add_argument("--output-dir", help="path to output directory to store vocab and numpy arrays")
    args = parser.parse_args()

    # read the smiles strings from the csv path 
    smiles_df = pd.read_csv(args.smiles_path, header=None) 
    smiles_list = smiles_df[0].values.tolist() 
  
    # extract the vocab  
    vocab = compute_vocab(smiles_list)

    # compute the integer representation of the smiles data
    data = compute_string_to_int(smiles_list, vocab, n_jobs=args.n_jobs, add_bos=args.add_bos, add_eos=args.add_eos)
 
    # compute the splits for train/test using the full data
    train_data, test_data = train_test_split(data, test_size=args.test_size)
    # compute the splits for train/val using the remaining data
    train_data, val_data = train_test_split(train_data, test_size=args.val_size)

    # if output directory does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save the vocab and the data splits to output directory
    torch.save(vocab, args.output_dir+"/vocab.pt")
    np.save(args.output_dir+"/train.npy", train_data)
    np.save(args.output_dir+"/val.npy", val_data)
    np.save(args.output_dir+"/test.npy", test_data)

if __name__ == "__main__":
    main()

