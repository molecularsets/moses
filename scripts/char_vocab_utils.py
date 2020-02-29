import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from moses.utils import CharVocab
from functools import partial


def compute_vocab_job(smiles):
    return set(smiles)

def compute_vocab(smiles_list, n_jobs=mp.cpu_count()):
    """
        simple function that can be used to create a vocabulary for an arbitrary set of smiles strings

        smiles_list: list of smiles strings
    """
    # extract all unique characters in smiles_list 
    #char_set = set.union(*[set(x) for x in smiles_list])

    with mp.Pool(n_jobs) as pool:
        result = list(tqdm(pool.imap_unordered(compute_vocab_job, smiles_list), total=len(smiles_list)))
        char_set = set.union(*result)


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


def merge_vocab(*args):
    """
        helper function to merge multiple vocab objects...helpful for cases that may require the processing of more data than 
        is able to held in memory or for getting a common vocab to use to merge multiple disjoint datasets, etc..

        *args: a list of an arbitrary number of vocab objects
    """

    # use this list to filter out 'characters' that we don't need to make the new dataset
    ignore_char_list = ['<bos>', '<eos>', '<pad>', '<unk>']
    merged_char_set = set()

    for vocab_path in args:
        vocab = torch.load(vocab_path)
        vocab_chars_set = set([x for x in vocab.c2i.keys() if x not in ignore_char_list]) 
        merged_char_set.update(vocab_chars_set)

    return CharVocab(merged_char_set)

