from char_vocab_utils import compute_string_to_int, merge_vocab


def main():
    import os
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--vocab-path",
        nargs="+",
        dest="vocab_path_list",
        help="path to the vocab(s) to use to featurize the smiles data. if more than one vocab path is given, the vocabs"
        "are merged and the result is used as the vocab to featurize with",
    )
    parser.add_argument("--smiles-path", help="path to csv of smiles strings")
    parser.add_argument(
        "--smiles-col", help="column name that contains smiles strings", default=None
    )
    parser.add_argument(
        "--smiles-sep",
        help="delimiter used to seperate smiles strings, default is set to pandas default for csv",
        default=",",
    )
    parser.add_argument(
        "--add-bos",
        help="add the begin of string character to smiles data",
        action="store_true",
    )
    parser.add_argument(
        "--add-eos",
        help="add the end of string character to smiles data",
        action="store_true",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="number of processes to use for parallel computations",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="if specified, saves the data into a seperate train/val/test split, where"
        "test set will be test-size %% of the full data, val is then selected from remaining train data"
        "using val-size %% of the train data",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="%% of the training data to hold out as validation or dev set",
    )
    parser.add_argument("--split-dataset", action="store_true")
    parser.add_argument(
        "--output-dir", help="path to output directory to store vocab and numpy arrays"
    )
    args = parser.parse_args()

    # read the smiles strings from the csv path, modin uses multiprocessing to do this more quickly
    import modin.pandas as pd

    if args.smiles_col is None:
        smiles_df = pd.read_csv(args.smiles_path, header=None, sep=args.smiles_sep)
        smiles_list = smiles_df[0].values

    else:
        smiles_df = pd.read_csv(args.smiles_path, sep=args.smiles_sep)
        smiles_list = smiles_df[args.smiles_col].values

    # if output directory does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # extract the vocab
    print("reading vocab...")

    if len(args.vocab_path_list) > 1:
        print("more than one vocab was specified...merging vocabs...")
        vocab = merge_vocab(args.vocab_path_list)

    else:
        vocab = torch.load(args.vocab_path_list[0])

    # compute the integer representation of the smiles data
    print("extracting dataset...")
    data = compute_string_to_int(
        smiles_list,
        vocab,
        n_jobs=args.n_jobs,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )
    np.save(args.output_dir + "/full_data.npy", data)

    if args.split_dataset:
        # compute the splits for train/test using the full data
        train_data, test_data = train_test_split(data, test_size=args.test_size)
        # compute the splits for train/val using the remaining data
        train_data, val_data = train_test_split(train_data, test_size=args.val_size)

        np.save(args.output_dir + "/train.npy", train_data)
        np.save(args.output_dir + "/val.npy", val_data)
        np.save(args.output_dir + "/test.npy", test_data)


if __name__ == "__main__":
    main()
