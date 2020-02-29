from char_vocab_utils import compute_vocab

def main():
    import os
    import torch 
    from sklearn.model_selection import train_test_split
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--smiles-path", help="path to csv of smiles strings")
    parser.add_argument("--smiles-col", help="column name that contains smiles strings", default=None)
    parser.add_argument("--smiles-sep", help="delimiter used to seperate smiles strings, default is set to pandas default for csv", default=",")
    parser.add_argument("--n-jobs", type=int, help="number of processes to use for parallel computations")
    
    parser.add_argument("--output-dir", help="path to output directory to store vocab and numpy arrays") 
    args = parser.parse_args()

    # read the smiles strings from the csv path 
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
    print("extracting the vocab...")
    vocab = compute_vocab(smiles_list, n_jobs=args.n_jobs)
    torch.save(vocab, args.output_dir+"/vocab.pt")
  

if __name__ == "__main__":
    main()

