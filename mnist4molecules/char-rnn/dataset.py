from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        super(SmilesDataset, self).__init__()

        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx]
