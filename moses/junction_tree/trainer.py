import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from multiprocessing import Pool
from collections import OrderedDict

from moses.utils import SmilesDataset
from moses.interfaces import MosesTrainer
from moses.junction_tree.jtnn.vocabulary import JTreeVocab
from moses.junction_tree.jtnn.mol_tree import MolTree

class JTreeTrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config

    def _train_epoch(self, model, tqdm_data, epoch, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix = OrderedDict()
        postfix['word_acc'] = 0
        postfix['topo_acc'] = 0
        postfix['assm_acc'] = 0
        postfix['steo_acc'] = 0
        postfix['kl'] = 0

        kl_w = 0 if epoch < self.config.kl_start else self.config.kl_w

        for i, batch in enumerate(tqdm_data):
            loss, kl_div, wacc, tacc, sacc, dacc = model(batch, kl_w)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['word_acc'] += (wacc * 100 - postfix['word_acc']) / (i + 1)
            postfix['topo_acc'] += (tacc * 100 - postfix['topo_acc']) / (i + 1)
            postfix['assm_acc'] += (sacc * 100 - postfix['assm_acc']) / (i + 1)
            postfix['steo_acc'] += (dacc * 100 - postfix['steo_acc']) / (i + 1)
            postfix['kl'] += (kl_div - postfix['kl']) / (i + 1)

            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        for field, value in postfix.items():
            self.log_file.write(field+' = '+str(value)+'\n')
        self.log_file.write('===\n')
        self.log_file.flush()

    def _train(self, model, train_loader, val_loader=None):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        optimizer = optim.Adam(get_params(), lr=self.config.lr)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            tqdm_data = tqdm(train_loader, desc='Train (epoch #{})'.format(epoch))

            self._train_epoch(model, tqdm_data, epoch, optimizer)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                self._train_epoch(model, tqdm_data, criterion)

            if epoch % self.config.save_frequency == 0:
                model = model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model = model.to(device)

    def get_vocabulary(self, data):
        clusters = set()
        with Pool(self.config.n_jobs) as pool:
            for mol in tqdm(pool.imap(MolTree, data),
                            total=len(data),
                            postfix=['Creating vocab']):
                for c in mol.nodes:
                    clusters.add(c.smiles)
        return JTreeVocab(sorted(list(clusters)))

    def get_dataloader(self, model, data, shuffle=True):
        n_workers = self.config.n_workers
        if n_workers == 1:
            n_workers = 0

        def parse_molecule(smiles):
            mol_tree = MolTree(smiles)
            mol_tree.recover()
            mol_tree.assemble()

            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
                    node.cand_mols.append(node.label_mol)

            return mol_tree

        def collate(smiles):
            return list(smiles)

        return DataLoader(SmilesDataset(data, transform=parse_molecule),
                          batch_size=self.config.n_batch, shuffle=shuffle,
                          num_workers=n_workers, collate_fn=collate,
                          drop_last=True,
                          worker_init_fn=set_torch_seed_to_all_gens if n_workers > 0 else None
                         )

    def fit(self, model, train_data, val_data=None):
        self.log_file = open(self.config.log_file, 'w')
        self.log_file.write(str(self.config)+'\n')
        self.log_file.write(str(model)+'\n')

        # model ??
        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)

        self._train(model, train_loader, val_loader)
        self.log_file.close()
        return model
