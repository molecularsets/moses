import torch
import torch.optim as optim

from tqdm import tqdm

from moses.utils import mapper, Logger
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

        postfix = { 'word_acc' : 0,
                    'topo_acc' : 0,
                    'assm_acc' : 0,
                    'steo_acc' : 0,
                    'kl' : 0,}

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
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        optimizer = optim.Adam(get_params(), lr=self.config.lr)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            tqdm_data = tqdm(train_loader, desc='Train (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, tqdm_data, epoch, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if epoch % self.config.save_frequency == 0:
                model = model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model = model.to(device)

    def get_vocabulary(self, data):
        clusters = set()
        for mol in tqdm(mapper(self.config.n_jobs)(MolTree, data),
                        total=len(data),
                        postfix=['Creating vocab']):
            for c in mol.nodes:
                clusters.add(c.smiles)
        return JTreeVocab(sorted(list(clusters)))

    def get_collate_fn(self, model):
        def collate(smiles):
            mol_trees = []
            for s in smiles:
                mol_tree = MolTree(s)
                mol_tree.recover()
                mol_tree.assemble()

                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)

                mol_trees.append(mol_tree)

            return mol_trees

        return collate

    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)

        self._train(model, train_loader, val_loader, logger)

        return model
