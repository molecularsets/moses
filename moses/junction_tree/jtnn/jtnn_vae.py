import copy

import rdkit.Chem as Chem
import torch
import torch.nn as nn
from collections import OrderedDict

from .chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, decode_stereo
from .jtmpn import JTMPN
from .jtnn_dec import JTNNDecoder
from .jtnn_enc import JTNNEncoder
from .mol_tree import MolTree
from .mpn import MPN, mol2graph


def set_batch_node_id(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


class JTNNVAE(nn.Module):

    def __init__(self, vocab, config):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = config.hidden
        self.latent_size = config.latent
        self.depth = config.depth

        self.embedding = nn.Embedding(vocab.size(), self.hidden_size)
        self.jtnn = JTNNEncoder(vocab, self.hidden_size, self.embedding)
        self.jtmpn = JTMPN(self.hidden_size, self.depth)
        self.mpn = MPN(self.hidden_size, self.depth)
        self.decoder = JTNNDecoder(vocab, self.hidden_size,
                                   self.latent_size // 2, self.embedding)

        self.T_mean = nn.Linear(self.hidden_size, self.latent_size // 2)
        self.T_var = nn.Linear(self.hidden_size, self.latent_size // 2)
        self.G_mean = nn.Linear(self.hidden_size, self.latent_size // 2)
        self.G_var = nn.Linear(self.hidden_size, self.latent_size // 2)

        self.assm_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stereo_loss = nn.CrossEntropyLoss(reduction='sum')

        # Xavier parameters initialization.
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, mol_batch):
        device = self.device

        set_batch_node_id(mol_batch, self.vocab)
        root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
        tree_mess, tree_vec = self.jtnn(root_batch)

        smiles_batch = [mol_tree.smiles for mol_tree in mol_batch]
        mol_vec = self.mpn(mol2graph(smiles_batch, device=device))
        return tree_mess, tree_vec, mol_vec

    def forward(self, mol_batch, beta=0):
        device = self.device

        batch_size = len(mol_batch)

        tree_mess, tree_vec, mol_vec = self.encode(mol_batch)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        z_mean = torch.cat([tree_mean, mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

        epsilon = torch.randn(batch_size, self.latent_size // 2, device=device)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = torch.randn(batch_size, self.latent_size // 2, device=device)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_batch, tree_vec)
        assm_loss, assm_acc = self.assm(mol_batch, mol_vec, tree_mess)
        stereo_loss, stereo_acc = self.stereo(mol_batch, mol_vec)

        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        return loss, kl_loss.item(), word_acc, topo_acc, assm_acc, stereo_acc

    def assm(self, mol_batch, mol_vec, tree_mess):
        device = self.device

        cands = []
        batch_idx = []
        for i, mol_tree in enumerate(mol_batch):
            for node in mol_tree.nodes:
                if node.is_leaf or len(node.cands) == 1:
                    continue
                cands.extend([(cand, mol_tree.nodes, node) for cand in node.cand_mols])
                batch_idx.extend([i] * len(node.cands))

        cand_vec = self.jtmpn(cands, tree_mess)
        cand_vec = self.G_mean(cand_vec)

        batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)

        mol_vec = mol_vec.index_select(0, batch_idx)

        mol_vec = mol_vec.view(-1, 1, self.latent_size // 2)
        cand_vec = cand_vec.view(-1, self.latent_size // 2, 1)
        scores = torch.bmm(mol_vec, cand_vec).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = torch.tensor([label], dtype=torch.long, device=device)
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        all_loss = torch.stack(all_loss, dim=0).to(device=device).sum() / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def stereo(self, mol_batch, mol_vec):
        device = self.device
        stereo_cands, batch_idx = [], []
        labels = []
        for i, mol_tree in enumerate(mol_batch):
            cands = mol_tree.stereo_cands
            if len(cands) == 1:
                continue
            if mol_tree.smiles3D not in cands:
                cands.append(mol_tree.smiles3D)
            stereo_cands.extend(cands)
            batch_idx.extend([i] * len(cands))
            labels.append((cands.index(mol_tree.smiles3D), len(cands)))

        if len(labels) == 0:
            return torch.tensor(0, dtype=torch.float32, device=device), 1.0  # TODO

        batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
        stereo_cands = self.mpn(mol2graph(stereo_cands, device=device))
        stereo_cands = self.G_mean(stereo_cands)
        stereo_labels = mol_vec.index_select(0, batch_idx)
        scores = torch.nn.CosineSimilarity()(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores.data[label] >= cur_scores.max().item():
                acc += 1
            label = torch.tensor([label], dtype=torch.long, device=device)
            all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label))
            st += le

        all_loss = torch.stack(all_loss, dim=0).to(device=device).sum() / len(mol_batch)
        return all_loss, acc * 1.0 / len(labels)

    def reconstruct(self, smiles, prob_decode=False):
        device = self.device
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        epsilon = torch.randn(1, self.latent_size // 2).to(device=device)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = torch.randn(1, self.latent_size // 2).to(device=device)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        return self.decode(tree_vec, mol_vec, prob_decode)

    def sample(self, n_batch, max_len=100):
        samples = []
        while len(samples) < n_batch:
            sample = self.sample_prior(prob_decode=True)
            if len(sample) <= max_len:
                samples.append(sample)
        return samples

    def sample_prior(self, prob_decode=False):
        device = self.device
        tree_vec = torch.randn(1, self.latent_size // 2, device=device)
        mol_vec = torch.randn(1, self.latent_size // 2, device=device)
        mol = self.decode(tree_vec, mol_vec, prob_decode)
        if mol is None:
            return self.sample_prior(prob_decode=True)  # TODO
        else:
            return mol

    def decode(self, tree_vec, mol_vec, prob_decode):
        device = self.device
        pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)

        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        tree_mess = self.jtnn([pred_root])[0]

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [OrderedDict()] + [OrderedDict() for _ in pred_nodes]
        global_amap[1] = OrderedDict([(atom.GetIdx(), atom.GetIdx()) for atom in cur_mol.GetAtoms()])

        cur_mol = self.dfs_assemble(tree_mess, mol_vec, pred_nodes, cur_mol, global_amap, [], pred_root, None,
                                    prob_decode)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None:
            return None

        smiles2D = Chem.MolToSmiles(cur_mol)
        stereo_cands = decode_stereo(smiles2D)
        if len(stereo_cands) == 1:
            return stereo_cands[0]
        stereo_vecs = self.mpn(mol2graph(stereo_cands, device=device))
        stereo_vecs = self.G_mean(stereo_vecs)
        scores = nn.CosineSimilarity()(stereo_vecs, mol_vec)
        _, max_id = scores.max(dim=0)
        return stereo_cands[max_id.item()]

    def dfs_assemble(self, tree_mess, mol_vec, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node,
                     prob_decode):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None
        cand_smiles, cand_mols, cand_amap = list(zip(*cands))

        cands = [(candmol, all_nodes, cur_node)
                 for candmol in cand_mols if len(candmol.GetBonds()) > 0]
        if len(cands) == 0:
            return None
        cand_vecs = self.jtmpn(cands, tree_mess)
        cand_vecs = self.G_mean(cand_vecs)
        mol_vec = mol_vec.squeeze()
        scores = torch.mv(cand_vecs, mol_vec) * 20

        if prob_decode:
            probs = nn.Softmax(dim=-1)(scores.view(1, -1)).squeeze() + 1e-5  # prevent prob = 0
            if probs.ndimension() == 0:
                probs = probs[None,]
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item() if cand_idx.numel() > 1 else cand_idx.item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                cur_mol = self.dfs_assemble(tree_mess, mol_vec, all_nodes, cur_mol, new_global_amap, pred_amap,
                                            nei_node, cur_node, prob_decode)
                if cur_mol is None:
                    result = False
                    break
            if result:
                return cur_mol

        return None
