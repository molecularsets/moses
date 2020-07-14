import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np 
import glob

class CharRNN(nn.Module):

    def __init__(self, vocabulary, config):
        super(CharRNN, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = config.hidden
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.vocab_size, padding_idx=vocabulary.pad)
        self.lstm_layer = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, hiddens

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,
                              device=self.device if device == 'model' else device)

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def load_lbann_weights(self,weights_dir,epoch_count=None):

       if epoch_count is None:
         epoch_count = '*'
 
        with torch.no_grad():
          #Load Embedding weights
          emb_weights = np.loadtxt(glob.glob(weights_dir+"*.epoch."+str(epoch_count)+"-emb_matrix-Weights.txt")[0])
          self.embedding_layer.weight.data.copy_(torch.from_numpy(np.transpose(emb_weights))) 

          #Load LSTM weights/biases
          param_idx = ['_ih_matrix','_hh_matrix','_ih_bias', '_hh_bias'] 
          for l in range(self.num_layers):
            for idx, val in enumerate(param_idx):
              param_tensor = np.loadtxt(glob.glob(weights_dir+"*.epoch."+str(epoch_count)+"*-gru"+str(l+1)+val+"-Weights.txt")[0])
              self.lstm_layer.all_weights[l][idx].copy_(torch.from_numpy(param_tensor))

          #Load Linear layer weights/biases
          linear_layer_weights = np.loadtxt(glob.glob(weights_dir+"*.epoch."+str(epoch_count)+"*-fcmodule"+str(2*self.num_layers+1)+"_matrix-Weights.txt")[0])
          self.linear_layer.weight.data.copy_(torch.from_numpy(linear_layer_weights))
          linear_layer_bias = np.loadtxt(glob.glob(weights_dir+"*.epoch."+str(epoch_count)+"*-fcmodule"+str(2*self.num_layers+1)+"_bias-Weights.txt")[0])
          self.linear_layer.bias.data.copy_(torch.from_numpy(linear_layer_bias))

          print("DONE loading LBANN weights ")
        return

    def sample(self, n_batch, max_length=100):
        with torch.no_grad():
            starts = [torch.tensor([self.vocabulary.bos], dtype=torch.long, device=self.device)
                      for _ in range(n_batch)]

            starts = torch.tensor(starts, dtype=torch.long, device=self.device).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.pad, dtype=torch.long, device=self.device).repeat(max_length + 2)
                for _ in range(n_batch)]

            for i in range(n_batch):
                new_smiles_list[i][0] = self.vocabulary.bos

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor([1 for _ in range(n_batch)], dtype=torch.long, device=self.device)
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            for i in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)

                # probabilities
                probs = [F.softmax(o, dim=-1) for o in output]

                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]

                for j, top in enumerate(ind_tops):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == self.vocabulary.eos:
                            end_smiles_list[j] = True

                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(ind_tops, dtype=torch.long, device=self.device).unsqueeze(1)

            new_smiles_list = [new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)]
            return [self.tensor2string(t) for t in new_smiles_list]
