import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class CharRNN(nn.Module):

    @staticmethod
    def _device(model):
        return next(model.parameters()).device

    def __init__(self, vocabulary, hidden_size, num_layers, dropout, device):
        super(CharRNN, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs_list, hidden=None):
        """
        Forward propagation in CharRNN

        :param inputs_list: list of Tensors (should be sorted and consists of indexes from vocabulary)
        :param hidden: hidden for LSTM layer
        :return: (x, state): output from last layer and hidden from LSTM layer
        """
        lengths = [t.shape[0] for t in inputs_list]

        x = rnn_utils.pad_sequence(inputs_list, batch_first=True, padding_value=self.vocabulary.pad)
        x = self.one_hot_encoding(x, lengths, self.device)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)

        x, state = self.lstm_layer(x, hidden)  # shape x is (seq_len, batch, num_directions * hidden_size)

        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True, padding_value=self.vocabulary.pad,
                                             total_length=lengths[0])

        x = torch.cat([t[:l, :] for l, t in zip(lengths, x)], dim=0)  # first dim - length, second dim - hidden
        x = self.linear_layer(x)

        ranges = np.cumsum([0] + lengths)
        x = [x[r1:r2, :] for r1, r2 in zip(ranges[0:-1], ranges[1:])]

        return x, state

    def sample_smiles(self, max_length, batch_size):
        starts = [torch.tensor(self.vocabulary.bos, dtype=torch.long, device=self.device).unsqueeze(0)
                  for _ in range(batch_size)
                  ]
        new_smiles_list = [torch.tensor(self.vocabulary.pad, dtype=torch.long, device=self.device).repeat(max_length)
                           for _ in range(batch_size)
                           ]
        end_smiles_list = [False for _ in range(batch_size)]
        for i in range(max_length):
            hidden = None if i == 0 else hidden
            output, hidden = self(starts, hidden)

            # probabilities
            probs = [F.softmax(o, dim=-1) for o in output]

            # sample from probabilities
            ind_tops = [torch.multinomial(p, 1) for p in probs]

            for j, top in enumerate(ind_tops):
                if not end_smiles_list[j]:
                    top_elem = top[0].item()
                    if top_elem == self.vocabulary.eos:
                        end_smiles_list[j] = True
                    else:
                        new_smiles_list[j][i] = top_elem

            starts = ind_tops

        return new_smiles_list

    def one_hot_encoding(self, x_tensor, lengths, device):
        x_size = x_tensor.size()
        x_one_hot = torch.zeros(x_size[0], x_size[1], self.vocab_size, device=device)
        for i, s in enumerate(x_tensor):
            for j, num in enumerate(s):
                if j < lengths[i]:
                    t = torch.zeros(self.vocab_size, device=device)
                    t[num] = 1
                    x_one_hot[i][j] = t
        return x_one_hot
