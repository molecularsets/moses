import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class CharRNN(nn.Module):

    @staticmethod
    def _device(model):
        return next(model.parameters()).device

    def __init__(self, vocabulary, hidden_size, num_layers, device):
        super(CharRNN, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs_list, hidden=None):
        """
        Forward propagation in CharLSTM

        :param inputs_list: list of Tensors (should be sorted and consists of indexes from vocabulary)
        :param hidden: hidden for LSTM layer
        :return: (x, state): output from last layer and hidden from LSTM layer
        """
        lengths = [len(t) for t in inputs_list]

        x = rnn_utils.pad_sequence(inputs_list, batch_first=True, padding_value=self.vocab_size)

        current_device = x.device

        x = self.one_hot_encoding(x, lengths, self.vocab_size, current_device)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)

        x, state = self.lstm_layer(x, hidden)  # shape x is (seq_len, batch, num_directions * hidden_size)

        # there is a simple way to get tensor
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True, padding_value=self.vocab_size,
                                             total_length=lengths[0])

        x = torch.cat([t[:l, :] for l, t in zip(lengths, x)], dim=0)  # first dim - length, second dim - hidden
        x = self.linear_layer(x)

        ranges = np.cumsum([0] + lengths)
        x = [x[r1:r2, :] for r1, r2 in zip(ranges[0:-1], ranges[1:])]

        return x, state

    def sample_smiles(self, num, max_length):
        return [self.sample_smile(max_length) for _ in range(num)]

    def sample_smile(self, max_length):
        start = torch.tensor(self.vocabulary.bos, dtype=torch.long, device=self.device)
        new_smile = torch.tensor(self.vocabulary.pad, dtype=torch.long, device=self.device).repeat(max_length)

        for i in range(max_length):
            hidden = None if i == 0 else hidden
            output, hidden = self([start.unsqueeze_(0)], hidden)

            # probabilities
            probs = F.softmax(output[0][0], dim=-1)

            # sample from probabilities
            topi = torch.multinomial(probs, 1).to(device=self.device)
            topi = topi[0].item()

            if topi == self.vocabulary.eos:
                new_smile = new_smile[:i]
                break
            else:
                new_smile[i] = topi

            start = torch.tensor(topi, dtype=torch.long, device=self.device)

        return new_smile

    def one_hot_encoding(self, y_tensor, lengths, vocab_dim, device):
        y_size = y_tensor.size()
        y_one_hot = torch.zeros(y_size[0], y_size[1], vocab_dim, device=device)
        for i, s in enumerate(y_tensor):
            for j, num in enumerate(s):
                if j < lengths[i]:
                    t = torch.zeros(vocab_dim, device=device)
                    t[num] = 1
                    y_one_hot[i][j] = t
        return y_one_hot
