import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from torch.utils.data import DataLoader
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

class CharLSTM(nn.Module):
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, hidden_size=32, num_layers=1, vocabulary=None, train_dataset=None, val_dataset=None):
        super(CharLSTM, self).__init__()

        self.device = torch.device('cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if vocabulary is None:
            if train_dataset is None and val_dataset is None:
                raise ValueError("You need to provide one of dataset if vocab_size isn't specified")

            if train_dataset is not None:
                dataset = train_dataset
                if val_dataset is not None:
                    dataset = itertools.chain(dataset, val_dataset)
            else:
                dataset = val_dataset

            data = tqdm(dataset)
            data.set_description('Dictionary initialization')

            vocabulary = {CharLSTM.SOS, CharLSTM.EOS}
            for smiles in dataset:
                vocabulary.update(smiles)
            vocabulary = sorted(vocabulary)
            print("Your vocabulary:")
            print(vocabulary)

        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.id2letter = {i: s for i, s in enumerate(vocabulary)}
        self.letter2id = {s: i for i, s in enumerate(vocabulary)}

        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
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

    def fit(self, train_dataset, n_epochs, batch_size, path_to_save, lr=1e-3, clip_grad=None, val_dataset=None,
            n_jobs=1):

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(list(self.parameters()), clip_grad)

        criterion = nn.CrossEntropyLoss()

        optimizers = optim.Adam(self.parameters(), lr=lr)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_func,
                                  num_workers=n_jobs)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_func,
                                    num_workers=n_jobs)

        best_val_loss = 100_000

        tqdm.monitor_interval = 0
        for epoch in range(n_epochs):
            self.train()

            train_dataloader = tqdm(train_loader)
            train_dataloader.set_description('Train (epoch #{})'.format(epoch))
            self.pass_data(train_dataloader, criterion, optimizers)

            if val_dataset is not None:
                self.eval()

                val_dataloader = tqdm(val_loader)
                val_dataloader.set_description('Validation (epoch #{})'.format(epoch))

                val_loss = self.pass_data(val_dataloader, criterion)

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), os.path.join(path_to_save, "epoch_{epoch}.pt".format(epoch=epoch)))
                    torch.save(self.state_dict(), os.path.join(path_to_save, "epoch_best.pt".format(epoch=epoch)))

    def pass_data(self, dataloader, criterion, optimizer=None):
        running_loss = 0

        for i, data in enumerate(dataloader):
            self.zero_grad()

            inputs = [t.to(self.device) for t in data[0]]
            targets = [t.to(self.device) for t in data[1]]

            outputs, _ = self(inputs)

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)

            loss = criterion(outputs, targets)

            postfix = {'loss': loss}
            dataloader.set_postfix(postfix)

            running_loss += loss * len(inputs)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        return running_loss / len(dataloader)

    def greedy_generate_smile(self, max_length):
        start = torch.tensor(self.letter2id["<SOS>"], dtype=torch.long, device=self.device)
        new_smile = ""
        for i in range(max_length):
            hidden = None if i == 0 else hidden
            output, hidden = self([start.unsqueeze_(0)], hidden)

            _, topi = output[0].topk(1)  # the largest element
            topi = topi[0][0].item()

            letter = self.id2letter[topi]
            if letter == "<EOS>":
                break
            else:
                new_smile += letter
            start = torch.tensor(topi, dtype=torch.long, device=self.device)

        return new_smile

    def sample_smiles(self, num, max_length=1000):
        return [self.sample_smile(max_length) for _ in range(num)]

    def sample_smile(self, max_length=1000):
        start = torch.tensor(self.letter2id["<SOS>"], dtype=torch.long, device=self.device)
        new_smile = ""
        for i in range(max_length):
            hidden = None if i == 0 else hidden
            output, hidden = self([start.unsqueeze_(0)], hidden)
            # probabilities
            probs = F.softmax(output[0][0], dim=-1)
            # sample from probabilities
            topi = torch.multinomial(probs, 1).to(device=self.device)
            topi = topi[0].item()

            letter = self.id2letter[topi]
            if letter == "<EOS>":
                break
            else:
                new_smile += letter
            start = torch.tensor(topi, dtype=torch.long, device=self.device)

        return new_smile

    def to(self, device):
        self.device = device
        return super(CharLSTM, self).to(device)

    def _collate_func(self, data):
        data.sort(key=lambda x: len(x), reverse=True)
        data = [[self.letter2id[s] for s in [CharLSTM.SOS] + list(smiles) + [CharLSTM.EOS]] for smiles in data]

        inputs = [torch.tensor(d[:-1], dtype=torch.long) for d in data]
        targets = [torch.tensor(d[1:], dtype=torch.long) for d in data]

        return inputs, targets

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
