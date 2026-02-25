import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, embedding_layer, hidden_size, num_layers,
                 bidirectional, dropout, latent_size):
        super(Encoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(embedding_layer.embedding_dim,
                                  hidden_size, num_layers,
                                  batch_first=True, dropout=dropout,
                                  bidirectional=bidirectional)
        self.linear_layer = nn.Linear(
            (int(bidirectional) + 1) * num_layers * hidden_size,
            latent_size
        )

    def forward(self, x, lengths):
        batch_size = x.shape[0]

        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        _, (_, x) = self.lstm_layer(x)
        x = x.permute(1, 2, 0).contiguous().view(batch_size, -1)
        x = self.linear_layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_layer, hidden_size,
                 num_layers, dropout, latent_size):
        super(Decoder, self).__init__()

        self.latent2hidden_layer = nn.Linear(latent_size, hidden_size)
        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(embedding_layer.embedding_dim,
                                  hidden_size, num_layers,
                                  batch_first=True, dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size,
                                      embedding_layer.num_embeddings)

    def forward(self, x, lengths, states, is_latent_states=False):
        if is_latent_states:
            c0 = self.latent2hidden_layer(states)
            c0 = c0.unsqueeze(0).repeat(self.lstm_layer.num_layers, 1, 1)
            h0 = torch.zeros_like(c0)
            states = (h0, c0)

        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, states = self.lstm_layer(x, states)
        x, lengths = pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, states


class Discriminator(nn.Module):
    def __init__(self, input_size, layers):
        super(Discriminator, self).__init__()

        in_features = [input_size] + layers
        out_features = layers + [1]

        self.layers_seq = nn.Sequential()
        for k, (i, o) in enumerate(zip(in_features, out_features)):
            self.layers_seq.add_module('linear_{}'.format(k), nn.Linear(i, o))
            if k != len(layers):
                self.layers_seq.add_module('activation_{}'.format(k),
                                           nn.ELU(inplace=True))

    def forward(self, x):
        return self.layers_seq(x)


class AAE(nn.Module):
    def __init__(self, vocabulary, config):
        super(AAE, self).__init__()

        self.vocabulary = vocabulary
        self.latent_size = config.latent_size

        self.embeddings = nn.Embedding(len(vocabulary),
                                       config.embedding_size,
                                       padding_idx=vocabulary.pad)
        self.encoder = Encoder(self.embeddings, config.encoder_hidden_size,
                               config.encoder_num_layers,
                               config.encoder_bidirectional,
                               config.encoder_dropout,
                               config.latent_size)
        self.decoder = Decoder(self.embeddings,
                               config.decoder_hidden_size,
                               config.decoder_num_layers,
                               config.decoder_dropout,
                               config.latent_size)
        self.discriminator = Discriminator(config.latent_size,
                                           config.discriminator_layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def encoder_forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decoder_forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def discriminator_forward(self, *args, **kwargs):
        return self.discriminator(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def sample_latent(self, n):
        return torch.randn(n, self.latent_size, device=self.device)

    def sample(self, n_batch, max_len=100):
        with torch.no_grad():
            samples = []
            lengths = torch.zeros(
                n_batch, dtype=torch.long, device=self.device
            )

            states = self.sample_latent(n_batch)
            prevs = torch.empty(
                n_batch, 1, dtype=torch.long, device=self.device
            ).fill_(self.vocabulary.bos)
            one_lens = torch.ones(n_batch, dtype=torch.long,
                                  device=self.device)
            is_end = torch.zeros(n_batch, dtype=torch.uint8,
                                 device=self.device)

            for i in range(max_len):
                logits, _, states = self.decoder(prevs, one_lens,
                                                 states, i == 0)
                logits = torch.softmax(logits, 2)
                shape = logits.shape[:-1]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                currents = torch.distributions.Categorical(logits).sample()
                currents = currents.view(shape)

                is_end[currents.view(-1) == self.vocabulary.eos] = 1
                if is_end.sum() == max_len:
                    break

                currents[is_end, :] = self.vocabulary.pad
                samples.append(currents.cpu())
                lengths[~is_end] += 1

                prevs = currents

            if len(samples):
                samples = torch.cat(samples, dim=-1)
                samples = [
                    self.tensor2string(t[:l])
                    for t, l in zip(samples, lengths)
                ]
            else:
                samples = ['' for _ in range(n_batch)]

            return samples
