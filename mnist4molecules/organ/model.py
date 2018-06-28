import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


__all__ = ['ORGAN']


class Generator(nn.Module):
    def __init__(self, embedding_layer, hidden_size, num_layers, dropout):
        super(Generator, self).__init__()

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(embedding_layer.embedding_dim, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size, embedding_layer.num_embeddings)

    def forward(self, x, lengths, states=None):
        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, states = self.lstm_layer(x, states)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, states


'''class Discriminator(nn.Module):
    def __init__(self, embedding_layer, convs):
        super(Discriminator, self).__init__()

        sum_filters = sum([f for f, _ in convs])
        self.min_seq_length = max(map(lambda x: x[1], convs))

        self.embedding_layer = embedding_layer
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, f, kernel_size=(n, embedding_layer.embedding_dim)) for f, n in convs])
        self.highway_layer = nn.Linear(sum_filters, sum_filters)
        self.output_layer = nn.Linear(sum_filters, 1)

    def forward(self, x):
        if x.shape[-1] < self.min_seq_length:
            padding = torch.empty(x.shape[0], self.min_seq_length - x.shape[1], dtype=x.dtype, device=x.device)
            padding.fill_(self.embedding_layer.padding_idx)
            x = torch.cat([x, padding], dim=-1)

        x = self.embedding_layer(x)
        x = x.unsqueeze(1) 
        convs = [F.elu(conv_layer(x)).squeeze(3) for conv_layer in self.conv_layers]
        x = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in convs]
        x = torch.cat(x, dim=1)
        h = self.highway_layer(x)
        t = F.sigmoid(h)
        x = t * F.elu(h) + (1 - t) * x
        out = self.output_layer(x)

        return out'''


class Discriminator(nn.Module):
    def __init__(self, embedding_layer, layers):
        super(Discriminator, self).__init__()

        self.embedding_layer = embedding_layer

        in_features = [embedding_layer.embedding_dim] + layers
        out_features = layers + [1]

        self.layers_seq = nn.Sequential()     
        for k, (i, o) in enumerate(zip(in_features, out_features)):
            self.layers_seq.add_module('linear_{}'.format(k), nn.Conv2d(i, o, kernel_size=(3, 1), padding=(1, 0)))
            if k != len(layers):
                self.layers_seq.add_module('activation_{}'.format(k), nn.ELU(inplace=True))
            else:
                self.layers_seq.add_module('average', nn.AdaptiveAvgPool2d(output_size=1))
        
    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.permute(0, 2, 1).unsqueeze(3)
        x = self.layers_seq(x)
        out = x.squeeze(-1).squeeze(-1)

        return out


class ORGAN(nn.Module):
    def __init__(self, config, vocabulary, reward_fn=None):
        super(ORGAN, self).__init__()

        self.reward_fn = reward_fn
        self.reward_weight = config.reward_weight

        self.vocabulary = vocabulary

        self.generator_embeddings = nn.Embedding(len(vocabulary), config.embedding_size, padding_idx=vocabulary.pad)
        self.discriminator_embeddings = nn.Embedding(len(vocabulary), config.embedding_size, padding_idx=vocabulary.pad)
        self.generator = Generator(self.generator_embeddings, config.hidden_size, config.num_layers, config.dropout)
        self.discriminator = Discriminator(self.discriminator_embeddings, config.discriminator_layers)

    @property
    def device(self):
        return next(self.generator_embeddings.parameters()).device

    def generator_forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def discriminator_forward(self, *args, **kwargs):
        return self.discriminator(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def string2tensor(self, string, add_bos=False, add_eos=False):
        ids = [self.vocabulary.stoi[c] for c in string]

        if add_bos:
            ids = [self.vocabulary.bos] + ids
        if add_eos:
            ids = ids + [self.vocabulary.eos]

        tensor = torch.tensor(ids, dtype=torch.long, device=self.device)

        return tensor

    def tensor2string(self, tensor, rem_bos=True, rem_eos=True):
        ids = tensor.tolist()

        if rem_bos and ids[0] == self.vocabulary.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.vocabulary.eos:
            ids = ids[:-1]

        chars = [self.vocabulary.itos[id] for id in ids]
        string = ''.join(chars)

        return string

    def _proceed_sequences(self, prevs, states, max_len):
        n_sequences = prevs.shape[0]

        sequences = []
        lengths = torch.zeros(n_sequences, dtype=torch.long, device=prevs.device)

        one_lens = torch.ones(n_sequences, dtype=torch.long, device=prevs.device)
        is_end = prevs.eq(self.vocabulary.eos).view(-1)

        for _ in range(max_len):
            logits, _, states = self.generator(prevs, one_lens, states)
            
            logits = logits.view(n_sequences, -1)
            currents = torch.multinomial(F.softmax(logits, dim=-1), 1)

            currents[is_end, :] = self.vocabulary.pad
            sequences.append(currents)
            lengths[~is_end] += 1

            is_end[currents.view(-1) == self.vocabulary.eos] = 1
            if is_end.sum() == n_sequences:
                break

            prevs = currents

        sequences = torch.cat(sequences, dim=-1)

        return sequences, lengths

    def rollout(self, n_samples, n_rollouts, max_len=100):
        sequences = []
        rewards = []
        lengths = torch.zeros(n_samples, dtype=torch.long, device=self.device)

        one_lens = torch.ones(n_samples, dtype=torch.long, device=self.device)
        prevs = torch.empty(n_samples, 1, dtype=torch.long, device=self.device).fill_(self.vocabulary.bos)
        is_end = torch.zeros(n_samples, dtype=torch.uint8, device=self.device)
        states = None

        sequences.append(prevs)
        lengths += 1

        for current_len in range(max_len):
            logits, _, states = self.generator(prevs, one_lens, states)
            currents = torch.multinomial(F.softmax(logits, dim=-1).view(n_samples, -1), 1)

            currents[is_end, :] = self.vocabulary.pad
            sequences.append(currents)
            lengths[~is_end] += 1

            rollout_prevs = currents[~is_end, :].repeat(n_rollouts, 1)
            rollout_states = (states[0][:, ~is_end, :].repeat(1, n_rollouts, 1), states[1][:, ~is_end, :].repeat(1, n_rollouts, 1))
            rollout_sequences, rollout_lengths = self._proceed_sequences(rollout_prevs, rollout_states, max_len - current_len)

            rollout_sequences = torch.cat([s[~is_end, :].repeat(n_rollouts, 1) for s in sequences] + [rollout_sequences], dim=-1)
            rollout_lengths += lengths[~is_end].repeat(n_rollouts)

            rollout_rewards = F.sigmoid(self.discriminator(rollout_sequences).detach())
            if self.reward_fn is not None:
                for k, (t, l) in enumerate(zip(rollout_sequences, rollout_lengths)):
                    string = self.vocabulary.tensor2string(t[:l])
                    rollout_rewards[k] = rollout_rewards[k] * (1 - self.reward_weight) + self.reward_fn(string) * self.reward_weight

            current_rewards = torch.zeros(n_samples, device=self.device)
            current_rewards[~is_end] = rollout_rewards.view(n_rollouts, -1).mean(dim=0)
            rewards.append(current_rewards.view(-1, 1))

            is_end[currents.view(-1) == self.vocabulary.eos] = 1
            if is_end.sum() == n_samples:
                break

            prevs = currents

        sequences = torch.cat(sequences, dim=1)
        rewards = torch.cat(rewards, dim=1)

        return sequences, rewards, lengths

    def sample_tensor(self, n, max_len=100):
        prevs = torch.empty(n, 1, dtype=torch.long, device=self.device).fill_(self.vocabulary.bos)
        samples, lengths = self._proceed_sequences(prevs, None, max_len)

        samples = torch.cat([prevs, samples], dim=-1)
        lengths += 1


        return samples, lengths

    def sample(self, n, max_len=100):
        samples, lengths = self.sample_tensor(n, max_len)
        samples = [t[:l] for t, l in zip(samples, lengths)]
        samples = [self.tensor2string(t) for t in samples]

        return samples