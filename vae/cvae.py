import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnVae(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.q, self.g = kwargs['q'], kwargs['g']
        self.d_z = kwargs['d_z']
        self.freeze_embeddings = kwargs['freeze_embeddings']
        x_vocab = kwargs['x_vocab']
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(x_vocab, ss))
        self.n_vocab = len(x_vocab)

        # Word embeddings layer
        if x_vocab is None:
            self.d_emb = q.d_h
            self.x_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)
        else:
            self.d_emb = x_vocab.vectors.size(1)
            self.x_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)

            # Set pretrained embeddings
            self.x_emb.weight.data.copy_(x_vocab.vectors)

            if self.freeze_embeddings:
                self.x_emb.weight.requires_grad = False

        # Encoder
        if self.q.cell == 'gru':
            self.encoder_rnn = nn.GRU(
                self.d_emb,
                self.q.d_h,
                num_layers=self.q.n_layers,
                batch_first=True,
                dropout=self.q.r_dropout if self.q.n_layers > 1 else 0,
                bidirectional=self.q.bidir
            )
        else:
            raise ValueError(
                "Invalid q.cell type, should be one of the ('gru',)"
            )

        q_d_h = self.q.d_h * (2 if self.q.bidir else 1)
        self.q_mu = nn.Linear(q_d_h, self.d_z)
        self.q_logvar = nn.Linear(q_d_h, self.d_z)

        # Decoder
        if self.g.cell == 'gru':
            self.decoder_rnn = nn.GRU(
                self.d_emb + self.d_z,
                self.d_z,
                num_layers=self.g.n_layers,
                batch_first=True,
                dropout=self.g.r_dropout if self.g.n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid g.cell type, should be one of the ('gru',)"
            )

        self.decoder_fc = nn.Linear(self.d_z, self.n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)

        # Decoder: x, z, c -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        return kl_loss, recon_loss

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.q.bidir)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss

    def forward_decoder(self, x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :param c: (n_batch, d_c) of floats, code c
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).expand(-1, x_emb.size(1), -1)
        input = torch.cat([x_emb, z_0], dim=-1)
        input = nn.utils.rnn.pack_padded_sequence(input, lengths,
                                                  batch_first=True)

        h_0 = z.unsqueeze(0).repeat(self.g.n_layers, 1, 1)

        output, _ = self.decoder_rnn(input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )

        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.d_z,
                           device=self.x_emb.weight.device)

    def sample_sentence(self, n_batch=1, n_len=100, z=None, temp=1.0):
        """Generating n_batch sentences in eval mode with values (could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param n_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: tuple of two:
            1. (n_batch, d_z) of floats, latent vector z
            2. list of tensors of longs, samples sequence x
        """

        self.eval()

        # `z`
        device = self.x_emb.weight.device
        if z is None:
            z = self.sample_z_prior(n_batch)
        z = z.to(device)
        z_0 = z.unsqueeze(1)

        # Initial values
        h = z.unsqueeze(0).repeat(self.g.n_layers, 1, 1)
        w = torch.tensor(self.bos, device=device).repeat(n_batch)
        x = torch.tensor([self.pad], device=device).repeat(n_batch, n_len)
        end_pads = torch.tensor([n_len], device=device).repeat(n_batch)
        eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=device)

        # Generating cycle
        for i in range(1, n_len):
            x_emb = self.x_emb(w).unsqueeze(1)
            input = torch.cat([x_emb, z_0], dim=-1)

            o, h = self.decoder_rnn(input, h)
            y = self.decoder_fc(o.squeeze(1))
            y = F.softmax(y / temp, dim=-1)

            w = torch.multinomial(y, 1)[:, 0]
            x[~eos_mask, i] = w[~eos_mask]
            i_eos_mask = (w == self.eos)
            end_pads[i_eos_mask] = i
            eos_mask = eos_mask | i_eos_mask

        # Converting `x` to list of tensors
        new_x = []
        for i in range(x.size(0)):
            new_x.append(x[i, :end_pads[i]])
        x = new_x

        self.train()

        return z, x
