import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_vocab, fps_len, config):
        super().__init__()

        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(x_vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(x_vocab), x_vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(x_vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # condition linear layer and output size
        if config.conditional:
            output_size = config.output_size
            self.c_lin = nn.Linear(fps_len, output_size)
        else:
            output_size = 0

        # Encoder
        if config.q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb + output_size,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, config.d_z)
        self.q_logvar = nn.Linear(q_d_last, config.d_z)

        # z linear layer
        self.z_lin = nn.Linear(config.d_z*2, config.d_z)

        # Decoder
        if config.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + config.d_z + output_size,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )

        self.decoder_lat = nn.Linear(config.d_z + output_size, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

    def forward(self, x, c):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x, c, config.conditional)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, c, z, config.conditional)

        return kl_loss, recon_loss

    def forward_encoder(self, x, c, conditional):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]

        # condition
        if conditional:
            c = [self.c_lin(i_c).unsqueeze(0).repeat(i_x.shape[0], 1) for (i_x, i_c) in zip(x, c)]
            # cat x and c
            x = [torch.cat((i_x, i_c), 1) for (i_x, i_c) in zip(x, c)]
        
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss

    def forward_decoder(self, x, c, z, conditional):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        # condition
        if conditional:
            c = self.c_lin(c)
            # cat z and c
            z = torch.cat((z, c), 1)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

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

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def sample(self, fps_center, conditional, n_batch=1, n_len=100, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param n_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: tuple of two:
            1. (n_batch, d_z) of floats, latent vector z
            2. list of tensors of longs, samples sequence x
        """
        with torch.no_grad():
            # `z`
            device = self.x_emb.weight.device
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(device)

            # condition
            if conditional:
                c = self.c_lin(fps_center).unsqueeze(0).repeat(n_batch, 1)
                # cat z and c
                z = torch.cat((z, c), 1)

            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=device).repeat(n_batch)
            x = torch.tensor([self.pad], device=device).repeat(n_batch, n_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([n_len], device=device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=device)

            # Generating cycle
            for i in range(1, n_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            x = new_x

            return z, x
