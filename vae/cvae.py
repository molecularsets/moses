import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_vae.sru import SRU
from text_vae.attention import SelfAttention
from text_vae.disc import CNNEncoder

from text_vae.utils import assert_check


class RnnVae(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        q, g, d = kwargs['q'], kwargs['g'], kwargs['d']
        self.n_len = kwargs['n_len']
        self.d_z = kwargs['d_z']
        self.d_c = kwargs['d_c']
        self.p_word_dropout = kwargs['p_word_dropout']
        freeze_embeddings = kwargs['freeze_embeddings']
        x_vocab = kwargs['x_vocab']
        self.unk = x_vocab.stoi['<unk>']
        self.pad = x_vocab.stoi['<pad>']
        self.bos = x_vocab.stoi['<bos>']
        self.eos = x_vocab.stoi['<eos>']
        self.n_vocab = len(x_vocab)
        self.attention = kwargs['attention']

        # Word embeddings layer
        if x_vocab is None:
            self.d_emb = q.d_h
            self.x_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)
        else:
            self.d_emb = x_vocab.vectors.size(1)
            self.x_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)

            # Set pretrained embeddings
            self.x_emb.weight.data.copy_(x_vocab.vectors)

            if freeze_embeddings:
                self.x_emb.weight.requires_grad = False

        # Encoder
        self.encoder_rnn = SRU(
            self.d_emb,
            q.d_h,
            num_layers=q.n_layers,
            dropout=q.s_dropout,
            rnn_dropout=q.r_dropout
        )
        self.q_mu = nn.Linear(q.d_h, self.d_z)
        self.q_logvar = nn.Linear(q.d_h, self.d_z)

        # Decoder
        self.decoder_rnn = SRU(
            self.d_emb + self.d_z + self.d_c,
            self.d_z + self.d_c,
            num_layers=g.n_layers,
            dropout=g.s_dropout,
            rnn_dropout=g.r_dropout
        )
        self.decoder_a = SelfAttention(self.d_z + self.d_c)
        self.decoder_fc = nn.Linear(self.d_z + self.d_c, self.n_vocab)

        # Discriminator
        self.disc_cnn = CNNEncoder(self.d_c, d.n_filters)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_a,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])
        self.discriminator = nn.ModuleList([
            self.disc_cnn
        ])

    def forward(self, x, use_c_prior=False):
        """Do the VAE forward step with prior c

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Input check
        assert_check(x, [-1, self.n_len], torch.long)

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)

        # Code: x -> c
        if use_c_prior:
            c = self.sample_c_prior(x.size(0))
        else:
            c = F.softmax(self.forward_discriminator(x), dim=1)

        # Decoder: x, z, c -> recon_loss
        recon_loss = self.forward_decoder(x, z, c)

        # Output check
        assert_check(kl_loss, [], torch.float, x.device)
        assert_check(recon_loss, [], torch.float, kl_loss.device)

        return kl_loss, recon_loss

    def forward_encoder(self, x, do_emb=True):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: (n_batch, n_len) of longs or (n_batch, n_len, d_emb) of
        floats, input sentence x
        :param do_emb: whether do embedding for x or not
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        # Input check
        if do_emb:
            assert_check(x, [-1, self.n_len], torch.long)
        else:
            assert_check(x, [-1, self.n_len, self.d_emb], torch.float)

        # Emb (n_batch, n_len, d_emb)
        if do_emb:
            x_emb = self.x_emb(x)
        else:
            x_emb = x

        # RNN
        x_emb = F.dropout(x_emb)
        h, _ = self.encoder_rnn(x_emb.t(), None)  # (n_len, n_batch, d_h)

        # Forward to latent
        h = h[-1]  # (n_batch, d_h)
        mu, logvar = self.q_mu(h), self.q_logvar(h)  # (n_batch, d_z)

        # Reparameterization trick: z = mu + std * eps; eps ~ N(0, I)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(logvar / 2) * eps

        # KL term loss
        kl_loss = 0.5 * (
                logvar.exp() + mu ** 2 - 1 - logvar
        ).sum(1).mean()  # 0

        # Output check
        assert_check(z, [x.size(0), self.d_z], torch.float, x.device)
        assert_check(kl_loss, [], torch.float, z.device)

        return z, kl_loss

    def forward_decoder(self, x, z, c):
        """Decoder step, emulating x ~ G(z, c)

        :param x: (n_batch, n_len) of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :param c: (n_batch, d_c) of floats, code c
        :return: float, recon component of loss
        """

        # Input check
        assert_check(x, [-1, self.n_len], torch.long)
        assert_check(z, [-1, self.d_z], torch.float, x.device)
        assert_check(c, [-1, self.d_c], torch.float, z.device)

        # Init
        h_init = torch.cat([
            z.unsqueeze(0),
            c.unsqueeze(0)
        ], 2)  # (1, n_batch, d_z + d_c)

        # Inputs
        x_drop = self.word_dropout(x)  # (n_batch, n_len)
        x_emb = self.x_emb(x_drop.t())  # (n_len, n_batch, d_emb)
        x_emb = torch.cat([
            x_emb,
            h_init.expand(x_emb.shape[0], -1, -1)
        ], 2)  # (n_len, n_batch, d_emb + d_z + d_c)

        # Rnn step
        outputs, _ = self.decoder_rnn(
            x_emb,
            h_init.expand(self.decoder_rnn.depth, -1, -1)
        )  # (n_len, n_batch, d_z + d_c)

        # Attention
        if self.attention:
            outputs, _ = self.decoder_a(outputs)

        # FC to vocab
        n_len, n_batch, _ = outputs.shape  # (n_len, n_batch)
        y = self.decoder_fc(
            outputs.view(n_len * n_batch, -1)
        ).view(n_len, n_batch, -1)  # (n_len, n_batch, n_vocab)

        # Loss
        recon_loss = F.cross_entropy(
            y.view(-1, y.size(2)),
            F.pad(x.t()[1:], (0, 0, 0, 1), 'constant', self.pad).view(-1)
        )  # 0

        # Output check
        assert_check(recon_loss, [], torch.float, x.device)

        return recon_loss

    def forward_discriminator(self, x, do_emb=True):
        """Discriminator step, emulating weights for c ~ D(x)

        :param x: (n_batch, n_len) of longs or (n_batch, n_len, d_emb) of
        floats, input sentence x
        :param do_emb: whether do embedding for x or not
        :return: (n_batch, d_c) of floats, sample of code c
        """

        # Input check
        if do_emb:
            assert_check(x, [-1, self.n_len], torch.long)
        else:
            assert_check(x, [-1, self.n_len, self.d_emb], torch.float)

        # Emb (n_batch, n_len, d_emb)
        if do_emb:
            x_emb = self.x_emb(x)
        else:
            x_emb = x

        # CNN
        c = self.disc_cnn(x)

        # Output check
        assert_check(c, [x.size(0), self.d_c], torch.float, x.device)

        return c

    def word_dropout(self, x):
        """
        Do word dropout: with prob `self.p_word_dropout`, set the word to
        `self.unk`, as initial Bowman et al. (2014) paper proposed.

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: (n_batch, n_len) of longs, x with drops
        """

        # Input check
        assert_check(x, [-1, self.n_len], torch.long)

        # Apply dropout mask
        mask = x.new_tensor(
            np.random.binomial(
                n=1,
                p=self.p_word_dropout,
                size=tuple(x.shape)
            ),
            dtype=torch.uint8
        )
        x_drop = x.clone()
        x_drop[mask] = self.unk

        # Output check
        assert_check(x_drop, [x.size(0), self.n_len], torch.long, x.device)

        return x_drop

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        # Input check
        assert isinstance(n_batch, int) and n_batch > 0

        # Sampling
        device = self.x_emb.weight.device
        z = torch.randn((n_batch, self.d_z), device=device)  # (n_batch, d_z)

        # Output check
        assert_check(z, [n_batch, self.d_z], torch.float, device)

        return z

    def sample_c_prior(self, n_batch):
        """Sampling prior, emulating c ~ P(c)

        :param n_batch: number of batches
        :return: (n_batch, d_c) of floats, sample of code c
        """

        # Input check
        assert isinstance(n_batch, int) and n_batch > 0

        # Sampling
        device = self.x_emb.weight.device
        inds = torch.multinomial(
            torch.ones(self.d_c, dtype=torch.float, device=device) / self.d_c,
            n_batch,
            replacement=True
        )
        ones = torch.eye(self.d_c, device=device)
        c = ones.index_select(0, inds)

        # Output check
        assert_check(c, [n_batch, self.d_c], torch.float, device)

        return c

    def sample_sentence(self, n_batch=1, z=None, c=None,
                        temp=1.0, pad=True, n_beam=5, coverage_penalty=True):
        """Generating n_batch sentences in eval mode with values (could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param c: (n_batch, d_c) of floats, code c or None
        :param temp: temperature of softmax
        :param pad: if do padding to n_len
        :param n_beam: size of beam search
        :param coverage_penalty:
        :return: tuple of four:
            1. (n_batch, d_z) of floats, latent vector z
            2. (n_batch, d_c) of floats, code c
            3. (n_batch, n_len) of longs if pad and list of n_batch len of
            tensors of longs: generated sents word ids if not
            4. (n_batch, n_len, n_len) of floats, attention probs
        """

        # Input check
        assert isinstance(n_batch, int) and n_batch > 0
        if z is not None:
            assert_check(z, [n_batch, self.d_z], torch.float)
        if c is not None:
            assert_check(c, [n_batch, self.d_c], torch.float)
        assert isinstance(temp, float)
        assert isinstance(n_beam, int) and n_beam > 0
        assert isinstance(coverage_penalty, bool)

        # Enable eval mode
        self.eval()

        # Params
        device = self.x_emb.weight.device
        n = n_batch * n_beam
        n_layers = self.decoder_rnn.depth

        # `z` and `c`, and then `h0`
        if z is None:
            z = self.sample_z_prior(n_batch)  # (n_batch, d_z)
        if c is None:
            c = self.sample_c_prior(n_batch)  # (n_batch, d_c)
        z, c = z.to(device), c.to(device)  # device change
        h0 = torch.cat([z.unsqueeze(0), c.unsqueeze(0)], dim=2)
        # (1, n, d_z + d_c)
        h0 = h0.unsqueeze(2).repeat(1, 1, n_beam, 1).view(1, n, -1)

        # Initial values
        w = torch.tensor(self.bos, device=device).repeat(n)  # n
        h = h0.repeat(n_layers, 1, 1)  # (n_layers, n, d_z + d_c)
        # Previous context vectors
        context = torch.empty(
            n, self.n_len - 1, self.d_z + self.d_c,
            device=device
        )
        # Attention matrix
        a = torch.zeros(n, self.n_len, self.n_len, device=device)
        # Candidates score for beam search
        H = torch.zeros(n, device=device)
        # X values
        x = torch.tensor(self.pad, device=device).repeat(n, self.n_len)
        x[:, 0] = self.bos
        eos_mask = torch.zeros(n, dtype=torch.uint8, device=device)
        end_pads = torch.tensor(self.n_len, device=device).repeat(n)

        # Cycle, word by word
        for i in range(1, self.n_len):
            # Init
            x_emb = self.x_emb(w)  # (n, d_emb)
            x_emb = x_emb.unsqueeze(0)  # (1, n, d_emb)
            x_emb = torch.cat([x_emb, h0], 2)  # (1, n, d_emb + d_z + d_c)

            # Step
            o, h = self.decoder_rnn(x_emb, h)  # o: (1, n, d_z + d_c)
            o = o.squeeze(0)
            context[:, i - 1, :] = o
            if self.attention:
                # o: (n, d_z + d_c), aw: (n, i)
                o, aw = self.decoder_a.forward_inference(
                    o, context[:, :i, :]
                )
                a[~eos_mask, i, :i] = aw[~eos_mask]
                if coverage_penalty:
                    H[~eos_mask] += aw.sum(1).clamp(max=1).log()[~eos_mask]
            y = F.softmax(self.decoder_fc(o) / temp, dim=-1)  # (n, n_vocab)

            # Generating
            nw = torch.multinomial(y, n_beam)  # (n, n_beam)
            pc = y.gather(1, nw)  # (n, n_beam)
            # (n_batch, n_beam, n_beam)
            pc = pc.view(n_batch, n_beam, -1).log()
            pc = H.view(n_batch, -1, 1) + pc  # (n_batch, n_beam, n_beam)
            aH, u = pc.view(n_batch, -1).topk(n_beam, 1)  # (n_batch, n_beam)
            w = nw.view(n_batch, -1).gather(1, u).view(-1)  # n

            # Masking new candidates
            parents = u.div(n_beam)  # (n_batch, n_beam)
            base_mask = torch.arange(n_batch, dtype=torch.long, device=device)
            base_mask *= n_beam
            base_mask = base_mask.unsqueeze(1).repeat(1, n_beam).view(-1)
            mask = base_mask + parents.view(-1)
            h = h[:, mask, :]
            context = context[mask]
            a = a[mask]
            H = H[mask]
            H[~eos_mask] += aH.view(-1)[~eos_mask]
            x = x[mask]
            eos_mask = eos_mask[mask]
            end_pads = end_pads[mask]
            x[~eos_mask, i] = w[~eos_mask]

            # Eos masks
            i_eos_mask = (w == self.eos)
            end_pads[i_eos_mask] = i
            eos_mask |= i_eos_mask

        # Choosing best candidate
        u = H.view(n_batch, -1).argmax(1).unsqueeze(-1).unsqueeze(-1)
        ux = u.repeat(1, 1, self.n_len)
        x = x.view(n_batch, -1, self.n_len)
        x = x.gather(1, ux).squeeze(1)
        ua = u.unsqueeze(-1).repeat(1, 1, self.n_len, self.n_len)
        a = a.view(n_batch, -1, self.n_len, self.n_len)
        a = a.gather(1, ua).squeeze(1)

        # Pad
        if not pad:
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            x = new_x

        # Back to train
        self.train()

        # Output check
        assert_check(z, [n_batch, self.d_z], torch.float, device)
        assert_check(c, [n_batch, self.d_c], torch.float, device)
        if pad:
            assert_check(x, [n_batch, self.n_len], torch.long, device)
        else:
            assert len(x) == n_batch
            for i_x in x:
                assert_check(i_x, [-1], torch.long, device)
                assert len(i_x) <= self.n_len
        assert_check(a, [n_batch, self.n_len, self.n_len],
                     torch.float, device)

        return z, c, x, a

    def perplexity(self, x, use_c_prior=False):
        """Calculating ppl score for input sequence x

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: n_batch of floats, ppl scores
        """

        # Input check
        assert_check(x, [-1, self.n_len], torch.long)

        # Eval mode
        self.eval()

        # Encoder: x -> z, kl_loss
        z, _ = self.forward_encoder(x)

        # Code: x -> c
        if use_c_prior:
            c = self.sample_c_prior(x.size(0))
        else:
            c = F.softmax(self.forward_discriminator(x), dim=1)

        # Decoder
        h_init = torch.cat([
            z.unsqueeze(0),
            c.unsqueeze(0)
        ], 2)  # (1, n_batch, d_z + d_c)
        x_emb = self.x_emb(x.t())  # (n_len, n_batch, d_emb)
        x_emb = torch.cat([
            x_emb,
            h_init.expand(x_emb.shape[0], -1, -1)
        ], 2)  # (n_len, n_batch, d_emb + d_z + d_c)
        outputs, _ = self.decoder_rnn(
            x_emb,
            h_init.expand(self.decoder_rnn.depth, -1, -1)
        )  # (n_len, n_batch, d_z + d_c)
        if self.attention:
            outputs, _ = self.decoder_a(outputs)
        n_len, n_batch, _ = outputs.shape  # (n_len, n_batch)
        y = self.decoder_fc(
            outputs.view(n_len * n_batch, -1)
        ).view(n_len, n_batch, -1)  # (n_len, n_batch, n_vocab)
        y = F.softmax(y, dim=2)

        # Calc ppl
        y = y[:-1]  # (n_len - 1, n_batch, n_vocab)
        rx = x.t()[1:].unsqueeze(2)  # (n_len - 1, n_batch, 1)
        ppl = y.gather(2, rx).squeeze(2)
        ppl = ppl.t()
        scores = []
        for i, xl in enumerate(x):
            threes = (xl == self.eos).nonzero()
            m = self.n_len - 1 if not len(threes) else threes.max().item()
            scores.append(ppl[i, :m].log().sum().exp() ** (-1.0 / (m + 1)))
        ppl = torch.tensor(scores, device=x.device)

        # Train mode
        self.train()

        # Output check
        assert_check(ppl, [x.size(0)], torch.float, x.device)

        return ppl

    def old_sample_sentence(self, n_batch=1, z=None, c=None,
                            temp=1.0, pad=True):
        """Generating n_batch sentences in eval mode with values (could be
        not on same device)
        :param n_batch: number of sentences to generate
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param c: (n_batch, d_c) of floats, code c or None
        :param temp: temperature of softmax
        :param pad: if do padding to n_len
        :return: tuple of four:
            1. (n_batch, d_z) of floats, latent vector z
            2. (n_batch, d_c) of floats, code c
            3. (n_batch, n_len) of longs if pad and list of n_batch len of
            tensors of longs: generated sents word ids if not
            4. (n_batch, n_len, n_len) of floats, attention probs
        """

        # Input check
        assert isinstance(n_batch, int) and n_batch > 0
        if z is not None:
            assert_check(z, [n_batch, self.d_z], torch.float)
        if c is not None:
            assert_check(c, [n_batch, self.d_c], torch.float)
        assert isinstance(temp, float) and 0 < temp <= 1
        assert isinstance(pad, bool)

        # Enable eval mode
        self.eval()

        # Initial values
        device = self.x_emb.weight.device
        if z is None:
            z = self.sample_z_prior(n_batch)  # (n_batch, d_z)
        if c is None:
            c = self.sample_c_prior(n_batch)  # (n_batch, d_c)
        z, c = z.to(device), c.to(device)  # device change
        z1, c1 = z.unsqueeze(0), c.unsqueeze(0)  # +1
        n_layers = self.decoder_rnn.depth
        h = torch.cat(
            [z1, c1], dim=2
        ).expand(n_layers, -1, -1)  # (n_layers, n_batch, d_z + d_c)
        w = torch.tensor(self.bos, device=device).expand(n_batch)  # n_batch
        x = torch.tensor(
            [self.pad], device=device
        ).repeat(n_batch, self.n_len)
        x[:, 0] = self.bos
        end_pads = torch.tensor(
            [self.n_len], device=device
        ).repeat(n_batch)
        eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=device)
        context = torch.empty(
            n_batch, self.n_len - 1, self.d_z + self.d_c,
            device=device
        )
        a = torch.zeros(
            n_batch, self.n_len, self.n_len,
            device=device
        )

        # Cycle, word by word
        for i in range(1, self.n_len):
            # Init
            x_emb = self.x_emb(w).expand(
                n_layers, -1, -1
            )  # (n_layers, n_batch, d_emb)
            x_emb = torch.cat(
                [x_emb, z1, c1], 2
            )  # (n_layers, n_batch, d_emb + d_z + d_c)

            # Step
            o, h = self.decoder_rnn(x_emb, h)  # (1, n_batch, d_z + d_c)
            context[:, i - 1, :] = o.t()
            o, aw = self.decoder_a.forward_inference(
                o.squeeze(0), context[:, :i, :]
            )
            a[~eos_mask, i, :i] = aw[~eos_mask]
            y = self.decoder_fc(o)
            y = F.softmax(y / temp, dim=1)

            # Generating
            w = torch.multinomial(y, 1)[:, 0]
            x[~eos_mask, i] = w[~eos_mask]
            i_eos_mask = (w == self.eos)
            end_pads[i_eos_mask] = i
            eos_mask = eos_mask | i_eos_mask

        # Pad
        if not pad:
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            x = new_x

        # Back to train
        self.train()

        # Output check
        assert_check(z, [n_batch, self.d_z], torch.float, device)
        assert_check(c, [n_batch, self.d_c], torch.float, device)
        if pad:
            assert_check(x, [n_batch, self.n_len], torch.long, device)
        else:
            assert len(x) == n_batch
            for i_x in x:
                assert_check(i_x, [-1], torch.long, device)
                assert len(i_x) <= self.n_len
        assert_check(a, [n_batch, self.n_len, self.n_len],
                     torch.float, device)

        return z, c, x, a

    def sample_soft_embed(self, n_batch=1, z=None, c=None, temp=1.0):
        """Generating single soft sample x
        TODO: Not working right now

        :param z: (n_batch, d_z) of floats, latent vector z / None
        :param c: (n_batch, d_c) of floats, code c / None
        :param temp: temperature of softmax
        :param device: device to run
        :return: (n_len, d_emb) of floats, sampled soft x
        """

        # Input check
        assert isinstance(n_batch, int) and n_batch > 0
        if z is not None:
            assert_check(z, [n_batch, self.d_z], torch.float)
        if c is not None:
            assert_check(c, [n_batch, self.d_c], torch.float)
        assert isinstance(temp, float) and 0 < temp <= 1

        # Initial values
        device = self.x_emb.weight.device
        if z is None:
            z = self.sample_z_prior(n_batch)  # (n_batch, d_z)
        if c is None:
            c = self.sample_c_prior(n_batch)  # (n_batch, d_c)
        z, c = z.to(device), c.to(device)  # device change
        z1, c1 = z.unsqueeze(0), c.unsqueeze(0)  # +1
        h = torch.cat(
            [z1, c1], dim=2
        ).expand(
            self.decoder_rnn.depth, -1, -1
        )  # (n_layers, n_batch, d_z + d_c)
        emb = self.x_emb(
            torch.tensor(self.bos, device=device).expand(n_batch)
        )  # (n_batch, d_emb)

        # Cycle, word by word
        outputs = [emb]
        for _ in range(1, self.n_len):
            # Init
            x_emb = emb.unsqueeze(0)
            x_emb = torch.cat([x_emb, z1, c1], 2)  # (1, 1, d_emb + d_z + d_c)

            # Step
            o, h = self.decoder_rnn(x_emb, h)
            y = self.decoder_fc(o).squeeze(0)
            y = F.softmax(y / temp, dim=1)

            emb = y @ self.x_emb.weight  # (n_batch, d_emb)
            outputs.append(emb)

        # Making x
        x = torch.stack(outputs, dim=1)  # (n_batch, n_len, d_emb)

        # Output check
        assert_check(z, [n_batch, self.d_z], torch.float, device)
        assert_check(c, [n_batch, self.d_c], torch.float, device)
        assert_check(x, [n_batch, self.n_len, self.d_emb], torch.float, device)

        return z, c, x
