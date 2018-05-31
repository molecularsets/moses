import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import transforms
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, d_in, h, d_out):
        super().__init__()

        self.linear1 = nn.Linear(d_in, h)
        self.linear2 = nn.Linear(h, d_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(nn.Module):
    def __init__(self, d_in, h, d_out):
        super().__init__()

        self.linear1 = nn.Linear(d_in, h)
        self.linear2 = nn.Linear(h, d_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.enc_mu = nn.Linear(100, 8)
        self.enc_log_sigma = nn.Linear(100, 8)
        self.device = device

    def forward(self, x):
        h = self.encoder(x)
        z = self._sample_z(h)
        return self.decoder(z)

    def _sample_z(self, h):
        mu, sigma = self.enc_mu(h), torch.exp(self.enc_log_sigma(h))

        std = torch.from_numpy(
            np.random.normal(0, 1, size=sigma.size())
        ).float().to(self.device)

        self.z_mu, self.z_sigma = mu, sigma

        return mu + sigma * std


def latent_loss(z_mu, z_sigma):
    mu_sq, sigma_sq = z_mu ** 2, z_sigma ** 2
    return 0.5 * torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq) - 1)


if __name__ == '__main__':
    n_batch = 32
    d_input = 28 * 28
    n_epoch = 5
    gpu = True

    mnist = torchvision.datasets.MNIST(
        './',
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    dataloader = torch.utils.data.DataLoader(
        mnist,
        batch_size=n_batch,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    encoder = Encoder(d_input, 100, 100)
    decoder = Decoder(8, 100, d_input)
    device = torch.device('cuda:3' if gpu else 'cpu')
    model = VAE(encoder, decoder, device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(n_epoch)):
        total_loss = 0
        for i, (x, _) in enumerate(dataloader):
            # data
            x = x.resize_(n_batch, d_input).to(device)

            # forward pass
            xh = model(x)

            # loss
            loss = criterion(xh, x) + latent_loss(model.z_mu, model.z_sigma)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            total_loss += loss.item()

        total_loss /= (len(dataloader) * n_batch)
        print(f'\nepoch={epoch + 1} loss={total_loss}')
