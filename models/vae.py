import torch
import torch.nn as nn

def reparametrize(mu, logsigma):
    sigma = logsigma.exp()
    eps = torch.randn_like(sigma)
    z = eps.mul(sigma).add_(mu)
    return z

class Decoder(nn.Module):
    def __init__(self, latent_size, m):
        super(Decoder, self).__init__()
        self.latent_size = latent_size

        self.fc = nn.Linear(latent_size, m)
        self.deconv1 = nn.ConvTranspose2d(m, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 6, stride=2)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))

        reconstr = torch.sigmoid(self.deconv4(x))
        return reconstr

class Encoder(nn.Module):
    def __init__(self, latent_size, m):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        
        #self.fc = nn.Linear(m, latent_size)
        self.fc_mu = nn.Linear(m, latent_size)
        self.fc_logsigma = nn.Linear(m, latent_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        m = 1024 #2*2*256
        self.encoder = Encoder(latent_size, m)
        self.decoder = Decoder(latent_size, m)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = reparametrize(mu, logsigma)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
