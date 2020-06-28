import json
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torchvision import transforms, datasets
from vae import VAE

num_epochs = 100
dataset_size = 10000
batch_size = 64
learning_rate = 1e-3
data_path = 'data/obs'

def load_dataset(data_path, dataset_size=5000, batch_size=64):
    print(f'Loading data from {data_path}...')
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform= transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )
    train_dataset_size_full = len(train_dataset.samples)
    print(f'{train_dataset_size_full} samples found. Sampling {dataset_size} for training.')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=dataset_size)
    )
    return train_loader

def loss_function(recon_x, x, mu, logsigma):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')   # reconstruction loss
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())       # KL divergence
    return BCE + KLD

def train(model, data_path, dataset_size=5000, batch_size=64, num_epochs=5, learning_rate=1e-3):
    train_loader = load_dataset(data_path, dataset_size, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    outputs = []
    for epoch in range(num_epochs):
        train_loss = 0
        for i, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x = batch[0].cuda()
            recon_batch, mu, logsigma = model(x)
            loss = loss_function(recon_batch, x, mu, logsigma)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # progress
            pct = (i*batch_size/dataset_size)*100.0
            print('\r[ %.2f%% ]' % round(pct, 2), end='', flush=True)

        train_loss_mean = float(train_loss)/i
        print('\nEpoch:{}, Loss:{:.4f}'.format(epoch+1, train_loss_mean))
        outputs.append((epoch, x, recon_batch),)

    print('Finished!')
    torch.save(model.state_dict(), 'models/compressor/conv_vae.pth')
    return outputs

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.fill_(0.01)

with open('config/creature.json') as f:
    config = json.load(f)
    latent_size = config.get('vae.latent.size')

torch.manual_seed(1)
model = VAE(latent_size).cuda()
model.apply(init_weights)

train(
    model=model, 
    data_path=data_path, 
    dataset_size=dataset_size, 
    batch_size=batch_size, 
    num_epochs=num_epochs, 
    learning_rate=learning_rate
)
