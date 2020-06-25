import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 16)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=0)
        x = torch.relu(self.fc1(cat_in))
        x = torch.tanh(self.fc2(x))
        return x
