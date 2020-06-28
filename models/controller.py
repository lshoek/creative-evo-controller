import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        self.fc = nn.Linear(latent_size, output_size)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=0)
        x = torch.tanh(self.fc(cat_in))
        return x
