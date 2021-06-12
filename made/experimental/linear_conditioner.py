import torch.nn as nn
import torch.autograd as autograd
import torch
from made.made import MADE


class LinearConditioner(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=True, n_ar_modules=2):
        super().__init__()
        self.layers = nn.Sequential([
            nn.Linear(nin * 2, hidden_sizes),
            nn.ELU(),
            nn.Linear(hidden_sizes, nin)]
        )

    def forward(self, x, u, delta):
        h = torch.cat([u, x], dim=-1)
        h = self.layers(h)
