import torch.nn as nn
import torch.autograd as autograd
import torch
from made.made import MADE


class MADEPreconditioner(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=True, n_ar_modules=2):
        super().__init__()
        self.mades = nn.ModuleList(
            [MADE(nin * 2, hidden_sizes, nout * 2, num_masks, natural_ordering) for _ in range(n_ar_modules)]
        )

    def baseline(self, x, u):
        x = torch.zeros_like(x)
        h = torch.cat([u, x], dim=-1)
        for made in self.mades:
            h = made(h)
            h = torch.chunk(h, 2, dim=-1)[-1]
            h = torch.cat([u, h], dim=-1)
        return torch.chunk(h, 2, dim=-1)[-1]

    def forward(self, x, u):
        h = torch.cat([u, x], dim=-1)
        for made in self.mades:
            h = made(h)
            h = torch.chunk(h, 2, dim=-1)[-1]
            h = torch.cat([u, h], dim=-1)
        return torch.chunk(h, 2, dim=-1)[-1] - self.baseline(x, u)

