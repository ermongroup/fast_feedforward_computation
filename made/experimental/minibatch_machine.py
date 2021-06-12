import torch
import numpy as np
from collections import deque

class MinibatchMachine(object):
    def __init__(self, config, ar_model, precon_model, replay_buffer, lambd=1e-5, bound=10.):
        self.config = config
        # Here the ar_model should directly output samples instead of latent space
        self.ar_model = ar_model
        self.precon_model = precon_model
        self.replay_buffer = replay_buffer
        self.lambd = lambd
        self.bound = bound

        self.batch_size = config.sample_batch_size
        if config.dataset == 'MNIST':
            self.data_dim = (784,)
        if config.dataset == 'CIFAR-10':
            self.data_dim = (3, 32, 32)

        self.device = config.device

        self.hs = [[torch.zeros(*self.data_dim, device=self.device)] for _ in range(self.batch_size)]
        self.fhs = [[] for _ in range(self.batch_size)]
        self.us = [torch.empty(*self.data_dim, device=self.device).uniform_(lambd, 1. - lambd) for _ in range(self.batch_size)]
        self.us = [torch.log(u) - torch.log(1. - u) for u in self.us]

        self.convergence = deque(maxlen=1000)
        self.cum_diff = deque(maxlen=1000)


    def next(self):
        hs = self.hs
        fhs = self.fhs
        us = self.us
        batched_h = torch.stack([self.hs[i][-1] for i in range(self.batch_size)], dim=0)
        batched_u = torch.stack(self.us, dim=0)
        with torch.no_grad():
            batched_f = self.ar_model(batched_h, u=batched_u).detach()
            batched_next_h = batched_f - self.precon_model(batched_f - batched_h, u=batched_u).detach()

        def reset(i):
            u = torch.empty(*self.data_dim, device=self.device).uniform_(self.lambd, 1. - self.lambd)
            u = torch.log(u) - torch.log(1. - u)
            us[i] = u
            fhs[i] = []
            hs[i] = [torch.zeros(*self.data_dim, device=self.device)]

        for i in range(self.batch_size):
            fhs[i].append(batched_f[i])

            if torch.isnan(batched_h[i]).any() or torch.isnan(batched_f[i]).any() or torch.isnan(batched_next_h[i]).any():
                reset(i)
            elif len(hs[i]) > int(np.prod(self.data_dim)):
                reset(i)
            elif torch.allclose(fhs[i][-1], hs[i][-1]):
                diffs = torch.cumsum(torch.stack([torch.norm(h - hs[i][-1]) for h in hs[i]]), dim=0).sum().item()
                self.cum_diff.append(diffs)
                self.convergence.append(len(hs[i]))
                self.replay_buffer.add(fhs[i], hs[i], us[i])
                reset(i)
            else:
                batched_next_h[i].data.clamp(-self.bound, self.bound)
                hs[i].append(batched_next_h[i])
