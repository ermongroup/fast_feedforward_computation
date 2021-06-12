import numpy as np
import torch
from itertools import product
import tqdm
import torch.autograd as autograd
from torch.nn import functional as F


def temperature_sampler(basic_sampler, x, u, Tmax=100, n_steps=10, hook=None):
    Ts = np.exp(np.linspace(np.log(Tmax), 0, 10))

    step = 0
    with torch.no_grad():
        for T, i in tqdm.tqdm(product(Ts, range(n_steps)), total=len(Ts) * n_steps):
            if hook is not None:
                hook(step, x)
            x = basic_sampler(x, u=u, T=T)
            step += 1

        return x


def sample_logistic(x, model, u=None, trunc_lambd=None):
    x = x.reshape(x.shape[0], -1)
    l = model(x)
    mean, log_scale = torch.chunk(l, 2, dim=-1)
    if u is None:
        u = torch.rand_like(mean) * (1. - 2e-5) + 1e-5
        u = torch.log(u) - torch.log(1. - u)

    # Use the following if u is uniformly distributed.
    # x = mean + torch.exp(log_scale) * (torch.log(u) - torch.log(1. - u))
    x = mean + torch.exp(log_scale) * u
    if trunc_lambd is not None:
        ub = 1. - trunc_lambd
        lb = trunc_lambd
        ub = np.log(ub) - np.log(1. - ub)
        lb = np.log(lb) - np.log(1. - lb)
        x = torch.clamp(x, lb, ub)
    return x


def sequential_sampler(basic_sampler, model, x, u=None, hook=None, debug_x=None):
    step = 0
    with torch.no_grad():
        for i in tqdm.tqdm(range(int(np.prod(x.shape[1:]))), desc="Sequential sampling"):
            if hook is not None:
                hook(step, x)
            step += 1

            sample = basic_sampler(x, model, u=u)
            x[:, i] = sample[:, i].data

        if hook is not None:
            hook(step, x)
        return x


def jacobi_sampler(basic_sampler, model, x, u, hook=None, debug_x=None):
    step = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(range(int(np.prod(x.shape[1:]))), desc="Jacobi sampling"):
            if hook is not None:
                hook(step, x)
            step += 1

            sample = basic_sampler(x, model, u=u)
            x = sample.data

        if hook is not None:
            hook(step, x)

        return x

def precondition_sampler(basic_sampler, model, precond_model, x, u, hook=None):
    step = 1
    xs = x.shape

    with torch.no_grad():

        if hook is not None:
            hook(0, x)

        for _ in tqdm.tqdm(range(int(np.prod(x.shape[1:])))):

            sample = basic_sampler(x, model, u=u)
            delta = sample.data - x.data
            x = sample.data - precond_model(delta, u)
            x = torch.clamp(x, -100., 100.)
            if torch.isnan(x).any():
                breakpoint()

            if hook is not None:
                hook(step, x)

            step += 1

        return x.reshape(*xs)


def anderson_sampler2(basic_sampler, x, u, hook=None, m=5, beta=1.):
    xs = x.shape
    with torch.no_grad():
        if hook is not None:
            hook(0, x)

        memory = []

        for k in tqdm.tqdm(range(1, x.shape[2] * x.shape[3] + 1)):
            if k <= 1:
                sample = basic_sampler(x, u=u)
                memory.append((x.permute(0, 2, 3, 1).reshape(xs[0], -1),
                               sample.permute(0, 2, 3, 1).reshape(xs[0], -1)))
                x = sample.data

                if len(memory) > m:
                    memory.pop(0)

            else:
                sample = basic_sampler(x, u=u)
                y = (sample - x).permute(0, 2, 3, 1).reshape(xs[0], -1)
                cum_y = y.abs().cumsum(dim=-1)
                cum_y = torch.cat([torch.zeros_like(cum_y[:, 0:1]), cum_y[:, :-1]], dim=-1)
                zero_indices = cum_y == torch.zeros_like(y)
                mask = torch.zeros_like(cum_y)
                mask[~zero_indices] = 1.

                y = y * mask
                diffs = [(pair[1] - pair[0]) * mask for pair in memory]
                X_mat = torch.stack([X - y for X in diffs], dim=1)
                X_mat_T = torch.transpose(X_mat, 1, 2)
                A_mat = torch.matmul(X_mat, X_mat_T)

                y = y[:, :, None]
                b = torch.matmul(X_mat, y)

                lambd = 1e-3 * torch.eye(A_mat.shape[-1], device=A_mat.device).unsqueeze(0)
                A_mat = A_mat + lambd.expand(A_mat.shape[0], -1, -1)
                alphas = torch.matmul(torch.inverse(A_mat), -b)
                alphas = torch.cat([alphas, 1. - torch.sum(alphas, dim=1, keepdim=True)], dim=1)
                G_mat = torch.stack([pair[1] for pair in memory] + [sample.permute(0, 2, 3, 1).reshape(xs[0], -1)], dim=1)
                all_x = torch.stack([pair[0] for pair in memory] + [x.permute(0, 2, 3, 1).reshape(xs[0], -1)], dim=1)
                new_x_1 = torch.sum(all_x * alphas, dim=1)
                new_x_2 = torch.sum(G_mat * alphas, dim=1)

                new_x = (1. - beta) * new_x_1 + beta * new_x_2
                flat_sample = sample.reshape(xs[0], -1)
                next_x = torch.zeros_like(new_x)
                next_x[zero_indices] = flat_sample[zero_indices]
                next_x[~zero_indices] = new_x[~zero_indices]

                memory.append((x.permute(0, 2, 3, 1).reshape(xs[0], -1),
                               sample.permute(0, 2, 3, 1).reshape(xs[0], -1)))

                if len(memory) > m:
                    memory.pop(0)

                x = next_x.reshape(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)

            if hook is not None:
                hook(k, x)

        return x.reshape(*xs)


def anderson_sampler3(basic_sampler, x, u, hook=None, m=5, beta=1.):
    xs = x.shape
    with torch.no_grad():
        if hook is not None:
            hook(0, x)

        memory = []

        # for k in tqdm.tqdm(range(1, x.shape[2] * x.shape[3] + 1)):
        for k in range(1, x.shape[2] * x.shape[3] + 1):
        # for k in range(1, xs[-1] + 1):
            if k <= 1:
                sample = basic_sampler(x, u=u)
                memory.append((x.reshape(xs[0], -1), sample.reshape(xs[0], -1)))
                x = sample.data

                if len(memory) > m:
                    memory.pop(0)

            else:
                sample = basic_sampler(x, u=u)
                y = (sample - x).reshape(x.shape[0], -1)
                diffs = [pair[1] - pair[0] for pair in memory] + [y]
                F_mat = torch.stack([n - p for n, p in zip(diffs[1:], diffs[:-1])], dim=-1)
                F_mat_T = torch.transpose(F_mat, 1, 2)
                A_mat = torch.matmul(F_mat_T, F_mat)

                y = y[:, :, None]
                b = torch.matmul(F_mat_T, y)

                lambd = 1e-5 * torch.eye(A_mat.shape[-1], device=A_mat.device).unsqueeze(0)
                A_mat = A_mat + lambd.expand(A_mat.shape[0], -1, -1)
                inv_A_mat = torch.inverse(A_mat)

                # inv_A_mat = A_mat.clone()
                # for i in range(inv_A_mat.shape[0]):
                #     inv_A_mat[i] = torch.pinverse(A_mat[i])

                norm = torch.norm(A_mat, dim=(1,2), p=2)
                inv_norm = torch.norm(inv_A_mat, dim=(1,2), p=2)
                mean_cond = torch.mean(norm * inv_norm)
                print(f"{k}, {mean_cond.item()}")

                gammas = torch.matmul(inv_A_mat, b)
                G_mat = [pair[1] for pair in memory] + [sample.reshape(x.shape[0], -1)]
                G_mat = torch.stack([n - p for n, p in zip(G_mat[1:], G_mat[:-1])], dim=-1)
                delta = torch.matmul(G_mat, gammas)
                new_x = sample.reshape(xs[0], -1) - delta.squeeze()
                new_x = new_x.reshape(*xs)

                memory.append((x.reshape(xs[0], -1), sample.reshape(xs[0], -1)))

                if len(memory) > m:
                    memory.pop(0)

                x = new_x.data

            if hook is not None:
                hook(k, x)

        return x.reshape(*xs)


def anderson_sampler(basic_sampler, x, u, hook=None, m=5, beta=1.):
    xs = x.shape
    with torch.no_grad():
        if hook is not None:
            hook(0, x)

        memory = []

        # for k in tqdm.tqdm(range(1, x.shape[2] * x.shape[3] + 1)):
        for k in range(1, x.shape[2] * x.shape[3] + 1):
            if k <= 1:
                sample = basic_sampler(x, u=u)
                memory.append((x.reshape(xs[0], -1), sample.reshape(xs[0], -1)))
                x = sample.data

                if len(memory) > m:
                    memory.pop(0)

            else:
                sample = basic_sampler(x, u=u)
                y = (sample - x).reshape(xs[0], -1)
                diffs = [pair[1] - pair[0] for pair in memory]
                X_mat = torch.stack([X - y for X in diffs], dim=1)
                X_mat_T = torch.transpose(X_mat, 1, 2)
                A_mat = torch.matmul(X_mat, X_mat_T)

                y = y[:, :, None]
                b = torch.matmul(X_mat, y)

                lambd = 1e-5 * torch.eye(A_mat.shape[-1], device=A_mat.device).unsqueeze(0)
                A_mat = A_mat + lambd.expand(A_mat.shape[0], -1, -1)
                norm = torch.norm(A_mat, p=2, dim=(1,2))
                inv_norm = torch.norm(torch.inverse(A_mat), p=2, dim=(1,2))
                cond_num = norm * inv_norm
                mean_cond_num = cond_num.mean()
                print(f"{k}, {mean_cond_num.item()}")
                alphas = torch.matmul(torch.inverse(A_mat), -b)
                alphas = torch.cat([alphas, 1. - torch.sum(alphas, dim=1, keepdim=True)], dim=1)
                G_mat = torch.stack([pair[1] for pair in memory] + [sample.reshape(xs[0], -1)], dim=1)
                all_x = torch.stack([pair[0] for pair in memory] + [x.reshape(xs[0], -1)], dim=1)
                new_x_1 = torch.sum(all_x * alphas, dim=1)
                new_x_1 = new_x_1.reshape(*xs)
                new_x_2 = torch.sum(G_mat * alphas, dim=1)
                new_x_2 = new_x_2.reshape(*xs)

                new_x = (1. - beta) * new_x_1 + beta * new_x_2

                memory.append((x.reshape(xs[0], -1), sample.reshape(xs[0], -1)))

                if len(memory) > m:
                    memory.pop(0)

                x = new_x.data

            if hook is not None:
                hook(k, x)

        return x.reshape(*xs)


def continuation_sampler(basic_sampler, x, u0, u, steps=50, hook=None):
    with torch.no_grad():
        lambds = np.linspace(1., 0., steps)

        if hook is not None:
            hook(0, x)
        for step in tqdm.tqdm(range(1, len(lambds) + 1)):

            lambd = lambds[step]
            cur_u = u0 * lambd + u * (1. - lambd)

            for _ in range(80):
                sample = basic_sampler(x, u=cur_u)
                x = sample.data

            if hook is not None:
                hook(step, x)

        return x


def newton_sampler(basic_sampler, x, u, hook=None, lr=1., n_iter=1000):
    with torch.no_grad():
        logit_u = torch.log(u) - torch.log(1. - u)

        if hook is not None:
            hook(0, x)
        for step in tqdm.tqdm(range(1, n_iter + 1)):

            f_x, grad = basic_sampler(x)

            delta = (f_x - logit_u) / grad

            delta = delta.reshape(*x.shape)

            x = x - lr * delta

            if hook is not None:
                hook(step, x)

        return x


def sequential_newton_sampler(basic_sampler, x, u=None, hook=None, lr=0.5, lambd=0.2):
    step = 1
    xs = x.shape
    if hook is not None:
        hook(0, x)

    with torch.no_grad():
        logit_u = torch.log(u) - torch.log(1. - u)

        for i, j in tqdm.tqdm(product(range(x.shape[2]),
                                      range(x.shape[3])),
                              total=int(np.prod(x.shape[2:]))):

            for _ in range(10):
                f_x, grad = basic_sampler(x)

                delta = (f_x - logit_u) / grad
                delta = delta.reshape(*x.shape)
                x[:, :, i, j] -= lr * delta[:, :, i, j]

            if hook is not None:
                hook(step, x)

            step += 1

        return x.reshape(*xs)


def gradient_sampler(basic_sampler, x, hook=None, n_iters=100, lr=0.001):
    step = 0
    for i in tqdm.tqdm(range(n_iters)):
        if hook is not None:
            hook(step, x)

        x.requires_grad_(True)
        sample = basic_sampler(x)
        flattened_sample = torch.flatten(sample, start_dim=1)
        flattened_x = torch.flatten(x, start_dim=1)
        distance = torch.sum((flattened_sample - flattened_x) ** 2, dim=-1)
        distance = distance.mean()
        grad = autograd.grad(distance, x)[0]

        new_x = x - lr * grad
        x = new_x
        step += 1

    return x



def inverse_sample_logistic(x, model):
    x = x.reshape(x.shape[0], -1)
    l = model(x)
    mean, log_scale = torch.chunk(l, 2, dim=-1)
    scale = torch.clamp(torch.exp(log_scale), max=15)
    normalized_x = (x - mean) / scale
    # return torch.sigmoid(normalized_x)
    return normalized_x


def newton_logistic_implicit(l, x, noise):
    mean, log_scale = torch.chunk(l, 2, dim=-1)
    centered_x = (x - mean) * torch.exp(-log_scale)
    z = torch.sigmoid(centered_x)
    J_z = torch.sigmoid(centered_x) * (1. - torch.sigmoid(centered_x)) * torch.exp(-log_scale)
    return z - noise, J_z


def newton_logistic_log_implicit(l, x, noise):
    mean, log_scale = torch.chunk(l, 2, dim=-1)
    centered_x = (x - mean) * torch.exp(-log_scale)
    val = -F.softplus(-centered_x) - torch.log(noise)
    grad = torch.sigmoid(centered_x) * torch.exp(-log_scale)
    return val, grad


def newton_logistic(l, x, noise):
    """
    clearly works better when the function is reformulated a bit to make gradients 1s.
    """
    mean, log_scale = torch.chunk(l, 2, dim=-1)
    inv_sigmoid_noise = torch.log(noise) - torch.log1p(-noise)
    scale = torch.clamp(torch.exp(log_scale), max=5)
    val = (x - mean) - scale * inv_sigmoid_noise
    return val, torch.ones_like(val)