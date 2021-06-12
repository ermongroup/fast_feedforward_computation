import os
from made.made import MADE
from made.samplers import sample_logistic, sequential_sampler, jacobi_sampler

from functools import partial
from torchvision.utils import save_image
import shutil
import torch
import numpy as np
import torch.nn.functional as F
import time


class MADESamplerRunner(object):
  def __init__(self, args, config):
    self.args = args
    self.config = config
    self.lambd = 1e-6 if config.dataset == 'MNIST' else 0.05

  def logit_transform(self, image):
    lambd = self.lambd
    image = lambd + (1 - 2 * lambd) * image

    latent_image = torch.log(image) - torch.log1p(-image)
    ll = F.softplus(-latent_image).sum() + F.softplus(latent_image).sum() + np.prod(
      image.shape) * np.log(1 - 2 * lambd)
    nll = -ll
    return latent_image, nll

  def train(self):
    self.sample()

  def sample(self):
    # load MADE model
    obs = (1, 28, 28) if 'MNIST' in self.config.dataset else (3, 32, 32)
    model = MADE(nin=np.prod(obs), hidden_sizes=[self.config.hidden_size] * self.config.hidden_layers,
                 nout=2 * np.prod(obs))

    model = model.to(self.config.device)

    ckpt = torch.load(self.config.ckpt_path, map_location=self.config.device)
    model.load_state_dict(ckpt)
    print('model parameters loaded')
    model.eval()

    sample_batch_size = 100

    x = torch.zeros(sample_batch_size, int(np.prod(obs)), device=self.config.device)
    u = x.new_empty(sample_batch_size, int(np.prod(obs)))
    u.uniform_(1e-5, 1. - 1e-5)
    u = torch.log(u) - torch.log(1. - u)
    rescaling_inv = lambda x: torch.sigmoid(x)

    if os.path.exists(os.path.join(self.args.log, 'images')):
      shutil.rmtree(os.path.join(self.args.log, 'images'))

    os.makedirs(os.path.join(self.args.log, 'images'))
    os.makedirs(self.config.save_folder, exist_ok=True)

    ###### Debugging: Test the inverse CDF sampling code by reconstruction
    # train_loader, _ = dataset.get_dataset(self.config)
    # X, y = next(iter(train_loader))
    # X = X.to('cuda')
    # X = X[:sample_batch_size]
    # with torch.no_grad():
    #     u = inverse_sample_from_discretized_mix_logistic_inverse_CDF(X, model, 10)

    ######
    all_images = []
    all_times = []

    def hook(step, image):
      image = image.reshape(*x.shape)
      torch.cuda.synchronize()
      all_times.append(time.time())
      all_images.append(torch.flatten(image, start_dim=1).cpu().numpy())
      image = rescaling_inv(image)
      image = image.reshape(*((image.shape[0],) + obs))

      save_image(image, os.path.join(self.args.log, 'images', f'sample_{step}.png'),
                 nrow=int(np.sqrt(sample_batch_size)))

    def save_hook(step, image):
      image = image.reshape(*x.shape)
      all_images.append(torch.flatten(image, start_dim=1).cpu().numpy())
      torch.cuda.synchronize()
      all_times.append(time.time())

    filename = 'made_{}_{}.npz'.format({
      'MNIST': 'mnist',
      'CIFAR10': 'cifar'
    }[self.config.dataset], self.config.algo)

    def save_data():
      images = np.stack(all_images, axis=0)
      times = np.stack(all_times, axis=0)
      np.savez(os.path.join(self.config.save_folder, filename), images=images, times=times)

    basic_sampler = partial(sample_logistic, trunc_lambd=self.lambd)
    # gt_x = sequential_sampler(basic_sampler, model, x, u=u, hook=None)

    x = torch.zeros_like(x)
    if self.config.algo == 'sequential':
      sequential_sampler(basic_sampler, model, x, u=u, hook=save_hook)
    elif self.config.algo == 'jacobi':
      jacobi_sampler(basic_sampler, model, x, u=u, hook=save_hook)
    else:
      raise NotImplementedError(f"Sampling algorithm {self.config.algo} does not exists.")

    save_data()