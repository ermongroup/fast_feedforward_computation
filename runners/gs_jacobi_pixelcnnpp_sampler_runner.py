import time
import os
from gs_jacobi_pixelcnn.gs_jacobi_pixelcnnpp import PixelCNN

from functools import partial
from gs_jacobi_pixelcnn.gs_jacobi_samplers import *
import jax.numpy as jnp
import jax
import flax
import numpy as np


class GSJacobiPixelCNNPPSamplerRunner(object):
  def __init__(self, args, config):
    self.args = args
    self.config = config

  def logit_transform(self, image, lambd=1e-6):
    image = .5 * image + .5
    image = lambd + (1 - 2 * lambd) * image

    latent_image = jnp.log(image) - jnp.log1p(-image)
    ll = jax.nn.softplus(-latent_image).sum() + jax.nn.softplus(latent_image).sum() + np.prod(
      image.shape) * (np.log(1 - 2 * lambd) + np.log(.5))
    nll = -ll
    return latent_image, nll

  def train(self):
    self.load_pixelcnnpp()
    self.sample()

  def sample(self):
    sample_batch_size = self.config.sample_batch_size

    self.args.rng, step_rng = jax.random.split(self.args.rng)

    def model_fn(cache, x_row_input, x_pixel_input, row_start):
      l, params = self.ar_model.apply({'params': self.ar_model_params, 'cache': cache}, x_row_input, x_pixel_input,
                                      row_start, mutable=['cache'], train=False)
      return l, params['cache']

    rescaling_inv = lambda x: .5 * x + .5
    rescaling = lambda x: (x - .5) * 2.
    max_rows = self.config.max_rows

    if self.config.dataset == 'CIFAR10':
      x = jnp.zeros((sample_batch_size, 32, 32, 3))
      clamp = True
      basic_sampler = partial(sample_from_discretized_mix_logistic,
                              nr_mix=self.config.nr_logistic_mix, clamp=clamp)
      pad_rng = jax.random.PRNGKey(0)
      pad_u = jax.random.uniform(pad_rng, (sample_batch_size, max_rows * 32 * (self.config.nr_logistic_mix + 3)),
                                 minval=1e-5, maxval=1. - 1e-5)
      u = jax.random.uniform(step_rng,
                             (sample_batch_size, 32 * 32 * (self.config.nr_logistic_mix + 3)),
                             minval=1e-5, maxval=1. - 1e-5)

    elif 'MNIST' in self.config.dataset:
      x = jnp.zeros((sample_batch_size, 28, 28, 1))
      clamp = True
      basic_sampler = partial(sample_from_discretized_mix_logistic_1d,
                              nr_mix=self.config.nr_logistic_mix, clamp=clamp)
      pad_rng = jax.random.PRNGKey(0)
      pad_u = jax.random.uniform(pad_rng, (sample_batch_size, max_rows * 28 * (self.config.nr_logistic_mix + 1)),
                                 minval=1e-5, maxval=1. - 1e-5)
      u = jax.random.uniform(step_rng,
                             (sample_batch_size, 28 * 28 * (self.config.nr_logistic_mix + 1)),
                             minval=1e-5, maxval=1. - 1e-5)

    if self.config.algo == 'gs_jacobi':
      sampler = get_gs_jacobi_sampler(basic_sampler, model_fn, self.initial_cache, x.shape,
                                      max_rows,
                                      nr_logistic_mix=self.config.nr_logistic_mix)

    else:
      raise NotImplementedError()

    def merge(new_u, pad_u):
      mixture_u, sample_u = jnp.split(new_u, [x.shape[1] * x.shape[2] * self.config.nr_logistic_mix], axis=-1)
      mixture_pad_u, sample_pad_u = jnp.split(pad_u, [max_rows * x.shape[2] * self.config.nr_logistic_mix], axis=-1)
      return jnp.concatenate([mixture_u, mixture_pad_u, sample_u, sample_pad_u], axis=-1)

    # Force jit compiling
    (samples, rows) = sampler(merge(u, pad_u))
    samples.block_until_ready()
    all_samples = []
    all_times = []
    all_row_ids = []

    for i in range(10):
      begin_time = time.time()
      (samples, row_ids) = sampler(merge(u, pad_u))
      samples.block_until_ready()
      end_time = time.time()
      all_samples.append(rescaling_inv(samples))
      all_times.append(end_time - begin_time)
      all_row_ids.append(row_ids)
      self.args.rng, step_rng = jax.random.split(self.args.rng)
      u = jax.random.uniform(step_rng, u.shape, minval=1e-5, maxval=1. - 1e-5)

    folder = os.path.join(self.config.save_folder, f'{self.config.dataset.lower()}_{self.config.algo.lower()}')
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, 'report.npz'), 'wb') as fout:
      np.savez_compressed(fout, all_samples=all_samples, all_times=all_times, all_row_ids=all_row_ids)

  def load_pixelcnnpp(self):
    obs = (28, 28, 1) if 'MNIST' in self.config.dataset else (32, 32, 3)
    input_channels = obs[-1]
    max_rows = self.config.max_rows
    model = PixelCNN(batch_size=self.config.sample_batch_size,
                     max_rows=max_rows,
                     image_width=obs[0],
                     nr_resnet=self.config.nr_resnet, nr_filters=self.config.nr_filters,
                     input_channels=input_channels, nr_logistic_mix=self.config.nr_logistic_mix)

    fake_row_input = jnp.zeros((self.config.sample_batch_size, max_rows, obs[1], obs[-1] + 1))
    fake_pixel_input = jnp.zeros((self.config.sample_batch_size, max_rows, obs[1], obs[-1] + 1))

    self.args.rng, step_rng = jax.random.split(self.args.rng)
    params_rng, dropout_rng = jax.random.split(step_rng)
    params = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_row_input, fake_pixel_input,
                        jnp.asarray(0))

    with open(self.config.ckpt_path, 'rb') as fin:
      ar_model_params = flax.serialization.from_bytes(params['params'], fin.read())

    print('model parameters loaded')
    self.ar_model = model
    self.ar_model_params = ar_model_params
    self.initial_cache = params['cache']

  def test(self):
    pass
