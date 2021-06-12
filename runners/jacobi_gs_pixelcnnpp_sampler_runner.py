import os
from jacobi_gs_pixelcnn.jacobi_gs_pixelcnnpp import PixelCNN
import time
import flax
from functools import partial
from jacobi_gs_pixelcnn.jacobi_gs_samplers import *


class JacobiGSPixelCNNPPSamplerRunner(object):
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

    def model_fn_first(cache, x_row_input, x_pixel_input, rows, cols):
      l, params = self.ar_model.apply({'params': self.ar_model_params, 'cache': cache}, x_row_input, x_pixel_input,
                                      rows, cols,
                                      first_index=True, mutable=['cache'], train=False)
      return l, params['cache']

    def model_fn(cache, x_row_input, x_pixel_input, rows, cols):
      l, params = self.ar_model.apply({'params': self.ar_model_params, 'cache': cache}, x_row_input, x_pixel_input,
                                      rows, cols,
                                      first_index=False, mutable=['cache'], train=False)
      return l, params['cache']

    rescaling_inv = lambda x: .5 * x + .5
    rescaling = lambda x: (x - .5) * 2.
    if self.config.dataset == 'CIFAR10':
      x = jnp.zeros((sample_batch_size, 32, 32, 3))
      clamp = True
      basic_sampler = partial(sample_from_discretized_mix_logistic,
                              nr_mix=self.config.nr_logistic_mix, clamp=clamp)
      u = jax.random.uniform(step_rng,
                             (sample_batch_size, 32 * 32 * (self.config.nr_logistic_mix + 3)),
                             minval=1e-5, maxval=1. - 1e-5)

    elif 'MNIST' in self.config.dataset:
      x = jnp.zeros((sample_batch_size, 28, 28, 1))
      clamp = True
      basic_sampler = partial(sample_from_discretized_mix_logistic_1d,
                              nr_mix=self.config.nr_logistic_mix, clamp=clamp)
      u = jax.random.uniform(step_rng,
                             (sample_batch_size, 28 * 28 * (self.config.nr_logistic_mix + 1)),
                             minval=1e-5, maxval=1. - 1e-5)

    if self.config.algo == 'jacobi_gs':
      sampler = get_jacobi_gs_sampler(basic_sampler, model_fn_first, model_fn,
                                      self.initial_cache, x.shape, self.config.block_size,
                                      nr_logistic_mix=self.config.nr_logistic_mix)
    else:
      raise NotImplementedError()

    # Force jit compiling
    results = sampler(u)
    results[0].block_until_ready()

    all_samples = []
    all_times = []
    for i in range(10):
      begin_time = time.time()
      final_sample, samples = sampler(u)
      final_sample.block_until_ready()
      end_time = time.time()
      all_samples.append(rescaling_inv(samples))
      all_times.append(end_time - begin_time)
      self.args.rng, step_rng = jax.random.split(self.args.rng)
      u = jax.random.uniform(step_rng, u.shape, minval=1e-5, maxval=1. - 1e-5)

    folder = os.path.join(self.config.save_folder, f'{self.config.dataset.lower()}_{self.config.algo.lower()}')
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, 'report.npz'), 'wb') as fout:
      np.savez_compressed(fout, all_samples=all_samples, all_times=all_times)

  def load_pixelcnnpp(self):
    obs = (28, 28, 1) if 'MNIST' in self.config.dataset else (32, 32, 3)
    input_channels = obs[-1]
    block_size = self.config.block_size

    assert np.prod(obs[:-1]) % block_size == 0, "The number of pixels must be divisible by the block size."
    num_blocks = np.prod(obs[:-1]) // block_size
    model = PixelCNN(batch_size=self.config.sample_batch_size,
                     num_blocks=num_blocks,
                     image_width=obs[0],
                     nr_resnet=self.config.nr_resnet, nr_filters=self.config.nr_filters,
                     input_channels=input_channels, nr_logistic_mix=self.config.nr_logistic_mix)

    fake_row_input = jnp.zeros((self.config.sample_batch_size, obs[0], obs[1], obs[-1] + 1))
    fake_pixel_input = jnp.zeros((self.config.sample_batch_size, obs[0], obs[1], obs[-1] + 1))
    fake_rows = jnp.arange(0, num_blocks)
    fake_cols = jnp.arange(0, num_blocks)

    self.args.rng, step_rng = jax.random.split(self.args.rng)
    params_rng, dropout_rng = jax.random.split(step_rng)
    params = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_row_input, fake_pixel_input,
                        fake_rows, fake_cols, first_index=True)

    with open(self.config.ckpt_path, 'rb') as fin:
      ar_model_params = flax.serialization.from_bytes(params['params'], fin.read())

    print('model parameters loaded')
    self.ar_model = model
    self.ar_model_params = ar_model_params
    self.initial_cache = params['cache']

  def test(self):
    pass
