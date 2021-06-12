import numpy as np
import jax.numpy as jnp
import jax

from pixelcnnpp.layers import to_one_hot
from pixelcnnpp.samplers import binary_search


def get_sequential_sampler(basic_sampler, model_fn, initial_cache, shape, nr_logistic_mix=10):
  image_size = shape[1]

  def sequential_sampler(u):
    def loop_fn(carry, index):
      output_images, cache = carry
      row = index // image_size
      col = index % image_size

      x_row_input = jnp.where(row == 0, jnp.zeros((shape[0], 1, image_size, shape[-1] + 1)),
                              jnp.pad(jax.lax.dynamic_slice(output_images, [0, row - 1, 0, 0],
                                                            [shape[0], 1, shape[2], shape[3]]),
                                      ((0, 0), (0, 0), (0, 0), (0, 1)),
                                      mode='constant', constant_values=1.))

      x_pixel_input = jnp.where(col == 0, jnp.zeros((shape[0], 1, 1, shape[-1] + 1)),
                                jnp.pad(jax.lax.dynamic_slice(output_images, [0, row, col - 1, 0],
                                                              [shape[0], 1, 1, shape[-1]]),
                                        ((0, 0), (0, 0), (0, 0), (0, 1)),
                                        mode='constant', constant_values=1.))

      mixture_u, sample_u = jnp.split(u, [image_size * image_size * nr_logistic_mix], axis=-1)

      start_mixture = row * image_size * nr_logistic_mix + col * nr_logistic_mix
      start_sample = row * image_size * shape[-1] + col * shape[-1]

      u_pixel_input = jnp.concatenate((
        jax.lax.dynamic_slice(mixture_u, [0, start_mixture], [mixture_u.shape[0], nr_logistic_mix]),
        jax.lax.dynamic_slice(sample_u, [0, start_sample], [sample_u.shape[0], shape[-1]])), axis=-1)

      l, cache = model_fn(cache, x_row_input, x_pixel_input, jnp.asarray(row), jnp.asarray(col))
      output_images = jax.lax.dynamic_update_slice(output_images,
                                                   basic_sampler(l, u=u_pixel_input),
                                                   [0, row, col, 0])
      return (output_images, cache), output_images

    output_images = jnp.zeros(shape)
    _, all_images = jax.lax.scan(loop_fn, (output_images, initial_cache), jnp.arange(0, image_size ** 2))
    return all_images

  return jax.jit(sequential_sampler)


def sample_from_discretized_mix_logistic_1d(l, u, nr_mix, clamp=True):
  ls = l.shape
  xs = ls[:-1] + (1,)  # [3]

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 2, ))  # for mean, scale

  mixture_u, sample_u = jnp.split(u, [l.shape[1] * l.shape[2] * nr_mix], axis=-1)
  mixture_u = mixture_u.reshape((l.shape[0], l.shape[1], l.shape[2], nr_mix))
  sample_u = sample_u.reshape((l.shape[0], l.shape[1], l.shape[2], 1))

  mixture_u = logit_probs - jnp.log(- jnp.log(mixture_u))
  argmax = jnp.argmax(mixture_u, axis=3)
  one_hot = to_one_hot(argmax, nr_mix)
  sel = one_hot.reshape(xs[:-1] + (1, nr_mix))
  # select logistic parameters
  means = jnp.sum(l[:, :, :, :, :nr_mix] * sel, axis=4)
  log_scales = jnp.clip(jnp.sum(
    l[:, :, :, :, nr_mix:2 * nr_mix] * sel, axis=4), a_min=-7.)

  x = means + jnp.exp(log_scales) * (jnp.log(sample_u) - jnp.log(1. - sample_u))
  if clamp:
    x0 = jnp.clip(x[:, :, :, 0], a_min=-1., a_max=1.)
  else:
    x0 = x[:, :, :, 0]

  out = jnp.expand_dims(x0, axis=-1)
  return out


def sample_from_discretized_mix_logistic(l, u, nr_mix, T=1, clamp=True):
  # Pytorch ordering
  ls = l.shape
  xs = ls[:-1] + (3,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix] / T
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 3,))
  # sample mixture indicator from softmax
  mixture_u, sample_u = jnp.split(u, [l.shape[1] * l.shape[2] * nr_mix], axis=-1)
  mixture_u = mixture_u.reshape((l.shape[0], l.shape[1], l.shape[2], nr_mix))
  sample_u = sample_u.reshape((l.shape[0], l.shape[1], l.shape[2], 3))

  mixture_u = logit_probs - jnp.log(- jnp.log(mixture_u))
  argmax = jnp.argmax(mixture_u, axis=3)

  one_hot = to_one_hot(argmax, nr_mix)
  sel = one_hot.reshape(xs[:-1] + (1, nr_mix))
  # select logistic parameters
  means = jnp.sum(l[:, :, :, :, :nr_mix] * sel, axis=4)
  log_scales = jnp.clip(jnp.sum(
    l[:, :, :, :, nr_mix:2 * nr_mix] * sel, axis=4), a_min=-7.) + np.log(T)
  coeffs = jnp.sum(jnp.tanh(
    l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, axis=4)
  # sample from logistic & clip to interval
  # we don't actually round to the nearest 8bit value when sampling

  x = means + jnp.exp(log_scales) * (jnp.log(sample_u) - jnp.log(1. - sample_u))
  if clamp:
    x0 = jnp.clip(x[:, :, :, 0], a_min=-1., a_max=1.)
  else:
    x0 = x[:, :, :, 0]

  if clamp:
    x1 = jnp.clip(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, a_min=-1., a_max=1.)
  else:
    x1 = x[:, :, :, 1] + coeffs[:, :, :, 0] * x0

  if clamp:
    x2 = jnp.clip(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, a_min=-1., a_max=1.)
  else:
    x2 = x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1

  out = jnp.concatenate([x0.reshape(xs[:-1] + (1,)), x1.reshape(xs[:-1] + (1,)), x2.reshape(xs[:-1] + (1,))], axis=3)
  return out


def sample_from_discretized_mix_logistic_inverse_CDF_1d(l, u, nr_mix, clamp=True, bisection_iter=15):
  ls = l.shape
  xs = ls[:-1] + (1,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 2,))
  u_r = u.reshape(ls[:-1])

  log_softmax = jax.nn.log_softmax(logit_probs, axis=-1)
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)
  if clamp is True:
    ubs = jnp.ones(ls[:-1])
    lbs = -ubs
  else:
    ubs = jnp.ones(ls[:-1]) * 30.
    lbs = -ubs

  means_r = means[..., 0, :]
  log_scales_r = log_scales[..., 0, :]

  def log_cdf_pdf_r(values, mode='cdf', mixtures=False):
    values = values[..., None]
    centered_values = (values - means_r) / jnp.exp(log_scales_r)

    if mode == 'cdf':
      log_logistic_cdf = -jax.nn.softplus(-centered_values)
      log_logistic_sf = -jax.nn.softplus(centered_values)
      log_cdf = jax.nn.logsumexp(log_softmax + log_logistic_cdf, axis=-1)
      log_sf = jax.nn.logsumexp(log_softmax + log_logistic_sf, axis=-1)
      logit = log_cdf - log_sf

      return logit if not mixtures else (logit, log_logistic_cdf)

    elif mode == 'pdf':
      log_logistic_pdf = -centered_values - log_scales_r - 2. * jax.nn.softplus(-centered_values)
      log_pdf = jax.nn.logsumexp(log_softmax + log_logistic_pdf, axis=-1)

      return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

  x0 = binary_search(u_r, lbs, ubs, lambda x: log_cdf_pdf_r(x, mode='cdf'), bisection_iter)

  out = x0.reshape(xs[:-1] + (1,))
  return out


def sample_from_discretized_mix_logistic_inverse_CDF_1_mixture_1d(l, u, nr_mix):
  ls = l.shape
  xs = ls[:-1] + (1,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 2,))
  u_r = u.reshape(ls[:-1])

  log_softmax = jax.nn.log_softmax(logit_probs, axis=-1)
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)

  means_r = means[..., 0, :]
  log_scales_r = log_scales[..., 0, :]

  x0 = jnp.exp(log_scales_r) * u_r[..., None] + means_r
  out = x0.reshape(xs[:-1] + (1,))
  return jnp.clip(out, a_min=-1., a_max=1.)


def sample_from_discretized_mix_logistic_inverse_CDF(l, u, nr_mix, clamp=True, bisection_iter=15, T=1):
  ls = l.shape
  xs = ls[:-1] + (3,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix] / T
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 3,))
  # sample mixture indicator from softmax
  u_r, u_g, u_b = jnp.split(u, 3, axis=-1)

  u_r = u_r.reshape(ls[:-1])
  u_g = u_g.reshape(ls[:-1])
  u_b = u_b.reshape(ls[:-1])

  log_softmax = jax.nn.log_softmax(logit_probs, axis=-1)
  coeffs = jnp.tanh(l[:, :, :, :, 2 * nr_mix: 3 * nr_mix])
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.) + np.log(T)
  if clamp:
    ubs = jnp.ones(ls[:-1])
    lbs = -ubs
  else:
    ubs = jnp.ones(ls[:-1]) * 20.
    lbs = -ubs

  means_r = means[..., 0, :]
  log_scales_r = log_scales[..., 0, :]

  def log_cdf_pdf_r(values, mode='cdf', mixtures=False):
    values = values[..., None]
    centered_values = (values - means_r) / jnp.exp(log_scales_r)

    if mode == 'cdf':
      log_logistic_cdf = -jax.nn.softplus(-centered_values)
      log_logistic_sf = -jax.nn.softplus(centered_values)
      log_cdf = jax.nn.logsumexp(log_softmax + log_logistic_cdf, axis=-1)
      log_sf = jax.nn.logsumexp(log_softmax + log_logistic_sf, axis=-1)
      logit = log_cdf - log_sf

      return logit if not mixtures else (logit, log_logistic_cdf)

    elif mode == 'pdf':
      log_logistic_pdf = -centered_values - log_scales_r - 2. * jax.nn.softplus(-centered_values)
      log_pdf = jax.nn.logsumexp(log_softmax + log_logistic_pdf, axis=-1)

      return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

  x0 = binary_search(u_r, lbs, ubs, lambda x: log_cdf_pdf_r(x, mode='cdf'), bisection_iter)

  means_g = x0[..., None] * coeffs[:, :, :, 0, :] + means[..., 1, :]
  log_scales_g = log_scales[..., 1, :]

  log_p_r, log_p_r_mixtures = log_cdf_pdf_r(x0, mode='pdf', mixtures=True)

  def log_cdf_pdf_g(values, mode='cdf', mixtures=False):
    values = values[..., None]
    centered_values = (values - means_g) / jnp.exp(log_scales_g)

    if mode == 'cdf':
      log_logistic_cdf = log_p_r_mixtures - log_p_r[..., None] - jax.nn.softplus(-centered_values)
      log_logistic_sf = log_p_r_mixtures - log_p_r[..., None] - jax.nn.softplus(centered_values)
      log_cdf = jax.nn.logsumexp(log_softmax + log_logistic_cdf, axis=-1)
      log_sf = jax.nn.logsumexp(log_softmax + log_logistic_sf, axis=-1)
      logit = log_cdf - log_sf

      return logit if not mixtures else (logit, log_logistic_cdf)

    elif mode == 'pdf':
      log_logistic_pdf = log_p_r_mixtures - log_p_r[..., None] - centered_values - log_scales_g - 2. * jax.nn.softplus(
        -centered_values)
      log_pdf = jax.nn.logsumexp(log_softmax + log_logistic_pdf, axis=-1)

      return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

  x1 = binary_search(u_g, lbs, ubs, lambda x: log_cdf_pdf_g(x, mode='cdf'), bisection_iter)

  means_b = x1[..., None] * coeffs[:, :, :, 2, :] + x0[..., None] * coeffs[:, :, :, 1, :] + means[..., 2, :]
  log_scales_b = log_scales[..., 2, :]

  log_p_g, log_p_g_mixtures = log_cdf_pdf_g(x1, mode='pdf', mixtures=True)

  def log_cdf_pdf_b(values, mode='cdf', mixtures=False):
    values = values[..., None]
    centered_values = (values - means_b) / jnp.exp(log_scales_b)

    if mode == 'cdf':
      log_logistic_cdf = log_p_g_mixtures - log_p_g[..., None] - jax.nn.softplus(-centered_values)
      log_logistic_sf = log_p_g_mixtures - log_p_g[..., None] - jax.nn.softplus(centered_values)
      log_cdf = jax.nn.logsumexp(log_softmax + log_logistic_cdf, axis=-1)
      log_sf = jax.nn.logsumexp(log_softmax + log_logistic_sf, axis=-1)
      logit = log_cdf - log_sf

      return logit if not mixtures else (logit, log_logistic_cdf)

    elif mode == 'pdf':
      log_logistic_pdf = log_p_g_mixtures - log_p_g[..., None] - centered_values - log_scales_b - 2. * jax.nn.softplus(
        -centered_values)
      log_pdf = jax.nn.logsumexp(log_softmax + log_logistic_pdf, axis=-1)

      return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

  x2 = binary_search(u_b, lbs, ubs, lambda x: log_cdf_pdf_b(x, mode='cdf'), bisection_iter)

  out = jnp.concatenate([x0.reshape(xs[:-1] + (1,)), x1.reshape(xs[:-1] + (1,)), x2.reshape(xs[:-1] + (1,))], axis=3)
  return out
