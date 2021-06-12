import numpy as np
from pixelcnnpp.layers import to_one_hot
import jax
import jax.random as random
import jax.numpy as jnp


def get_sequential_sampler(basic_sampler, shape):

  def sequential_sampler(rng, u):
    def loop_fn(carry, index):
      (x, rng) = carry
      rng, step_rng = random.split(rng)
      sample = basic_sampler(step_rng, x, u=u)
      i = index // shape[1]
      j = index % shape[1]
      x = x.at[:, i, j, :].set(sample[:, i, j, :])
      return (x, rng), x

    init_x = jnp.zeros(shape)
    _, all_imgs = jax.lax.scan(loop_fn, (init_x, rng), jnp.arange(0, shape[1] * shape[2]))
    return all_imgs

  return jax.jit(sequential_sampler)


def get_jacobi_sampler(basic_sampler, shape):

  def jacobi_sampler(rng, u):
    def loop_fn(carry, index):
      (x, rng) = carry
      rng, step_rng = random.split(rng)
      new_x = basic_sampler(step_rng, x, u=u)
      return (new_x, rng), new_x

    init_x = jnp.zeros(shape)
    _, all_imgs = jax.lax.scan(loop_fn, (init_x, rng), jnp.arange(0, shape[1] * shape[2]))
    return all_imgs

  return jax.jit(jacobi_sampler)


def sample_from_discretized_mix_logistic_1d(rng, x, model, nr_mix, u=None, clamp=True):
  l = model(x)
  ls = l.shape
  xs = ls[:-1] + (1, )  # [3]

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 2,))  # for mean, scale

  # sample mixture indicator from softmax
  if u is None:
    rng, step_rng = random.split(rng)
    u = random.uniform(step_rng, (l.shape[0], l.shape[1] * l.shape[2] * (nr_mix + 1)),
                       minval=1e-5, maxval=1. - 1e-5)

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


def sample_from_discretized_mix_logistic(rng, x, model, nr_mix, u=None, T=1, clamp=True):
  # Pytorch ordering
  l = model(x)
  ls = l.shape
  xs = ls[:-1] + (3,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix] / T
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 3,))
  # sample mixture indicator from softmax
  if u is None:
    rng, step_rng = random.split(rng)
    u = random.uniform(step_rng, (l.shape[0], l.shape[1] * l.shape[2] * (nr_mix + 3)),
                       minval=1e-5, maxval=1. - 1e-5)

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


def binary_search(log_cdf, lb, ub, cdf_fun, n_iter=15):
  def loop_fn(carry, _):
    lb, ub = carry
    mid = (lb + ub) / 2.
    mid_cdf_value = cdf_fun(mid)
    right_idxes = mid_cdf_value < log_cdf
    left_idxes = ~right_idxes
    lb = jnp.where(right_idxes, jnp.minimum(mid, ub), lb)
    ub = jnp.where(left_idxes, jnp.maximum(mid, lb), ub)
    return (lb, ub), _

  (lb, ub), _ = jax.lax.scan(loop_fn, (lb, ub), None, length=n_iter)

  return (lb + ub) / 2.


def sample_from_discretized_mix_logistic_inverse_CDF_1d(rng, x, model, nr_mix, u=None, clamp=True, bisection_iter=15):
  # Pytorch ordering

  l = model(x)
  ls = l.shape
  xs = ls[:-1] + (1,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 2,))
  # sample mixture indicator from softmax
  if u is None:
    rng, step_rng = random.split(rng)
    u = random.uniform(step_rng, (l.shape[0], l.shape[1] * ls.shape[2]),
                       minval=1e-5, maxval=1. - 1e-5)
    u = jnp.log(u) - jnp.log(1. - u)

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


def sample_from_discretized_mix_logistic_inverse_CDF_1_mixture_1d(rng, x, model, nr_mix, u=None):
  # Pytorch ordering
  l = model(x)
  ls = l.shape
  xs = ls[:-1] + (1,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 2,))
  # sample mixture indicator from softmax
  if u is None:
    rng, step_rng = random.split(rng)
    u = random.uniform(step_rng,
                       (l.shape[0], l.shape[1] * l.shape[2]),
                       minval=1e-5, maxval=1. - 1e-5)
    u = jnp.log(u) - jnp.log(1. - u)

  u_r = u.reshape(ls[:-1])

  log_softmax = jax.nn.log_softmax(logit_probs, axis=-1)
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)

  means_r = means[..., 0, :]
  log_scales_r = log_scales[..., 0, :]

  x0 = jnp.exp(log_scales_r) * u_r[..., None] + means_r
  out = x0.reshape(xs[:-1] + (1,))
  return jnp.clip(out, a_min=-1., a_max=1.)


def sample_from_discretized_mix_logistic_inverse_CDF(rng, x, model, nr_mix, u=None, clamp=True, bisection_iter=15, T=1):
  # Pytorch ordering
  l = model(x)
  ls = l.shape
  xs = ls[:-1] + (3,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix] / T
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 3,))
  # sample mixture indicator from softmax
  if u is None:
    rng, step_rng = random.split(rng)
    u = random.uniform(step_rng, (l.shape[0], l.shape[1] * l.shape[2] * 3),
                       minval=1e-5, maxval=1. - 1e-5)
    u = jnp.log(u) - jnp.log(1. - u)

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


def inverse_sample_from_discretized_mix_logistic_inverse_CDF_1d(sample, model, nr_mix):
  # Pytorch ordering
  l = model(sample)
  ls = l.shape
  xs = ls[:-1] + (1,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + (2 * nr_mix,))

  # sample mixture indicator from softmax
  log_softmax = jax.nn.log_softmax(logit_probs, axis=-1)
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)
  x0 = sample[:, 0, ...]

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

  u0 = log_cdf_pdf_r(x0, mode='cdf', mixtures=False)

  u = u0.reshape(u0.shape[0], -1)
  return u


def inverse_sample_from_discretized_mix_logistic_inverse_CDF(sample, model, nr_mix, T=1):
  # Pytorch ordering
  l = model(sample)
  ls = l.shape
  xs = ls[:-1] + (3,)

  # unpack parameters
  logit_probs = l[:, :, :, :nr_mix] / T
  l = l[:, :, :, nr_mix:].reshape(xs + (nr_mix * 3,))

  # sample mixture indicator from softmax
  log_softmax = jax.nn.log_softmax(logit_probs, axis=-1)
  coeffs = jnp.tanh(l[:, :, :, :, 2 * nr_mix: 3 * nr_mix])
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.) + np.log(T)
  x0 = sample[:, 0, ...]
  x1 = sample[:, 1, ...]
  x2 = sample[:, 2, ...]

  means_r = means[..., 0, :]
  log_scales_r = log_scales[..., 0, :]

  def log_cdf_pdf_r(values, mode='cdf', mixtures=False):
    values = values[..., None]
    centered_values = (values - means_r) / jnp.exp(log_scales_r)

    if mode == 'cdf':
      log_logistic_cdf = centered_values - jax.nn.softplus(centered_values)
      log_logistic_sf = -jax.nn.softplus(centered_values)
      log_cdf = jax.nn.logsumexp(log_softmax + log_logistic_cdf, axis=-1)
      log_sf = jax.nn.logsumexp(log_softmax + log_logistic_sf, axis=-1)
      logit = log_cdf - log_sf

      return logit if not mixtures else (logit, log_logistic_cdf)

    elif mode == 'pdf':
      log_logistic_pdf = centered_values - log_scales_r - 2. * jax.nn.softplus(centered_values)
      log_pdf = jax.nn.logsumexp(log_softmax + log_logistic_pdf, axis=-1)

      return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

  u0 = log_cdf_pdf_r(x0, mode='cdf', mixtures=False)

  means_g = x0[..., None] * coeffs[:, :, :, 0, :] + means[..., 1, :]
  log_scales_g = log_scales[..., 1, :]

  log_p_r, log_p_r_mixtures = log_cdf_pdf_r(x0, mode='pdf', mixtures=True)

  def log_cdf_pdf_g(values, mode='cdf', mixtures=False):
    values = values[..., None]
    centered_values = (values - means_g) / jnp.exp(log_scales_g)

    if mode == 'cdf':
      log_logistic_cdf = log_p_r_mixtures - log_p_r[..., None] + centered_values - jax.nn.softplus(centered_values)
      log_logistic_sf = log_p_r_mixtures - log_p_r[..., None] - jax.nn.softplus(centered_values)
      log_cdf = jax.nn.logsumexp(log_softmax + log_logistic_cdf, axis=-1)
      log_sf = jax.nn.logsumexp(log_softmax + log_logistic_sf, axis=-1)
      logit = log_cdf - log_sf

      return logit if not mixtures else (logit, log_logistic_cdf)

    elif mode == 'pdf':
      log_logistic_pdf = log_p_r_mixtures - log_p_r[..., None] + centered_values - log_scales_g - 2. * jax.nn.softplus(
        centered_values)
      log_pdf = jax.nn.logsumexp(log_softmax + log_logistic_pdf, axis=-1)

      return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

  u1 = log_cdf_pdf_g(x1, mode='cdf', mixtures=False)
  means_b = x1[..., None] * coeffs[:, :, :, 2, :] + x0[..., None] * coeffs[:, :, :, 1, :] + means[..., 2, :]
  log_scales_b = log_scales[..., 2, :]

  log_p_g, log_p_g_mixtures = log_cdf_pdf_g(x1, mode='pdf', mixtures=True)

  def log_cdf_pdf_b(values, mode='cdf', mixtures=False):
    values = values[..., None]
    centered_values = (values - means_b) / jnp.exp(log_scales_b)

    if mode == 'cdf':
      log_logistic_cdf = log_p_g_mixtures - log_p_g[..., None] + centered_values - jax.nn.softplus(centered_values)
      log_logistic_sf = log_p_g_mixtures - log_p_g[..., None] - jax.nn.softplus(centered_values)
      log_cdf = jax.nn.logsumexp(log_softmax + log_logistic_cdf, axis=-1)
      log_sf = jax.nn.logsumexp(log_softmax + log_logistic_sf, axis=-1)
      logit = log_cdf - log_sf

      return logit if not mixtures else (logit, log_logistic_cdf)

    elif mode == 'pdf':
      log_logistic_pdf = log_p_g_mixtures - log_p_g[..., None] + centered_values - log_scales_b - 2. * jax.nn.softplus(
        centered_values)
      log_pdf = jax.nn.logsumexp(log_softmax + log_logistic_pdf, axis=-1)

      return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

  u2 = log_cdf_pdf_b(x2, mode='cdf', mixtures=False)

  u = jnp.concatenate([u0.reshape((u0.shape[0], -1)),
                       u1.reshape((u1.shape[0], -1)),
                       u2.reshape((u2.shape[0], -1))], axis=-1)
  return u

