from pixelcnnpp.layers import *
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
import numpy as np


class PixelCNNLayer_up(nn.Module):
  nr_resnet: int
  nr_filters: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, train=True):
    u_list, ul_list = [], []
    for i in range(self.nr_resnet):
      u = gated_resnet(self.nr_filters,
                       down_shifted_conv2d,
                       self.resnet_nonlinearity)(u, train=train)
      ul = gated_resnet(self.nr_filters,
                        down_right_shifted_conv2d,
                        self.resnet_nonlinearity)(ul, a=u, train=train)
      u_list += [u]
      ul_list += [ul]

    return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
  nr_resnet: int
  nr_filters: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, u_list, ul_list, train=True):
    for i in range(self.nr_resnet):
      u = gated_resnet(self.nr_filters,
                       down_shifted_conv2d,
                       self.resnet_nonlinearity)(u, a=u_list.pop(), train=train)
      ul = gated_resnet(self.nr_filters,
                        down_right_shifted_conv2d,
                        self.resnet_nonlinearity)(ul, a=jnp.concatenate((u, ul_list.pop()), axis=-1),
                                                  train=train)
    return u, ul


class PixelCNN(nn.Module):
  nr_resnet: int = 5
  nr_filters: int = 160
  nr_logistic_mix: int = 10
  resnet_nonlinearity: Any = 'concat_elu'
  input_channels: int = 3
  init_padding: Any = None

  @nn.compact
  def __call__(self, x, sample=False, train=True):
    if self.resnet_nonlinearity == 'concat_elu':
      resnet_nonlinearity = lambda x: concat_elu(x)
    else:
      raise Exception('right now only concat elu is supported as resnet nonlinearity.')

    down_nr_resnet = [self.nr_resnet] + [self.nr_resnet + 1] * 2

    if self.init_padding is None and not sample:
      xs = x.shape
      padding = jnp.ones((xs[0], xs[1], xs[2], 1))
      init_padding = padding

    if sample:
      xs = x.shape
      padding = jnp.ones((xs[0], xs[1], xs[2], 1))
      x = jnp.concatenate((x, padding), axis=-1)

    ### UP PASS ###
    x = x if sample else jnp.concatenate((x, init_padding), axis=-1)
    u_list = [down_shifted_conv2d(self.nr_filters, filter_size=(2, 3),
                                  shift_output_down=True)(x)]
    ul_list = [down_shifted_conv2d(self.nr_filters,
                                   filter_size=(1, 3), shift_output_down=True)(x) +
               down_right_shifted_conv2d(
                 self.nr_filters,
                 filter_size=(2, 1),
                 shift_output_right=True)(x)]

    for i in range(3):
      # resnet block
      u_out, ul_out = PixelCNNLayer_up(self.nr_resnet, self.nr_filters, resnet_nonlinearity)(
        u_list[-1], ul_list[-1], train=train)
      u_list += u_out
      ul_list += ul_out

      if i != 2:
        # downscale (only twice)
        u_list += [down_shifted_conv2d(self.nr_filters, stride=(2, 2))(u_list[-1])]
        ul_list += [down_right_shifted_conv2d(self.nr_filters, stride=(2, 2))(ul_list[-1])]

    ### DOWN PASS ###
    u = u_list.pop()
    ul = ul_list.pop()

    for i in range(3):
      # resnet block
      u, ul = PixelCNNLayer_down(down_nr_resnet[i], self.nr_filters,
                                 resnet_nonlinearity)(u, ul, u_list, ul_list, train=train)

      # upscale (only twice)
      if i != 2:
        u = down_shifted_deconv2d(self.nr_filters, stride=(2, 2))(u)
        ul = down_right_shifted_deconv2d(self.nr_filters, stride=(2, 2))(ul)

    num_mix = 3 if self.input_channels == 1 else 10
    x_out = nin(num_mix * self.nr_logistic_mix)(jax.nn.elu(ul))

    assert len(u_list) == len(ul_list) == 0, breakpoint()

    return x_out


def discretized_mix_logistic_loss(x, l):
  """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
  xs = x.shape
  ls = l.shape

  # here and below: unpacking the params of the mixture of logistics
  nr_mix = int(ls[-1] / 10)
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + [nr_mix * 3])  # 3 for mean, scale, coef
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)

  coeffs = jnp.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
  # here and below: getting the means and adjusting them based on preceding
  # sub-pixels
  x = jnp.expand_dims(x, axis=-1) + jnp.zeros(xs + [nr_mix])
  m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
        * x[:, :, :, 0, :]).reshape((xs[0], xs[1], xs[2], 1, nr_mix))
  m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
        coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).reshape((xs[0], xs[1], xs[2], 1, nr_mix))

  means = jnp.concatenate((jnp.expand_dims(means[:, :, :, 0, :], axis=3), m2, m3), axis=3)
  centered_x = x - means
  inv_stdv = jnp.exp(-log_scales)
  plus_in = inv_stdv * (centered_x + 1. / 255.)
  cdf_plus = jax.nn.sigmoid(plus_in)
  min_in = inv_stdv * (centered_x - 1. / 255.)
  cdf_min = jax.nn.sigmoid(min_in)
  # log probability for edge case of 0 (before scaling)
  log_cdf_plus = plus_in - jax.nn.softplus(plus_in)
  # log probability for edge case of 255 (before scaling)
  log_one_minus_cdf_min = -jax.nn.softplus(min_in)
  cdf_delta = cdf_plus - cdf_min  # probability for all other cases
  mid_in = inv_stdv * centered_x
  # log probability in the center of the bin, to be used in extreme cases
  # (not actually used in our code)
  log_pdf_mid = mid_in - log_scales - 2. * jax.nn.softplus(mid_in)

  inner_inner_cond = (cdf_delta > 1e-5).astype(jnp.float32)
  inner_inner_out = inner_inner_cond * jnp.log(jnp.clip(cdf_delta, a_min=1e-12)) + (1. - inner_inner_cond) * (
      log_pdf_mid - np.log(127.5))
  inner_cond = (x > 0.999).astype(jnp.float32)
  inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
  cond = (x < -0.999).astype(jnp.float32)
  log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
  log_probs = jnp.sum(log_probs, axis=3) + log_prob_from_logits(logit_probs)

  return -jnp.sum(log_sum_exp(log_probs))


def mix_logistic_loss(x, l):
  """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
  xs = x.shape
  ls = l.shape

  # here and below: unpacking the params of the mixture of logistics
  nr_mix = int(ls[-1] / 10)
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + [nr_mix * 3])  # 3 for mean, scale, coef
  means = l[:, :, :, :, :nr_mix]
  # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)

  coeffs = jnp.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
  # here and below: getting the means and adjusting them based on preceding
  # sub-pixels
  x = jnp.expand_dims(x, axis=-1) + jnp.zeros(xs + [nr_mix])
  m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
        * x[:, :, :, 0, :]).reshape((xs[0], xs[1], xs[2], 1, nr_mix))

  m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
        coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).reshape((xs[0], xs[1], xs[2], 1, nr_mix))

  means = jnp.concatenate((jnp.expand_dims(means[:, :, :, 0, :], axis=3), m2, m3), axis=3)
  centered_x = x - means
  inv_stdv = jnp.exp(-log_scales)
  mid_in = inv_stdv * centered_x
  log_probs = mid_in - log_scales - 2. * jax.nn.softplus(mid_in)
  log_probs = jnp.sum(log_probs, axis=3) + log_prob_from_logits(logit_probs)

  return -jnp.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d(x, l):
  """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
  # Pytorch ordering
  xs = x.shape
  ls = l.shape

  # here and below: unpacking the params of the mixture of logistics
  nr_mix = int(ls[-1] / 3)
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + [nr_mix * 2])  # 2 for mean, scale
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)
  # here and below: getting the means and adjusting them based on preceding
  # sub-pixels
  x = jnp.expand_dims(x, axis=-1) + jnp.zeros(xs + [nr_mix])

  # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
  centered_x = x - means
  inv_stdv = jnp.exp(-log_scales)
  plus_in = inv_stdv * (centered_x + 1. / 255.)
  cdf_plus = jax.nn.sigmoid(plus_in)
  min_in = inv_stdv * (centered_x - 1. / 255.)
  cdf_min = jax.nn.sigmoid(min_in)
  # log probability for edge case of 0 (before scaling)
  log_cdf_plus = plus_in - jax.nn.softplus(plus_in)
  # log probability for edge case of 255 (before scaling)
  log_one_minus_cdf_min = -jax.nn.softplus(min_in)
  cdf_delta = cdf_plus - cdf_min  # probability for all other cases
  mid_in = inv_stdv * centered_x
  # log probability in the center of the bin, to be used in extreme cases
  # (not actually used in our code)
  log_pdf_mid = mid_in - log_scales - 2. * jax.nn.softplus(mid_in)

  inner_inner_cond = (cdf_delta > 1e-5).astype(jnp.float32)
  inner_inner_out = inner_inner_cond * jnp.log(jnp.clip(cdf_delta, a_min=1e-12)) + (1. - inner_inner_cond) * (
      log_pdf_mid - np.log(127.5))
  inner_cond = (x > 0.999).astype(jnp.float32)
  inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
  cond = (x < -0.999).astype(jnp.float32)
  log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
  log_probs = jnp.sum(log_probs, axis=3) + log_prob_from_logits(logit_probs)

  return -jnp.sum(log_sum_exp(log_probs))


def mix_logistic_loss_1d(x, l):
  """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
  xs = x.shape
  ls = l.shape

  # here and below: unpacking the params of the mixture of logistics
  nr_mix = int(ls[-1] / 3)
  logit_probs = l[:, :, :, :nr_mix]
  l = l[:, :, :, nr_mix:].reshape(xs + [nr_mix * 2])  # 2 for mean, scale
  means = l[:, :, :, :, :nr_mix]
  log_scales = jnp.clip(l[:, :, :, :, nr_mix:2 * nr_mix], a_min=-7.)
  # here and below: getting the means and adjusting them based on preceding
  # sub-pixels
  x = jnp.expand_dims(x, axis=-1) + jnp.zeros(xs + [nr_mix])

  # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
  centered_x = x - means
  inv_stdv = jnp.exp(-log_scales)
  mid_in = inv_stdv * centered_x
  log_pdf_mid = mid_in - log_scales - 2. * jax.nn.softplus(mid_in)
  log_probs = jnp.sum(log_pdf_mid, axis=3) + log_prob_from_logits(logit_probs)

  return -jnp.sum(log_sum_exp(log_probs))