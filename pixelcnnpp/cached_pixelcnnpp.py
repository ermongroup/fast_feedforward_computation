from pixelcnnpp.cached_layers import *
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class PixelCNNLayer_up(nn.Module):
  nr_resnet: int
  nr_filters: int
  batch_size: int
  image_width: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, row, col, cache_every, run_every, train=True):
    u_list, ul_list = [], []

    for i in range(self.nr_resnet):
      u = gated_resnet(self.nr_filters,
                       down_shifted_conv2d,
                       self.batch_size,
                       self.image_width,
                       self.resnet_nonlinearity)(u, row, col, cache_every, run_every, vstack=True, train=train)
      ul = gated_resnet(self.nr_filters,
                        down_right_shifted_conv2d,
                        self.batch_size,
                        self.image_width,
                        self.resnet_nonlinearity)(ul, row, col, cache_every, run_every, a=u, train=train)
      u_list += [u]
      ul_list += [ul]

    return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
  nr_resnet: int
  nr_filters: int
  batch_size: int
  image_width: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, u_list, ul_list, row, col, cache_every, run_every, train=True):
    for i in range(self.nr_resnet):
      u = gated_resnet(self.nr_filters,
                       down_shifted_conv2d,
                       self.batch_size,
                       self.image_width,
                       self.resnet_nonlinearity)(u, row, col, cache_every, run_every, vstack=True, a=u_list.pop(),
                                                 train=train)
      ul = gated_resnet(self.nr_filters,
                        down_right_shifted_conv2d,
                        self.batch_size,
                        self.image_width,
                        self.resnet_nonlinearity)(ul, row, col, cache_every, run_every, a=u, b=ul_list.pop(),
                                                  train=train)

    return u, ul


def undo_zeroth_row_bias_when_downshifting(row_output, row):
  return jnp.where(row == 0, jnp.zeros_like(row_output), row_output)


def undo_zeroth_column_bias_when_rightshifting(pixel_output, col):
  return jnp.where(col == 0, jnp.zeros_like(pixel_output), pixel_output)


class PixelCNN(nn.Module):
  nr_resnet: int = 5
  nr_filters: int = 160
  batch_size: int = 36
  image_width: int = 32
  nr_logistic_mix: int = 10
  resnet_nonlinearity: str = 'concat_elu'
  input_channels: int = 3

  @nn.compact
  def __call__(self, row_input, pixel_input, row, col, train=False):
    if self.resnet_nonlinearity == 'concat_elu':
      resnet_nonlinearity = lambda x: concat_elu(x)
    else:
      raise Exception('right now only concat elu is supported as resnet nonlinearity.')

    down_nr_resnet = [self.nr_resnet] + [self.nr_resnet + 1] * 2

    ###      UP PASS    ###
    cache_every, run_every = 1, 1
    u_list_input = down_shifted_conv2d(self.nr_filters,
                                       self.batch_size,
                                       self.image_width, filter_size=(2, 3))(
      row_input, row, col, cache_every, run_every)
    u_list = [undo_zeroth_row_bias_when_downshifting(u_list_input, row)]

    downshift_hstack_input = down_shifted_conv2d(self.nr_filters,
                                                 self.batch_size,
                                                 self.image_width,
                                                 filter_size=(1, 3))(
      row_input, row, col, cache_every, run_every)

    downshift_hstack_input = undo_zeroth_row_bias_when_downshifting(downshift_hstack_input, row)

    rightshift_hstack_input = down_right_shifted_conv2d(self.nr_filters,
                                                        self.batch_size,
                                                        self.image_width,
                                                        filter_size=(2, 1))(
      pixel_input, row, col, cache_every, run_every)
    rightshift_hstack_input = undo_zeroth_column_bias_when_rightshifting(rightshift_hstack_input, col)
    ul_list = [sum_rightshift_downshift(rightshift_hstack_input, downshift_hstack_input, col)]

    u_out, ul_out = PixelCNNLayer_up(self.nr_resnet,
                                     self.nr_filters,
                                     self.batch_size,
                                     self.image_width,
                                     resnet_nonlinearity)(u_list[-1], ul_list[-1], row, col,
                                                          cache_every, run_every, train=train)
    u_list += u_out
    ul_list += ul_out

    cache_every, run_every = 1, 2
    u_list.append(down_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width, stride=(2, 2))
                  (u_list[-1], row, col, cache_every, run_every))
    ul_list.append(down_right_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width, stride=(2, 2))
                   (ul_list[-1], row, col, cache_every, run_every))

    cache_every, run_every = 2, 2
    u_out, ul_out = PixelCNNLayer_up(self.nr_resnet,
                                     self.nr_filters,
                                     self.batch_size,
                                     self.image_width // 2,
                                     resnet_nonlinearity)(u_list[-1], ul_list[-1], row, col, cache_every, run_every,
                                                          train=train)
    u_list += u_out
    ul_list += ul_out

    cache_every, run_every = 2, 4
    u_list.append(down_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width // 2, stride=(2, 2))
                  (u_list[-1], row, col, cache_every, run_every))
    ul_list.append(down_right_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width // 2, stride=(2, 2))
                   (ul_list[-1], row, col, cache_every, run_every))

    cache_every, run_every = 4, 4
    u_out, ul_out = PixelCNNLayer_up(self.nr_resnet,
                                     self.nr_filters,
                                     self.batch_size,
                                     self.image_width // 4,
                                     resnet_nonlinearity)(u_list[-1], ul_list[-1], row, col, cache_every, run_every,
                                                          train=train)
    u_list += u_out
    ul_list += ul_out

    ###    DOWN PASS    ###
    u = u_list.pop()
    ul = ul_list.pop()
    u, ul = PixelCNNLayer_down(down_nr_resnet[0], self.nr_filters, self.batch_size, self.image_width // 4,
                               resnet_nonlinearity)(u, ul, u_list, ul_list, row, col, cache_every, run_every,
                                                    train=train)

    cache_every, run_every = 4, 2
    u = down_shifted_deconv2d(self.nr_filters, self.batch_size, self.image_width // 4, stride=(2, 2))(
      u, row, col, cache_every, run_every)
    ul = down_right_shifted_deconv2d(self.nr_filters, self.batch_size, self.image_width // 4, stride=(2, 2))(
      ul, row, col, cache_every, run_every)

    cache_every, run_every = 2, 2
    u, ul = PixelCNNLayer_down(down_nr_resnet[1], self.nr_filters, self.batch_size, self.image_width // 2,
                               resnet_nonlinearity)(u, ul, u_list, ul_list, row, col, cache_every, run_every,
                                                    train=train)

    cache_every, run_every = 2, 1
    u = down_shifted_deconv2d(self.nr_filters, self.batch_size, self.image_width // 2, stride=(2, 2))(
      u, row, col, cache_every, run_every)
    ul = down_right_shifted_deconv2d(self.nr_filters, self.batch_size, self.image_width // 2, stride=(2, 2))(
      ul, row, col, cache_every, run_every)

    cache_every, run_every = 1, 1
    u, ul = PixelCNNLayer_down(down_nr_resnet[2], self.nr_filters, self.batch_size, self.image_width,
                               resnet_nonlinearity)(u, ul, u_list, ul_list, row, col, cache_every, run_every,
                                                         train=train)

    num_mix = 3 if self.input_channels == 1 else 10
    x_out = nin(num_mix * self.nr_logistic_mix)(jax.nn.elu(ul))

    assert len(u_list) == len(ul_list) == 0, breakpoint()

    return x_out