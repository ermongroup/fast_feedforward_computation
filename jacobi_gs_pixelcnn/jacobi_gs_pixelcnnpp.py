from jacobi_gs_pixelcnn.jacobi_gs_layers import *
import jax
import jax.numpy as jnp
import flax.linen as nn


class PixelCNNLayer_up(nn.Module):
  num_blocks: int
  original_image_width: int
  nr_resnet: int
  nr_filters: int
  batch_size: int
  image_width: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, rows, cols, cache_every, run_every, first_index=True, train=False):
    u_list, ul_list = [], []

    for i in range(self.nr_resnet):
      u = gated_resnet(self.num_blocks, self.original_image_width, self.nr_filters, down_shifted_conv2d,
                       self.batch_size, self.image_width, self.resnet_nonlinearity)(
        u, rows, cols, cache_every, run_every, vstack=True, first_index=first_index, train=train)
      ul = gated_resnet(self.num_blocks, self.original_image_width, self.nr_filters, down_right_shifted_conv2d,
                        self.batch_size, self.image_width, self.resnet_nonlinearity)(
        ul, rows, cols, cache_every, run_every, a=u, first_index=first_index, train=train)
      u_list += [u]
      ul_list += [ul]

    return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
  num_blocks: int
  original_image_width: int
  nr_resnet: int
  nr_filters: int
  batch_size: int
  image_width: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, u_list, ul_list, rows, cols, cache_every, run_every, first_index=True, train=False):
    for i in range(self.nr_resnet):
      u = gated_resnet(self.num_blocks, self.original_image_width, self.nr_filters, down_shifted_conv2d,
                       self.batch_size, self.image_width, self.resnet_nonlinearity)(
        u, rows, cols, cache_every, run_every, vstack=True, a=u_list.pop(), first_index=first_index, train=train)
      ul = gated_resnet(self.num_blocks, self.original_image_width, self.nr_filters, down_right_shifted_conv2d,
                        self.batch_size, self.image_width, self.resnet_nonlinearity)(
        ul, rows, cols, cache_every, run_every, a=u, b=ul_list.pop(), first_index=first_index, train=train)

    return u, ul


def undo_zeroth_row_bias_when_downshifting(row_output, rows, num_blocks, first_index=True):
  if first_index:
    row_output = row_output.at[:, 0, :, :].set(0.0)

  else:
    row_view = row_output.reshape((num_blocks, -1, *row_output.shape[1:]))
    zero_out = (rows == 0)
    row_view = jax.vmap(jnp.where)(zero_out, jnp.zeros_like(row_view), row_view)
    row_output = row_view.reshape(row_output.shape)

  return row_output


def undo_zeroth_column_bias_when_rightshifting(pixel_output, cols, num_blocks, first_index=True):
  if first_index:
    pixel_output = pixel_output.at[:, :, 0, :].set(0.0)

  else:
    zero_out = (cols == 0)
    pixel_view = pixel_output.reshape((num_blocks, -1, *pixel_output.shape[1:]))
    pixel_view = jax.vmap(jnp.where)(zero_out, jnp.zeros_like(pixel_view), pixel_view)
    pixel_output = pixel_view.reshape(pixel_output.shape)

  return pixel_output


# Here we always assume that the line widths can be fully divided by the block size.
class PixelCNN(nn.Module):
  num_blocks: int
  nr_resnet: int = 5
  nr_filters: int = 160
  batch_size: int = 36
  image_width: int = 32
  nr_logistic_mix: int = 10
  resnet_nonlinearity: str = 'concat_elu'
  input_channels: int = 3

  @nn.compact
  def __call__(self, row_input, pixel_input, rows, cols, first_index=True, train=False):
    ###      UP PASS    ###
    if self.resnet_nonlinearity == 'concat_elu':
      resnet_nonlinearity = lambda x: concat_elu(x)
    else:
      raise Exception('right now only concat elu is supported as resnet nonlinearity.')

    cache_every, run_every = 1, 1
    u_list_input = down_shifted_conv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                                       self.image_width, filter_size=(2, 3))(
      row_input, rows, cols, cache_every, run_every, first_index=first_index)
    u_list = [
      undo_zeroth_row_bias_when_downshifting(u_list_input, rows, self.num_blocks, first_index=first_index)
    ]

    downshift_hstack_input = down_shifted_conv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                                                 self.image_width, filter_size=(1, 3))(
      row_input, rows, cols, cache_every, run_every, first_index=first_index)
    downshift_hstack_input = undo_zeroth_row_bias_when_downshifting(downshift_hstack_input, rows, self.num_blocks,
                                                                    first_index=first_index)

    rightshift_hstack_input = down_right_shifted_conv2d(self.num_blocks, self.image_width, self.nr_filters,
                                                        self.batch_size,
                                                        self.image_width, filter_size=(2, 1))(
      pixel_input, rows, cols, cache_every, run_every, first_index=first_index)
    rightshift_hstack_input = undo_zeroth_column_bias_when_rightshifting(rightshift_hstack_input, cols,
                                                                         self.num_blocks, first_index=first_index)
    ul_list = [
      sum_rightshift_downshift(rightshift_hstack_input, downshift_hstack_input, cols, self.num_blocks,
                               first_index=first_index)
    ]

    u_out, ul_out = PixelCNNLayer_up(self.num_blocks, self.image_width, self.nr_resnet, self.nr_filters,
                                     self.batch_size, self.image_width, resnet_nonlinearity)(
      u_list[-1], ul_list[-1], rows, cols, cache_every, run_every, first_index=first_index, train=train)
    u_list += u_out
    ul_list += ul_out

    cache_every, run_every = 1, 2
    u_list.append(down_shifted_conv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                                      self.image_width, stride=(2, 2))(
      u_list[-1], rows, cols, cache_every, run_every, first_index=first_index))
    ul_list.append(down_right_shifted_conv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                                             self.image_width, stride=(2, 2))(
      ul_list[-1], rows, cols, cache_every, run_every, first_index=first_index))

    cache_every, run_every = 2, 2
    u_out, ul_out = PixelCNNLayer_up(self.num_blocks, self.image_width, self.nr_resnet, self.nr_filters,
                                     self.batch_size,
                                     self.image_width // 2, resnet_nonlinearity)(
      u_list[-1], ul_list[-1], rows, cols, cache_every, run_every, first_index=first_index, train=train)
    u_list += u_out
    ul_list += ul_out

    cache_every, run_every = 2, 4
    u_list.append(down_shifted_conv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                                      self.image_width // 2, stride=(2, 2))(
      u_list[-1], rows, cols, cache_every, run_every, first_index=first_index))
    ul_list.append(down_right_shifted_conv2d(self.num_blocks, self.image_width, self.nr_filters,
                                             self.batch_size, self.image_width // 2, stride=(2, 2))(
      ul_list[-1], rows, cols, cache_every, run_every, first_index=first_index))

    cache_every, run_every = 4, 4
    u_out, ul_out = PixelCNNLayer_up(self.num_blocks, self.image_width, self.nr_resnet, self.nr_filters,
                                     self.batch_size, self.image_width // 4, resnet_nonlinearity)(
      u_list[-1], ul_list[-1], rows, cols, cache_every, run_every, first_index=first_index, train=train)

    u_list += u_out
    ul_list += ul_out

    down_nr_resnet = [self.nr_resnet] + [self.nr_resnet + 1] * 2
    ###    DOWN PASS    ###
    u = u_list.pop()
    ul = ul_list.pop()
    u, ul = PixelCNNLayer_down(self.num_blocks, self.image_width, down_nr_resnet[0], self.nr_filters,
                               self.batch_size, self.image_width // 4, resnet_nonlinearity)(
      u, ul, u_list, ul_list, rows, cols, cache_every, run_every, first_index=first_index, train=train)

    cache_every, run_every = 4, 2
    u = down_shifted_deconv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                              self.image_width // 4, stride=(2, 2))(
      u, rows, cols, cache_every, run_every, first_index=first_index)
    ul = down_right_shifted_deconv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                                     self.image_width // 4, stride=(2, 2))(
      ul, rows, cols, cache_every, run_every, first_index=first_index)

    cache_every, run_every = 2, 2
    u, ul = PixelCNNLayer_down(self.num_blocks, self.image_width, down_nr_resnet[1], self.nr_filters,
                               self.batch_size, self.image_width // 2, resnet_nonlinearity)(
      u, ul, u_list, ul_list, rows, cols, cache_every, run_every, first_index=first_index, train=train)

    cache_every, run_every = 2, 1
    u = down_shifted_deconv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                              self.image_width // 2, stride=(2, 2))(
      u, rows, cols, cache_every, run_every, first_index=first_index)
    ul = down_right_shifted_deconv2d(self.num_blocks, self.image_width, self.nr_filters, self.batch_size,
                                     self.image_width // 2, stride=(2, 2))(
      ul, rows, cols, cache_every, run_every, first_index=first_index)

    cache_every, run_every = 1, 1
    u, ul = PixelCNNLayer_down(self.num_blocks, self.image_width, down_nr_resnet[2], self.nr_filters,
                               self.batch_size, self.image_width, resnet_nonlinearity)(
      u, ul, u_list, ul_list, rows, cols, cache_every, run_every, first_index=first_index, train=train)

    num_mix = 3 if self.input_channels == 1 else 10
    x_out = nin(num_mix * self.nr_logistic_mix)(jax.nn.elu(ul))

    assert len(u_list) == len(ul_list) == 0, breakpoint()

    return x_out
