from gs_jacobi_pixelcnn.gs_jacobi_layers import *
import jax
import jax.numpy as jnp


class PixelCNNLayer_up(nn.Module):
  nr_resnet: int
  nr_filters: int
  batch_size: int
  image_width: int
  max_rows: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, row_start, cache_every, run_every, train=False):
    u_list, ul_list = [], []

    for i in range(self.nr_resnet):
      u = gated_resnet(self.nr_filters,
                       down_shifted_conv2d,
                       self.batch_size,
                       self.image_width,
                       self.max_rows,
                       self.resnet_nonlinearity)(u, row_start, cache_every, run_every, vstack=True, train=train)
      ul = gated_resnet(self.nr_filters,
                        down_right_shifted_conv2d,
                        self.batch_size,
                        self.image_width,
                        self.max_rows,
                        self.resnet_nonlinearity)(ul, row_start, cache_every, run_every, a=u, train=train)
      u_list += [u]
      ul_list += [ul]

    return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
  nr_resnet: int
  nr_filters: int
  batch_size: int
  image_width: int
  max_rows: int
  resnet_nonlinearity: Any

  @nn.compact
  def __call__(self, u, ul, u_list, ul_list, row_start, cache_every, run_every, train=False):
    for i in range(self.nr_resnet):
      u = gated_resnet(self.nr_filters,
                       down_shifted_conv2d,
                       self.batch_size,
                       self.image_width,
                       self.max_rows,
                       self.resnet_nonlinearity)(u, row_start, cache_every, run_every, vstack=True, a=u_list.pop(),
                                                 train=train)
      ul = gated_resnet(self.nr_filters,
                        down_right_shifted_conv2d,
                        self.batch_size,
                        self.image_width,
                        self.max_rows,
                        self.resnet_nonlinearity)(ul, row_start, cache_every, run_every, a=u, b=ul_list.pop(),
                                                  train=train)

    return u, ul


def undo_zeroth_row_bias_when_downshifting(row_output, row_start):
  return jnp.where(row_start == 0, row_output.at[:, 0, :, :].set(0.0), row_output)


def undo_zeroth_column_bias_when_rightshifting(pixel_output):
  return pixel_output.at[:, :, 0, :].set(0.0)


## The cache implementation here is not optimal, and we still redo many computations.
class PixelCNN(nn.Module):
  nr_resnet: int = 5
  nr_filters: int = 160
  batch_size: int = 36
  image_width: int = 32
  max_rows: int = 5
  nr_logistic_mix: int = 10
  resnet_nonlinearity: str = 'concat_elu'
  input_channels: int = 3

  @nn.compact
  def __call__(self, row_input, pixel_input, row_start, train=False):
    if self.resnet_nonlinearity == 'concat_elu':
      resnet_nonlinearity = lambda x: concat_elu(x)
    else:
      raise Exception('right now only concat elu is supported as resnet nonlinearity.')

    down_nr_resnet = [self.nr_resnet] + [self.nr_resnet + 1] * 2

    ###      UP PASS    ###
    cache_every, run_every = 1, 1
    u_list_input = down_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width, self.max_rows,
                                       filter_size=(2, 3))(row_input, row_start, cache_every, run_every)
    u_list = [
      undo_zeroth_row_bias_when_downshifting(u_list_input, row_start)
    ]

    downshift_hstack_input = down_shifted_conv2d(self.nr_filters, self.batch_size,
                                                 self.image_width, self.max_rows,
                                                 filter_size=(1, 3))(row_input, row_start, cache_every, run_every)

    downshift_hstack_input = undo_zeroth_row_bias_when_downshifting(downshift_hstack_input, row_start)

    rightshift_hstack_input = down_right_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width,
                                                        self.max_rows, filter_size=(2, 1))(
      pixel_input, row_start, cache_every, run_every
    )
    rightshift_hstack_input = undo_zeroth_column_bias_when_rightshifting(rightshift_hstack_input)
    ul_list = [
      sum_rightshift_downshift(rightshift_hstack_input, downshift_hstack_input)
    ]

    u_out, ul_out = PixelCNNLayer_up(self.nr_resnet, self.nr_filters, self.batch_size, self.image_width,
                                     self.max_rows, resnet_nonlinearity)(
      u_list[-1], ul_list[-1], row_start, cache_every, run_every, train=train
    )
    u_list += u_out
    ul_list += ul_out

    cache_every, run_every = 1, 2
    u_list.append(down_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width, self.max_rows,
                                      stride=(2, 2))(
      u_list[-1], row_start, cache_every, run_every
    ))
    ul_list.append(down_right_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width,
                                             self.max_rows, stride=(2, 2))(
      ul_list[-1], row_start, cache_every, run_every
    ))

    cache_every, run_every = 2, 2
    u_out, ul_out = PixelCNNLayer_up(self.nr_resnet, self.nr_filters, self.batch_size, self.image_width // 2,
                                     self.max_rows, resnet_nonlinearity)(
      u_list[-1], ul_list[-1], row_start, cache_every, run_every, train=train
    )
    u_list += u_out
    ul_list += ul_out

    cache_every, run_every = 2, 4
    u_list.append(down_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width // 2,
                                      self.max_rows, stride=(2, 2))(
      u_list[-1], row_start, cache_every, run_every
    ))

    ul_list.append(down_right_shifted_conv2d(self.nr_filters, self.batch_size, self.image_width // 2,
                                             self.max_rows, stride=(2, 2))(
      ul_list[-1], row_start, cache_every, run_every
    ))

    cache_every, run_every = 4, 4
    u_out, ul_out = PixelCNNLayer_up(self.nr_resnet, self.nr_filters, self.batch_size, self.image_width // 4,
                                     self.max_rows, resnet_nonlinearity)(
      u_list[-1], ul_list[-1], row_start, cache_every, run_every, train=train
    )
    u_list += u_out
    ul_list += ul_out

    ###    DOWN PASS    ###
    u = u_list.pop()
    ul = ul_list.pop()
    u, ul = PixelCNNLayer_down(down_nr_resnet[0], self.nr_filters, self.batch_size, self.image_width // 4,
                               self.max_rows,
                               resnet_nonlinearity)(
      u, ul, u_list, ul_list, row_start, cache_every, run_every, train=train
    )

    cache_every, run_every = 4, 2
    u = down_shifted_deconv2d(self.nr_filters, self.batch_size, self.image_width // 4,
                              self.max_rows, stride=(2, 2))(
      u, row_start, cache_every, run_every
    )
    ul = down_right_shifted_deconv2d(self.nr_filters, self.batch_size,
                                     self.image_width // 4, self.max_rows, stride=(2, 2))(
      ul, row_start, cache_every, run_every
    )

    cache_every, run_every = 2, 2
    u, ul = PixelCNNLayer_down(down_nr_resnet[1], self.nr_filters, self.batch_size, self.image_width // 2,
                               self.max_rows, resnet_nonlinearity)(
      u, ul, u_list, ul_list, row_start, cache_every, run_every, train=train
    )

    cache_every, run_every = 2, 1
    u = down_shifted_deconv2d(self.nr_filters, self.batch_size, self.image_width // 2,
                              self.max_rows, stride=(2, 2))(
      u, row_start, cache_every, run_every
    )
    ul = down_right_shifted_deconv2d(self.nr_filters, self.batch_size, self.image_width // 2,
                                     self.max_rows, stride=(2, 2))(
      ul, row_start, cache_every, run_every
    )

    cache_every, run_every = 1, 1
    u, ul = PixelCNNLayer_down(down_nr_resnet[2], self.nr_filters, self.batch_size, self.image_width, self.max_rows,
                               resnet_nonlinearity)(
      u, ul, u_list, ul_list, row_start, cache_every, run_every, train=train
    )

    num_mix = 3 if self.input_channels == 1 else 10
    x_out = nin(num_mix * self.nr_logistic_mix)(jax.nn.elu(ul))
    assert len(u_list) == len(ul_list) == 0, breakpoint()

    return x_out
