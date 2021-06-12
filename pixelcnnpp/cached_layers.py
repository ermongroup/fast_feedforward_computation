import math
from pixelcnnpp.layers import *


def _init_cache(batch, cache_height, cache_width, channels):
  '''Creates a cache, which is used to avoid redundant computation.'''
  cache = jnp.zeros((batch, cache_height, cache_width, channels))
  return cache


def _init_deconv_cache(batch, filter_height, filter_width, filter_channels, image_width, stride):
  '''Creates the cache for the two deconv layers.'''
  cache_height = filter_height  # Just large enough to fit the filter.
  # The deconv will increases the number of outputs `stride` times.
  # The extra width comes from the tf.nn.conv2d_transpose() operation.
  cache_width = image_width * stride + filter_width - 1
  cache = _init_cache(batch, cache_height, cache_width, filter_channels)
  return cache, cache_height, cache_width


def _roll_cache(cache):
  '''Pop off the oldest row of the cache to make space for the newest row of input.'''
  batch, _, cache_width, channels = cache.shape
  without_dropped_row = cache[:, 1:, :, :]
  zero_row = jnp.zeros((batch, 1, cache_width, channels))
  rolled_cache = jnp.concatenate([without_dropped_row, zero_row], axis=1)
  return rolled_cache


def sum_rightshift_downshift(rightshifted_pixel, downshifted_row, col):
  '''Sums the vertical and horizontal stack.'''
  s = downshifted_row.shape
  downshifted_pixel = jax.lax.dynamic_slice(downshifted_row, [0, 0, col, 0],
                                            [s[0], s[1], 1, s[-1]])
  return rightshifted_pixel + downshifted_pixel


class down_shifted_conv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 3)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, row_input, row, col, cache_every, run_every):
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    cache_height = filter_height
    padding = filter_width // 2
    cache_width = self.image_width + 2 * padding
    output_width = int(math.ceil(self.image_width / float(self.stride[0])))

    cache = self.variable('cache', 'cache', _init_cache,
                          self.batch_size, cache_height,
                          cache_width, row_input.shape[-1])

    output_cache = self.variable('cache', 'output_cache', _init_cache,
                                 self.batch_size, 1, output_width,
                                 self.num_filters_out)

    cache.value = jnp.where((row == 0) & (col == 0), jnp.zeros_like(cache.value), cache.value)

    cache.value = jnp.where((row % cache_every == 0) & (col == 0),
                            jax.lax.dynamic_update_slice(cache.value, row_input, (0, cache.value.shape[1] - 1,
                                                                                  padding, 0)),
                            cache.value)
    outputs = WNConv(features=self.num_filters_out,
                     kernel_size=self.filter_size,
                     strides=self.stride)(cache.value)

    output_cache.value = jnp.where((row % run_every == 0) & (col == 0), outputs, output_cache.value)

    cache.value = jnp.where((row % cache_every == 0) & (col == 0), _roll_cache(cache.value), cache.value)

    return output_cache.value


def deconv_output_length(input_length,
                         filter_size,
                         padding,
                         output_padding=None,
                         stride=0,
                         dilation=1):
  """Determines output length of a transposed convolution given input length.

  Function copied from Keras. Original documentation below:

  Arguments:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"same"`, `"valid"`, `"full"`.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.
  Returns:
      The output length (integer).
  """
  assert padding in {'same', 'valid', 'full'}
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'valid':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'full':
      length = input_length * stride - (stride + filter_size - 2)
    elif padding == 'same':
      length = input_length * stride

  else:
    if padding == 'same':
      pad = filter_size // 2
    elif padding == 'valid':
      pad = 0
    elif padding == 'full':
      pad = filter_size - 1

    length = ((input_length - 1) * stride + filter_size - 2 * pad +
              output_padding)
  return length


class down_shifted_deconv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 3)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, row_input, row, col, cache_every, run_every):
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]

    init_cache, cache_height, cache_width = _init_deconv_cache(self.batch_size, filter_height, filter_width,
                                                               self.num_filters_out, self.image_width, self.stride[0])
    cache = self.variable('cache', 'cache', lambda: init_cache)
    output_cache = self.variable('cache', 'output_cache',
                                 lambda: jnp.zeros((row_input.shape[0],
                                                   1, cache_width - 2,
                                                   self.num_filters_out)))
    cache.value = jnp.where((row == 0) & (col == 0), jnp.zeros_like(cache.value), cache.value)

    xs = row_input.shape
    target_shape = [xs[0], None, None, self.num_filters_out]
    target_shape[1] = deconv_output_length(xs[1], self.filter_size[0], 'valid', output_padding=0,
                                           stride=self.stride[0])
    target_shape[2] = deconv_output_length(xs[2], self.filter_size[1], 'valid', output_padding=1,
                                           stride=self.stride[1])
    pad_before_h = self.filter_size[0] - 1
    pad_before_w = self.filter_size[1] - 1
    pad_after_h = target_shape[1] + self.filter_size[0] - 1 - (xs[1] - 1) * self.stride[0] - 1 - pad_before_h
    pad_after_w = target_shape[2] + self.filter_size[1] - 1 - (xs[2] - 1) * self.stride[1] - 1 - pad_before_w

    outputs = WNConvTranspose(features=self.num_filters_out,
                              kernel_size=self.filter_size,
                              strides=self.stride,
                              padding=((pad_before_h, pad_after_h),
                                       (pad_before_w, pad_after_w)))(row_input)

    cache.value = jnp.where((row % cache_every == 0) & (col == 0), cache.value + outputs, cache.value)
    output_cache.value = jnp.where((row % run_every == 0) & (col == 0),
                                   cache.value[:, 0:1, 1:-1, :], output_cache.value)
    cache.value = jnp.where((row % run_every == 0) & (col == 0), _roll_cache(cache.value), cache.value)

    return output_cache.value


class down_right_shifted_conv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)
  shift_output_right: bool = False

  @nn.compact
  def __call__(self, pixel_input, row, col, cache_every, run_every):
    cache_height = filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    left_pad = filter_width - 1
    cache_width = self.image_width + left_pad
    cache = self.variable('cache', 'cache',
                          lambda: _init_cache(self.batch_size, cache_height, cache_width, pixel_input.shape[-1]))
    output_cache = self.variable('cache', 'output_cache',
                                 lambda: _init_cache(self.batch_size, 1, 1, self.num_filters_out))

    cache.value = jnp.where((row == 0) & (col == 0), jnp.zeros_like(cache.value), cache.value)

    cache_col = col // cache_every
    # update cache
    should_cache = (row % cache_every == 0) & (col % cache_every == 0)
    should_run = (row % run_every == 0) & (col % run_every == 0)

    pixel_col = cache_col + left_pad
    cache.value = jnp.where(should_cache,
                            jax.lax.dynamic_update_slice(cache.value, pixel_input,
                                                         [0, cache.value.shape[1] - 1,
                                                          pixel_col, 0]),
                            cache.value)

    width_start = cache_col
    cs = cache.value.shape
    cache_neighborhood = jax.lax.dynamic_slice(cache.value, [0, 0, width_start, 0],
                                               [cs[0], cs[1], filter_width, cs[-1]])
    outputs = WNConv(features=self.num_filters_out,
                     kernel_size=self.filter_size,
                     strides=self.stride)(cache_neighborhood)

    output_cache.value = jnp.where(should_run, outputs, output_cache.value)

    is_end_of_row = (cache_col == self.image_width - 1)
    cache.value = jnp.where(is_end_of_row & should_cache, _roll_cache(cache.value), cache.value)

    return output_cache.value


class down_right_shifted_deconv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)
  shift_output_right: bool = False

  @nn.compact
  def __call__(self, pixel_input, row, col, cache_every, run_every):
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    cache, cache_height, cache_width = _init_deconv_cache(self.batch_size, filter_height, filter_width,
                                                          self.num_filters_out, self.image_width, self.stride[0])
    cache = self.variable('cache', 'cache', lambda: cache)
    output_cache = self.variable('cache', 'output_cache',
                                 lambda: jnp.zeros((self.batch_size, 1, 1, self.num_filters_out)))

    cache.value = jnp.where((row == 0) & (col == 0), jnp.zeros_like(cache.value), cache.value)
    should_cache = (row % cache_every == 0) & (col % cache_every == 0)

    xs = pixel_input.shape
    target_shape = [xs[0], None, None, self.num_filters_out]
    target_shape[1] = deconv_output_length(xs[1], self.filter_size[0], 'valid', output_padding=0,
                                           stride=self.stride[0])
    target_shape[2] = deconv_output_length(xs[2], self.filter_size[1], 'valid', output_padding=0,
                                           stride=self.stride[1])
    pad_before_h = self.filter_size[0] - 1
    pad_before_w = self.filter_size[1] - 1
    pad_after_h = target_shape[1] + self.filter_size[0] - 1 - (xs[1] - 1) * self.stride[0] - 1 - pad_before_h
    pad_after_w = target_shape[2] + self.filter_size[1] - 1 - (xs[2] - 1) * self.stride[1] - 1 - pad_before_w

    outputs = WNConvTranspose(features=self.num_filters_out,
                              kernel_size=self.filter_size,
                              strides=self.stride,
                              padding=((pad_before_h, pad_after_h),
                                       (pad_before_w, pad_after_w)))(pixel_input)
    cache_col = col // cache_every

    cache.value = jnp.where(should_cache,
                            jax.lax.dynamic_update_slice(cache.value, outputs,
                                                         [0, 0, self.stride[1] * cache_col, 0]),
                            cache.value)

    should_run = (row % run_every == 0) & (col % run_every == 0)

    output_col = col // run_every
    cs = cache.value.shape
    output_cache.value = jnp.where(should_run,
                                   jax.lax.dynamic_slice(cache.value, [0, 0, output_col, 0],
                                                         [cs[0], 1, 1, cs[-1]]),
                                   output_cache.value)

    is_end_of_row = (output_col == cache_width - filter_width)
    cache.value = jnp.where(should_run & is_end_of_row, _roll_cache(cache.value), cache.value)

    return output_cache.value


class gated_resnet(nn.Module):
  num_filters: int
  conv_op: Any
  batch_size: int
  image_width: int
  nonlinearity: Any

  @nn.compact
  def __call__(self, og_x, row, col, cache_every, run_every, vstack=False, a=None, b=None, train=True):
    if vstack:
      x = self.conv_op(self.num_filters, self.batch_size, self.image_width)(
        self.nonlinearity(og_x), row, col, cache_every, run_every)
      if a is not None:
        x += nin(self.num_filters)(self.nonlinearity(a))
      x = self.nonlinearity(x)
      x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
      x = self.conv_op(2 * self.num_filters, self.batch_size, self.image_width)(
        x, row, col, cache_every, run_every)
      a, b = jnp.split(x, 2, axis=-1)
      c3 = a * jax.nn.sigmoid(b)
      return og_x + c3
    else:

      x = self.conv_op(self.num_filters, self.batch_size, self.image_width)(
        self.nonlinearity(og_x), row, col, cache_every, run_every)

      cache_col = col // cache_every
      v_stack_pixel = jax.lax.dynamic_slice(a, [0, 0, cache_col, 0],
                                            [a.shape[0], a.shape[1], 1, a.shape[-1]])

      if b is not None:
        v_stack_pixel = jnp.concatenate([v_stack_pixel, b], axis=-1)

      x += nin(self.num_filters)(self.nonlinearity(v_stack_pixel))
      x = self.nonlinearity(x)
      x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
      x = self.conv_op(2 * self.num_filters, self.batch_size, self.image_width)(
        x, row, col, cache_every, run_every)
      a, b = jnp.split(x, 2, axis=-1)
      c3 = a * jax.nn.sigmoid(b)
      return og_x + c3