from pixelcnnpp.cached_layers import *
from pixelcnnpp.cached_layers import _init_cache
import math


def _init_deconv_cache(batch, filter_height, filter_width, filter_channels, image_width, max_rows, stride):
  '''Creates the cache for the two deconv layers.'''
  cache_height = (image_width + max_rows) * stride + filter_height - 1
  # cache_height = max_rows * stride + filter_height - 1
  # The deconv will increases the number of outputs `stride` times.
  # The extra width comes from the tf.nn.conv2d_transpose() operation.
  cache_width = image_width * stride + filter_width - 1
  cache = _init_cache(batch, cache_height, cache_width, filter_channels)
  return cache, cache_height, cache_width


def sum_rightshift_downshift(rightshifted_pixel, downshifted_row):
  '''Sums the vertical and horizontal stack.'''
  return rightshifted_pixel + downshifted_row


class down_shifted_conv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  max_rows: int
  filter_size: Tuple[int, int] = (2, 3)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, row_inputs, row_start, cache_every, run_every):
    row_end = row_start + self.max_rows - 1
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    padding = filter_width // 2
    top_padding = filter_height - 1
    cache_width = self.image_width + 2 * padding
    cache_height = self.image_width + top_padding + math.ceil(self.max_rows / cache_every)
    cache = self.variable('cache', 'cache', lambda: _init_cache(self.batch_size, cache_height,
                                                                cache_width, row_inputs.shape[-1]))

    cache_init = jnp.ceil(row_start / cache_every).astype(jnp.int32)
    cache_end = jnp.floor(row_end / cache_every).astype(jnp.int32)

    should_cache = cache_init <= cache_end
    cache.value = jnp.where(should_cache,
                            jax.lax.dynamic_update_slice(cache.value, row_inputs,
                                                         [0, top_padding + cache_init, padding, 0]),
                            cache.value)

    output_init = jnp.ceil(row_start / run_every).astype(jnp.int32)
    output_end = jnp.floor(row_end / run_every).astype(jnp.int32)

    should_run = output_end >= output_init
    slice_init = output_init * self.stride[0]
    slice_height = math.ceil((self.max_rows - 1) / run_every) * self.stride[0] + filter_height
    cs = cache.value.shape
    cache_slice = jax.lax.dynamic_slice(cache.value, [0, slice_init, 0, 0],
                                        [cs[0], slice_height, cs[2], cs[-1]])

    outputs = WNConv(features=self.num_filters_out,
                     kernel_size=self.filter_size,
                     strides=self.stride)(cache_slice)

    outputs = jnp.where(should_run, outputs, jnp.zeros_like(outputs))
    return outputs


class down_shifted_deconv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  max_rows: int
  filter_size: Tuple[int, int] = (2, 3)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, row_inputs, row_start, cache_every, run_every):
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    cache, cache_height, cache_width = _init_deconv_cache(self.batch_size,
                                                          filter_height, filter_width,
                                                          self.num_filters_out,
                                                          self.image_width,
                                                          math.ceil(self.max_rows / cache_every),
                                                          self.stride[0])
    cache = self.variable('cache', 'cache', lambda: cache)

    xs = row_inputs.shape
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
                                       (pad_before_w, pad_after_w)))(row_inputs)

    row_end = row_start + self.max_rows - 1
    cache_init = jnp.ceil(row_start / cache_every).astype(jnp.int32)
    cache_end = jnp.floor(row_end / cache_every).astype(jnp.int32)

    cache.value = jnp.where(cache_end >= cache_init,
                            jax.lax.dynamic_update_slice(cache.value,
                                                         outputs,
                                                         [0, cache_init * self.stride[0], 0, 0]),
                            cache.value)

    output_init = jnp.ceil(row_start / run_every).astype(jnp.int32)
    output_height = math.ceil((self.max_rows - 1) / run_every) + 1
    output_end = jnp.floor(row_end / run_every).astype(jnp.int32)
    cs = cache.value.shape
    cache_output = jax.lax.dynamic_slice(cache.value, [0, output_init, 1, 0],
                                         [cs[0], output_height, cs[2] - 2, cs[3]])
    return jnp.where(output_end >= output_init, cache_output, jnp.zeros_like(cache_output))


class down_right_shifted_conv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  max_rows: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)
  shift_output_right: bool = False

  @nn.compact
  def __call__(self, pixel_inputs, row_start, cache_every, run_every):
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    left_pad = filter_width - 1
    top_pad = filter_height - 1
    cache_width = self.image_width + left_pad
    cache_height = self.image_width + top_pad + math.ceil(self.max_rows / cache_every)
    cache = self.variable('cache', 'cache', lambda: _init_cache(
      self.batch_size, cache_height, cache_width, pixel_inputs.shape[-1]))

    row_end = row_start + self.max_rows - 1
    cache_init = jnp.ceil(row_start / cache_every).astype(jnp.int32)
    cache_end = jnp.floor(row_end / cache_every).astype(jnp.int32)

    cache.value = jnp.where(cache_end >= cache_init,
                            jax.lax.dynamic_update_slice(cache.value, pixel_inputs,
                                                         [0, top_pad + cache_init, left_pad, 0]),
                            cache.value)

    output_init = jnp.ceil(row_start / run_every).astype(jnp.int32)
    output_end = jnp.floor(row_end / run_every).astype(jnp.int32)

    slice_init = output_init * self.stride[0]
    slice_height = math.ceil((self.max_rows - 1) / run_every) * self.stride[0] + filter_height
    cs = cache.value.shape
    slice = jax.lax.dynamic_slice(cache.value, [0, slice_init, 0, 0],
                                  [cs[0], slice_height, cs[2], cs[3]])
    outputs = WNConv(features=self.num_filters_out,
                     kernel_size=self.filter_size,
                     strides=self.stride)(slice)

    return jnp.where(output_end >= output_init, outputs, jnp.zeros_like(outputs))


class down_right_shifted_deconv2d(nn.Module):
  num_filters_out: int
  batch_size: int
  image_width: int
  max_rows: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, pixel_inputs, row_start, cache_every, run_every):
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    cache, cache_height, cache_width = _init_deconv_cache(self.batch_size,
                                                          filter_height, filter_width,
                                                          self.num_filters_out,
                                                          self.image_width,
                                                          math.ceil(self.max_rows / cache_every),
                                                          self.stride[0])
    cache = self.variable('cache', 'cache', lambda: cache)

    row_end = row_start + self.max_rows - 1
    cache_init = jnp.ceil(row_start / cache_every).astype(jnp.int32)
    cache_end = jnp.floor(row_end / cache_every).astype(jnp.int32)

    xs = pixel_inputs.shape
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
                                       (pad_before_w, pad_after_w)))(pixel_inputs)
    cache.value = jnp.where(cache_end >= cache_init,
                            jax.lax.dynamic_update_slice(cache.value,
                                                         outputs,
                                                         [0, cache_init * self.stride[0], 0, 0]),
                            cache.value)

    output_init = jnp.ceil(row_start / run_every).astype(jnp.int32)
    output_end = jnp.floor(row_end / run_every).astype(jnp.int32)

    cs = cache.value.shape
    output_height = math.ceil((self.max_rows - 1) / run_every) + 1
    cache_output = jax.lax.dynamic_slice(cache.value, [0, output_init, 0, 0],
                                         [cs[0], output_height, cs[2] - 1, cs[3]])
    return jnp.where(output_end >= output_init, cache_output, jnp.zeros_like(cache_output))


class gated_resnet(nn.Module):
  num_filters: int
  conv_op: Any
  batch_size: int
  image_width: int
  max_rows: int
  nonlinearity: Any

  @nn.compact
  def __call__(self, og_x, row_start, cache_every, run_every, vstack=False, a=None, b=None, train=True):
    if vstack:
      x = self.conv_op(self.num_filters, self.batch_size, self.image_width,
                       self.max_rows)(self.nonlinearity(og_x), row_start, cache_every, run_every)
      if a is not None:
        x += nin(self.num_filters)(self.nonlinearity(a))
      x = self.nonlinearity(x)
      x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
      x = self.conv_op(2 * self.num_filters, self.batch_size, self.image_width,
                       self.max_rows)(x, row_start, cache_every, run_every)
      a, b = jnp.split(x, 2, axis=-1)
      c3 = a * jax.nn.sigmoid(b)
      return og_x + c3
    else:
      x = self.conv_op(self.num_filters, self.batch_size, self.image_width,
                       self.max_rows)(self.nonlinearity(og_x), row_start, cache_every, run_every)
      v_stack_pixel = a
      if b is not None:
        v_stack_pixel = jnp.concatenate([v_stack_pixel, b], axis=-1)
      x += nin(self.num_filters)(self.nonlinearity(v_stack_pixel))
      x = self.nonlinearity(x)
      x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
      x = self.conv_op(2 * self.num_filters, self.batch_size,
                       self.image_width, self.max_rows)(x, row_start, cache_every, run_every)
      a, b = jnp.split(x, 2, axis=-1)
      c3 = a * jax.nn.sigmoid(b)
      return og_x + c3
