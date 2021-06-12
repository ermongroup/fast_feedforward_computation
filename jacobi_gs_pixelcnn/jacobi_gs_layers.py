from pixelcnnpp.cached_layers import *
import math


def sum_rightshift_downshift(rightshifted_pixel, downshifted_row, cols, num_blocks, first_index=True):
  '''Sums the vertical and horizontal stack.'''
  if first_index:
    return rightshifted_pixel + downshifted_row
  else:
    # num_block x B x H x W x C
    downshifted_row = downshifted_row.reshape((num_blocks, -1, *downshifted_row.shape[1:]))
    rightshifted_pixel = rightshifted_pixel.reshape((num_blocks, -1, *rightshifted_pixel.shape[1:]))

    # num_block x 1 x B x H x C
    downshifted_row_pixel = downshifted_row[jnp.arange(num_blocks)[:, None], ..., cols[:, None], :]

    # num_block x B x H x 1 x C
    downshifted_row_pixel = downshifted_row_pixel.transpose((0, 2, 3, 1, 4))
    sums = downshifted_row_pixel + rightshifted_pixel
    sums = sums.reshape((-1, *sums.shape[2:]))

    return sums


class down_shifted_conv2d(nn.Module):
  num_blocks: int
  original_image_width: int
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 3)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, all_inputs, rows, cols, cache_every, run_every, first_index=True):
    output_width = int(math.ceil(self.image_width // float(self.stride[1])))
    output_cache = self.variable('cache', 'output_cache', lambda: jnp.zeros(
      (self.num_blocks * self.batch_size, 1, output_width, self.num_filters_out)
    ))
    conv = WNConv(features=self.num_filters_out,
                  kernel_size=self.filter_size,
                  strides=self.stride)
    if first_index:
      pad_left = int((self.filter_size[1] - 1) / 2)
      pad_right = int((self.filter_size[1] - 1) / 2)
      pad_top = self.filter_size[0] - 1
      pad_down = 0

      padded_input = jnp.pad(all_inputs, ((0, 0), (pad_top, pad_down), (pad_left, pad_right), (0, 0)),
                             mode='constant', constant_values=0.)
      outputs = conv(padded_input)
      os = output_cache.value.shape
      should_update_cache = (rows % run_every == 0)
      run_rows = rows // run_every
      blockwise_outputs = outputs[:, run_rows[:, None], :, :].transpose((1, 0, 2, 3, 4))
      blockwise_cache = output_cache.value.reshape((self.num_blocks, -1, *os[1:]))
      new_blockwise_cache = jax.vmap(jnp.where)(should_update_cache, blockwise_outputs, blockwise_cache)
      output_cache.value = new_blockwise_cache.reshape(os)
      return outputs
    else:
      return output_cache.value


class down_shifted_deconv2d(nn.Module):
  num_blocks: int
  original_image_width: int
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 3)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, all_inputs, rows, cols, cache_every, run_every, first_index=True):
    filter_width = self.filter_size[1]
    left_pad = filter_width - 1
    cache_width = self.image_width * self.stride[0] + left_pad
    output_cache = self.variable('cache', 'output_cache', lambda: jnp.zeros(
      (self.num_blocks * self.batch_size, 1, cache_width - 2, self.num_filters_out)
    ))
    xs = all_inputs.shape
    target_shape = [xs[0], None, None, self.num_filters_out]
    target_shape[1] = deconv_output_length(xs[1], self.filter_size[0], 'valid', output_padding=0,
                                           stride=self.stride[0])
    target_shape[2] = deconv_output_length(xs[2], self.filter_size[1], 'valid', output_padding=1,
                                           stride=self.stride[1])
    pad_before_h = self.filter_size[0] - 1
    pad_before_w = self.filter_size[1] - 1
    pad_after_h = target_shape[1] + self.filter_size[0] - 1 - (xs[1] - 1) * self.stride[0] - 1 - pad_before_h
    pad_after_w = target_shape[2] + self.filter_size[1] - 1 - (xs[2] - 1) * self.stride[1] - 1 - pad_before_w

    deconv = WNConvTranspose(features=self.num_filters_out,
                             kernel_size=self.filter_size,
                             strides=self.stride,
                             padding=((pad_before_h, pad_after_h),
                                      (pad_before_w, pad_after_w)))

    if first_index:
      outputs = deconv(all_inputs)
      os = output_cache.value.shape
      blockwise_cache = output_cache.value.reshape((self.num_blocks, -1, *os[1:]))
      should_update_cache = (rows % run_every == 0)
      run_rows = rows // run_every
      blockwise_outputs = outputs[:, run_rows[:, None], 1:-1, :].transpose((1, 0, 2, 3, 4))
      new_blockwise_cache = jax.vmap(jnp.where)(should_update_cache, blockwise_outputs, blockwise_cache)
      output_cache.value = new_blockwise_cache.reshape(os)
      return outputs[:, :, 1:-1, :]

    else:
      return output_cache.value


class down_right_shifted_conv2d(nn.Module):
  num_blocks: int
  original_image_width: int
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)
  shift_output_right: bool = False

  @nn.compact
  def __call__(self, pixel_inputs, rows, cols, cache_every, run_every, first_index=True):
    filter_height = self.filter_size[0]
    filter_width = self.filter_size[1]
    left_pad = filter_width - 1
    top_pad = filter_height - 1
    cache_height = self.filter_size[0]
    cache_width = self.image_width + left_pad
    cache = self.variable('cache', 'cache', lambda: jnp.zeros(
      (self.batch_size * self.num_blocks, cache_height, cache_width, pixel_inputs.shape[-1])
    ))
    conv = WNConv(features=self.num_filters_out,
                  kernel_size=self.filter_size,
                  strides=self.stride)

    if first_index:
      cs = cache.value.shape
      blockwise_cache = cache.value.reshape((self.num_blocks, -1, *cs[1:]))
      padded_pixel_inputs = jnp.pad(pixel_inputs, ((0, 0), (self.filter_size[0] - 1, 0),
                                                   (self.filter_size[1] - 1, 0), (0, 0)),
                                    mode='constant', constant_values=0.)
      should_cache = (rows % cache_every == 0)
      cache_row = rows // cache_every
      pads = jnp.arange(0, top_pad + 1)
      cache_row_ranges = cache_row[:, None] + pads[None, :]
      new_blockwise_cache = padded_pixel_inputs[:, cache_row_ranges, :, :].transpose((1, 0, 2, 3, 4))
      new_blockwise_cache = jax.vmap(jnp.where)(should_cache, new_blockwise_cache, blockwise_cache)
      cache.value = new_blockwise_cache.reshape(cs)

      return conv(padded_pixel_inputs)

    else:
      # num_blocks x B x H x W x C
      pixel_inputs = pixel_inputs.reshape((self.num_blocks, -1, *pixel_inputs.shape[1:]))
      cs = cache.value.shape
      # num_blocks x B x H x W x C
      blockwise_cache = cache.value.reshape((self.num_blocks, -1, *cs[1:]))
      should_cache = (rows % cache_every == 0) & (cols % cache_every == 0)
      should_run = (rows % run_every == 0) & (cols % run_every == 0)

      cache_col = cols // cache_every
      pixel_col = cache_col + left_pad
      new_blockwise_cache = jax.vmap(jnp.where)(
        should_cache,
        # num_blocks x 1 x B x H x C
        blockwise_cache.at[jnp.arange(0, pixel_col.shape[0])[:, None], :, -1:, pixel_col[:, None], :].set(
          pixel_inputs.transpose((0, 3, 1, 2, 4))),
        blockwise_cache
      )
      cache.value = new_blockwise_cache.reshape(cs)
      ## Run
      outputs = jnp.zeros((self.num_blocks * self.batch_size, 1, 1, self.num_filters_out))
      outputs_view = outputs.reshape((self.num_blocks, -1, *outputs.shape[1:]))
      width_start = cols // cache_every
      width_range = width_start[:, None] + jnp.arange(0, filter_width)
      # num_blocks x filter_width x B x H x C
      cache_neighborhood = new_blockwise_cache[jnp.arange(0, self.num_blocks)[:, None], :, :, width_range, :]
      # num_blocks x B x H x filter_width x C
      cache_neighborhood = cache_neighborhood.transpose((0, 2, 3, 1, 4))
      cache_neighborhood = cache_neighborhood.reshape((-1, *cache_neighborhood.shape[2:]))
      after_conv = conv(cache_neighborhood)
      after_conv = after_conv.reshape((self.num_blocks, -1, *after_conv.shape[1:]))
      outputs_view = jax.vmap(jnp.where)(should_run, after_conv, outputs_view)

      return outputs_view.reshape(outputs.shape)


class down_right_shifted_deconv2d(nn.Module):
  num_blocks: int
  original_image_width: int
  num_filters_out: int
  batch_size: int
  image_width: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, pixel_inputs, rows, cols, cache_every, run_every, first_index=True):
    cache = self.variable('cache', 'cache', lambda: jnp.zeros(
      (self.num_blocks * self.batch_size, 1, self.image_width * self.stride[1], self.num_filters_out)
    ))

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

    deconv = WNConvTranspose(features=self.num_filters_out,
                             kernel_size=self.filter_size,
                             strides=self.stride,
                             padding=((pad_before_h, pad_after_h),
                                      (pad_before_w, pad_after_w)))
    if first_index:
      outputs = deconv(pixel_inputs)
      cs = cache.value.shape
      blockwise_cache = cache.value.reshape((self.num_blocks, -1, *cs[1:]))
      should_run = (rows % run_every == 0)
      cache_row = rows // run_every
      new_blockwise_cache = jax.vmap(jnp.where)(
        should_run, outputs[:, cache_row[:, None], :, :].transpose((1, 0, 2, 3, 4)), blockwise_cache
      )
      cache.value = new_blockwise_cache.reshape(cs)
      return outputs

    else:
      output_for_cache = deconv(pixel_inputs)
      cs = cache.value.shape
      blockwise_cache = cache.value.reshape((self.num_blocks, -1, *cs[1:]))

      should_cache = (rows % cache_every == 0) & (cols % cache_every == 0)
      should_run = (rows % run_every == 0) & (cols % run_every == 0)

      cache_col = cols // cache_every
      output_for_cache = output_for_cache.reshape((-1, self.batch_size, *output_for_cache.shape[1:]))
      cache_range_start = cache_col * self.stride[1]
      cache_range = cache_range_start[:, None] + jnp.arange(0, self.stride[1])
      output_for_cache = output_for_cache[:, :, 0:1, :, :]
      new_blockwise_cache = jax.vmap(jnp.where)(
        should_cache, blockwise_cache.at[jnp.arange(0, self.num_blocks)[:, None], :, :, cache_range, :].set(
          output_for_cache.transpose((0, 3, 1, 2, 4))
        ),
        blockwise_cache
      )
      cache.value = new_blockwise_cache.reshape(cs)
      outputs = jnp.zeros((self.num_blocks * self.batch_size, 1, 1, self.num_filters_out))

      output_col = cols // run_every
      outputs_view = outputs.reshape((self.num_blocks, -1, *outputs.shape[1:]))
      cache_for_outputs = new_blockwise_cache[jnp.arange(0, self.num_blocks)[:, None], :, :, output_col[:, None], :]
      outputs_view = jax.vmap(jnp.where)(should_run, cache_for_outputs.transpose((0, 2, 3, 1, 4)), outputs_view)
      outputs = outputs_view.reshape(outputs.shape)

      return outputs


'''
skip connection parameter : 0 = no skip connection 
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''


class gated_resnet(nn.Module):
  num_blocks: int
  original_image_width: int
  num_filters: int
  conv_op: Any
  batch_size: int
  image_width: int
  nonlinearity: Any

  @nn.compact
  def __call__(self, og_x, rows, cols, cache_every, run_every, vstack=False, a=None, b=None, first_index=True,
               train=False):
    if first_index:
      if vstack:
        x = self.conv_op(self.num_blocks, self.original_image_width, self.num_filters,
                         self.batch_size, self.image_width)(self.nonlinearity(og_x), rows, cols, cache_every, run_every,
                                                            first_index=first_index)
        if a is not None:
          x += nin(self.num_filters)(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
        x = self.conv_op(self.num_blocks, self.original_image_width, 2 * self.num_filters,
                         self.batch_size, self.image_width)(x, rows, cols, cache_every, run_every, first_index)
        a, b = jnp.split(x, 2, axis=-1)
        c3 = a * jax.nn.sigmoid(b)
        return og_x + c3
      else:
        x = self.conv_op(self.num_blocks, self.original_image_width, self.num_filters,
                         self.batch_size, self.image_width)(self.nonlinearity(og_x), rows, cols, cache_every, run_every,
                                                            first_index=first_index)
        v_stack_pixel = a
        if b is not None:
          v_stack_pixel = jnp.concatenate([a, b], axis=-1)

        x += nin(self.num_filters)(self.nonlinearity(v_stack_pixel))
        x = self.nonlinearity(x)
        x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
        x = self.conv_op(self.num_blocks, self.original_image_width, 2 * self.num_filters,
                         self.batch_size, self.image_width)(x, rows, cols, cache_every, run_every, first_index)
        a, b = jnp.split(x, 2, axis=-1)
        c3 = a * jax.nn.sigmoid(b)
        return og_x + c3
    else:
      if vstack:
        x = self.conv_op(self.num_blocks, self.original_image_width, self.num_filters,
                         self.batch_size, self.image_width)(self.nonlinearity(og_x), rows, cols, cache_every, run_every,
                                                            first_index=first_index)
        if a is not None:
          x += nin(self.num_filters)(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
        x = self.conv_op(self.num_blocks, self.original_image_width, 2 * self.num_filters,
                         self.batch_size, self.image_width)(x, rows, cols, cache_every, run_every, first_index)
        a, b = jnp.split(x, 2, axis=-1)
        c3 = a * jax.nn.sigmoid(b)
        return og_x + c3
      else:
        x = self.conv_op(self.num_blocks, self.original_image_width, self.num_filters,
                         self.batch_size, self.image_width)(self.nonlinearity(og_x), rows, cols, cache_every, run_every,
                                                            first_index=first_index)

        a_view = a.reshape((self.num_blocks, -1, *a.shape[1:]))
        cache_col = cols // cache_every
        v_stack_pixel = a_view[jnp.arange(0, self.num_blocks)[:, None], :, :, cache_col[:, None], :]
        v_stack_pixel = v_stack_pixel.transpose((0, 2, 3, 1, 4))

        if b is not None:
          b_view = b.reshape((self.num_blocks, -1, *b.shape[1:]))
          v_stack_pixel = jnp.concatenate([v_stack_pixel, b_view], axis=-1)

        v_stack_pixel = v_stack_pixel.reshape((-1, *v_stack_pixel.shape[2:]))

        x += nin(self.num_filters)(self.nonlinearity(v_stack_pixel))
        x = self.nonlinearity(x)
        x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
        x = self.conv_op(self.num_blocks, self.original_image_width, 2 * self.num_filters,
                         self.batch_size, self.image_width)(x, rows, cols, cache_every, run_every, first_index)
        a, b = jnp.split(x, 2, axis=-1)
        c3 = a * jax.nn.sigmoid(b)
        return og_x + c3
