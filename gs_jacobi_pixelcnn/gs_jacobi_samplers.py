from pixelcnnpp.cached_samplers import *


def get_gs_jacobi_sampler(basic_sampler, model_fn, initial_cache, shape, max_rows, nr_logistic_mix=10):
  image_size = shape[1]
  input_channels = shape[-1]
  sample_batch_size = shape[0]

  def sampler(u):
    output_images = jnp.zeros((shape[0], shape[1] + max_rows, shape[2], shape[3]))

    def loop_fn(carry, step):
      cache, row_start, output_images = carry
      padded_images = jnp.pad(output_images, ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      row_inputs = jax.lax.dynamic_slice(padded_images, [0, row_start, 0, 0], [shape[0], max_rows, shape[2], shape[3]])
      row_inputs = jnp.pad(row_inputs, ((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1.)
      row_inputs = jnp.where(row_start == 0, row_inputs.at[:, 0, :, -1].set(0.), row_inputs)

      first_col_input = jnp.zeros((sample_batch_size, max_rows, 1, input_channels + 1))

      other_col_inputs = jax.lax.dynamic_slice(output_images, [0, row_start, 0, 0],
                                               [shape[0], max_rows, shape[2] - 1, shape[3]])
      other_col_inputs = jnp.pad(other_col_inputs, ((0, 0), (0, 0), (0, 0), (0, 1)),
                                 mode='constant', constant_values=1.)

      pixel_inputs = jnp.concatenate([first_col_input, other_col_inputs], axis=2)
      l, cache = model_fn(cache, row_inputs, pixel_inputs, row_start)

      mixture_u, sample_u = jnp.split(u, [(image_size + max_rows) * image_size * nr_logistic_mix], axis=-1)

      start_mixture = row_start * image_size * nr_logistic_mix
      start_sample = row_start * image_size * input_channels
      u_start_end = jnp.concatenate((
        jax.lax.dynamic_slice(mixture_u, [0, start_mixture],
                              [mixture_u.shape[0], max_rows * image_size * nr_logistic_mix]),
        jax.lax.dynamic_slice(sample_u, [0, start_sample],
                              [sample_u.shape[0], max_rows * image_size * input_channels])
      ), axis=-1)

      samples = basic_sampler(l, u=u_start_end)

      original = jax.lax.dynamic_slice(output_images, [0, row_start, 0, 0],
                                       [shape[0], max_rows, shape[2], shape[3]])
      matches = jnp.isclose(samples, original, atol=1e-5, rtol=1e-3)
      matches = jnp.cumprod(matches.reshape((matches.shape[0], -1)), axis=-1)
      n_matches = jnp.sum(jnp.prod(matches, axis=0)) // input_channels
      output_images = jax.lax.dynamic_update_slice(output_images, samples, [0, row_start, 0, 0])
      row_start += n_matches // image_size
      row_start = jnp.where(row_start >= image_size, image_size, row_start)

      return (cache, row_start, output_images), (output_images[:, :image_size, :, :], row_start)

    _, (all_images, all_row_ids) = jax.lax.scan(loop_fn, (initial_cache, jnp.asarray(0), output_images),
                                                jnp.arange(0, image_size**2), length=image_size ** 2)
    return all_images, all_row_ids

  return jax.jit(sampler)