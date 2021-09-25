from pixelcnnpp.cached_samplers import *
import tqdm


def get_jacobi_gs_sampler(basic_sampler, model_fn_first, model_fn, initial_cache, shape, block_size, nr_logistic_mix=10):
  image_size = shape[1]
  input_channels = shape[-1]
  sample_batch_size = shape[0]
  num_blocks = image_size ** 2 // block_size

  def sampler(u):
    output_images = jnp.zeros((sample_batch_size, image_size, image_size, input_channels))

    def jacobi_loop(carry, index):
      cache, output_images = carry
      first_row_input = jnp.zeros((sample_batch_size, 1, image_size, input_channels + 1))
      other_row_inputs = jnp.concatenate(
        [output_images[:, :-1, :, :],
         jnp.ones((sample_batch_size, image_size - 1, image_size, 1))], axis=-1)
      row_inputs = jnp.concatenate([first_row_input, other_row_inputs], axis=1)

      first_col_input = jnp.zeros((sample_batch_size, image_size, 1, input_channels + 1))
      other_col_inputs = jnp.concatenate(
        [output_images[:, :, :-1, :],
         jnp.ones((sample_batch_size, image_size, image_size - 1, 1))], axis=-1)

      pixel_inputs = jnp.concatenate([first_col_input, other_col_inputs], axis=2)
      rows = jnp.asarray([(block * block_size) // image_size for block in range(num_blocks)])
      cols = jnp.asarray([(block * block_size) % image_size for block in range(num_blocks)])

      l, cache = model_fn_first(cache, row_inputs, pixel_inputs, rows=rows, cols=cols)
      output_images = basic_sampler(l, u=u)
      jacobi_images = [output_images[None, ...]]

      def gs_loop(carry, block_idx):
        cache, output_images = carry

        def input_gen(block):
          row = (block * block_size + block_idx) // image_size
          col = (block * block_size + block_idx) % image_size

          padded_output_image = jnp.pad(output_images, ((0, 0), (1, 0), (0, 0), (0, 0)),
                                        mode='constant', constant_values=0.)
          x_row_input = jax.lax.dynamic_slice(padded_output_image, [0, row, 0, 0],
                                              [shape[0], 1, shape[2], shape[3]])
          x_row_input = jnp.pad(x_row_input, ((0, 0), (0, 0), (0, 0), (0, 1)),
                                mode='constant', constant_values=1.)

          x_row_input = jnp.where(row == 0, x_row_input.at[..., -1].set(0.), x_row_input)

          x_pixel_input = jnp.where(col == 0,
                                    jnp.zeros((sample_batch_size, 1, 1, input_channels + 1)),
                                    jnp.pad(jax.lax.dynamic_slice(output_images, [0, row, col - 1, 0],
                                                                  [shape[0], 1, 1, shape[-1]]),
                                            ((0, 0), (0, 0), (0, 0), (0, 1)),
                                            mode='constant', constant_values=1.))

          return (row, col, x_row_input, x_pixel_input)

        (rows, cols, row_inputs, pixel_inputs) = jax.vmap(input_gen)(jnp.arange(0, num_blocks))
        row_inputs = row_inputs.reshape((-1, *row_inputs.shape[2:]))
        pixel_inputs = pixel_inputs.reshape((-1, *pixel_inputs.shape[2:]))

        l, cache = model_fn(cache, row_inputs, pixel_inputs, rows, cols)

        mixture_u, sample_u = jnp.split(u, [image_size * image_size * nr_logistic_mix], axis=-1)
        start_mixture = rows * image_size * nr_logistic_mix + cols * nr_logistic_mix
        mixture_range = start_mixture[:, None] + jnp.arange(nr_logistic_mix)
        start_sample = rows * image_size * input_channels + cols * input_channels
        sample_range = start_sample[:, None] + jnp.arange(input_channels)
        mixture_u = mixture_u[:, mixture_range].transpose((1, 0, 2))
        sample_u = sample_u[:, sample_range].transpose((1, 0, 2))
        input_u = jnp.concatenate([mixture_u, sample_u], axis=-1)
        input_u = input_u.reshape((-1, *input_u.shape[2:]))


        samples = basic_sampler(l, u=input_u)
        samples = samples.reshape((num_blocks, -1, *samples.shape[1:]))

        output_images = output_images.at[:, rows[:, None, None], cols[:, None, None], :].set(
          samples.transpose((1, 0, 2, 3, 4))
        )
        return (cache, output_images), output_images

      (cache, output_images), gs_output_images = jax.lax.scan(gs_loop, (cache, output_images),
                                                              jnp.arange(1, block_size))
      jacobi_output_images = jnp.concatenate(jacobi_images + [gs_output_images], axis=0)
      return (cache, output_images), jacobi_output_images

    (_, final_sample), all_images = jax.lax.scan(jacobi_loop, (initial_cache, output_images),
                                                 jnp.arange(0, num_blocks))
    return final_sample, all_images

  return jax.jit(sampler)
