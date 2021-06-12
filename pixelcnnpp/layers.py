import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Any, Optional, Tuple, Union, Iterable
from functools import partial
from jax import lax


def concat_elu(x):
  """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
  return nn.elu(jnp.concatenate([x, -x], axis=-1))


def log_sum_exp(x):
  """ numerically stable log_sum_exp implementation that prevents overflow """
  # TF ordering
  axis = - 1
  m, _ = jnp.max(x, axis=axis)
  m2, _ = jnp.max(x, axis=axis, keepdims=True)
  return m + jnp.log(jnp.sum(jnp.exp(x - m2), axis=axis))


def down_shift(x, pad=None):
  xs = x.shape
  # when downshifting, the last row is removed
  x = x[:, :xs[1] - 1, :, :]
  if pad is None:
    pad = lambda f: jnp.pad(f, ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
  return pad(x)


def right_shift(x, pad=None):
  xs = x.shape
  # when downshifting, the last column is removed
  x = x[:, :, :xs[2] - 1, :]
  if pad is None:
    pad = lambda f: jnp.pad(f, ((0, 0), (0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
  return pad(x)


def log_prob_from_logits(x):
  """ numerically stable log_softmax implementation that prevents overflow """
  # TF ordering
  axis = -1
  m, _ = jnp.max(x, axis=axis, keepdims=True)
  return x - m - jnp.log(jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True))


def to_one_hot(tensor, n, fill_with=1.):
  # we perform one hot encore with respect to the last axis
  return jax.nn.one_hot(tensor, n) * fill_with


class WNDense(nn.Module):
  features: int
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    def weight_v_init_fn(rng):
      in_features = x.shape[-1]
      weight_v = jax.random.uniform(rng, (self.features, in_features),
                                    minval=-1. / jnp.sqrt(in_features),
                                    maxval=1. / jnp.sqrt(in_features))
      return weight_v

    def weight_g_init_fn(rng):
      weight_g = weight_v_init_fn(rng)
      g = jnp.sqrt(jnp.sum(jnp.square(weight_g), axis=-1, keepdims=True))
      return g

    def bias_init_fn(rng):
      in_features = x.shape[-1]
      bias = jax.random.uniform(rng, (self.features,),
                                minval=-1. / jnp.sqrt(in_features),
                                maxval=1. / jnp.sqrt(in_features))
      return bias

    weight_v = self.param('weight_v', weight_v_init_fn)
    weight_g = self.param('weight_g', weight_g_init_fn)

    weight_v = weight_v / jnp.sqrt(jnp.sum(jnp.square(weight_v), axis=-1, keepdims=True))
    weight = weight_v * weight_g

    output = x @ weight.T
    if self.use_bias:
      bias = self.param('bias', bias_init_fn)
      output += bias
    return output


class nin(nn.Module):
  dim_out: int

  @nn.compact
  def __call__(self, x):
    shp = list(x.shape)
    lin_a = WNDense(self.dim_out)
    out = lin_a(x.reshape((shp[0] * shp[1] * shp[2], shp[3])))
    shp[-1] = self.dim_out
    out = out.reshape(shp)
    return out


class WNConv(nn.Module):
  """2D convolution Modules with weightnorm."""
  features: int
  kernel_size: Tuple[int, int]
  strides: Optional[Tuple[int, int]] = None
  padding: Union[str, Iterable[Iterable[int]]] = 'VALID'
  init_scale: float = 1.
  dtype: Any = jnp.float32
  precision: Any = None
  use_bias: bool = True

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    strides = self.strides or (1,) * (inputs.ndim - 2)

    conv = partial(lax.conv_general_dilated, window_strides=strides,
                   padding=self.padding,
                   dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
                   precision=self.precision)

    in_features = inputs.shape[-1]
    kernel_shape = (self.features, in_features) + self.kernel_size

    def weight_v_initializer(rng):
      k = in_features * self.kernel_size[0] * self.kernel_size[1]
      weight_v = jax.random.uniform(rng, kernel_shape,
                                    minval=-1. / jnp.sqrt(k),
                                    maxval=1. / jnp.sqrt(k))
      return weight_v

    def weight_g_initializer(rng):
      weight_g = weight_v_initializer(rng)
      return jnp.sqrt(jnp.sum(jnp.square(weight_g), axis=(1, 2, 3), keepdims=True))

    def bias_initializer(rng):
      k = in_features * self.kernel_size[0] * self.kernel_size[1]
      bias = jax.random.uniform(rng, (self.features,),
                                minval=-1. / jnp.sqrt(k),
                                maxval=1. / jnp.sqrt(k))
      return bias

    weight_v = self.param('weight_v', weight_v_initializer)
    weight_g = self.param('weight_g', weight_g_initializer)
    normed_weight_v = weight_v / jnp.sqrt(jnp.sum(jnp.square(weight_v), axis=(1, 2, 3), keepdims=True))
    weight = normed_weight_v * weight_g

    output = conv(inputs, weight)
    if self.use_bias:
      bias = self.param('bias', bias_initializer)
      output = output + bias[..., :]

    return output


class WNConvTranspose(nn.Module):
  """2D convolution Modules with weightnorm."""
  features: int
  kernel_size: Tuple[int, int]
  strides: Optional[Tuple[int, int]] = None
  padding: Union[str, Iterable[Iterable[int]]] = 'VALID'
  init_scale: float = 1.
  dtype: Any = jnp.float32
  precision: Any = None
  use_bias: bool = True

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    strides = self.strides or (1,) * (inputs.ndim - 2)

    conv = partial(lax.conv_transpose, strides=strides, padding=self.padding,
                   dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
                   precision=self.precision,
                   transpose_kernel=True)

    in_features = inputs.shape[-1]
    kernel_shape = (in_features, self.features) + self.kernel_size

    def weight_v_initializer(rng):
      k = self.features * self.kernel_size[0] * self.kernel_size[1]
      weight_v = jax.random.uniform(rng, kernel_shape,
                                    minval=-1. / jnp.sqrt(k),
                                    maxval=1. / jnp.sqrt(k))
      return weight_v

    def weight_g_initializer(rng):
      weight_g = weight_v_initializer(rng)
      return jnp.sqrt(jnp.sum(jnp.square(weight_g), axis=(1, 2, 3), keepdims=True))

    def bias_initializer(rng):
      k = self.features * self.kernel_size[0] * self.kernel_size[1]
      bias = jax.random.uniform(rng, (self.features,),
                                minval=-1. / jnp.sqrt(k),
                                maxval=1. / jnp.sqrt(k))
      return bias

    weight_v = self.param('weight_v', weight_v_initializer)
    weight_g = self.param('weight_g', weight_g_initializer)
    normed_weight_v = weight_v / jnp.sqrt(jnp.sum(jnp.square(weight_v), axis=(1, 2, 3), keepdims=True))
    weight = normed_weight_v * weight_g

    output = conv(inputs, weight)
    if self.use_bias:
      bias = self.param('bias', bias_initializer)
      output = output + bias[..., :]

    return output


class down_shifted_conv2d(nn.Module):
  num_filters_out: int
  filter_size: Tuple[int, int] = (2, 3)
  stride: Tuple[int, int] = (1, 1)
  shift_output_down: bool = False

  @nn.compact
  def __call__(self, x):
    x = jnp.pad(x, ((0, 0), (self.filter_size[0] - 1, 0),
                    ((self.filter_size[1] - 1) // 2, (self.filter_size[1] - 1) // 2),
                    (0, 0)), mode='constant', constant_values=0)

    x = WNConv(features=self.num_filters_out,
               kernel_size=self.filter_size,
               strides=self.stride)(x)

    pad_fn = lambda f: jnp.pad(f, ((0, 0), (1, 0), (0, 0), (0, 0)),
                               mode='constant', constant_values=0)
    if self.shift_output_down:
      return down_shift(x, pad=pad_fn)
    else:
      return x


class down_shifted_deconv2d(nn.Module):
  num_filters_out: int
  filter_size: Tuple[int] = (2, 3)
  stride: Tuple[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    xs = x.shape
    target_shape = [xs[0], xs[1] * self.stride[0] + self.filter_size[0] - 1,
                    xs[2] * self.stride[1] + self.filter_size[1] - 1, self.num_filters_out]
    pad_before_h = self.filter_size[0] - 1
    pad_before_w = self.filter_size[1] - 1
    pad_after_h = target_shape[1] + self.filter_size[0] - 1 - (xs[1] - 1) * self.stride[0] - 1 - pad_before_h
    pad_after_w = target_shape[2] + self.filter_size[1] - 1 - (xs[2] - 1) * self.stride[1] - 1 - pad_before_w
    x = WNConvTranspose(features=self.num_filters_out,
                        kernel_size=self.filter_size,
                        strides=self.stride,
                        padding=((pad_before_h, pad_after_h),
                                 (pad_before_w, pad_after_w)))(x)
    xs = x.shape
    return x[:, :(xs[1] - self.filter_size[0] + 1),
           int((self.filter_size[1] - 1) / 2):(xs[2] - int((self.filter_size[1] - 1) / 2)), :]


class down_right_shifted_conv2d(nn.Module):
  num_filters_out: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)
  shift_output_right: bool = False

  @nn.compact
  def __call__(self, x):
    x = jnp.pad(x, (
      (0, 0),
      (self.filter_size[0] - 1, 0),
      (self.filter_size[1] - 1, 0),
      (0, 0)
    ), mode='constant', constant_values=0)
    x = WNConv(features=self.num_filters_out,
               kernel_size=self.filter_size,
               strides=self.stride)(x)
    if self.shift_output_right:
      def pad_fn(data):
        return jnp.pad(data, ((0, 0), (0, 0), (1, 0), (0, 0)))

      return right_shift(x, pad=pad_fn)
    else:
      return x


class down_right_shifted_deconv2d(nn.Module):
  num_filters_out: int
  filter_size: Tuple[int, int] = (2, 2)
  stride: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    xs = x.shape
    target_shape = [xs[0], xs[1] * self.stride[0] + self.filter_size[0] - 1,
                    xs[2] * self.stride[1] + self.filter_size[1] - 1, self.num_filters_out]
    pad_before_h = self.filter_size[0] - 1
    pad_before_w = self.filter_size[1] - 1
    pad_after_h = target_shape[1] + self.filter_size[0] - 1 - (xs[1] - 1) * self.stride[0] - 1 - pad_before_h
    pad_after_w = target_shape[2] + self.filter_size[1] - 1 - (xs[2] - 1) * self.stride[1] - 1 - pad_before_w
    x = WNConvTranspose(features=self.num_filters_out,
                        kernel_size=self.filter_size,
                        strides=self.stride,
                        padding=((pad_before_h, pad_after_h),
                                 (pad_before_w, pad_after_w)))(x)
    xs = x.shape
    x = x[:, :(xs[1] - self.filter_size[0] + 1):, :(xs[2] - self.filter_size[1] + 1), :]
    return x


'''
skip connection parameter : 0 = no skip connection 
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''


class gated_resnet(nn.Module):
  num_filters: int
  conv_op: Any
  nonlinearity: Any

  @nn.compact
  def __call__(self, og_x, a=None, train=True):
    x = self.nonlinearity(og_x)
    x = self.conv_op(self.num_filters)(x)
    if a is not None:
      x += nin(self.num_filters)(self.nonlinearity(a))
    x = self.nonlinearity(x)
    x = nn.Dropout(0.5, deterministic=not train, broadcast_dims=(1, 2))(x)
    x = self.conv_op(2 * self.num_filters)(x)
    a, b = jnp.split(x, 2, axis=-1)
    c3 = a * jax.nn.sigmoid(b)
    return og_x + c3