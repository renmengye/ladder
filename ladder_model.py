"""Re-implements ladder networks.
Need to write unit test to ensure the exact re-producibility, forward +
backward.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logger
import numpy as np
import tensorflow as tf

from collections import namedtuple

log = logger.get()


def bi(name, shape, inits=0.0, dtype=tf.float32):
  """Declares a bias variable with random normal initialization."""
  return tf.get_variable(
      name, shape, dtype=dtype, initializer=tf.constant_initializer(inits))


def wi(name, shape, dtype=tf.float32):
  """Declares a weight variable with random normal initialization."""
  return tf.get_variable(
      name,
      # shape,
      dtype=dtype,
      initializer=tf.truncated_normal(
          shape, mean=0.0, stddev=1 / np.sqrt(np.prod(shape[:-1]))))


def gauss_denoise(z_corrupt, u, size):
  """Gaussian denoising function proposed in the original paper.

  Args:
  """
  a1 = bi("a1", [size], 0.)
  a2 = bi("a2", [size], 1.)
  a3 = bi("a3", [size], 0.)
  a4 = bi("a4", [size], 0.)
  a5 = bi("a5", [size], 0.)

  a6 = bi("a6", [size], 0.)
  a7 = bi("a7", [size], 1.)
  a8 = bi("a8", [size], 0.)
  a9 = bi("a9", [size], 0.)
  a10 = bi("a10", [size], 0.)

  mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
  v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

  z_recon = (z_corrupt - mu) * v + mu
  return z_recon


def batch_norm(x,
               is_training,
               gamma=None,
               beta=None,
               axes=[0, 1, 2],
               eps=1e-3,
               scope="bn",
               name="bn_out",
               mean=None,
               var=None,
               decay=0.9):
  """Applies batch normalization.
    Collect mean and variances on x except the last dimension. And apply
    normalization as below:
    x_ = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
      x: Input tensor, [B, ...].
      n_out: Integer, depth of input variable.
      gamma: Scaling parameter.
      beta: Bias parameter.
      axes: Axes to collect statistics.
      eps: Denominator bias.
      mean: Manually calculated mean and var.

    Returns:
      normed: Batch-normalized variable.
      mean: Mean used for normalization (optional).
  """
  with tf.variable_scope(scope):
    n_out = x.get_shape()[-1]
    emean = tf.get_variable("ema_mean", [n_out], trainable=False)
    evar = tf.get_variable("ema_var", [n_out], trainable=False)
    if is_training:
      if mean is None or var is None:
        mean, var = tf.nn.moments(x, axes, name="moments")
      # batch_mean.set_shape([n_out])
      # batch_var.set_shape([n_out])
      ema_mean_op = tf.assign(emean, emean * decay + mean * (1 - decay))
      ema_var_op = tf.assign(evar, evar * decay + var * (1 - decay))
      with tf.control_dependencies([ema_mean_op, ema_var_op]):
        normed = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, eps, name=name)
    else:
      if mean is None or var is None:
        mean, var = emean, evar
      normed = tf.nn.batch_normalization(
          x, mean, var, beta, gamma, eps, name=name)
  return normed


class LadderModel(object):
  """Ladder network object."""

  def __init__(self, config, is_training=True):
    """Initializes model, assuming architecture is feed-forward, layer-wise."""
    self._dtype = tf.float32
    self._config = config
    self._is_training = is_training

    inputs = tf.placeholder(
        tf.float32, shape=[None, config.layer_sizes[0]], name="inputs")
    labels = tf.placeholder(tf.int64, shape=[None], name="labels")
    self._inputs = inputs
    self._labels = labels
    
    with tf.name_scope("clean"):
      with tf.variable_scope("encoder"):
        y_clean, act_clean, moments_clean = self.encoder(inputs, 0.0)

    with tf.name_scope("corrupt"):
      with tf.variable_scope("encoder", reuse=True):
        y_corrupt, act_corrupt, _ = self.encoder(inputs, self.config.noise_std)

    with tf.variable_scope("decoder"):
      recon_cost = self.decoder(y_corrupt, act_corrupt, act_clean,
                                moments_clean)

    correct = tf.equal(tf.argmax(y_clean, 1), labels)
    acc = tf.reduce_mean(tf.cast(correct, "float")) * 100.0
    self._acc = acc

    if is_training:
      train_cost = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              self.slice_labeled(y_corrupt), labels))
      train_cost += tf.add_n(recon_cost)
      pred_cost = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              self.slice_labeled(y_clean), labels))
      lr = tf.Variable(0.0, trainable=False)
      train_step = tf.train.AdamOptimizer(lr).minimize(train_cost)
      new_lr = tf.placeholder(self.dtype, None)
      assign_lr = tf.assign(lr, new_lr)
      self._lr = lr
      self._new_lr = new_lr
      self._assign_lr = assign_lr
      self._train_op = train_step
    pass

  @property
  def config(self):
    return self._config

  @property
  def is_training(self):
    return self._is_training

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def dtype(self):
    """Type of the floating precision."""
    return self._dtype

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def acc(self):
    return self._acc

  def slice_unlabeled(self, x):
    batch_size = self.config.batch_size
    return tf.slice(x, [batch_size, 0], [-1, -1])

  def slice_labeled(self, x):
    batch_size = self.config.batch_size
    return tf.slice(x, [0, 0], [batch_size, -1])

  def split_lu(self, x):
    """Splits labeled and unlabeled data."""
    batch_size = self.config.batch_size
    labeled = self.slice_labeled(x)
    unlabeled = self.slice_unlabeled(x)
    return labeled, unlabeled

  def add_noise(self, x, noise_std):
    """Adds Gaussian noise to activation.

    Args:
      x: clean activation.
      noise_std: Gaussian noise standard deviation.

    Returns:
      x_corrupt: Corrupted activation.
    """
    return x + tf.random_normal(tf.shape(x)) * noise_std

  def encoder(self, inputs, noise_std):
    """Builds encoder part.

    Args:
      inputs: Inputs to the encoder.
      noise_std: Standard deviation of the additive Gaussian noise.

    Returns:
      output:
      act:
      moments:
    """
    # Add noise to input
    h = self.add_noise(inputs, noise_std)

    # Store intermediate activations.
    act = {}
    act["labeled"] = {}
    act["unlabeled"] = {}
    moments = {"mean": {}, "var": {}}
    act["labeled"][0], act["unlabeled"][0] = self.split_lu(h)
    L = len(self.config.layer_sizes) - 1

    for ll in range(1, L + 1):
      with tf.variable_scope("layer_{}".format(ll)):
        log.info("Layer {}: {} -> {}".format(ll, self.config.layer_sizes[
            ll - 1], self.config.layer_sizes[ll]))

        # Recognition weights.
        w_shape = [self.config.layer_sizes[ll - 1], self.config.layer_sizes[ll]]
        w = wi("w", w_shape, dtype=self.dtype)

        # Pre-activation
        z_pre = tf.matmul(h, w)

        # Split labeled and unlabeled examples.
        z_pre_l, z_pre_u = self.split_lu(z_pre)

        # Calculate batch statistics for unlabeled examples.
        # Change "axes" to [0, 1, 2] for conv nets.
        mean, var = tf.nn.moments(z_pre_u, axes=[0], name="moments")

        # In the original implementation, there is no gamma in BN until
        # the very last layer. In any case, affine transformation is
        # performed after noise injection.
        z_l_bn = batch_norm(
            z_pre_l, axes=[0], is_training=self.is_training, scope="bn_labeled")
        z_u_bn = batch_norm(
            z_pre_u,
            axes=[0],
            is_training=self.is_training,
            scope="bn_unlabeled",
            mean=mean,
            var=var)
        z = tf.concat(0, [z_l_bn, z_u_bn])

        # Add random Gaussian noise.
        z = self.add_noise(z, noise_std)

        beta = bi("beta", [self.config.layer_sizes[ll]], 0.0, dtype=self.dtype)
        if ll == L:
          # Use softmax activation in output layer.
          gamma = bi("gamma", [self.config.layer_sizes[ll]],
                     1.0,
                     dtype=self.dtype)

          # Just compute the logits here.
          h = gamma * (z + beta)
        else:
          # Use ReLU activation in hidden layers.
          h = tf.nn.relu(z + beta)

        # Save intermediate activation for reconstruction.
        act["labeled"][ll], act["unlabeled"][ll] = self.split_lu(z)
        # save mean and variance of unlabeled examples for decoding
        moments["mean"][ll] = mean
        moments["var"][ll] = var
    return h, act, moments

  def decoder(self, y_corrupt, act_corrupt, act_clean, moments_clean):
    """Builds decoder part.

    Args:
      y_corrupt: The activation of last layer through the corrupt path.
      act_corrupt: Intermediate layer activations (before non-linearity).

    Returns:
    """
    # To store the denoising cost of all layers
    recon_cost = []
    L = len(self.config.layer_sizes) - 1
    for ll in range(L, -1, -1):
      with tf.variable_scope("layer_{}".format(ll)):
        log.info("Layer {}: {} -> {}, denoising cost: {}".format(
            ll, self.config.layer_sizes[ll + 1] if ll < L else None,
            self.config.layer_sizes[ll], self.config.denoising_cost[ll]))

        z_clean = act_clean["unlabeled"][ll]
        z_corrupt = act_corrupt["unlabeled"][ll]
        mean = moments_clean["mean"].get(ll, 0.0)
        var = moments_clean["var"].get(ll, 1 - 1e-10)
        if ll == L:
          u = self.slice_unlabeled(y_corrupt)
        else:
          v_shape = [
              self.config.layer_sizes[ll + 1], self.config.layer_sizes[ll]
          ]
          print(v_shape)
          v = wi("v", v_shape, dtype=self.dtype)
          u = tf.matmul(z_recon, v)
        u = batch_norm(u, is_training=self.is_training, axes=[0], scope="bn_u")
        z_recon = gauss_denoise(z_corrupt, u, self.config.layer_sizes[ll])
        z_recon_bn = (z_recon - mean) / var
        _cost = tf.reduce_mean(tf.square(z_recon_bn - z_clean))
        _cost *= self.config.denoising_cost[ll]
        recon_cost.append(_cost)
    return recon_cost

  def assign_lr(self, sess, new_lr):
    sess.run(self._assign_lr, feed_dict={self._new_lr: new_lr})


LadderConfig = namedtuple(
    "LadderConfig",
    ["layer_sizes", "batch_size", "denoising_cost", "noise_std"])

if __name__ == "__main__":
  LadderModel(
      LadderConfig(
          layer_sizes=[784, 1000, 500, 250, 250, 250, 10],
          batch_size=100,
          denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10],
          noise_std=0.3))