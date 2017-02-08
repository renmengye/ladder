"""Re-implements ladder networks.
Need to write unit test to ensure the exact re-producibility, forward +
backward.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import tensorflow as tf


def bi(inits, size, name, dtype=tf.float32):
  """Declares a bias variable with random normal initialization."""
  return tf.get_variable(
      name, shape, dtype=dtype, initializer=tf.constant_initializer(inits))


def wi(shape, name, dtype=tf.float32):
  """Declares a weight variable with random normal initialization."""
  return tf.get_variable(
      name,
      shape,
      dtype=dtype,
      initializer=tf.truncated_normal(
          shape, mean=0.0, stddev=1 / np.sqrt(shape[0])))


def batch_norm(x,
               is_training,
               gamma=None,
               beta=None,
               axes=[0, 1, 2],
               eps=1e-3,
               scope="bn",
               name="bn_out",
               return_mean=False):
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
      return_mean: Whether to also return the computed mean.

    Returns:
      normed: Batch-normalized variable.
      mean: Mean used for normalization (optional).
  """
  with tf.variable_scope(scope):
    n_out = tf.shape(x)[-1]
    emean = tf.get_variable("ema_mean", [n_out], trainable=False)
    evar = tf.get_variable("ema_var", [n_out], trainable=False)
    if is_training:
      batch_mean, batch_var = tf.nn.moments(x, axes, name="moments")
      batch_mean.set_shape([n_out])
      batch_var.set_shape([n_out])
      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      ema_apply_op_local = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op_local]):
        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
      emean_val = ema.average(batch_mean)
      evar_val = ema.average(batch_var)
      with tf.control_dependencies(
          [tf.assign(emean, emean_val), tf.assign(evar, evar_val)]):
        normed = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, eps, name=name)
    else:
      normed = tf.nn.batch_normalization(
          x, emean, evar, beta, gamma, eps, name=name)
  if return_mean:
    if is_training:
      return normed, mean
    else:
      return normed, emean
  else:
    return normed


class LadderModel(object):
  """Ladder network object."""

  def __init__(self, config):
    self._dtype = tf.float32
    pass

  @property
  def dtype(self):
    """Type of the floating precision."""
    return self._dtype

  def encoder(self, inputs, noise_std):
    """Build encoder part.

    Args:
      inputs: Inputs to the encoder.
      noise_std: Standard deviation of the additive Gaussian noise.

    Returns:

    """
    join = lambda l, u: tf.concat(0, [l, u])
    labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
    unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
    split_lu = lambda x: (labeled(x), unlabeled(x))

    h = inputs + tf.random_normal(tf.shape(
        inputs)) * noise_std  # add noise to input
    # To store the pre-activation, activation, mean and variance for each layer
    d = {}
    # The data for labeled and unlabeled examples are stored separately
    d["labeled"] = {"z": {}, "m": {}, "v": {}, "h": {}}
    d["unlabeled"] = {"z": {}, "m": {}, "v": {}, "h": {}}
    d["labeled"]["z"][0], d["unlabeled"]["z"][0] = split_lu(h)
    for l in range(1, L + 1):
      print "Layer ", l, ": ", layer_sizes[l - 1], " -> ", layer_sizes[l]
      d["labeled"]["h"][l - 1], d["unlabeled"]["h"][l - 1] = split_lu(h)
      z_pre = tf.matmul(h, weights["W"][l - 1])  # pre-activation
      z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

      m, v = tf.nn.moments(z_pre_u, axes=[0])

      def training_batch_norm():
        # Training batch normalization
        # batch normalization for labeled and unlabeled examples is performed
        # separately
        if noise_std > 0:
          # Corrupted encoder
          # batch normalization + noise
          z = join(
              batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
          z += tf.random_normal(tf.shape(z_pre)) * noise_std
        else:
          # Clean encoder
          # batch normalization + update the average mean and variance using
          # batch mean and variance of labeled examples
          z = join(
              update_batch_normalization(z_pre_l, l),
              batch_normalization(z_pre_u, m, v))
        return z

      def eval_batch_norm():
        # Evaluation batch normalization
        # obtain average mean and variance and use it to normalize the batch
        mean = ewma.average(running_mean[l - 1])
        var = ewma.average(running_var[l - 1])
        z = batch_normalization(z_pre, mean, var)
        # Instead of the above statement, the use of the following 2 statements
        # containing a typo
        # consistently produces a 0.2% higher accuracy for unclear reasons.
        # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
        # z = join(batch_normalization(z_pre_l, m_l, mean, var),
        # batch_normalization(z_pre_u, mean, var))
        return z

      # perform batch normalization according to value of boolean "training"
      # placeholder:
      z = tf.cond(training, training_batch_norm, eval_batch_norm)

      if l == L:
        # use softmax activation in output layer
        h = tf.nn.softmax(weights["gamma"][l - 1] *
                          (z + weights["beta"][l - 1]))
      else:
        # use ReLU activation in hidden layers
        h = tf.nn.relu(z + weights["beta"][l - 1])
      d["labeled"]["z"][l], d["unlabeled"]["z"][l] = split_lu(z)
      # save mean and variance of unlabeled examples for decoding
      d["unlabeled"]["m"][l], d["unlabeled"]["v"][l] = m, v
    d["labeled"]["h"][l], d["unlabeled"]["h"][l] = split_lu(h)
    return h, d

  def decoder(self):
    pass
