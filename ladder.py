from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import csv
import math
import os
import tensorflow as tf

import input_data
import logger

from tqdm import tqdm

log = logger.get()
layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
# number of layers
L = len(layer_sizes) - 1
num_examples = 60000
num_epochs = 150
num_labeled = 100
starter_learning_rate = 0.02
# epoch after which to begin learning rate decay
decay_after = 15
batch_size = 100
# number of loop iterations
num_iter = int((num_examples / batch_size) * num_epochs)

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)


def bi(inits, size, name):
  return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
  return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])


# shapes of linear layers
shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))

weights = {
    # Encoder weights
    "W": [wi(s, "W") for s in shapes],
    # Decoder weights
    "V": [wi(s[::-1], "V") for s in shapes],
    # batch normalization parameter to shift the normalized value
    "beta": [bi(0.0, layer_sizes[l + 1], "beta") for l in range(L)],
    # batch normalization parameter to scale the normalized value
    "gamma": [bi(1.0, layer_sizes[l + 1], "beta") for l in range(L)]
}

# scaling factor for noise used in corrupted encoder
noise_std = 0.3
# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

join = lambda l, u: tf.concat(0, [l, u])
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))

training = tf.placeholder(tf.bool)
# to calculate the moving averages of mean and variance
ewma = tf.train.ExponentialMovingAverage(decay=0.99)
# this list stores the updates to be made to average mean and variance
bn_assigns = []


def batch_normalization(batch, mean=None, var=None):
  if mean is None or var is None:
    mean, var = tf.nn.moments(batch, axes=[0])
  return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


# average mean and variance of all layers
running_mean = [
    tf.Variable(
        tf.constant(
            0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]
]
running_var = [
    tf.Variable(
        tf.constant(
            1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]
]


def update_batch_normalization(batch, l):
  "batch normalize + update average mean and variance of layer l"
  mean, var = tf.nn.moments(batch, axes=[0])
  assign_mean = running_mean[l - 1].assign(mean)
  assign_var = running_var[l - 1].assign(var)
  bn_assigns.append(ewma.apply([running_mean[l - 1], running_var[l - 1]]))
  with tf.control_dependencies([assign_mean, assign_var]):
    return (batch - mean) / tf.sqrt(var + 1e-10)


def encoder(inputs, noise_std):
  # add noise to input
  h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std
  # To store the pre-activation, activation, mean and variance for each layer
  d = {}
  # The data for labeled and unlabeled examples are stored separately
  d["labeled"] = {"z": {}, "m": {}, "v": {}, "h": {}}
  d["unlabeled"] = {"z": {}, "m": {}, "v": {}, "h": {}}
  d["labeled"]["z"][0], d["unlabeled"]["z"][0] = split_lu(h)
  for l in range(1, L + 1):
    log.info("Layer {}: {} -> {}".format(l, layer_sizes[l - 1], layer_sizes[l]))
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
      return z

    # perform batch normalization according to value of boolean "training"
    # placeholder:
    z = tf.cond(training, training_batch_norm, eval_batch_norm)

    if l == L:
      # use softmax activation in output layer
      h = tf.nn.softmax(weights["gamma"][l - 1] * (z + weights["beta"][l - 1]))
    else:
      # use ReLU activation in hidden layers
      h = tf.nn.relu(z + weights["beta"][l - 1])
    d["labeled"]["z"][l], d["unlabeled"]["z"][l] = split_lu(z)
    # save mean and variance of unlabeled examples for decoding
    d["unlabeled"]["m"][l], d["unlabeled"]["v"][l] = m, v
  d["labeled"]["h"][l], d["unlabeled"]["h"][l] = split_lu(h)
  return h, d


log.info("=== Corrupted Encoder ===")
y_c, corr = encoder(inputs, noise_std)

log.info("=== Clean Encoder ===")
y, clean = encoder(inputs, 0.0)  # 0.0 -> do not add noise

log.info("=== Decoder ===")


def g_gauss(z_c, u, size):
  "gaussian denoising function proposed in the original paper"
  wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
  a1 = wi(0., "a1")
  a2 = wi(1., "a2")
  a3 = wi(0., "a3")
  a4 = wi(0., "a4")
  a5 = wi(0., "a5")

  a6 = wi(0., "a6")
  a7 = wi(1., "a7")
  a8 = wi(0., "a8")
  a9 = wi(0., "a9")
  a10 = wi(0., "a10")

  mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
  v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

  z_est = (z_c - mu) * v + mu
  return z_est


# Decoder
z_est = {}
d_cost = []  # to store the denoising cost of all layers
for l in range(L, -1, -1):
  log.info("Layer {}: {} -> {}, denoising cost: {}".format(
      l, layer_sizes[l + 1]
      if l + 1 < len(layer_sizes) else None, layer_sizes[l], denoising_cost[l]))
  z, z_c = clean["unlabeled"]["z"][l], corr["unlabeled"]["z"][l]
  m, v = clean["unlabeled"]["m"].get(l, 0), clean["unlabeled"]["v"].get(
      l, 1 - 1e-10)
  if l == L:
    u = unlabeled(y_c)
  else:
    u = tf.matmul(z_est[l + 1], weights["V"][l])
  u = batch_normalization(u)
  z_est[l] = g_gauss(z_c, u, layer_sizes[l])
  z_est_bn = (z_est[l] - m) / v
  # append the cost of this layer to d_cost
  d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) /
                 layer_sizes[l]) * denoising_cost[l])

# calculate total unsupervised cost by adding the denoising cost of all layers
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)
# supervised cost
cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(y_N), 1))
# total cost
loss = cost + u_cost
# cost used for prediction
pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(y), 1))
# no of correct predictions
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(
    100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
  train_step = tf.group(bn_updates)

log.info("===  Loading Data ===")
mnist = input_data.read_data_sets(
    "MNIST_data", n_labeled=num_labeled, one_hot=True)

saver = tf.train.Saver()

log.info("===  Starting Session ===")
sess = tf.Session()

i_iter = 0
# get latest checkpoint (if any)
ckpt = tf.train.get_checkpoint_state("checkpoints/")
if ckpt and ckpt.model_checkpoint_path:
  # if checkpoint exists, restore the parameters and set epoch_n and i_iter
  saver.restore(sess, ckpt.model_checkpoint_path)
  epoch_n = int(ckpt.model_checkpoint_path.split("-")[1])
  i_iter = int((epoch_n + 1) * (num_examples / batch_size))
  log.info("Restored Epoch ", epoch_n)
else:
  # no checkpoint exists. create checkpoints directory if it does not exist.
  if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
  sess.run(tf.global_variables_initializer())

log.info("=== Training ===")
test_acc = sess.run(accuracy,
                    feed_dict={
                        inputs: mnist.test.images,
                        outputs: mnist.test.labels,
                        training: False
                    })
log.info("Initial Accuracy: {:.2f}%".format(test_acc))

for i in tqdm(range(i_iter, num_iter)):
  images, labels = mnist.train.next_batch(batch_size)
  sess.run(train_step,
           feed_dict={inputs: images,
                      outputs: labels,
                      training: True})
  if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0):
    epoch_n = int(i / (num_examples / batch_size))
    if (epoch_n + 1) >= decay_after:
      # decay learning rate
      # lr = starter_lr * ((num_epochs - epoch_n) / (num_epochs - decay_after))
      # epoch_n + 1 because learning rate is set for next epoch
      ratio = 1.0 * (num_epochs - (epoch_n + 1))
      ratio = max(0, ratio / (num_epochs - decay_after))
      sess.run(learning_rate.assign(starter_learning_rate * ratio))
    saver.save(sess, "checkpoints/model.ckpt", epoch_n)
    test_acc = sess.run(accuracy,
                        feed_dict={
                            inputs: mnist.test.images,
                            outputs: mnist.test.labels,
                            training: False
                        })
    log.info("Epoch {}, Accuracy: {:.2f}%".format(epoch_n, test_acc))
    with open("train_log", "a") as train_log:
      # write test accuracy to file "train_log"
      train_log_w = csv.writer(train_log)
      log_i = [epoch_n, test_acc]
      train_log_w.writerow(log_i)

test_acc = sess.run(accuracy,
                    feed_dict={
                        inputs: mnist.test.images,
                        outputs: mnist.test.labels,
                        training: False
                    })
log.info("Final Accuracy: {:.2f}%".format(test_acc))
sess.close()
