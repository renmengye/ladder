from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import input_data
import logger
import os
import tensorflow as tf

from ladder_model import LadderModel, LadderConfig
from tqdm import tqdm

log = logger.get()


def test(sess, model, images, labels):
  """Runs test accuracy."""
  test_acc = sess.run(model.acc,
                      feed_dict={model.inputs: images,
                                 model.labels: labels})
  return test_acc


def main():
  """Trains a ladder network."""
  num_examples = 60000
  num_epochs = 150
  num_labeled = 100
  starter_lr = 0.02
  layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
  batch_size = 100
  denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]
  noise_std = 0.3
  config = LadderConfig(
      layer_sizes=layer_sizes,
      denoising_cost=denoising_cost,
      noise_std=noise_std)
  decay_after = 15
  num_iter = int((num_examples / batch_size) * num_epochs)

  log.info("=== Loading Data ===")
  mnist = input_data.read_data_sets(
      "MNIST_data", n_labeled=num_labeled, one_hot=False)

  log.info("=== Build Model ===")
  with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
      m = LadderModel(config, is_training=True)

  num_test_examples = 10000
  with tf.name_scope("Valid"):
    with tf.variable_scope("Model", reuse=True):
      mvalid = LadderModel(config, is_training=False)

  saver = tf.train.Saver()

  log.info("=== Starting Session ===")
  with tf.Session() as sess:
    i_iter = 0
    # get latest checkpoint (if any)
    ckpt = tf.train.get_checkpoint_state("checkpoints_new/")
    if ckpt and ckpt.model_checkpoint_path:
      # if checkpoint exists, restore the parameters and set epoch_n and i_iter
      saver.restore(sess, ckpt.model_checkpoint_path)
      epoch_n = int(ckpt.model_checkpoint_path.split("-")[1])
      i_iter = int((epoch_n + 1) * (num_examples / batch_size))
      log.info("Restored Epoch ", epoch_n)
    else:
      # no checkpoint exists. create checkpoints directory if it does not exist.
      if not os.path.exists("checkpoints_new"):
        os.makedirs("checkpoints_new")
      sess.run(tf.global_variables_initializer())
      [print(vv.name) for vv in tf.all_variables()]
    log.info("=== Training ===")
    test_acc = test(sess, mvalid, mnist.test.images, mnist.test.labels)
    log.info("Initial Accuracy: {:.2f}%".format(test_acc))
    m.assign_lr(sess, starter_lr)
    for i in tqdm(range(i_iter, num_iter)):
      images, labels = mnist.train.next_batch(batch_size)
      sess.run(m.train_op, feed_dict={m.inputs: images, m.labels: labels})
      if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0):
        epoch_n = int(i / (num_examples / batch_size))
        if (epoch_n + 1) >= decay_after:
          # decay learning rate
          ratio = 1.0 * (num_epochs - (epoch_n + 1))
          ratio = max(0, ratio / (num_epochs - decay_after))
          m.assign_lr(sess, starter_lr * ratio)
        saver.save(sess, "checkpoints_new/model.ckpt", epoch_n)
        test_acc = test(sess, mvalid, mnist.test.images, mnist.test.labels)
        log.info("Epoch {}, Accuracy: {:.2f}%".format(epoch_n, test_acc))
        # with open("train_log", "a") as train_log:
        #   train_log_w = csv.writer(train_log)
        #   log_i = [epoch_n, test_acc]
        #   train_log_w.writerow(log_i)
    test_acc = test(sess, mvalid, mnist.test.images, mnist.test.labels)
    log.info("Final Accuracy: {:.2f}%".format(test_acc))


if __name__ == "__main__":
  main()
