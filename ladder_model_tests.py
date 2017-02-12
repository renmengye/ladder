"""
Checks the newer ladder implementation by reading the weights from the older
implementation.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import input_data
import logger
import numpy as np
import tensorflow as tf

from train_ladder import test
from ladder_model import LadderModel, LadderConfig

OLD_CKPT = "checkpoints_old"
NEW_CKPT = "checkpoints_new"
log = logger.get()


class LadderModelTests(tf.test.TestCase):

  def get_config(self):
    return LadderConfig(
        layer_sizes=[784, 1000, 500, 250, 250, 250, 10],
        denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10],
        noise_std=0.3)

  def get_dataset(self):
    return input_data.read_data_sets("MNIST_data", n_labeled=100, one_hot=False)

  def test_forward_pass(self):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        config = self.get_config()

        with tf.name_scope("Valid"):
          with tf.variable_scope("Model", reuse=None):
            mvalid = LadderModel(config, is_training=False)
            mvalid.load_weights(sess, np.load("weights.npz"))

        mnist = self.get_dataset()
        test_acc = test(sess, mvalid, mnist.test.images, mnist.test.labels)
      self.assertEqual(np.round(float(test_acc), decimals=2), 98.66)

  def test_train_step(self):
    with tf.Graph().as_default():
      with tf.Session() as sess, tf.device("/cpu:0"):
        # with tf.Session() as sess:
        tf.set_random_seed(0)
        config = self.get_config()
        with tf.name_scope("Train"):
          with tf.variable_scope("Model", reuse=None):
            m = LadderModel(
                config, dtype=tf.float64, is_training=True, debug=True)
            sess.run(tf.global_variables_initializer())
            weights_init = np.load("weights_init.npz")
            weights_act = m.get_weights()
            m.load_weights(sess, weights_init)

        # Check same initialization.
        for key in weights_act.keys():
          idx = np.argmax(
              np.abs(weights_act[key].eval() - weights_init[key]).ravel())
          np.testing.assert_allclose(
              weights_act[key].eval(), weights_init[key], rtol=1e-5, atol=1e-6)

        # Load same test data.
        test_data = np.load("test_images.npz")
        images, labels = test_data["images"], test_data["labels"]

        labels = labels.argmax(axis=1)
        m.assign_lr(sess, 0.02)
        loss, _ = sess.run([m.cost, m.train_op],
                           feed_dict={m.inputs: images,
                                      m.labels: labels})

        with tf.name_scope("Train"):
          with tf.variable_scope("Model", reuse=True):
            weights_act = m.get_weights()

        # Check same weight updates.
        weights_exp = np.load("weights_step1.npz")
        for key in weights_act.keys():
          idx = np.argmax(
              np.abs(weights_act[key].eval() - weights_exp[key]).ravel())
          # print(key, weights_act[key].eval().ravel()[idx],
          #       weights_exp[key].ravel()[idx], weights_act[key].eval().shape,
          #       weights_exp[key].shape, idx)
          np.testing.assert_allclose(
              weights_act[key].eval(), weights_exp[key], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
