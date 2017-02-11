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

with tf.Graph().as_default():
  with tf.Session() as sess:
    config = LadderConfig(
        layer_sizes=[784, 1000, 500, 250, 250, 250, 10],
        denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10],
        noise_std=0.3)
    # with tf.name_scope("Train"):
    #   with tf.variable_scope("Model"):
    #     mvalid = LadderModel(config, is_training=True)
    #     variable_map = mvalid.get_variable_map()
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=None):
        mvalid = LadderModel(config, is_training=False)
        variable_map = mvalid.get_variable_map()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(variable_map)
    ckpt = tf.train.get_checkpoint_state(OLD_CKPT)
    
    # saver = tf.train.Saver()
    # ckpt = tf.train.get_checkpoint_state(NEW_CKPT)
    
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      raise Exception("Unable to restore checkpoint.")

    # print(variable_map["running_mean"].eval())
    L = len(config.layer_sizes) - 1
    wws = np.load("weights.npz")
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True):
        with tf.variable_scope("encoder"):
          for ll in range(1, L + 1):
            with tf.variable_scope("layer_{}".format(ll)):
              if ll > 1:
                w = tf.get_variable("w")
                beta = tf.get_variable("beta")
              else:
                w = tf.get_variable("w")
                beta = tf.get_variable("beta")
              with tf.variable_scope("bn_labeled"):
                if ll > 1:
                  mean = tf.get_variable("ema_mean")
                  var = tf.get_variable("ema_var")
                else:
                  mean = tf.get_variable("ema_mean")
                  var = tf.get_variable("ema_var")
              print(ll)
              print("w", w.eval().mean())
              print("w", w.eval().ravel()[:4])
              sess.run(tf.assign(w, wws[str(ll - 1)]))
              print("w", w.eval().mean())
              print("beta", beta.eval().mean())
              # print("mean", mean.eval().mean())
              # print("var", var.eval().mean())

    mnist = input_data.read_data_sets(
        "MNIST_data", n_labeled=100, one_hot=False)
    test_acc = test(sess, mvalid, mnist.test.images, mnist.test.labels)

  log.info("Final Accuracy: {:.2f}%".format(test_acc))
  # print(variable_map)
