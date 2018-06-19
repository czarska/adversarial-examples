from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tqdm import tqdm
from keras.datasets import cifar10
from keras.utils import np_utils

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
EPOCHES = 10
BATCH_SIZE = 16
LEARING_RATE = 1e-4

def agu(x, y, training=False):
    result_x = []
    result_y = []

    times = 1
    if training:
        times = 10

    for i in range(x.shape[0]):
        for _ in range(times):
            r1 = np.random.randint(0, 7)
            r2 = np.random.randint(0, 7)
            result_x.append([s[r2:(r2+24)] for s in x[i][r1:(r1+24)]])
            result_y.append(y[i])

    return (np.array(result_x), np.array(result_y))

def build_network(x, training=False, logits=False, reuse=False):

  with tf.variable_scope('conv1', reuse=reuse):
    W_conv1 = weight_variable([3, 3, 3, 128])
    b_conv1 = bias_variable([128])
    x = conv2d(x, W_conv1)
    x = batch_norm(x, training)
    x = tf.nn.relu(x + b_conv1)

  x_old = x

  with tf.variable_scope('conv2', reuse=reuse):
    W_conv2 = weight_variable([3, 3, 128, 128])
    b_conv2 = bias_variable([128])
    x = conv2d(x, W_conv2)
    x = batch_norm(x, training)
    x = tf.nn.relu(x + b_conv2)

  with tf.variable_scope('conv3', reuse=reuse):
    W_conv3 = weight_variable([3, 3, 128, 64])
    b_conv3 = bias_variable([64])
    x = conv2d(x, W_conv3)
    x = batch_norm(x, training)
    x += b_conv3

  with tf.variable_scope('conv33', reuse=reuse):
    W_conv33 = weight_variable([1, 1, 128, 64])
    x_old = conv2d(x_old, W_conv33)
  x += x_old
  x = max_pool_2x2(x)

  x_old = x

  with tf.variable_scope('conv4', reuse=reuse):
      W_conv4 = weight_variable([3, 3, 64, 64])
      b_conv4 = bias_variable([64])
      x = conv2d(x, W_conv4)
      x = batch_norm(x, training)
      x = tf.nn.relu(x + b_conv4)

  with tf.variable_scope('conv5', reuse=reuse):
      W_conv5 = weight_variable([3, 3, 64, 32])
      b_conv5 = bias_variable([32])
      x = conv2d(x, W_conv5)
      x = batch_norm(x, training)
      x += b_conv5

  with tf.variable_scope('conv55', reuse=reuse):
    W_conv55 = weight_variable([1, 1, 64, 32])
    x_old = conv2d(x_old, W_conv55)
  x += x_old
  x = max_pool_2x2(x)

  x_old = x

  with tf.variable_scope('conv6', reuse=reuse):
      W_conv6 = weight_variable([3, 3, 32, 32])
      b_conv6 = bias_variable([32])
      x = conv2d(x, W_conv6)
      x = batch_norm(x, training)
      x = tf.nn.relu(x + b_conv6)

  with tf.variable_scope('conv7', reuse=reuse):
      W_conv7 = weight_variable([3, 3, 32, 16])
      b_conv7 = bias_variable([16])
      x = conv2d(x, W_conv7)
      x = batch_norm(x, training)
      x += b_conv7

  with tf.variable_scope('conv77', reuse=reuse):
    W_conv77 = weight_variable([1, 1, 32, 16])
    x_old = conv2d(x_old, W_conv77)
  x += x_old

  with tf.variable_scope('pool1', reuse=reuse):
    x = max_pool_2x2(x)

  with tf.variable_scope('fc1', reuse=reuse):
    W_fc1 = weight_variable([3 * 3 * 16, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(x, [-1, 3 * 3 * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.variable_scope('fc2', reuse=reuse):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    log = tf.matmul(h_fc1, W_fc2) + b_fc2
    y_conv = tf.nn.softmax(log)
  if logits:
      return y_conv, log

  return y_conv


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def batch_norm(x, training):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.99,
                                        epsilon=0.001,
                                        is_training=training,
                                        fused=True,
                                        updates_collections=None)

def weight_variable(shape):
  return tf.get_variable("weight", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

def bias_variable(shape):
  return tf.get_variable("bias", shape=shape, initializer=tf.initializers.constant(0.1))

def evaluation(y, y_):
    temp = tf.argmax(y, 1)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    return acc, temp

def loss_func(y, y_):
    r = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    return tf.reduce_mean(r)

def main(_):
  # Import data
  (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
  train_features = train_features.astype('float32')
  test_features = test_features.astype('float32')
  train_features /= 255
  test_features /= 255
  (train_features, train_labels) = agu(train_features, train_labels, True)
  (test_features, test_labels) = agu(test_features, test_labels)
  train_labels = np_utils.to_categorical(train_labels, 10)
  test_labels = np_utils.to_categorical(test_labels, 10)

  with tf.variable_scope('model'):
    x = tf.placeholder(tf.float32, [None, 24, 24, 3])
    y_ = tf.placeholder(tf.float32, [None, 10])
    tr = tf.placeholder_with_default(False, (), name='mode')

    y_conv, logits = build_network(x, training=tr, logits=True)

    loss = loss_func(logits, y_)
    train_step = tf.train.AdamOptimizer(LEARING_RATE).minimize(loss)
    accuracy,_ = evaluation(y_conv, y_)

    saver = tf.train.Saver(max_to_keep=1)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in tqdm(range(EPOCHES)):
        perm = np.random.permutation(len(train_features))
        train_features = train_features[perm]
        train_labels = train_labels[perm]
        for i in range(int(len(train_features) / BATCH_SIZE)):
            train_step.run(feed_dict={x: train_features[i*BATCH_SIZE:(i+1)*BATCH_SIZE], y_: train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE], tr: True})
        print('test accuracy %g' % accuracy.eval(feed_dict={x: test_features[:100], y_: test_labels[:100]}))

    # Save the variables and model to disk.
    save_path = saver.save(sess, './tmp/original_cifar_model', global_step=8)
    print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/cifar/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)