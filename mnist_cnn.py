from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

import sys
import argparse

import tensorflow as tf

def build_network(x, training=False, logits=False, reuse=False):
  with tf.variable_scope('reshape', reuse=reuse):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.variable_scope('conv1', reuse=reuse):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  with tf.variable_scope('pool1', reuse=reuse):
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.variable_scope('fc1', reuse=reuse):
    W_fc1 = weight_variable([14 * 14 * 32, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1 = tf.layers.dropout(h_fc1, rate=0.5, training=training)

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
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
  mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

  with tf.variable_scope('model'):
    x = tf.placeholder(tf.float32, [None, 784])
    tr = tf.placeholder_with_default(False, (), name='mode')

    y_conv, logits = build_network(x, training=tr, logits=True)

    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = loss_func(y_conv, y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    accuracy,_ = evaluation(y_conv, y_)

    saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in tqdm(range(20000)):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], tr: True})

    save_path = saver.save(sess, './tmp/original_mnist_model', global_step=8)
    print("Model saved in file: %s" % save_path)

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)