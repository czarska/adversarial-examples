import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from attacks import fgsm_it, cw
from utils import stats, stats_cari, heat, heat_cari
from mnist_cnn import build_network, loss_func, evaluation
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from keras.datasets import cifar10
from tqdm import tqdm

ITERS = 100
ALPHA = 0.01
EPSI = 0.1

CW_ITERS = 100
BIN_STEPS = 20
NORM = 'l2' #l0/l2/linf

mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)

if len(sys.argv)<2 or sys.argv[1]=="fgsm_it":
    perturbed_accuracy = fgsm_it(mnist.test.images[:1000], ITERS, EPSI, ALPHA, build_network, loss_func, evaluation)
    print([s[0] for s in perturbed_accuracy])
    stats([s[1] for s in perturbed_accuracy], mnist.test.labels[:1000])
    heat([s[1] for s in perturbed_accuracy], mnist.test.labels[:1000], "mnist_fgsm_")
else:
    perturbed_norms, was = cw(mnist.test.images[:1000], mnist.test.labels[:1000], CW_ITERS, BIN_STEPS, build_network, loss_func, evaluation, NORM)
    stats_cari(perturbed_norms, was)
    heat_cari(perturbed_norms, was)
