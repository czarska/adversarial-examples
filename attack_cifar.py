from utils import stats, heat
from attacks import fgsm, fgsm_it, cw
from cifar_resnet import agu, build_network, loss_func, evaluation
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils
import sys

ITERS = 100
ALPHA = 0.01
EPSI = 0.1

CW_ITERS = 100
BIN_STEPS = 20

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
(test_features, test_labels) = agu(test_features, test_labels)
train_features = train_features.astype('float32')
test_features = test_features.astype('float32')
train_features /= 255
test_features /= 255
test_labels = np_utils.to_categorical(test_labels, 10)

if len(sys.argv)<2 or sys.argv[1]=="fgsm_it":
    perturbed_accuracy = fgsm_it(test_features[:100], ITERS, EPSI, ALPHA, build_network, loss_func, evaluation, './tmp/original_cifar_model-8')
    print([s[0] for s in perturbed_accuracy])
    stats([s[1] for s in perturbed_accuracy], test_labels[:1000])
    heat([s[1] for s in perturbed_accuracy], test_labels[:1000], "cifar10_fgsm_")
else:
    cw(test_features[:100], test_labels[:100], CW_ITERS, BIN_STEPS, build_network, loss_func, evaluation, './tmp/original_cifar_model-8')