1. First train clasificator (data will be stored in tmp directory) - python mnist_cnn.py / python cifar10_resnet.py
2. To run attacks - python attack_mnist.py / python attack_cifar.py -> will start FGSM attack
python attack_mnist.py cw / python attack_cifar.py cw -> will start Carlini and Wagner attack
