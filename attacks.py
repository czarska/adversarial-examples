import numpy as np
import tensorflow as tf
from utils import make_picture, disc
from tqdm import tqdm

def fgsm_it(images, ITERS, EPSI, ALPHA, build_network, loss_func, evaluation, restore_path="./tmp/original_mnist_model-8"):
    tf.reset_default_graph()
    with tf.variable_scope('model'):
        x = tf.placeholder(tf.float32, images.shape, name="input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="labels")
        tr = tf.placeholder_with_default(False, (), name='mode')

        advx = tf.identity(x)
        y_conv, logits = build_network(x, training=tr, logits=True)

        saver = tf.train.Saver()

    cond = lambda advx, i: i < ITERS
    def body(advx, i):
        y_conv, logits = build_network(advx, logits=True)
        loss = loss_func(logits, y_)
        loss = -loss
        gr, = tf.gradients(loss, advx)
        pertubation = tf.sign(gr)
        advx = tf.stop_gradient(ALPHA * pertubation + advx)
        advx = tf.clip_by_value(advx, x - EPSI, x + EPSI)
        advx = tf.clip_by_value(advx, 0, 1)
        return advx, i+1

    with tf.variable_scope('model', reuse=True):
        advx, _ = tf.while_loop(cond, body, (advx, tf.constant(0)), back_prop=False)

    sess=tf.Session()
    saver.restore(sess, restore_path)

    n = (images.shape)[0]
    accuracy = evaluation(y_conv, y_)
    perturbed_accuracy = []

    for i in tqdm(range(10)):
        indices = np.empty(n, dtype=np.int)
        indices.fill(i)
        target = np.zeros((n, 10))
        target[np.arange(n), indices] = 1

        perturbed_images = sess.run(advx, feed_dict={x: images, y_:target})
        #for itr in range(10):
        #    make_picture(perturbed_images[itr], images[itr], i, itr)
        perturbed_accuracy.append(sess.run(accuracy, feed_dict={x: perturbed_images, y_: target}))

    return perturbed_accuracy

def fgsm(images, EPSI, build_network, loss_func, evaluation, restore_path="./tmp/original_mnist_model-8"):
    tf.reset_default_graph()
    with tf.variable_scope('model'):
        x = tf.placeholder(tf.float32, images.shape, name="input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="labels")
        tr = tf.placeholder_with_default(False, (), name='mode')
        y_conv, logits = build_network(x, training=tr, logits=True)

    loss = loss_func(logits, y_)
    loss = -loss

    gr, = tf.gradients(loss, x)
    pertubation = tf.sign(gr)
    perturbed_op = tf.stop_gradient(EPSI * pertubation + images)
    perturbed_op = tf.clip_by_value(perturbed_op, 0, 1)

    sess=tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, restore_path)

    n = (images.shape)[0]
    accuracy = evaluation(y_conv, y_)
    perturbed_accuracy = []

    for i in tqdm(range(10)):
        indices = np.empty(n, dtype=np.int)
        indices.fill(i)
        target = np.zeros((n, 10))
        target[np.arange(n), indices] = 1

        perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y_: target})
        perturbed_accuracy.append(sess.run(accuracy, feed_dict={x: perturbed_images, y_: target}))

    return perturbed_accuracy

def loss_f6(logits, x_before, x_after, target, c):
    l2 = tf.sqrt(tf.reduce_sum(tf.square(x_before - x_after)))
    Z_t = tf.reduce_sum(target[0]*logits[0])
    temp = logits[0] - 100*target[0]
    Z_max = tf.reduce_max(temp)
    f6 = tf.maximum(Z_max-Z_t, 0.0)
    return l2 + f6*c[0]

def cw(images, labels, ITERS, BIN_STEPS, build_network, loss_func, evaluation, restore_path="./tmp/original_mnist_model-8"):
    tf.reset_default_graph()
    imp_sh = [1]
    for i in images.shape[1:]:
        imp_sh.append(i)
    print("SHAPE:", imp_sh)
    with tf.variable_scope('model'):
        x = tf.placeholder(tf.float32, imp_sh, name="input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="labels")
        tr = tf.placeholder_with_default(False, (), name='mode')
    with tf.variable_scope('modify'):
        d = tf.get_variable("bias", shape=images.shape[1:], initializer=tf.initializers.constant(0.001))
    advx = tf.clip_by_value(x+d, 0, 1)
    const = tf.placeholder(tf.float32, [1], name="const")
    with tf.variable_scope('model'):
        y_conv, logits = build_network(advx, training=tr, logits=True)

    loss = loss_f6(logits, x, advx, y_, const)
    opt = tf.train.GradientDescentOptimizer(1e-3).minimize(loss, var_list=[d])
    myinit = tf.variables_initializer(var_list=[d])

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars[2:])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, restore_path)

        afterd = 0
        for itr in tqdm(range(images.shape[0])):
            for i in range(10):
                target = np.zeros(10)
                target[i] = 1.0
                begin = 0.01
                end = 20.00
                for _ in range(BIN_STEPS):
                    sess.run(myinit)
                    c = (end + begin) / 2
                    for _ in range(ITERS):
                        _,l = sess.run([opt,loss], feed_dict={x: [images[itr]], y_: [target], const: [c]})
                    perturbed = y_conv.eval(feed_dict={x: [images[itr]], y_: [target], const: [c]})
                    if np.argmax(perturbed) == i:
                        end = c
                    else:
                        begin = c
                c = (begin+end)/2
                #if itr < 10:
                #    perturbed_im = advx.eval(feed_dict={x: [images[itr]], y_: [target], const: [c]})
                #    make_picture(perturbed_im, images[itr], i, itr)
                perturbed = y_conv.eval(feed_dict={x: disc([images[itr]]), y_: [target], const: [c]})
                if np.argmax(perturbed) == i:
                    afterd+=1
                print("target:", i, " c=", ((begin+end)/2), " dics:", (afterd/(10*itr+i+1)))
    return
