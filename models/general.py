from random import sample

import tensorflow as tf
import numpy as np

from roadhouse.utils import config as cfg


def one_hot(labels, num_classes):
    base = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        base[i][labels[i]] = 1
    return base


def global_norm_pooling(inputs):
    return tf.reduce_sum(tf.square(inputs), axis=1, name='global_l2_norm_pooling')


def global_max_pooling(inputs):
    return tf.reduce_max(inputs, axis=1, name='global_max_pooling')


def global_avg_pooling(inputs):
    return tf.reduce_mean(inputs, axis=1, name='global_avg_pooling')


def build_loss(logits, labels):
    loss_sep = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    )
    loss_mean = tf.reduce_mean(loss_sep, axis=[0, 1])
    return loss_mean, tf.reduce_mean(loss_sep, axis=0)


def build_simple_adam(loss):
    opt = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return opt


def train_classification(optimizer, loss, x_ph, y_ph, X=None, y=None,
                         session=None, epochs=100, batch_size=32, data_generator=None,
                         val_data=None):
    if session is None:
        session = tf.InteractiveSession()

    assert (data_generator is not None)^(X is not None
                                         and y is not None)

    session.run(tf.global_variables_initializer())

    if X is not None:
        print('samples shape: ', X.shape)
        print('labels shape: ', y.shape)
        print('steps per epoch: ', X.shape[0] // batch_size + 1)

    losses = []
    val_losses = []

    num_classes = y_ph.get_shape()[-1].value

    def evaluate_single_batch(x_, y_, with_training=False):
        feeder = {x_ph: x_, y_ph: one_hot(y_, num_classes)}
        if with_training:
            return session.run(fetches=[optimizer, loss], feed_dict=feeder)[1]
        else:
            return session.run(fetches=loss, feed_dict=feeder)

    losses = []
    for e in range(epochs):

        epoch_losses = []

        for b, (x, y) in enumerate(data_generator()):
            L = evaluate_single_batch(x, y, True)
            epoch_losses.append(L)

            if b % cfg.TRAINING_DISPLAY == 0:
                print('loss: ', L)
                print('moving avg: ', np.mean(epoch_losses[-10:]))

        losses.append(epoch_losses)
