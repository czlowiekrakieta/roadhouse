import tensorflow as tf
from tensorflow.python.layers.convolutional import conv1d
from tensorflow.python.layers.pooling import max_pooling1d, average_pooling1d
from tensorflow.python.layers.core import dense
from tensorflow.contrib.rnn import (LSTMCell,
                                    GRUCell,
                                    DropoutWrapper,
                                    LSTMStateTuple,
                                    MultiRNNCell)
from tensorflow.contrib.layers import xavier_initializer
from roadhouse.models.general import global_avg_pooling, global_max_pooling, global_norm_pooling


def benanne_like_net(init_placeholder, num_classes, return_dict=False):

    layer_dict = {}
    net = conv1d(init_placeholder,
                 filters=256,
                 kernel_size=4,
                 activation=tf.nn.relu)
    layer_dict['conv_1'] = net
    net = max_pooling1d(net,
                        pool_size=4,
                        strides=1)
    layer_dict['max_pool1'] = net
    net = conv1d(net,
                 filters=512,
                 kernel_size=2,
                 activation=tf.nn.relu)
    layer_dict['conv_2'] = net
    net = max_pooling1d(net,
                        pool_size=2,
                        strides=1)
    layer_dict['max_pool2'] = net
    net = conv1d(net,
                 filters=512,
                 kernel_size=2,
                 activation=tf.nn.relu)
    layer_dict['conv_3'] = net
    net = max_pooling1d(net,
                        pool_size=2,
                        strides=1)
    layer_dict['max_pool2'] = net
    net = tf.concat(
        [global_norm_pooling(net), global_max_pooling(net), global_avg_pooling(net)],
        axis=1
    )
    layer_dict['pooled'] = net
    net = dense(net, 1024, activation=tf.nn.relu)
    net = dense(net, 1024, activation=tf.nn.relu)
    net = dense(net, num_classes)
    if return_dict:
        return net, layer_dict
    return net


def benanne_on_rnns(init_placeholder, num_classes, return_dict=False):

    layer_dict = {}
    cell = GRUCell(num_units=512,
                   kernel_initializer=xavier_initializer(),
                   activation=tf.nn.relu)
    net = conv1d(init_placeholder,
                 filters=256,
                 kernel_size=4,
                 activation=tf.nn.relu)
    layer_dict['conv_1'] = net
    net = tf.nn.dynamic_rnn(cell=cell, inputs=net, dtype=tf.float32)[0]
    layer_dict['gru'] = net
    net = max_pooling1d(net,
                        pool_size=4,
                        strides=1)
    layer_dict['pool_1'] = net
    net = conv1d(net,
                 filters=512,
                 kernel_size=2,
                 activation=tf.nn.relu)
    layer_dict['conv_2'] = net
    net = max_pooling1d(net,
                        pool_size=2,
                        strides=1)
    layer_dict['pool_2'] = net
    net = conv1d(net,
                 filters=512,
                 kernel_size=2,
                 activation=tf.nn.relu)
    layer_dict['conv_3'] = net
    net = max_pooling1d(net,
                        pool_size=2,
                        strides=1)
    layer_dict['pool_3'] = net
    net = tf.concat(
        [global_norm_pooling(net), global_max_pooling(net), global_avg_pooling(net)],
        axis=1
    )
    layer_dict['pooled'] = net
    net = dense(net, 1024, activation=tf.nn.relu)
    net = dense(net, 1024, activation=tf.nn.relu)
    net = dense(net, num_classes)
    if return_dict:
        return net, layer_dict

    return net

zoo = {
    'benanne_like_net': benanne_like_net,
    'benanne_on_rnns': benanne_on_rnns
}