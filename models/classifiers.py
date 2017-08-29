import tensorflow as tf
from tensorflow.python.layers.convolutional import conv1d
from tensorflow.python.layers.pooling import max_pooling1d, average_pooling1d
from tensorflow.python.layers.core import dense
from roadhouse.models.general import global_avg_pooling, global_max_pooling, global_norm_pooling


def benanne_like_net(init_placeholder, num_classes):

    net = conv1d(init_placeholder,
                 filters=256,
                 kernel_size=4,
                 activation=tf.nn.relu)
    net = max_pooling1d(net,
                        pool_size=4,
                        strides=1)
    net = conv1d(net,
                 filters=512,
                 kernel_size=2,
                 activation=tf.nn.relu)
    net = max_pooling1d(net,
                        pool_size=2,
                        strides=1)
    net = conv1d(net,
                 filters=512,
                 kernel_size=2,
                 activation=tf.nn.relu)
    net = max_pooling1d(net,
                        pool_size=2,
                        strides=1)
    net = tf.concat(
        [global_norm_pooling(net), global_max_pooling(net), global_avg_pooling(net)],
        axis=1
    )
    net = dense(net, 1024, activation=tf.nn.relu)
    net = dense(net, 1024, activation=tf.nn.relu)
    net = dense(net, num_classes, activation=tf.nn.sigmoid)

    return net