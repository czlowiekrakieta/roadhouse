import argparse
import os
import pickle

import tensorflow as tf

from roadhouse.models.classifiers import benanne_like_net
from roadhouse.models.general import train_classification, build_loss, build_simple_adam
from roadhouse.utils.utils import fetch_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--spectro_pickle', default='data')
    parser.add_argument('--data_pts_nr', default=1000, type=int)
    parser.add_argument('--channels', default=2, type=int)
    parser.add_argument('--frames_nr', default=600, type=int)
    parser.add_argument('--crops', default=4, type=int)
    parser.add_argument('--target_label', default='artist_genre')
    parser.add_argument('--max_pickle_size', default=1000, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    generator, labels, shape = fetch_data(args)
    num_classes = len(labels)

    x_ph = tf.placeholder('float', [None, args.frames_nr, shape])
    y_ph = tf.placeholder('float', [None, num_classes])

    net = benanne_like_net(x_ph, num_classes=num_classes)
    loss_mean, loss_per_class = build_loss(net, y_ph)
    opt = build_simple_adam(loss_mean)

    train_classification(opt, loss_mean, data_generator=generator, x_ph=x_ph, y_ph=y_ph)