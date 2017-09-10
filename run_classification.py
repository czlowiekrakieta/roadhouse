import argparse
import os
import pickle
import sys

import tensorflow as tf

from roadhouse.models.classifiers import zoo
from roadhouse.models.general import train_classification, build_loss, build_simple_adam
from roadhouse.utils.utils import fetch_data, Logger
from roadhouse.utils import config as cfg

os.environ['TF_CPP_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--spectro_pickle', default='data')
    parser.add_argument('--data_pts_nr', default=1000, type=int)
    parser.add_argument('--channels', default=2, type=int)
    parser.add_argument('--frames_nr', default=600, type=int)
    parser.add_argument('--crops', default=4, type=int)
    parser.add_argument('--target_label', default='artist_genre')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log', default='true', choices=['true', 'false'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--net_name', required=True)
    parser.add_argument('--steps', default=8, type=int)
    args = parser.parse_args()

    if args.log != 'false':
        if not os.path.exists('logs'):
            os.mkdir('logs')

        sys.stdout = Logger('logs/' + args.name)

    generator, labels, shape, val_data = fetch_data(args)
    num_classes = len(labels)

    x_ph = tf.placeholder('float', [None, args.frames_nr, shape])
    y_ph = tf.placeholder('float', [None, num_classes])

    net = zoo[args.net_name](x_ph, num_classes=num_classes)
    loss_mean, loss_per_class = build_loss(net, y_ph)
    opt = build_simple_adam(loss_mean)

    path = os.path.join(cfg.SPECTROGRAMS_PICKLE_STORAGE,
                        args.spectro_pickle + str(args.data_pts_nr))
    train_classification(opt, loss_mean, data_generator=generator,
                         x_ph=x_ph, y_ph=y_ph, val_data=val_data,
                         mode_name=args.name)
