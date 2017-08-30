from requests import HTTPError
import requests
import pickle
import numpy as np
from glob import glob
from random import sample
from pydub import AudioSegment
import os
from functools import partial, reduce
from pydub.exceptions import CouldntDecodeError

from roadhouse.utils import config as cfg
from roadhouse.utils.spectrogram import build_spectrograms

from time import sleep
import os.path as op
import json


class Container(object):
    def __init__(self,
                 number_cap=None,
                 read_on_init=False,
                 target_label='artist',
                 truncate_classes=True):

        self.artist_to_genre = read_json('artist_genres.json')
        self.album_to_genre = read_json('album_genres.json')
        self.artist_id_map = read_json('id_to_artist.json')
        self.album_id_map = read_json('id_to_album.json')

        self.bag = []
        self.number_cap = number_cap
        self.target_label = target_label
        self.truncate_classes = truncate_classes and target_label.endswith('genre')

        if target_label not in ['artist', 'artist_genre', 'album_genre', 'album']:
            raise ValueError

        if read_on_init:
            self._read_files_and_get_data()

    def _mapper(self, song_id):

        artist = self.artist_id_map[song_id]
        album = self.album_id_map[song_id]

        genre_artist = self.artist_to_genre[artist]
        genre_album = self.album_to_genre[album]

        return artist, album, genre_artist, genre_album

    def _read_files_and_get_data(self):
        songs = glob(op.join(cfg.DUMP_FOLDER, 'part_*', '*'))
        if self.number_cap is not None and len(songs) > self.number_cap:
            songs = sample(songs, self.number_cap)
        BuildSong = partial(Song, mapper=self._mapper)
        print('beginning loading')
        for i, song in enumerate(songs, 1):
            try:
                self.bag.append(BuildSong(song))
                if i % cfg.BUILDER_DISPLAY == 0:
                    print('%s songs so far' % i)
            except CouldntDecodeError:
                print('song nr %s failed to decode' % i)

    def build_trainable_dataset(self):

        labels_map = {}
        X, y = [], []

        def assign_number(label):
            if label not in labels_map:
                curr = len(labels_map)
                labels_map[label] = curr
                return curr
            else:
                return labels_map[label]

        for song in self.bag:

            array = song.ampls

            label = song.__getattribute__(self.target_label)
            labels = []

            if isinstance(label, list):
                label = [label]

            if self.truncate_classes:
                label = set(reduce(lambda x, y: x+y,
                                 [[x for x in cfg.GENRES if x in y]
                                  for y in label]))

            for x in label:
                lab = assign_number(x)
                labels.append(lab)


            X.append(array)
            y.append(labels)

        return X, y, labels_map


class Song(object):

    def __init__(self, filename, mapper):

        song = AudioSegment.from_mp3(file=filename)
        ampls = song.get_array_of_samples()
        ampls = np.array(ampls).reshape(-1, 2)

        self.ampls = ampls

        song_id = filename.split('/')[-1].split('.')[0]

        self.artist, self.album, self.artist_genre, self.album_genre = mapper(song_id)
        self.id = song_id

    def __hash__(self):
        return self.id


def dump_preview(url, filename):
    resp = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        f.write(resp.content)


def sleep_when_hit_boundary(func):
    def retry(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except HTTPError:
                print('too many requests, sleeping, ', cfg.WAIT_TIME)
                sleep(cfg.WAIT_TIME)

    return retry


def read_json(fname):
    fname = op.join(cfg.MAPPINGS, fname)
    with open(fname, 'r') as f:
        x = json.load(f)
    return x


def data_generator(path, batch_size):
    for file in glob(os.path.join(path, 'data_*.pkl')):
        with open(file, 'rb') as f:
            X, y, label_map = pickle.load(f)

        offset = 0
        while offset < X.shape[0]:
            offset += batch_size
            yield X[offset - batch_size:offset], y[offset - batch_size:offset]


def fetch_data(args):

    path = os.path.join(cfg.SPECTROGRAMS_PICKLE_STORAGE,
                        args.spectro_pickle + str(args.data_pts_nr))
    if os.path.exists(path) and len(glob(os.path.join(path, 'data_*.pkl'))):
        print('loading pre-created data')
        with open(os.path.join(path, 'meta.json'), 'r') as f:
            meta = json.load(f)

        return partial(data_generator, path=path, batch_size=args.batch_size), \
               meta['label_map'], meta['shape']

    else:
        print('creating data')
        if not os.path.exists(path):
            os.makedirs(path)

        cont = Container(number_cap=args.data_pts_nr,
                         read_on_init=True,
                         target_label=args.target_label)
        X, y, label_map = cont.build_trainable_dataset()
        sh = build_spectrograms(X[0],
                                channels=args.channels,
                                crops_per_song=args.crops,
                                frames_nr=args.frames_nr).shape[-1] # can't calculate it by hand yet
        multiplier = (1 if args.channels < 2 else 2)*args.crops


        offset = 0
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump({'label_map': label_map, 'shape': sh}, f)

        while offset < len(X):

            x_spectr = np.random.randn(args.max_pickle_size * multiplier,
                                       args.frames_nr,
                                       sh)
            y_spectr = []

            for i in range(args.max_pickle_size):
                x_spectr[i * multiplier:(i + 1) * multiplier] = build_spectrograms(X[offset+i],
                                                                                   crops_per_song=args.crops,
                                                                                   channels=args.channels,
                                                                                   frames_nr=args.frames_nr)
                y_spectr.extend(y[i] for i in range(multiplier))

            y_spectr = np.asarray(y_spectr)
            perm = np.random.permutation(x_spectr.shape[0])
            x_spectr = x_spectr[perm]
            y_spectr = y_spectr[perm]

            with open(os.path.join(path, 'data_{}.pkl'.format(offset)), 'wb') as f:
                pickle.dump((x_spectr, y_spectr, label_map), f)

            offset += args.max_pickle_size


        return partial(data_generator, path=path, batch_size=args.batch_size), label_map, sh
