from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import numpy as np
from glob import glob
import json
from roadhouse.utils import config as cfg
from roadhouse.utils.utils import read_json
import os.path as op
from random import sample
from functools import partial


class Container(object):
    def __init__(self, number_cap=None, read_on_init=False):

        self.artist_to_genre = read_json('artist_genres.json')
        self.album_to_genre = read_json('album_genres.json')
        self.artist_id_map = read_json('id_to_artist.json')
        self.album_id_map = read_json('id_to_album.json')

        self.bag = []
        self.number_cap = number_cap

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

        for i, song in enumerate(songs, 1):
            try:
                self.bag.append(BuildSong(song))
                if i % cfg.BUILDER_DISPLAY == 0:
                    print('%s songs so far' % i)
            except CouldntDecodeError:
                print('song nr %s failed to decode' % i)

    def build_trainable_dataset(self, target_label='artist'):

        labels_map = {}
        X, y = [], []

        try:
            self.bag[0].__getattribute__(target_label)
        except AttributeError:
            print('possible attributes are: "artist", '
                  '"artist_genre", "album", "album_genre"')

        def assign_number(label):
            if label not in labels_map:
                curr = len(labels_map)
                labels_map[label] = curr
                return curr
            else:
                return labels_map[label]

        for song in self.bag:

            array = song.ampls

            label = song.__getattribute__(target_label)
            labels = []

            if isinstance(label, list):

                for x in label:
                    lab = assign_number(x)
                    labels.append(lab)
            else:
                labels = [assign_number(label)]

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
