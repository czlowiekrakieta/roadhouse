import argparse
import os

import tensorflow as tf
import numpy as np
from pydub import AudioSegment
from scipy.signal import istft

from roadhouse.models.transferer import run_transfer
from roadhouse.utils.spectrogram import build_spectrograms
from roadhouse.utils.utils import save_song


def fetch_songs(content_path, style_path, frames=600):

    content = AudioSegment.from_mp3(content_path).get_array_of_samples()
    style = AudioSegment.from_mp3(style_path).get_array_of_samples()

    content = np.asarray(content).reshape(-1, 2)
    style = np.asarray(style).reshape(-1, 2)

    spectro_content = build_spectrograms(content,
                                         channels=0,
                                         crops_per_song=1,
                                         frames_nr=frames)
    spectro_style = build_spectrograms(style,
                                       channels=0,
                                       crops_per_song=1,
                                       frames_nr=frames)

    return spectro_content, spectro_style


def run(args):
    spectro_content, spectro_style = fetch_songs(args.content_path, args.style_path, args.frames)
    print('spectrograms fetched')
    song = run_transfer(args.net_name, args.weights_filename, spectro_content, spectro_style,
                        args.style_mult, args.content_mult, args.iters, args.frames)
    print('got new song baked')
    inverse_content = istft(spectro_content)[1]
    inverse_style = istft(spectro_style)[1]

    inverse_song = istft(song)[1]

    if not os.path.exists('songs'):
        os.mkdir('songs')

    for arr, name in zip([inverse_content, inverse_style, inverse_song],
                         ['content', 'style', 'song']):

        save_song(arr, 'songs/{}_{}.wav'.format(name, args.name))


def main():
    import argparse


    parser = argparse.ArgumentParser()

    def add_arg(name, default=None, type=str, req=False, parser=parser):
        parser.add_argument('--' + name, default=default, type=type, required=req)

    add_arg('iters', 1000, int)
    add_arg('net_name', 'benanne_like_net', str)
    add_arg('style_mult', 5, float)
    add_arg('content_mult', 4., float)
    add_arg('content_path', req=True)
    add_arg('style_path', req=True)
    add_arg('weights_filename', req=True)
    add_arg('frames', 600, int)
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()