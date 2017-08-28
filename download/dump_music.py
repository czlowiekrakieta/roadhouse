from roadhouse.utils import config as cfg
from roadhouse.utils.utils import dump_preview
import pickle
from requests import HTTPError
from os.path import join, exists
from os import mkdir, makedirs
from time import sleep, time
from roadhouse.download.downloader import Downloader

with open(cfg.FILENAME, 'rb') as f:
    database = pickle.load(f)


def fetch(args):

    times = []
    curr_folder = join(cfg.DUMP_FOLDER, 'part_0')
    if not exists(curr_folder):
        makedirs(curr_folder)

    with open(join(curr_folder, 'list_of_files.txt'), 'r') as f:

        already_fetched = set(f.read().split('\n'))

    f = open(join(curr_folder, 'list_of_files.txt'), 'a')
    for i, record in enumerate(database.list_of_songs[args.start:args.end], args.start):

        if record['id'] in already_fetched:
            continue

        if i % args.split == 1:
            curr_folder = join(cfg.DUMP_FOLDER, 'part_%s' % (i // args.split))
            if not exists(curr_folder):
                mkdir(curr_folder)
            f.close()
            list_file = join(curr_folder, 'list_of_files.txt')
            if exists(list_file):
                with open(list_file, 'r') as f:

                    already_fetched = set(f.read().split('\n'))
            f = open(join(curr_folder, 'list_of_files.txt'), 'a')

        if record['preview_url'] is None:
            continue

        while True:
            try:
                t = time()
                dump_preview(record['preview_url'], join(curr_folder, record['id'] + '.mp3'))
                t = time() - t
                f.write(record['id'] + '\n')
                already_fetched.add(record['id'])
                break
            except HTTPError:
                sleep(cfg.WAIT_TIME)
                print('slept 2 seconds due to http error')

        times.append(t)

        if i % cfg.DOWNLOAD_DISPLAY_PROGRESS == 0:
            print('downloaded %s files so far, approximately %s seconds left' % (i, (args.end - args.start) * (sum(times)/i)))

    f.close()
if __name__ == '__main__':
    import argparse

    if not exists(cfg.DUMP_FOLDER):
        makedirs(cfg.DUMP_FOLDER)

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=len(database.list_of_songs))
    parser.add_argument('--split', type=int, default=10000)

    fetch(parser.parse_args())