from ..utils import config as cfg
import argparse
from requests import HTTPError
import pickle
import requests
from time import sleep
import os.path as op
from spotify import OAuth, Client
from roadhouse.utils.utils import sleep_when_hit_boundary

auth = OAuth(client_id=cfg.CLIENT_ID, client_secret=cfg.CLIENT_SECRET)
auth.request_client_credentials()

client = Client(auth)


class Downloader:
    def __init__(self):
        self.artists_in_memory = set()
        self.albums_in_memory = set()
        self.songs_in_memory = set()

        self.artists_mapping = {}
        self.album_mapping = {}
        self.song_mapping = {}

        self.last_saved_at = 0

        self.album_genres = {}
        self.artist_genres = {}

        self.list_of_songs = []

    def fetch(self):
        playlists = Downloader._fetch_songs_by_categories()

        for category, p_lists in playlists:

            print('fetching: ', category)

            for p_l in p_lists['playlists']['items']:
                print(p_l['name'])
                self.fetch_single_playlist(p_l['href'])

    @staticmethod
    def dump_preview(url, filename):
        resp = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            f.write(filename)

    @staticmethod
    def _fetch_songs_by_categories():

        categories = client.api.browse_categories()['categories']['items']
        categories_ids = [(x['id'], x['name']) for x in categories]
        return [(x[1], client.api.browse_category_playlists(x[0])) for x in categories_ids]

    def fetch_single_playlist(self, playlist_url, fetch_artist=True, fetch_album=True):

        playlist_data = client.api.request('get', playlist_url)

        def fetch_single_pass():

            data = playlist_data['tracks']['items']
            for datum in data:
                track = datum['track']
                artists = track['artists']
                album = track['album']

                artists = [(x['name'], x['id']) for x in artists]

                if fetch_artist:
                    for name, id in artists:
                        if id not in self.artists_mapping:
                            self.artists_mapping[id] = name

                        if id not in self.artists_in_memory:
                            artist_data = self.fetch_single_artist(id, name)
                            self.artists_in_memory.add(id)
                            artist_genre = artist_data['genres']
                            self.artist_genres[name] = artist_genre
                            albums = artist_data['albums']

                            for id, album_data in albums:

                                if id not in self.albums_in_memory:
                                    self.albums_in_memory.add(id)
                                    self.album_mapping[id] = album_data[0]['album']
                                    self.list_of_songs += album_data

                if fetch_album and album['id'] not in self.albums_in_memory:
                    album_data = self.fetch_single_album(album['id'], list(zip(*artists))[0], album['name'])
                    self.albums_in_memory.add(album['id'])
                    self.list_of_songs += album_data

                if len(self.list_of_songs) - self.last_saved_at > cfg.SAVES_DIFF:
                    self.last_saved_at = len(self.list_of_songs)
                    with open(cfg.FILENAME, 'wb') as f:
                        pickle.dump(self, f)

        while playlist_data['tracks']['next'] is not None:
            fetch_single_pass()

            playlist_data = client.api.request('get', playlist_data['next'])

        fetch_single_pass()

    @sleep_when_hit_boundary
    def fetch_single_artist(self, artist_id, artist_name):

        print('fetching,', artist_name)
        print('so far gathered references to', len(self.list_of_songs), 'songs')
        artist_albums = client.api.artist_albums(artist_id)['items']
        albs = [(x['id'], self.fetch_single_album(x['id'],
                                                  artist_name=artist_name,
                                                  album_name=x['name'])) for x in artist_albums]
        artist_data = client.api.artist(artist_id)
        return {'genres': artist_data['genres'], 'albums': albs}

    @sleep_when_hit_boundary
    def fetch_single_album(self, album_id, artist_name, album_name):

        tracks = client.api.album_tracks(album_id)
        genres = client.api.album(album_id)['genres']
        self.album_genres[album_name] = genres

        return [{'name': x['name'],
                 'id': x['id'],
                 'preview_url': x['preview_url'],
                 'album': album_name,
                 'artists': artist_name} for x in tracks['items']]

# if not op.exists(cfg.FILENAME):
if __name__ == '__main__':
    downloader = Downloader()
    downloader.fetch()
