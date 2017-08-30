import os

CLIENT_SECRET = os.environ['SPOTIFY_CLIENT_ID']
CLIENT_ID = os.environ['SPOTIFY_CLIENT_SECRET'] # because fuck scrapers
WAIT_TIME = 2
FILENAME = 'downloader_object.pkl'
SAVES_DIFF = 100
DOWNLOAD_DISPLAY_PROGRESS = 1000
DUMP_FOLDER = '/media/lukasz/ML_DATA/remixer_music'
MAPPINGS = '/home/lukasz/Documents/roadhouse/mappings'
BUILDER_DISPLAY = 50
SPECTROGRAMS_PICKLE_STORAGE = '/media/lukasz/ML_DATA/roadhouse_spectrograms'
TRAINING_DISPLAY = 10
EPS = 1e-4
GENRES = [
    'rock', 'pop', 'electro', 'latin', 'funk', 'folk',
    'christian', 'hip hop', 'rap', 'techno', 'house', 'dubstep',
    'punk', 'country', 'dance', 'metal', 'r&b', 'blues', 'soul' ,
    'african', 'rave', 'disco', 'piano', 'reggae', 'stoner',
    'trance', 'tropical', 'soft', 'grunge'
]