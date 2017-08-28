from requests import HTTPError
import requests
import roadhouse.utils.config as cfg
from time import sleep
import os.path as op
import json


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


def parse_architecture(fname, type):
    """

    :param type: 'generator', 'classifier'
    :return:
    """
    pass

def _parse_generator(fname):
    pass

def _parse_classfier(fname):
    pass