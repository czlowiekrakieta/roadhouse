from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from roadhouse.utils.spectrogram import stft, logscale_spec
from roadhouse.utils.loaders import Container
from glob import glob


def compile_classifier(model, opt):
    model.compile(optimizer=opt, loss=binary_crossentropy)
    return model


def run_model():
    