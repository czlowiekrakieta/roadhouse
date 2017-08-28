from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from roadhouse.utils.spectrogram import stft, logscale_spec
from glob import glob


def compile_model(model, opt):
    model.compile(optimizer=opt, loss=binary_crossentropy)
    return model
