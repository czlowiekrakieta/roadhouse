from numpy.lib import stride_tricks
import numpy as np

"""

direct ripoff from 
http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html?i=1

with some minor typecasting from float to int where necessary

someday I will figure this out ( hopefully by the end of month )

"""

# TODO: inverse spectrogram

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize / 2.0).astype(int)), sig)
    # cols for windowing
    cols = (np.ceil((len(samples) - frameSize) / float(hopSize)) + 1).astype(int)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))
    scale = scale.astype(int)

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i + 1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i + 1]])]

    return newspec, freqs


def build_spectrograms(song_array,
                       crops_per_song=4,
                       channels=0,
                       binsize=2 ** 10,
                       frames_nr=600):
    def _build_single_spectr(signal):
        s = stft(signal, frameSize=binsize)
        sshow, freq = logscale_spec(s, factor=1.0, sr=44100)
        ims = 20. * np.log10(np.abs(sshow) / 10e-6)
        return ims

    def _get_crops(freq):

        low, high = 0, freq.shape[0] - frames_nr
        if low >= high:
            return None
        st = np.random.randint(low=low, high=high, size=crops_per_song).flatten().tolist()
        return np.stack([freq[st[i]:st[i] + frames_nr] for i in range(crops_per_song)], axis=0)/300.

    if channels == 2:

        channel_one = _build_single_spectr(song_array[:, 0])
        channel_two = _build_single_spectr(song_array[:, 1])
        channel_one = _get_crops(channel_one)
        channel_two = _get_crops(channel_two)
        not_nones = [channel_two is not None, channel_one is not None]
        if all(not_nones):
            return np.concatenate((channel_one, channel_two), axis=0)
        elif any(not_nones):
            return [channel_two, channel_one][not_nones.index(True)]
        else:
            return None

    else:

        return _get_crops(
            _build_single_spectr(song_array[:, channels])
        )