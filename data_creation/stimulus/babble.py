import glob
import os
from typing import List

import numpy as np

from scipy.io.wavfile import read
from scipy.signal import resample_poly


def get_n_sound_files(data_folder: str, n_talkers: int) -> List[str]:
    """

    :param data_folder:
    :param n_talkers:
    :return:
    @raises ValueError:
        When n_talkers is larger than the count of files in data_folder
    """
    wav_files = glob.glob(os.path.join(data_folder, '*.wav'))

    if n_talkers > len(wav_files):
        raise ValueError(f'desired number of talkers ({n_talkers}) exceed sound files found in {data_folder}')

    return [wav_files[x] for x in np.random.permutation(len(wav_files))[:n_talkers]]


def generate_babble(data_folder: str, n_talkers: int, fs: int = None) -> (int, np.ndarray):
    audio_files = get_n_sound_files(data_folder=data_folder, n_talkers=n_talkers)

    return combine_audio_files(audio_files=audio_files, fs=fs)


def combine_audio_files(audio_files: List[str], fs: int) -> (int, np.ndarray):
    """
    Combines the audio-files and resamples to a desired sampling frequency. If fs is None, then the sampling frequency
    will be of the first loaded audio-file.

    @arg audio_files:
        list of audio-files to combine
    @arg fs:
        sampling frequency, if None, sampling frequency will be the same as the first audio-file in audio_files
    @return:
        the sampling frequency and the combined sound in a numpy array normalised by the max-value of the array
    """
    x = None  # Output audio
    fs_loaded = None
    for fn in audio_files:
        fs_loaded, data = read(fn)
        if fs is None:
            fs = fs_loaded

        # Resample, if sampling frequencies of audio file are different from target
        if fs_loaded != fs:
            data = resample_poly(data, fs, fs_loaded)

        if x is None:  # If nothing has been loaded, re-allocate empty vector to fill
            x = np.zeros((data.shape[0],))

        if x.shape[0] > data.shape[0]:  # Use the size of the shortest audio-clip
            x = x[:data.shape[0]]

        for i in range(data.shape[1]):  # Mix in all channels of the audio (one if mono, two if stereo, etc.)
            x += data[:x.shape[0], i]

    if x is None:
        raise ValueError(f'No audio loaded from audio_files: {audio_files} (x is None)')

    if fs_loaded is None:
        raise ValueError(f'No audio loaded from audio_files: {audio_files} (fs_loaded is None)')

    # Normalise
    x /= x.max(initial=0.0)

    return fs, x
