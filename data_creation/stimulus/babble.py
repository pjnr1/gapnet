from typing import List

import numpy as np

import scipy.io.wavfile
import scipy.signal


def combine_audio_files(audio_files: List[str], fs: int) -> np.ndarray:
    x = None  # Output audio
    fs_loaded = None
    for fn in audio_files:
        fs_loaded, data = scipy.io.wavfile.read(fn)

        # Resample, if sampling frequencies of audio file are different from target
        if fs_loaded != fs:
            a = scipy.signal.resample_poly(data, fs, fs_loaded)

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

    return x
