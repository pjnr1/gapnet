import glob
import os
import re
from pathlib import Path

import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    if isinstance(text, Path):
        return [atoi(c) for c in re.split(r'(\d+)', text.name.replace('p', '.'))]
    return [atoi(c) for c in re.split(r'(\d+)', text.replace('p', '.'))]


def get_gaplength_from_filename(x: str):
    if 'nogap' in x: return 0.0
    return float(re.split(r'(\d+\.\d+ms)', x.replace('p', '.'))[1].replace('ms', ''))


def get_gapposition_from_filename(x: str):
    if 'nogap' in x: return np.nan
    m = re.split(r'(\d+\.\d+ms)', x.replace('p', '.'))
    if len(m) > 3:
        return float(m[3].replace('ms', ''))
    else:
        return np.nan


def get_spikeogram_files(path):
    return glob.glob(os.path.join(path, '*.pt'))


def get_spikeogram_gap_nogap(path):
    files = get_spikeogram_files(path)

    gap = [x for x in files if get_gaplength_from_filename(x) != 0.0]
    no_gap = [x for x in files if get_gaplength_from_filename(x) == 0.0]

    gap.sort(key=natural_keys)
    no_gap.sort(key=natural_keys)

    return gap, no_gap


def get_gap_closest_to_duration(file_list, duration_in_ms):
    x = np.array([get_gaplength_from_filename(x) for x in file_list])

    x = np.abs(x - duration_in_ms)
    i = np.argmin(x)

    return i


def get_file_and_gaplength(file_list, idx):
    return file_list[idx], get_gaplength_from_filename(file_list[idx])
