from typing import List

import h5py

import torch

from .tools import get_sampling_frequency
from .tools import historize
from .tools import get_tensor_from_h5py_numpy


def spikeogram_2D(data: h5py.File,
                  data_field: str = 'an_out_sum',
                  bin_width: float = 1e-3,
                  dtype='float32') -> torch.Tensor:
    x = data.get(data_field, default=None)
    if x is None:
        raise ValueError(f'Field {data_field} not found in {data.filename}')
    fs = get_sampling_frequency(data)
    return historize(get_tensor_from_h5py_numpy(x, dtype),
                     bin_width=int(bin_width * fs))


def spikeogram_3D(data: h5py.File,
                  data_fields: List[str] = None,
                  bin_width: float = 1e-3,
                  dtype='float32'):
    if data_fields is None:
        data_fields = ['an_out_hs', 'an_out_ms', 'an_out_ls']
    data_list = list()
    for data_field in data_fields:
        x = data.get(data_field, default=None)
        if x is None:
            raise ValueError(f'Field {data_field} not found in {data.filename}')
        else:
            data_list.append(get_tensor_from_h5py_numpy(x, dtype))
    fs = get_sampling_frequency(data)

    mat = torch.zeros(3, data_list[0].shape[0], data_list[0].shape[1])
    for i in range(3):
        mat[i, :, :] = data_list[i]
    return historize(mat, bin_width=int(bin_width * fs))
