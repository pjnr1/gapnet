"""
Spikeogram generator functions
"""
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
    """
    Generates a 2D spikeogram from the .mat-file.

    @param data:
        the open .mat-file object
    @param data_field:
        the datafield to use, e.g. 'an_out_sum'
    @param bin_width:
        binwidth for downsampling, 1e-3 (1 ms)
    @param dtype:
        datatype, default is 'float32'
    @return:
        tensor with 2 dimensions, i.e. [frequency, time]
    @raise ValueError:
        when <h5py.File>.Get() fails and return None
    """
    x = data.get(data_field, default=None)
    if x is None:
        raise ValueError(f'Field {data_field} not found in {data.filename}')
    fs = get_sampling_frequency(data)
    return historize(get_tensor_from_h5py_numpy(x, dtype),
                     bin_width=int(bin_width * fs))


def spikeogram_3D(data: h5py.File,
                  data_fields: List[str] = None,
                  bin_width: float = 1e-3,
                  dtype='float32') -> torch.Tensor:
    """
    Generates a 3D spikeogram from the .mat-file.

    @param data:
        the open .mat-file object
    @param data_fields:
        the datafields to use. Default is ['an_out_hs', 'an_out_ms', 'an_out_ls']
    @param bin_width:
        binwidth for downsampling. Default is 1e-3 (1 ms)
    @param dtype:
        datatype. Default is 'float32'
    @return:
        tensor with 3 dimensions, i.e. [channels, frequency, time]
    @raise ValueError:
        when <h5py.File>.Get() fails and return None
    """
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

    # Preallocate output array
    mat = torch.zeros(3, data_list[0].shape[0], data_list[0].shape[1])
    for i in range(3):
        mat[i, :, :] = data_list[i]
    return historize(mat, bin_width=int(bin_width * fs))
