"""
Various tools for interfacing with the AN-simulations and more
"""
import h5py

import numpy as np

import torch
import torch.nn.functional as F


def as_torch_type(type_string: str) -> torch.dtype:
    """
    Get torch datatype from string

    @arg type_string: string version of a datatype, e.g. 'int32'
    @return: torch.dtype
    """
    type_map = {
        'int': torch.int,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'double': torch.double,
        'float': torch.float,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64
    }
    if type_string not in type_map.keys():
        raise ValueError(f'Type-string "{type_string}" not found. Supported types are {type_map.keys()}')

    return type_map[type_string]


def open_an_simulation_data(fp, mode='r') -> h5py.File:
    return h5py.File(fp, mode)


def get_tensor_from_h5py_numpy(x, dtype) -> torch.Tensor:
    """
    Get torch.Tensor from h5py data array

    @arg x:
        input data
    @arg dtype:
        target datatype
    @return:
    """
    return torch.from_numpy(np.array(x).astype(dtype)).T


def get_sampling_frequency(mat) -> int:
    """

    @arg mat: AN-simulation structure (read from h5py)
    @return:
    """
    return mat['params']['fs_model'][0]


def get_freq_vect(mat):
    """

    @arg mat: AN-simulation structure (read from h5py)
    @return:
    """
    return mat['params']['freq_vect']


def historize(data: torch.Tensor, bin_width: int = 1) -> torch.Tensor:
    """
    Takes a 2D or 3D tensor and bins along the time axis.
    If the tensor is 2D, the dimensions should be [frequency, time]
    If the tensor is 3D, the dimensions should be [channels, frequency, time]

    @arg data:
        input data
    @arg bin_width:
        bin width in samples

    @return:
        torch.Tensor
    """
    channels = 1
    if len(data.shape) == 3:
        channels = data.shape[0]
        data = data[None, :, :, :]  # Extend by one dimension
    else:
        data = data[None, None, :, :]  # Extend by two dimensions

    # Create a kernel for using convolution to bin 'spikes'
    kernel = torch.ones(channels, 1, 1, bin_width, dtype=data.dtype)
    return torch.squeeze(F.conv2d(input=data,
                                  weight=kernel,
                                  groups=channels,
                                  stride=(1, bin_width)))
