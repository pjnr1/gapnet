from functools import partial
from math import floor

import torch
import torch.nn
import torch.nn.functional as F

from fastai.data.transforms import get_files
from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import error_rate
from fastcore.foundation import L

from .datasets import get_gaplength_from_filename


get_spikeograms = partial(get_files, extensions='.pt')


def get_gap_nogap_vocab():
    return ['gap', 'no gap']


def label_func__gap():
    def _func(fn):
        if get_gaplength_from_filename(fn.name.lower()) == 0.0:
            return [1.]
        else:
            return [0.]

    return _func, partial(RegressionBlock, n_out=1)


def label_func__gap_nogap():
    def _func(fn):
        if get_gaplength_from_filename(fn.name.lower()) == 0.0:
            return [get_gap_nogap_vocab()[1]]
        else:
            return [get_gap_nogap_vocab()[0]]

    return _func, GapNoGapCategoryBlock


def label_func__gap_length():
    def _func(fn):
        return [get_gaplength_from_filename(fn.name.lower())]

    return _func, partial(RegressionBlock, n_out=1)


def decode_model_output(x, vocab):
    y = torch.zeros(x.shape)
    y[torch.argmax(x)] = 1.0
    return EncodedMultiCategorize(vocab=vocab).decodes(y), x[torch.argmax(x)]


def Tensor3Dload(*args):
    return torch.unsqueeze(torch.load(*args), 0)


def TensorBlock3D():
    return TransformBlock(type_tfms=Tensor3Dload)


def TensorZeroPadLoad(*args, pad=(0, 0), mode='constant', value=0.0):
    X = F.pad(torch.load(*args),
              pad=pad,
              mode=mode,
              value=value)
    if len(X.shape) == 2:
        X = torch.unsqueeze(X, dim=0)
    return X


def TensorBlock(pad=None, item_tfms=None):
    if pad is None: pad = (0, 0)
    return TransformBlock(type_tfms=partial(TensorZeroPadLoad,
                                            pad=pad),
                          item_tfms=item_tfms)


class TensorTemporalSubsamplingTransform(DisplayedTransform):
    def __init__(self, subsampling_percent=0.5, desired_length=700):
        super().__init__()
        self.subsampling_percent = subsampling_percent
        self.desired_length = desired_length

    def encodes(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        if not len(x.shape) > 1:
            return x
        max_shift = floor((x.shape[-1] - self.desired_length) * self.subsampling_percent)
        if max_shift <= 0:
            return x

        cut_index = random.randint(0, max_shift)
        return x[..., cut_index:(cut_index + self.desired_length)]


def GapNoGapCategoryBlock(*args, **kwargs):
    return partial(MultiCategoryBlock, vocab=get_gap_nogap_vocab())(*args, **kwargs)


def get_datablock(valid_pct=0.1, splitter=None, x_block=None, label_func=None):
    if x_block is None:
        x_block = TensorBlock
    if splitter is None:
        splitter = RandomSplitter(valid_pct=valid_pct)
    if label_func is None:
        label_func = label_func__gap_nogap

    # Note: label_func is a function that returns a labelling function and a datablock class
    _label_func, _label_block = label_func()

    return DataBlock(blocks=(x_block, _label_block),
                     get_items=get_spikeograms,
                     get_y=_label_func,
                     splitter=splitter)


def get_subsampling_datablock(valid_pct=0.1,
                              subsample_pct=0.0,
                              desired_length=700,
                              splitter=None,
                              x_block=None,
                              label_func=None):
    return get_datablock(valid_pct=valid_pct,
                         splitter=splitter,
                         x_block=partial(TensorBlock,
                                         item_tfms=TensorTemporalSubsamplingTransform(
                                             subsampling_percent=subsample_pct,
                                             desired_length=desired_length)),
                         label_func=label_func)
