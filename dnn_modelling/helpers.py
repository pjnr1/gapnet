import functools
import operator
import os
from pathlib import Path

import torch
import torch.nn as nn

from fastcore.foundation import mask2idxs, L


def get_2d_zeropadding(x):
    """
    Helper function to determine the amount of padding for obtaining same-size pooling and conv
    (see modules.py for help)

    Goes through x (e.g. kernel-sizes) backwards

    @param x:
    @return:
    """
    return functools.reduce(operator.__add__,
                            [[k // 2 + (k - 2 * (k // 2)) - 1, k // 2] for k in x[::-1]])


def weight_initialisation(m, func=nn.init.xavier_normal_) -> None:
    """
    Helper function to apply for initialising weights of the networks

    @param m:
        Model
    @param func:
        Function use for initialisation, takes Tensor, returns Tensor of same size
    @return:
    """
    if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
        func(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()


def print_sizes(model, input_tensor):
    """
    Print shape of layer for each layer in Sequential model
    """
    output = input_tensor
    with torch.no_grad():
        was_training = model.training
        if was_training:
            model.eval()
        for name, m in model.named_children():
            output = m(output)
            print('Layer Name: {:15}{}'.format(name + ',', output.shape))
            print(m)
        if was_training:
            model.train()
    return output


def GrandparentRegexpSplitter(train_regexp='train', valid_regexp='valid'):
    """
    Split `items` from the grandparent folder names matching the regexp's (`train_regexp` and `valid_regexp`).

    @param train_regexp:
    @param valid_regexp:
    @return:
        Function for splitting items into train and validation
    """

    def _grandparent_regexp_idxs(items, name):
        def _inner(i, n):
            return mask2idxs(Path(o).parent.parent.name == n for o in i)

        return [i for n in L(name) for i in _inner(items, n)]

    def f(x):
        return _grandparent_regexp_idxs(x, train_regexp), \
               _grandparent_regexp_idxs(x, valid_regexp)

    return f


def extract_from_path(path, key, default=None, split_char='_'):
    extracted = [x for x in path.split(split_char) if key in x]
    if len(extracted) > 0:
        return extracted[-1][len(key):]
    return default


def extract_from_meta(fp, keys):
    """
    Loads info from meta file with path _fp_ with the keys and transformers in _keys_

    @param fp:
        path to the meta file
    @param keys:
        a dict(), key is the line-identifier, value is functions to convert rest of the line to the desired value
        if value is a list, the list will be applied to the line

    @return:
    """
    output = dict()
    with open(fp, 'r') as f:
        for line in f:
            for key, transform in keys.items():
                if key not in line:
                    continue

                # Grab 'value' part of line
                line = [x for x in line.split(' ') if x != ''][1]
                if transform is None:
                    output[key] = line
                else:
                    if isinstance(transform, list):
                        result = list()
                        for tf in transform:
                            result.append(tf(line))
                        output[key] = result
                    else:
                        output[key] = transform(line)

    if len(keys) > 0:
        print(f'didn\'t extract following: {[x for x in keys.keys() if x not in output.keys()]}')
    return output


def get_datapath(folder, mode, bw):
    return os.path.join(folder,
                        'data',
                        'cnn_data',
                        f'mode_{mode}',
                        f'bw_{bw}')


def load_history(model_output_folder):
    history_path = os.path.join(model_output_folder, f'history.csv')
    return pd.read_csv(history_path)
