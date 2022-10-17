import os
from typing import Any
from functools import partial

from dnn_modelling.models.saddler_et_al_2020 import get_model as get_saddler_2020_model
from dnn_modelling.models.haro_2019 import get_model as get_haro_2019_model
from dnn_modelling.models.kell_et_al_2018 import KellEtAl2018_smaller
from dnn_modelling.models.lindahl_2022 import Lindahl_model_1
from dnn_modelling.models.lindahl_2022 import Lindahl_model_2
from dnn_modelling.models.lindahl_2022 import Lindahl_model_3
from dnn_modelling.models.lindahl_2022 import Lindahl_model_3b
from dnn_modelling.models.lindahl_2022 import Lindahl_model_3c
from dnn_modelling.models.lindahl_2022 import Lindahl_model_4
from dnn_modelling.models.lindahl_2022 import Lindahl_model_5
from dnn_modelling.models.lindahl_2022 import Lindahl_model_6
from dnn_modelling.models.moore_and_glasberg_1996 import MooreAndGlasberg1996
from dnn_modelling.models.neural_metric__lindahl_2022 import NeuralMetric

from dnn_modelling.helpers import extract_from_path, extract_from_meta

import torch

models_with_channel_factor_dict = {
    'kell': KellEtAl2018_smaller,
    'lindahl1': Lindahl_model_1,
    'lindahl2': Lindahl_model_2,
    'lindahl3': Lindahl_model_3,
    'lindahlB3': Lindahl_model_3b,
    'lindahlC3': Lindahl_model_3c,
    'lindahl4': Lindahl_model_4,
    'lindahl5': Lindahl_model_5,
    'lindahl6': Lindahl_model_6,
}

custom_models_dict = {
    'mooreglasberg1996': MooreAndGlasberg1996,
    'neuralmetric': NeuralMetric,
    'snell_neural': None,  # TODO
    'identifier': None,  # Replace None with model class
    'saddler': get_saddler_2020_model,
    'haro': get_haro_2019_model,
}


def get_model(model_string, **kwargs) -> (Any, bool):
    """
    Get model class from identifier string:


    Saddler et al 2020 dnn_modelling:
        model_string = "saddler-???" where ??? should match one of the following:
            TODO

    Kell et al 2018 dnn_modelling:
        model_string = "kell"
            returns a model with the same architecture but with no input split and number of output classes
        model_string = "kell-???"
            returns same as above but with the number of channels reduced as $n // ??$, ??=1 makes no changes
        model_string = "kell-1?-2?"
            same as above for 1?, where 2? sets the size of the hidden fully connected layer


    @param model_string:
        model identifier string
    @param kwargs:


    @return:
        [0] the actual model as a class
        [1] boolean flag, indicating whether the model is based on PyTorch

    """

    def _get_model_with_channel_factor_hl_size(mc, model_basename, ms, **kwargs):
        if ms == model_basename:
            return mc(**kwargs,
                      channel_factor=1,
                      hidden_layer_size=1024)
        elif ms[:len(model_basename)] == model_basename:
            parameters = model_string.split('-')[1:]
            return mc(**kwargs,
                      channel_factor=int(parameters[0]),
                      hidden_layer_size=int(parameters[1]) if len(parameters) > 1 else 1024)
        return None

    for key, model_class in models_with_channel_factor_dict.items():
        model = _get_model_with_channel_factor_hl_size(model_class, key, model_string, **kwargs)
        if model is not None:
            return model, True

    for key, model_class in custom_models_dict.items():
        if model_string[:len(key)] == key:
            if len(model_string) > len(key):
                args = model_string.split('-')[1:]
                return model_class(*args, **kwargs), False
            if model_class is None:
                raise NotImplementedError('the model with identifier', key, 'hasn\'t been implemented yet')
            return model_class(**kwargs), False


def get_summary(model_string, input_shape, batch_size, **kwargs):
    """
    Print summary of model with identifier `model_string`. Input-shape and batch-size are needed to compute the
    estimated memory-footprint for forward- and backward-passes.


    Parameters
    ----------
    model_string
    input_shape
    batch_size
    kwargs

    Returns
    -------

    """
    from torchsummary import summary

    model, _ = get_model(model_string, input_shape=input_shape, **kwargs)

    summary(model, input_shape, batch_size)

    return model


def load_model_from_state_dict(model_string, state_dict_path=None, device=None, **kwargs):
    model, is_torch_model = get_model(model_string=model_string, **kwargs)

    if is_torch_model and state_dict_path is not None:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(state_dict=torch.load(state_dict_path,
                                                    map_location=torch.device(device)))
    return model


def load_model(saved_model_string, device=None, **kwargs):
    if 'statedict' in saved_model_string:
        model_string = saved_model_string.split(os.path.sep)[-2].split('_')[0]
    else:
        model_string = saved_model_string.split(os.path.sep)[-1].split('_')[0]

    bias = extract_from_path(path=saved_model_string,
                             key='bias',
                             default='True')
    bias = True if bias == 'True' else False
    dropout_rate = float(extract_from_path(path=saved_model_string,
                                           key='dr',
                                           default=0.0))
    output_classes = int(extract_from_path(path=saved_model_string,
                                           key='outs',
                                           default=2))

    return load_model_from_state_dict(model_string,
                                      saved_model_string,
                                      device,
                                      output_classes=output_classes,
                                      dropout_rate=dropout_rate,
                                      bias=bias,
                                      **kwargs)


def load_model_from_metafile(metafile, state_dict_name, device=None, **kwargs):
    metadict = {
        'model-id': None,
        'input-length': int,
        'dropout': float,
        'data-input': [lambda x: extract_from_path(x, key='mode_', split_char=os.path.sep),
                       lambda x: float(extract_from_path(x, key='bw_', split_char=os.path.sep))],
        'output-classes': int,
        'bias': lambda x: True if extract_from_path(path=x, key='bias', default='True') == 'True' else False,
    }
    values = extract_from_meta(fp=metafile, keys=metadict)

    model_string = values['model-id']
    if 'beta' not in values.keys(): values['beta'] = True
    if 'dropout' not in values.keys(): values['dropout'] = 0.0
    if 'output-classes' not in values.keys(): values['output-classes'] = 2

    state_dict_path = os.path.join(os.path.dirname(metafile),
                                   f'{state_dict_name}.statedict') if state_dict_name is not None else None

    return load_model_from_state_dict(model_string,
                                      state_dict_path,
                                      device,
                                      output_classes=values['output-classes'],
                                      dropout_rate=values['dropout'],
                                      bias=values['beta'],
                                      **kwargs)
