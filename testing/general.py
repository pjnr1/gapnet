from __future__ import annotations

import os
import re
from typing import Annotated, List
import glob
import pandas as pd
import torch

from dnn_modelling.dataloader.datasets import get_spikeogram_files, natural_keys
from psychoacoustics.dprime import dprime_empirical_jones, ideal_threshold
from psychoacoustics.psychometrics import psychometric_func, fit_psychometric_func
from typing_tools.annotations import check_annotations
from typing_tools.annotation_checkers import PathExists


def get_levels_from_path(path: str | os.PathLike[str], regexp: str) -> List[int]:
    """

    @param path:
    @param regexp:
    @return: list of levels derived from folder names
    """
    levels = [int(re.findall(regexp, x)[0]) for x in glob.glob(os.path.join(path, '*'))]
    levels.sort()
    return levels


def get_model_output_dataframe() -> pd.DataFrame:
    """
    Simple constructor of the model-output dataframe

    """
    return pd.DataFrame(columns=['experiment_parameter',
                                 'gap_length',
                                 'model_output'
                                 'impairment'])


def get_gaplength_from_stimulus_filename(fn: str) -> float:
    """
    Matches the number in the format $.$ms

    If multiple $.$ms exists, the last one will be used.

    Example::
        /somefolder/gdt__12.5ms.pt              -> 12.5
        /somefolder/51.3/542.2ms/gdt__2.5ms.pt  -> 2.5

    @param fn:
        filepath

    @return:
        gap length as a float

    """
    return float(re.findall(r'\d+\.\d+ms', fn)[-1][:-2])


@check_annotations
def apply_model(model: torch.nn.Module, path: Annotated[os.PathLike, PathExists], input_length: int) -> torch.Tensor:
    try:
        X = torch.load(path)
        while len(X.shape) < 4:
            X = torch.unsqueeze(X, dim=0)
        X = X[..., :input_length]

        with torch.no_grad():
            return model.forward(X)
    except EOFError as e:
        print(f'{path} couldn\'t be loaded, check that it\'s not corrupt ({e})')


def process_parameter(model,
                      experiment_parameter,
                      basepath,
                      parent_regexp,
                      input_length,
                      experiment='',
                      impairment='none'):
    print(f'Processing experiment-parameter: {experiment_parameter}, {impairment}, {basepath}')
    df = get_model_output_dataframe()
    stimuli_path = os.path.join(basepath,
                                parent_regexp.replace('${experiment_parameter}',
                                                      str(experiment_parameter)))
    stimuli_list = get_spikeogram_files(stimuli_path)
    stimuli_list.sort(key=lambda x: natural_keys(x.replace('-', '')))

    for stimulus_path in stimuli_list:
        y = apply_model(model=model,
                        path=stimulus_path,
                        input_length=input_length)
        if y is not None:
            df = df.append({'experiment_parameter': experiment_parameter,
                            'gap_length': get_gaplength_from_stimulus_filename(stimulus_path),
                            'model_output': float(y[0][0]),
                            'impairment': impairment,
                            'experiment': experiment},
                           ignore_index=True)
    print(f'Returning from experiment-parameter: {experiment_parameter}, {impairment}, {basepath}'
          f'\t(files: {len(stimuli_list)})')
    return df


def process_dprime(df, experiment_parameter, gap_lengths, impairment='none'):
    psychometric_methods = {
        'd\' .5': lambda s, n: dprime_empirical_jones(s, n, threshold=0.5),
        'd\' ideal': lambda s, n: dprime_empirical_jones(s, n, threshold=ideal_threshold(s, n)),
    }

    local_df = pd.DataFrame(columns=['experiment_parameter',
                                     'impairment',
                                     'gap_length',
                                     'dprime',
                                     'dprime_type'])
    df_at_param = df[df['experiment_parameter'] == experiment_parameter]
    for gap in gap_lengths:
        signal = df_at_param[df_at_param['gap_length'] == gap]['model_output']
        noise = df_at_param[df_at_param['gap_length'] == 0.0]['model_output']

        for k, v in psychometric_methods.items():
            local_df = local_df.append({
                'experiment_parameter': experiment_parameter,
                'impairment': impairment,
                'gap_length': gap,
                'dprime': v(signal, noise),
                'dprime_type': k,
            }, ignore_index=True)

    # Fit psychometric to d'-estimates
    for dprime_type in local_df['dprime_type'].unique():
        df_at_dprime_type = local_df[local_df['dprime_type'] == dprime_type]
        p = fit_psychometric_func(df_at_dprime_type['gap_length'].to_numpy(),
                                  df_at_dprime_type['dprime'].to_numpy(),
                                  scale_bounds=[0., 10.],
                                  with_loc=False)  # Don't elevate from zero
        if p is None:
            print(f'Couldn\'t fit psychometric function to dprime_type: {dprime_type}')
            continue

        for gap in gap_lengths:
            local_df = local_df.append({
                'experiment_parameter': experiment_parameter,
                'impairment': impairment,
                'gap_length': gap,
                'dprime': psychometric_func(gap, *p),
                'dprime_type': dprime_type + ' (psychometric fit)',
            }, ignore_index=True)

    return local_df
