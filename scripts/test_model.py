import pandas as pd
import numpy as np
import argparse
import os
from functools import partial
import re
import glob

from joblib import Parallel, delayed
from threading import Thread

# Make imports work, even though script is in sub-folder "scripts"
import sys
sys.path.insert(0, os.getcwd())

from dnn_modelling.helpers import extract_from_path, get_datapath
from dnn_modelling.model import load_model
from dnn_modelling.model import load_model_from_metafile
from psychoacoustics.dprime import ideal_threshold
from psychoacoustics.psychometrics import get_psychometric_point, fit_psychometric_func

from testing.general import process_parameter
from testing.general import process_dprime
from testing.plots.model import plot_history
from testing.plots.general_plots import scatter_and_psychometric_fit_plot
from testing.plots.general_plots import plot_thresholds, plot_dprime
from data_creation.files.naming import threshold_pickle, model_output_pickle
from testing.print import TimeAndPrintContext

from testing.experiments.zeng_et_al_2005 import external_results as external__zeng_et_al_2005
from testing.experiments.shailer_and_moore_1983 import external_results as external__shailer_and_moore_1983

"""
Description here
"""
parser = argparse.ArgumentParser(prog='test_model')

parser.add_argument('-tf',
                    dest='thesis_folder',
                    type=str, default='/zhome/e8/0/78560/msc_thesis')
parser.add_argument('-m',
                    dest='model_path',
                    type=str, default=None,
                    help='path to model state-dict for loading model class')
parser.add_argument('-mf',
                    dest='models_folder',
                    type=str, default='/work3/s124347/msc_thesis/model_output/',
                    help='path to model output folder')
parser.add_argument('-e',
                    dest='test_experiment',
                    type=str, default='zeng_et_al_2005')
parser.add_argument('-er',
                    dest='external_results',
                    type=str, default='external_results',
                    help='path to external results, absolute or relative to thesis_folder')
parser.add_argument('-sd',
                    dest='state_dict',
                    type=str, default=None)
parser.add_argument('-o',
                    dest='output_folder',
                    type=str, default='/zhome/e8/0/78560/public_html')
parser.add_argument('-w',
                    dest='workers',
                    type=int, default=20)
parser.add_argument('--noplots',
                    dest='no_plots',
                    action='store_const',
                    const=True, default=False
                    )
parser.add_argument('-load',
                    dest='load_model_output_from_pickle',
                    action='store_const',
                    const=True, default=False)
parser.add_argument('--use_meta',
                    dest='use_meta_file',
                    action='store_const',
                    const=True, default=False)
args = parser.parse_args()

# Check inputs
if args.model_path is None:
    print(parser.print_help())
    raise ValueError('Model-path is missing from input arguments')

# Startup print
print('Running test_model with arguments:')
for k, v in vars(args).items():
    print('\t', k, ': ', v, sep='')

# Set output folder
output_folder = os.path.join(args.output_folder,
                             os.path.basename(args.model_path))
print('Saving to folder:', output_folder)
os.makedirs(output_folder, exist_ok=True)

if os.path.sep not in args.model_path:
    args.model_path = os.path.join(args.models_folder, args.model_path)
if os.path.sep != args.external_results[0]:
    args.external_results = os.path.join(args.thesis_folder, args.external_results)

# Setup experiment look-ups
experiment_folders = {
    'zeng_et_al_2005': 'zeng_et_al_2005__sensitivity_test',
    'shailer_and_moore_1983': '',
    'moore_er_al_1989': 'moore_er_al_1989',
    'moore_er_al_1993': 'moore_er_al_1993',
}

experiment_parameters = {
    'zeng_et_al_2005': 'level',
    'shailer_and_moore_1983': 'frequency'
}

experiment_parent_regexps = {
    'zeng_et_al_2005': 'lvl_${experiment_parameter}_db_spl',
    'shailer_and_moore_1983': 'freq_${experiment_parameter}hz'
}

external_results_loaders = {
    'zeng_et_al_2005': external__zeng_et_al_2005,
    'shailer_and_moore_1983': external__shailer_and_moore_1983,
}

external_results_path = os.path.join(args.external_results, args.test_experiment)

# Other constants
impairments = ['none',
               'cs_20pct',
               'cs_40pct',
               'cs_60pct',
               'cs_80pct',
               'zeng_23ohc_13ihc__cs_0pct',
               'zeng_23ohc_13ihc__cs_20pct',
               'zeng_23ohc_13ihc__cs_40pct',
               'zeng_23ohc_13ihc__cs_60pct',
               'zeng_23ohc_13ihc__cs_80pct',
               'zeng_all_ihc__cs_0pct',
               'zeng_all_ohc__cs_0pct',
               ]

input_mode_input_lengths = {
    '2d': 1,
    '3d': 3
}

# Load hyper parameters
input_length = None
mode_str = None
bandwidth = None
if args.use_meta_file:
    with open(os.path.join(args.model_path, 'meta.txt')) as metafile:
        for line in metafile:
            line = line.strip()  # remove surrounding
            if 'input-length' in line:
                input_length = int([x for x in line.split(' ') if x != ''][1])
            if 'data-input' in line:
                _data_input_path = [x for x in line.split(' ') if x != ''][1]
                mode_str = extract_from_path(_data_input_path, 'mode_', split_char=os.path.sep)
                bandwidth = extract_from_path(_data_input_path, 'bw_', split_char=os.path.sep)

else:
    input_length = int(extract_from_path(args.model_path, 'dl'))
    mode_str = extract_from_path(args.model_path, "m-")
    bandwidth = extract_from_path(args.model_path, "bw-")

if input_length is None:
    raise ValueError('input length not set')
if mode_str is None:
    raise ValueError('mode not set')
if bandwidth is None:
    raise ValueError('bandwidth  not set')

# Set input channels
in_channels = input_mode_input_lengths[mode_str]

# paths
data_folder = get_datapath(args.thesis_folder, mode=mode_str, bw=bandwidth)
an_model_basepath = os.path.join(data_folder, experiment_folders[args.test_experiment])
parent_regexp = experiment_parent_regexps[args.test_experiment]

#
# Plot history
#
if not args.no_plots:
    history_path = os.path.join(args.model_path, 'history.csv')
    if os.path.exists(history_path):
        print('plotting history')
        Thread(target=plot_history, kwargs={'model_path': args.model_path,
                                            'output_folder': output_folder}).start()
    else:
        print(f'no history found ({history_path})')

#
# Load Model
#
load_model_path = os.path.join(os.path.basename(args.model_path), f'{args.state_dict}.statedict')
with TimeAndPrintContext(f'Loading model from {load_model_path}'):
    if args.use_meta_file:
        model = load_model_from_metafile(os.path.join(args.model_path, 'meta.txt'),
                                         state_dict_name=args.state_dict,
                                         input_shape=(in_channels, 200, input_length))
    else:
        model_string_path = args.model_path
        if args.state_dict is not None:
            model_string_path = os.path.join(model_string_path, f'{args.state_dict}.statedict')
        model = load_model(model_string_path,
                           input_shape=(in_channels, 200, input_length))

local_process_parameter = partial(process_parameter,
                                  model=model,
                                  parent_regexp=parent_regexp,
                                  input_length=input_length)

# Get number of workers
print('Checking levels from data-folder', an_model_basepath)
regexp_r = r'\d+'.join(parent_regexp.split('${experiment_parameter}'))
levels = [int(re.findall(regexp_r, x)[0].split('_')[1]) for x in glob.glob(os.path.join(an_model_basepath, '*'))]
levels.sort()
print('Preparing to test for levels', levels)
if len(levels) == 0:
    levels = [x for x in range(20, 105, 5)]  # Artificial levels

model_output_path = os.path.join(args.model_path, model_output_pickle(args.state_dict))
if args.load_model_output_from_pickle and os.path.exists(model_output_path):
    df_mo = pd.read_pickle(model_output_path)

    gap_lengths = df_mo['gap_length'].unique()
    df_mo = df_mo.sort_values(by=['experiment',
                                  'impairment',
                                  'experiment_parameter',
                                  'gap_length',
                                  'model_output'])
else:
    parallel_argument_list = list()

    for impairment in impairments:
        if impairment == 'none':
            for lvl in levels:
                parallel_argument = (
                    lvl,
                    impairment,
                    an_model_basepath
                )
                parallel_argument_list.append(parallel_argument)
        else:
            impairment_path = os.path.join(data_folder, impairment,
                                           experiment_folders[args.test_experiment])
            cs_levels = [int(re.findall(regexp_r, x)[0].split('_')[1]) for x in
                         glob.glob(os.path.join(impairment_path, '*'))]
            for lvl in cs_levels:
                parallel_argument = (
                    lvl,
                    impairment,
                    impairment_path
                )
                parallel_argument_list.append(parallel_argument)

    df_mo = pd.concat(
        Parallel(n_jobs=args.workers)(
            delayed(local_process_parameter)(experiment_parameter=lvl,
                                             impairment=impairment,
                                             basepath=path) for lvl, impairment, path in
            parallel_argument_list),
        ignore_index=True)

    gap_lengths = df_mo['gap_length'].unique()
    df_mo = df_mo.sort_values(by=['experiment',
                                  'impairment',
                                  'experiment_parameter',
                                  'gap_length',
                                  'model_output'])

    with TimeAndPrintContext(f'Saving model output to {model_output_path}'):
        df_mo.to_pickle(model_output_path)

#
# Psychometric fit to model output
params = list()
for lvl in levels:
    at_lvl = df_mo[df_mo['experiment_parameter'] == lvl]

    params.append(fit_psychometric_func(at_lvl['gap_length'].to_numpy(),
                                        at_lvl['model_output'].to_numpy()))

#
# percentage correct
with TimeAndPrintContext('compute percentage-correct (0.5)'):
    df_pct = pd.DataFrame(columns=['impairment',
                                   'experiment_parameter',
                                   'gap_length',
                                   'percentage_correct',
                                   'threshold'])
    for impairment in impairments:
        df_impair = df_mo[df_mo['impairment'] == impairment]
        for lvl in levels:
            df_at_lvl = df_impair[df_impair['experiment_parameter'] == lvl]

            for gap in gap_lengths:
                X = df_at_lvl[df_at_lvl['gap_length'] == gap]['model_output'].to_numpy()

                if len(X) == 0:
                    continue


                # Outputs True/False on whether the model output 'x' surpass the threshold 'p'
                def correct_func(x, p):
                    return x >= p if gap > 0 else x < p


                df_pct = df_pct.append({'impairment': impairment,
                                        'experiment_parameter': lvl,
                                        'gap_length': gap,
                                        'percentage_correct': np.mean(correct_func(X, 0.5)),
                                        'threshold': 0.5},
                                       ignore_index=True)

with TimeAndPrintContext('compute percentage-correct (ideal)'):
    df_pct_ideal = pd.DataFrame(columns=['impairment',
                                         'experiment_parameter',
                                         'gap_length',
                                         'percentage_correct',
                                         'threshold'])
    for impairment in impairments:
        df_impair = df_mo[df_mo['impairment'] == impairment]
        for lvl in levels:
            df_at_lvl = df_impair[df_impair['experiment_parameter'] == lvl]

            N = df_at_lvl[df_at_lvl['gap_length'] == 0]['model_output'].to_numpy()
            for gap in gap_lengths:
                S = df_at_lvl[df_at_lvl['gap_length'] == gap]['model_output'].to_numpy()

                if len(S) == 0:
                    continue


                # Outputs True/False on whether the model output 'x' surpass the threshold 'p'
                def correct_func(x, p):
                    return x >= p if gap > 0 else x < p


                threshold = ideal_threshold(S, N)
                df_pct_ideal = df_pct_ideal.append({'impairment': impairment,
                                                    'experiment_parameter': lvl,
                                                    'gap_length': gap,
                                                    'percentage_correct': np.mean(correct_func(S, threshold)),
                                                    'threshold': threshold},
                                                   ignore_index=True)


def percentage_correct_plot(df, threshold, impairment):
    if len(df[df['impairment'] == impairment]) == 0:
        print(f'percentage_correct_plot ({threshold}) didnt find data for {impairment}')
        return
    fn = '_'.join([str(args.state_dict), f'percentage_correct-{threshold}-{impairment}.html'])
    scatter_and_psychometric_fit_plot(df[df['impairment'] == impairment],
                                      y='percentage_correct',
                                      outputpath=os.path.join(output_folder, fn),
                                      title=f'Percentage Correct (with threshold: {threshold})',
                                      with_scale=False,
                                      with_loc=False,
                                      ylimit=[0, 1],
                                      fit_offset=1)


def model_output_plot(impairment):
    if len(df_mo[df_mo['impairment'] == impairment]) == 0:
        print(f'model_output_plot didnt find data for {impairment}')
        return
    fn = '_'.join([str(args.state_dict), f'model_output-{impairment}.html'])
    scatter_and_psychometric_fit_plot(df_mo[df_mo['impairment'] == impairment],
                                      y='model_output',
                                      outputpath=os.path.join(output_folder, fn),
                                      title='Raw model-output fit fitted psychometric functions',
                                      with_scale=True,
                                      with_loc=True,
                                      ylimit=[0, 1])


if not args.no_plots:
    # Setup plot jobs
    job_list = list()
    [job_list.append(delayed(percentage_correct_plot)(df_pct, '0.5', impairment)) for impairment in impairments]
    [job_list.append(delayed(percentage_correct_plot)(df_pct_ideal, 'ideal', impairment)) for impairment in impairments]
    [job_list.append(delayed(model_output_plot)(impairment)) for impairment in impairments]

    Parallel(n_jobs=args.workers)(job_list)


#
def append_dataframe_with_psychometric_fit_threshold(df,
                                                     psychometric_params,
                                                     impairment,
                                                     experiment_parameter,
                                                     method,
                                                     pct=0.707):
    p = psychometric_params
    t = -1
    if p is not None:
        t = get_psychometric_point(pct, p[0], p[1])
    if t <= 0.0:
        t = np.nan

    if method == 'percentage_correct_ideal':
        print(experiment_parameter, method, impairment, t)
    return df.append({'experiment_parameter': experiment_parameter,
                      'gap_threshold': t,
                      'method': method,
                      'impairment': impairment,
                      'psy a': p[0] if p is not None else p,
                      'psy b': p[1] if p is not None else p,
                      'psy scale': p[2] if p is not None and len(p) > 2 else p,
                      'psy loc': p[3] if p is not None and len(p) > 3 else p},
                     ignore_index=True)


def threshold_for_impairment(impairment):
    thres_df = pd.DataFrame(columns=['experiment_parameter',
                                     'gap_threshold',
                                     'method',
                                     'impairment',
                                     'psy a',
                                     'psy b',
                                     'psy scale',
                                     'psy loc'])
    df_mo_local = df_mo[df_mo['impairment'] == impairment]
    df_pct_local = df_pct[df_pct['impairment'] == impairment]
    df_pct_local2 = df_pct_ideal[df_pct_ideal['impairment'] == impairment]
    for lvl in levels:
        df_at_lvl = df_mo_local[df_mo_local['experiment_parameter'] == lvl]
        if len(df_at_lvl) == 0 or len(df_at_lvl['model_output'].to_numpy()) == 0:
            continue
        thres_df = append_dataframe_with_psychometric_fit_threshold(df=thres_df,
                                                                    psychometric_params=fit_psychometric_func(
                                                                        df_at_lvl['gap_length'].to_numpy(),
                                                                        df_at_lvl['model_output'].to_numpy()),
                                                                    method=f'model_output',
                                                                    impairment=impairment,
                                                                    experiment_parameter=lvl,
                                                                    pct=0.707)
        df_at_lvl = df_pct_local[df_pct_local['experiment_parameter'] == lvl]
        if len(df_at_lvl) == 0 or len(df_at_lvl['percentage_correct'].to_numpy()) == 0:
            continue
        thres_df = append_dataframe_with_psychometric_fit_threshold(df=thres_df,
                                                                    psychometric_params=fit_psychometric_func(
                                                                        xdata=df_at_lvl['gap_length'].to_numpy()[1:],
                                                                        ydata=df_at_lvl[
                                                                                  'percentage_correct'].to_numpy()[1:],
                                                                        with_scale=False,
                                                                        with_loc=False),
                                                                    method=f'percentage_correct',
                                                                    impairment=impairment,
                                                                    experiment_parameter=lvl,
                                                                    pct=0.707)
        df_at_lvl = df_pct_local2[df_pct_local2['experiment_parameter'] == lvl]
        if len(df_at_lvl) == 0 or len(df_at_lvl['percentage_correct'].to_numpy()) == 0:
            continue
        thres_df = append_dataframe_with_psychometric_fit_threshold(df=thres_df,
                                                                    psychometric_params=fit_psychometric_func(
                                                                        xdata=df_at_lvl['gap_length'].to_numpy()[1:],
                                                                        ydata=df_at_lvl[
                                                                                  'percentage_correct'].to_numpy()[1:],
                                                                        with_scale=False,
                                                                        with_loc=False),
                                                                    method=f'percentage_correct_ideal',
                                                                    impairment=impairment,
                                                                    experiment_parameter=lvl,
                                                                    pct=0.707)
    print(f'threshold for impairment {impairment} computed')
    return thres_df


thres_df = pd.concat(
    Parallel(n_jobs=len(impairments))(delayed(threshold_for_impairment)(impairment) for impairment in impairments))

# Save to disk (before adding external results)
thres_df_path = os.path.join(args.model_path, threshold_pickle(args.state_dict))

with TimeAndPrintContext(f'Saving thresholds to {thres_df_path}'):
    thres_df.to_pickle(thres_df_path)

if not args.no_plots:
    # Add external results
    if os.path.exists(external_results_path):
        thres_df = external_results_loaders[args.test_experiment](thres_df, external_results_path)
    else:
        print(f'couldnt load external results from path: {external_results_path}')

    threshold_plot_path = os.path.join(output_folder, '_'.join([args.state_dict,
                                                                'gap_threshold.html']))
    with TimeAndPrintContext(f'Plotting threshold plot to {threshold_plot_path}'):
        plot_thresholds(df=thres_df,
                        x_label='Stimulus Level [mixed dB]',
                        output_path=threshold_plot_path)

parallel_argument_list = list()
for impairment in impairments:
    levels_impairment = df_mo[df_mo['impairment'] == impairment]['experiment_parameter'].unique()
    for lvl in levels_impairment:
        parallel_argument_list.append((lvl, impairment))

df_dprime = pd.concat(Parallel(n_jobs=args.workers)(
    delayed(process_dprime)(df=df_mo[df_mo['impairment'] == impairment],
                            experiment_parameter=lvl,
                            gap_lengths=gap_lengths,
                            impairment=impairment) for lvl, impairment in
    parallel_argument_list))
df_dprime = df_dprime.sort_values(by=['experiment_parameter',
                                      'gap_length'])

if not args.no_plots:
    Parallel(n_jobs=args.workers)(
        delayed(plot_dprime)(df_dprime[df_dprime['impairment'] == impairment],
                             os.path.join(output_folder,
                                          '_'.join([args.state_dict,
                                                    f'dprime-({impairment}).html']))) for impairment in
        impairments)

print('thres_df')
print(thres_df.sort_values(by=['method', 'experiment_parameter', 'impairment']))
print('df_dprime')
print(df_dprime)

print()
print('DONE')
