import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
import argparse
import os
from functools import partial
import re
import glob
from datetime import datetime

from joblib import Parallel, delayed
from dnn_modelling.helpers import load_history

from testing.plots.helpers import get_color_with_alpha
from testing.general import process_parameter
from testing.general import process_dprime
from testing.plots.general_plots import plot_history, scatter_and_psychometric_fit_plot
from testing.plots.general_plots import plot_thresholds
from data_creation.files.naming import threshold_pickle
from testing.print import TimeAndPrintContext

parser = argparse.ArgumentParser(prog='test_ensemble')

parser.add_argument('-tf',
                    dest='thesis_folder',
                    type=str, default='/zhome/e8/0/78560/msc_thesis')
parser.add_argument('-m',
                    dest='model_paths',
                    type=str, default=None,
                    help='path to model state-dicts for loading model class. '
                         'Models must be comma-seperated and without spaces.')
parser.add_argument('-mf',
                    dest='models_folder',
                    type=str, default='/work3/s124347/msc_thesis/model_output/',
                    help='path to model output folder')
parser.add_argument('-sd',
                    dest='state_dict',
                    type=str, default='best-model-train')
parser.add_argument('-e',
                    dest='test_experiment',
                    type=str, default='zeng_et_al_2005')
parser.add_argument('-er',
                    dest='external_results',
                    type=str, default='external_results',
                    help='path to external results, absolute or relative to thesis_folder')
parser.add_argument('-o',
                    dest='output_folder',
                    type=str, default='/zhome/e8/0/78560/public_html')
parser.add_argument('-os',
                    dest='output_subfolder',
                    type=str, default=None)
parser.add_argument('-w',
                    dest='workers',
                    type=int, default=20)
parser.add_argument('--history',
                    dest='plot_history',
                    action='store_const',
                    const=True, default=False)
args = parser.parse_args()

# Check model-paths is provided
if args.model_paths is None:
    print(parser.print_help())
    raise ValueError('Model-paths is missing from input arguments')

# Add subfolder
model_output_folder = None
if args.output_subfolder is not None:
    args.output_folder = os.path.join(args.output_folder, args.output_subfolder)
    model_output_folder = os.path.join(args.models_folder, args.output_subfolder)

# Check model-paths exist and create list of absolute paths
absolute_model_paths = list()
for model in args.model_paths.split(','):
    if model[0] == os.path.sep:
        absolute_model_path = model
    else:
        absolute_model_path = os.path.join(args.models_folder, model)

    if not os.path.exists(absolute_model_path):
        raise FileNotFoundError(
            f'couldn\'t find the folder for the model output for {model}, (fullpath: {absolute_model_path})')
    pickle_path = os.path.join(absolute_model_path, threshold_pickle(args.state_dict))
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f'couldn\'t find the thresholds dataframe for {model}, (fullpath: {pickle_path})')

    absolute_model_paths.append(absolute_model_path)

os.makedirs(args.output_folder, exist_ok=True)
if model_output_folder is not None:
    os.makedirs(model_output_folder, exist_ok=True)

if args.plot_history:
    # Setup concurrent function
    def load_history_for_concat(path, model_idx):
        df_out = pd.DataFrame(columns=[
            'model_index',
            'epoch',
            'loss',
            'loss_type',
            'time'
        ])
        for _, row in load_history(path).iterrows():
            for loss_type in ['valid_loss', 'train_loss']:
                df_out = df_out.append({
                    'model_index': model_idx,
                    'epoch': row['epoch'],
                    'loss': row[loss_type],
                    'loss_type': loss_type,
                    'time': row['time'],
                }, ignore_index=True)
        return df_out

    # Load histories in parallel
    with TimeAndPrintContext('Load training history'):
        history_df = pd.concat(Parallel(n_jobs=args.workers)(
            delayed(load_history_for_concat)(path=path, model_idx=idx) for idx, path in enumerate(absolute_model_paths)))

    with TimeAndPrintContext('Plot training history'):
        fig = px.line(history_df,
                      x='epoch',
                      y='loss',
                      color='model_index',
                      facet_row='loss_type')
        fig.write_html(os.path.join(args.output_folder, 'combined_history.html'),
                       include_plotlyjs='directory')


def load_thresholds_for_concat(path, model_idx):
    df = pd.read_pickle(os.path.join(path, threshold_pickle(args.state_dict)))
    # Add model_index column
    df['model_index'] = pd.Series([model_idx for _ in range(len(df.index))],
                                  index=df.index)
    return df


# Combine thresholds
with TimeAndPrintContext('Load thresholds'):
    threshold_df = pd.concat(Parallel(n_jobs=args.workers)(
        delayed(load_thresholds_for_concat)(path=path, model_idx=idx) for idx, path in enumerate(absolute_model_paths)))

if model_output_folder is not None:
    path = os.path.join(model_output_folder, f'thresholds-{args.state_dict}.df')
    with TimeAndPrintContext(f'Save concat-thresholds: {path}'):
        threshold_df.to_pickle(path)

# Compute mean thresholds
methods = threshold_df['method'].unique()
mean_threshold_df = pd.DataFrame(columns=['method',
                                          'impairment',
                                          'experiment_parameter',
                                          'gdt_mean',
                                          'gdt_std'])
for method_idx, method in enumerate(methods):
    df_at_method = threshold_df[threshold_df['method'] == method]
    for impairment in df_at_method['impairment'].unique():
        df_at_imp = df_at_method[df_at_method['impairment'] == impairment]
        if len(df_at_imp['model_index'].unique()) > 1:
            for experiment_parameter in df_at_imp['experiment_parameter'].unique():
                df_at_param = df_at_imp[df_at_imp['experiment_parameter'] == experiment_parameter]
                gdt = list()
                for model_index in df_at_param['model_index'].unique():
                    gdt.append(df_at_param[df_at_param['model_index'] == model_index]['gap_threshold'])
                gdt = np.array(gdt)
                mean_threshold_df = mean_threshold_df.append({'method': method,
                                                              'impairment': impairment,
                                                              'experiment_parameter': experiment_parameter,
                                                              'gdt_mean': gdt.mean(),
                                                              'gdt_std': gdt.std()}, ignore_index=True)
if model_output_folder is not None:
    path = os.path.join(model_output_folder, f'mean-thresholds-{args.state_dict}.df')
    with TimeAndPrintContext(f'Save mean-thresholds: {path}'):
        mean_threshold_df.to_pickle(path)

# Plot
with TimeAndPrintContext('Plot mean-thresholds'):
    methods = mean_threshold_df['method'].unique()
    fig = make_subplots(rows=1, cols=len(methods),
                        shared_yaxes=True,
                        shared_xaxes=True,
                        subplot_titles=[str(x) for x in methods])
    for method_idx, method in enumerate(methods):
        color_index_counter = 0
        df_at_method = mean_threshold_df[mean_threshold_df['method'] == method]
        for impairment in df_at_method['impairment'].unique():
            x = df_at_method[df_at_method['impairment'] == impairment]['experiment_parameter']
            y = df_at_method[df_at_method['impairment'] == impairment]['gdt_mean']
            gdt_std = df_at_method[df_at_method['impairment'] == impairment]['gdt_std']
            y_lower = [y - std for y, std in zip(y, gdt_std)]
            y_upper = [y + std for y, std in zip(y, gdt_std)]
            fig.add_trace(go.Scatter(x=pd.concat((x, x[::-1]), ignore_index=True),
                                     y=y_upper + y_lower[::-1],
                                     fill='toself',
                                     fillcolor=get_color_with_alpha(color_index_counter, 0.2),
                                     line=dict(color='rgba(0,0,0,0)'),
                                     showlegend=False,
                                     name=impairment,
                                     legendgroup=impairment),
                          row=1, col=method_idx + 1)
            fig.add_trace(go.Scatter(x=x,
                                     y=y,
                                     name=impairment,
                                     line=dict(color=get_color_with_alpha(color_index_counter, 1.0)),
                                     legendgroup=impairment),
                          row=1, col=method_idx + 1)

            # Loop and counter updates
            color_index_counter += 1

    fig.write_html(os.path.join(args.output_folder, f'combined_threshold-{args.state_dict}.html'),
                   include_plotlyjs='directory')

# python3 test_ensemble.py -m "eotd_1,eotd_2,eotd_3,eotd_4" -sd "model-at-epoch-80" -os "eotd_1-4"
# python3 -i test_ensemble.py -m "eotd_1,eotd_2,eotd_3,eotd_4eotd_5" -sd "model-at-epoch-40"
# python3 -i test_ensemble.py -m "eotd_1,eotd_2,eotd_3,eotd_4eotd_5" -sd "model-at-epoch-60"
# python3 -i test_ensemble.py -m "eotd_1,eotd_2,eotd_3,eotd_4eotd_5" -sd "model-at-epoch-80"
# python3 -i test_ensemble.py -m "eotd_1,eotd_2,eotd_3,eotd_4eotd_5" -sd "model-at-epoch-100"

# python3 -i test_ensemble.py -tf /Users/jenslindahl/repos/msc_thesis/ \
#         -mf /Users/jenslindahl/repos/msc_thesis/model_output \
#         -er /Users/jenslindahl/repos/msc_thesis/data/external_results \
#         -o /Users/jenslindahl/repos/msc_thesis/test_model_outputs \
#         -m "\
# lindahl3-32-512_bs64_lossBCE_optadam_wd0.5_dr0.0_nepochs1000_da0.5_dl700_lr0.0001_m-2d_bw-0.001_biasTrue_targetbinary_outs2_20220128_105343,\
# lindahl3-32-512_bs64_lossBCE_optadam_wd0.5_dr0.0_nepochs1000_da0.5_dl700_lr0.0001_m-2d_bw-0.001_biasTrue_targetbinary_outs2_20220128_105046,\
# lindahl3-32-512_bs64_lossBCE_optadam_wd0.5_dr0.0_nepochs1000_da0.5_dl700_lr0.0001_m-2d_bw-0.001_biasTrue_targetbinary_outs2_20220128_105140,\
# lindahl3-32-512_bs64_lossBCE_optadam_wd0.5_dr0.0_nepochs1000_da0.5_dl700_lr0.0001_m-2d_bw-0.001_biasTrue_targetbinary_outs2_20220128_105256"


# for epoch in 20 40 60 80 100