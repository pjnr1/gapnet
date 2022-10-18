"""
Command line script for training DNN models

Usage::
    > python scripts/train_model.py --help

"""
import argparse
import os
from functools import partial
from datetime import datetime
import subprocess

# Pytorch
import torch

# FastAI
import fastai.optimizer
from fastai.data.transforms import GrandparentSplitter
from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.learner import Learner
from fastai.losses import BCELossFlat

# Make local imports work, even though script is in sub-folder "scripts"
import sys
sys.path.insert(0, os.getcwd())

# Local
from typing_tools.argparsers import string_from_valid_list_type
from dnn_modelling.model import get_model
from dnn_modelling.callbacks import SaveStatedictCallback, SaveStatedictEveryNEpochCallback
from dnn_modelling.dataloader.dataloader import get_dataloaders
from dnn_modelling.dataloader.datablock import label_func__gap_nogap, label_func__gap_length


loss_func_dict = {
    'bce': torch.nn.BCELoss,
    'bce_sum': partial(torch.nn.BCELoss, reduction='sum'),
    'bce_none': partial(torch.nn.BCELoss, reduction='none'),
    'bceflat': BCELossFlat,
    'bcelogits': torch.nn.BCEWithLogitsLoss,
    'mse': torch.nn.MSELoss,
    'mseflat': fastai.losses.MSELossFlat,
}

opt_func_dict = {
    'adam': fastai.optimizer.Adam,
    'adam_l2': partial(fastai.optimizer.Adam, decouple_wd=False),
    'sgd': fastai.optimizer.SGD,
    'lamb': fastai.optimizer.Lamb,
}

early_stopping_variable_dict = {
    'valid': 'valid_loss',
    'train': 'train_loss'
}

label_function_dict = {
    'binary': (label_func__gap_nogap, 2),  # Default
    'gap_length': (label_func__gap_length, 1),
}


parser = argparse.ArgumentParser(prog='train_model',
                                 description='Description: TODO',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--input',
                    dest='input_folder',
                    type=str,
                    default='/work3/s124347/msc_thesis/cnn_data',
                    help='list of folders to include in training')
parser.add_argument('-dw', '--dataworkers',
                    dest='dataloader_workers',
                    type=int,
                    default=0,
                    help='Number of dataworkers to use')
parser.add_argument('-n', '--nepochs',
                    dest='n_epochs',
                    type=int,
                    default=10,
                    help='Number of epochs to train the model')
parser.add_argument('-m', '--model',
                    dest='model',
                    type=str,
                    default='kell',
                    help='list of folders to include in training')
parser.add_argument('-w', '--weights',
                    dest='weights',
                    type=str,
                    default=None,
                    help='path of pretrained weights to load model with')
parser.add_argument('-o', '--output',
                    dest='output_folder',
                    type=str,
                    default='/work3/s124347/msc_thesis/model_output',
                    help='sets the folder to save the model in')
parser.add_argument('-on', '--output_name',
                    dest='output_name',
                    type=str,
                    default=None,
                    help='sets the name of the model')
parser.add_argument('--test',
                    dest='run_test',
                    action='store_const',
                    const=True,
                    default=False,
                    help='flag for running test script after training')
parser.add_argument('-er', '--externalresults',
                    dest='external_results_folder',
                    type=str,
                    default='/work3/s124347/msc_thesis/external_results',
                    help='sets the folder from which to pull external results (see test_model.py for reference)')

# ** hyper-parameters **
parser.add_argument('-l', '--loss',
                    dest='loss_func',
                    type=str,
                    default='BCE',
                    help=f'Set the loss function. Valid inputs are: {loss_func_dict.keys()}')
parser.add_argument('-op', '--opt',
                    dest='opt_func',
                    type=str,
                    default='adam',
                    help=f'Set the loss function. Valid inputs are: {opt_func_dict.keys()}')
parser.add_argument('-bs', '--batchsize',
                    dest='batch_size',
                    type=int,
                    default=16,
                    help='')
parser.add_argument('-vp', '--valid',
                    dest='valid_pct',
                    type=float,
                    default=0.1,
                    help='')
parser.add_argument('-cp', '--cutpct', '--cutaugment',
                    dest='data_augmentation_pct',
                    type=float,
                    default=0.0,
                    help='Use data augmentation by cutting a percentage of the time between start of sample and onset '
                         'of noise')
parser.add_argument('-il', '--inputlength',
                    dest='input_length',
                    type=int,
                    default=800,
                    help='Use for data augmentation: sets the desired output length and thus how much of the original '
                         'sample can be truncated')
parser.add_argument('-es', '--earlystop',
                    dest='early_stopping',
                    type=int,
                    default=100,
                    help='')
parser.add_argument('-nts', '--n_to_save',
                    dest='n_to_save',
                    type=int,
                    default=20,
                    help='Every n_to_save epoch, the model is saved to disk')
parser.add_argument('-esv', '--earlystopvariable',
                    dest='early_stopping_variable',
                    type=string_from_valid_list_type(early_stopping_variable_dict.keys()),
                    default='valid',
                    help=f'Set the variable for early stopping. Valid inputs are: {early_stopping_variable_dict.keys()}')
parser.add_argument('-lr',
                    dest='learning_rate',
                    type=float,
                    default=0.001,
                    help='')
parser.add_argument('-dr',
                    dest='dropout_rate',
                    type=float,
                    default=0.,
                    help='')
parser.add_argument('-nb', '--nobias',
                    dest='bias',
                    action='store_const',
                    const=False,
                    default=True,
                    help='')
parser.add_argument('-lf',
                    dest='label_function',
                    type=string_from_valid_list_type(label_function_dict.keys()),
                    default='binary',
                    help='')
parser.add_argument('-wd',
                    dest='weight_decay',
                    type=float,
                    default=None,
                    help='')
parser.add_argument('--gpf_t',
                    dest='grandparent_train_folders',
                    type=str,
                    default='')
parser.add_argument('--gpf_v',
                    dest='grandparent_valid_folders',
                    type=str,
                    default='')
parser.add_argument('-bw', '--binwidth',
                    dest='spikeogram_binwidth',
                    type=float,
                    default=1e-3)
parser.add_argument('-sm', '--spikemode',
                    dest='spikeogram_mode',
                    type=string_from_valid_list_type([
                        '2d',
                        '3d'
                    ]),
                    default='2d')

args = parser.parse_args()

if args.input_folder is None:
    raise ValueError("You need to set an input folder")

input_folder = os.path.join(args.input_folder, f'mode_{args.spikeogram_mode}', f'bw_{str(args.spikeogram_binwidth)}')

output_name = args.output_name
if output_name is None or output_name == '':
    output_name = '_'.join([args.model,
                            'bs' + str(args.batch_size),
                            'loss' + str(args.loss_func),
                            'opt' + str(args.opt_func),
                            'wd' + str(args.weight_decay),
                            'dr' + str(args.dropout_rate),
                            'nepochs' + str(args.n_epochs),
                            'da' + str(args.data_augmentation_pct),
                            'dl' + str(args.input_length),
                            'lr' + str(args.learning_rate),
                            'm-' + str(args.spikeogram_mode),
                            'bw-' + str(args.spikeogram_binwidth),
                            'bias' + str('True' if args.bias else 'False'),
                            'target' + args.label_function,
                            'outs' + str(label_function_dict[args.label_function][1]),
                            datetime.now().strftime("%Y%m%d_%H%M%S")])

output_path = os.path.join(args.output_folder, output_name)
os.makedirs(output_path, exist_ok=True)

# Parse the optimizer function, add weight decay if found in argument
opt_func_str = args.opt_func.lower()
opt_func = opt_func_dict[opt_func_str]
opt_func = opt_func if args.weight_decay is None else partial(opt_func, wd=args.weight_decay)

splitter = None
if args.grandparent_train_folders != '' and args.grandparent_valid_folders != '':
    train_folders_split = args.grandparent_train_folders.split(',')
    valid_folders_split = args.grandparent_valid_folders.split(',')
    splitter = GrandparentSplitter(train_name=train_folders_split,
                                   valid_name=valid_folders_split)
    print('Using GrandparentSplitter with:')
    print('  train parents:', train_folders_split)
    print('  valid parents:', valid_folders_split)

# Get dataloader
print('Dataloader', end='...')
dls = get_dataloaders(input_folder=input_folder,
                      batch_size=args.batch_size,
                      valid_pct=args.valid_pct,
                      subsample_pct=args.data_augmentation_pct,
                      input_length=args.input_length,
                      num_workers=args.dataloader_workers,
                      splitter=splitter,
                      label_func=label_function_dict[args.label_function][0])
print('done')

# Get model
input_shape = dls.one_batch()[0][0].shape

model, _ = get_model(args.model,
                     input_shape=input_shape,
                     output_classes=label_function_dict[args.label_function][1],
                     dropout_rate=args.dropout_rate,
                     bias=args.bias)

parameter_str = '\n'.join(['Model loaded with:',
                           '  model-id        ' + str(args.model),
                           '  input-shape     ' + str(input_shape),
                           '  target          ' + str(args.label_function),
                           '  label-function  ' + str(label_function_dict[args.label_function][0]),
                           '  output-classes  ' + str(label_function_dict[args.label_function][1]),
                           '',
                           'Training on:',
                           '  ' + args.grandparent_train_folders,
                           ''
                           'Validating on:',
                           '  ' + args.grandparent_valid_folders,
                           '',
                           'HyperParameters:',
                           '  loss-func       ' + str(args.loss_func) + ' ' + str(
                               loss_func_dict[args.loss_func.lower()]),
                           '  opt-func        ' + str(args.opt_func) + ' ' + str(opt_func),
                           '  weight-decay    ' + str(args.weight_decay),
                           '  batch-size      ' + str(args.batch_size),
                           '  cut-pct         ' + str(args.data_augmentation_pct),
                           '  input-length    ' + str(args.input_length),
                           '  n-epoch-saving  ' + str(args.n_to_save),
                           '  early-stopping  ' + str(args.early_stopping),
                           '  early-stopping variable ' + str(args.early_stopping_variable) + ' -> ' + str(
                               early_stopping_variable_dict[args.early_stopping_variable]),
                           '  learning-rate   ' + str(args.learning_rate),
                           '  dropout         ' + str(args.dropout_rate),
                           '  bias            ' + str('True' if args.bias else 'False'),
                           '',
                           'Paths:',
                           '  data-input      ' + input_folder,
                           '  output          ' + output_path,
                           ''])
with open(os.path.join(output_path, 'meta.txt'), 'a') as f:
    f.write(parameter_str)  # print to disk
    print(parameter_str)  # print to screen

callbacks = list()
if 0 < args.early_stopping < args.n_epochs:
    callbacks.append(EarlyStoppingCallback(monitor=early_stopping_variable_dict[args.early_stopping_variable],
                                           min_delta=0.0,
                                           patience=args.early_stopping))

callbacks.append(SaveStatedictCallback(monitor='valid_loss',
                                       path=output_path,
                                       fname='best-model-valid'))
callbacks.append(SaveStatedictCallback(monitor='train_loss',
                                       path=output_path,
                                       fname='best-model-train'))
callbacks.append(SaveStatedictEveryNEpochCallback(path=output_path,
                                                  n_to_save=args.n_to_save,  # TODO make hyper-parameter
                                                  fname='model-at-epoch'))
callbacks.append(CSVLogger())

# Setup learner
print('Setting up FastAI-Learner', end='...')
learner = Learner(path=output_path,
                  dls=dls,
                  model=model,
                  lr=args.learning_rate,
                  opt_func=opt_func,
                  loss_func=loss_func_dict[args.loss_func.lower()](),
                  cbs=callbacks)
print('done')

# Train
learner.fit(args.n_epochs)
print('Learning complete')

# Export model (full learner)
print('Exporting Learner', end='...')
learner_export_path = os.path.join(output_path, 'learner.statedict')
torch.save(learner.model.state_dict(), learner_export_path)
print('done')
print(f'({learner_export_path})')
print(f'{os.path.basename(output_path)}')

# Launch test script
if args.run_test:
    subprocess.call(' '.join([f'python3 test_model.py',
                              f'-er {args.external_results_folder}',
                              f'-mf {os.path.dirname(output_path)}',
                              f'-m {os.path.basename(output_path)}',
                              '-sd best-model-valid']),
                    shell=True)
    subprocess.call(' '.join([f'python3 test_model.py',
                              f'-er {args.external_results_folder}',
                              f'-mf {os.path.dirname(output_path)}',
                              f'-m {os.path.basename(output_path)}',
                              '-sd best-model-train']),
                    shell=True)
    subprocess.call(' '.join([f'python3 test_model.py',
                              f'-er {args.external_results_folder}',
                              f'-mf {os.path.dirname(output_path)}',
                              f'-m {os.path.basename(output_path)}',
                              '-sd learner']),
                    shell=True)
