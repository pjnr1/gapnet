"""
Command line script for generating spikeograms of ANF simulations

Usage::
    > python scripts/spikeogramorator.py --help

"""
import argparse
import glob
import os

from joblib import Parallel, delayed

import torch

# Make imports work, even though script is in sub-folder "scripts"
import sys
sys.path.insert(0, os.getcwd())

# Import local packages
from data_creation.files.naming import get_condition_folders
from data_creation.spikeogram.generators import spikeogram_2D, spikeogram_3D
from data_creation.spikeogram.tools import open_an_simulation_data
from typing_tools.argparsers import ranged_type, string_from_valid_list_type

generator_dict = {
    '2d': spikeogram_2D,
    '3d': spikeogram_3D,
}

parser = argparse.ArgumentParser(prog='spikeogramorator',
                                 description="""
                                 Tool for generating spikeograms prior to presenting the ANF stimulation to a model.
                                 """,
                                 epilog="""
                                 Part of GapNet package. Visit https://github.com/pjnr1/gapnet for more information.
                                 """,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--regexp',
                    dest='file_regexp',
                    type=str,
                    default='*.mat',
                    help='Specify the regexp pattern to match files with')
parser.add_argument('-w', '--binwidth',
                    dest='binwidth',
                    type=ranged_type(float,
                                     min_value=0.0,
                                     min_inclusive=False),
                    default=1e-3,
                    help='set the width of the bin/window of the "psth" / spikeogram')
parser.add_argument('-p', '--overlap',
                    dest='overlap',
                    type=ranged_type(float,
                                     min_value=0.0,
                                     max_value=0.99),
                    default=0.0,
                    help='set the ratio of overlap for each bin/window')
parser.add_argument('-m', '--mode',
                    dest='mode',
                    type=string_from_valid_list_type(['2d',
                                                      '3d']),
                    default='2d',
                    help='set spikeogram mode (2d or 3d)')
parser.add_argument('-r', '--recursive',
                    dest='recursive',
                    action='store_const',
                    const=True,
                    default=False,
                    help='flag for searching input folders recursively for .mat files to convert')
parser.add_argument('-f', '--overwrite',
                    dest='overwrite', action='store_const',
                    const=True,
                    default=False,
                    help='flag for enabling overwriting existing spikeograms')
parser.add_argument('-j', '--workers',
                    dest='workers',
                    type=int,
                    default=1,
                    help='number of workers for parallel processing')
parser.add_argument('-i', '--input',
                    dest='input_folders',
                    type=str,
                    nargs='+',
                    required=True,
                    help='list of folders to include in generation')
parser.add_argument('-o', '--output',
                    dest='output_folder',
                    type=str,
                    default='',
                    help='sets the folder to save the spikeograms in')

# Parse arguments
args = parser.parse_args()
condition_folders = get_condition_folders(args.mode, args.binwidth)
generator = generator_dict[args.mode]

print('Starting generation for the following input folders:')
for i, folder in enumerate(args.input_folders):
    print('  ', i, folder)
print('Mode:       ', args.mode)
print('Binwidth:   ', args.binwidth)
print('Destination:', os.path.join(args.output_folder,
                                   *condition_folders))


#
# Setup worker functions and local helpers
#
def get_destination_path(mat_file, input_folder):
    """
    Generates output folder for the given mat_file and input_folder::
        <output_folder> / <*condition_folders> /
    @param mat_file:
    @param input_folder:
    @return:
    """
    return os.path.join(args.output_folder,
                        *condition_folders,
                        *(mat_file.split(input_folder)[-1].split(os.path.sep)[:-1]),
                        os.path.splitext(mat_file.split(os.path.sep)[-1])[0] + '.pt')


def process_simulation(mat_file, input_folder):
    filepath = get_destination_path(mat_file, input_folder)
    if os.path.exists(filepath):
        print('File exists, returning')
        return
    filepath_folder = os.path.dirname(filepath)
    os.makedirs(filepath_folder, exist_ok=True)
    print('Reading:', mat_file)
    try:
        output = generator(open_an_simulation_data(mat_file),
                           bin_width=args.binwidth)

        print('Saving: ', filepath, end=' ')
        torch.save(output, filepath)
        print('done!')
    except OSError as e:
        print(f'error reading {mat_file}. Got {e}')
        return


for f in args.input_folders:
    mat_files = list()  # initialise file list

    path_string = f + (os.path.sep + '**' if args.recursive else '') + os.path.sep + args.file_regexp
    print('Searching for', path_string)
    for file in glob.glob(path_string, recursive=args.recursive):
        mat_files.append(file)

    if not args.overwrite:
        mat_files = [x for x in mat_files if not os.path.exists(get_destination_path(x, f))]

    print('Generating spikeograms for', len(mat_files), 'files')
    Parallel(n_jobs=args.workers)(delayed(process_simulation)(mf, f) for mf in mat_files)
