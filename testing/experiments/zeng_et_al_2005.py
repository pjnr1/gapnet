import os

import h5py
import numpy as np
import pandas as pd

impairments = {
    'Zeng et al. 2005, NH': 'fig5_normal.csv',
    'Zeng et al. 2005, NP': 'fig5_neuropathy.csv',
}


def external_results(df: pd.DataFrame, path: str) -> pd.DataFrame:
    methods = df['method'].unique()

    for impairment, v in impairments.items():
        fp = os.path.join(path, v)
        for i, row in pd.read_csv(fp, sep=',', delimiter=';', header=None).iterrows():
            level = round(float(row[0].replace(',', '.')))
            threshold = round(float(row[1].replace(',', '.')), 2)
            for method in methods:
                df = df.append({'experiment_parameter': level,
                                'gap_threshold': threshold,
                                'impairment': impairment,
                                'method': method},
                               ignore_index=True)

    return df


def get_external_results(path: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=['experiment_parameter',
                               'gap_threshold',
                               'gdt_se',
                               'impairment'])
    for impairment, v in impairments.items():
        fp = os.path.join(path, v)
        for i, row in pd.read_csv(fp, sep=',', delimiter=';', header=None).iterrows():
            level = round(float(row[0].replace(',', '.')))
            threshold = round(float(row[1].replace(',', '.')), 2)
            upper_se_bound = round(float(row[2].replace(',', '.')), 2)
            df = df.append({'experiment_parameter': level,
                            'gap_threshold': threshold,
                            'gdt_se': upper_se_bound - threshold,
                            'impairment': impairment},
                           ignore_index=True)
    return df


def mean_audiogram(path: str, group: str = 'np') -> (np.array, np.array):
    df = pd.DataFrame(columns=['frequency',
                               'threshold'])
    if group.lower() == 'np':
        fp = os.path.join(path, 'fig1_mean.csv')
        for i, row in pd.read_csv(fp, sep=',', delimiter=';', header=None).iterrows():
            frequency = float(row[0])
            threshold = round(float(row[1].replace(',', '.')), 2)

            # TODO Convert into dB SPL

            df = df.append({'frequency': frequency,
                            'threshold': threshold},
                           ignore_index=True)
    return df['frequency'].to_numpy(), df['threshold'].to_numpy()


def get_simulated_audiogram(folder: str, condition: str):
    mfs = {
        'nh': 'aux__an_mdl_human__thres.mat',
        'np_23ohc_13ihc': 'aux__mdl_h_thres__gap_detect__zeng2005__23ohc_13ihc.mat',
        'np_all_ihc': 'aux__mdl_h_thres__gap_detect__zeng2005__all_ihc.mat',
        'np_all_ohc': 'aux__mdl_h_thres__gap_detect__zeng2005__all_ohc.mat',
    }
    if condition not in mfs.keys():
        raise ValueError('condition must match following keys:', mfs.keys())

    mfilename = os.path.join(folder, mfs[condition])
    with h5py.File(mfilename) as f:
        return np.array(f['struct_out']['freq_vect'])[:, 0], np.array(f['struct_out']['lvl_at_thres__smth'])[:, 0]
