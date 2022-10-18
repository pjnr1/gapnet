import os

import h5py
import numpy as np
import pandas as pd

impairments = {
    'HI, PM': ('fig7_PM.csv', 'pm'),
    'HI, RT': ('fig7_RT.csv', 'rt')
}


def get_external_results(path: str) -> pd.DataFrame:
    def _convert(r):
        if isinstance(r, int):
            x = r
        else:
            x = r.replace(',', '.')
        return round(float(x), 2)
    df = pd.DataFrame(columns=['experiment_parameter',
                               'gap_threshold',
                               'impairment'])
    for impairment, (fn, method_label) in impairments.items():
        fp = os.path.join(path, fn)
        for i, row in pd.read_csv(fp, sep=',', delimiter=';', header=None).iterrows():
            level = _convert(row[0])
            threshold = _convert(row[1])
            df = df.append({'experiment_parameter': level,
                            'gap_threshold': threshold,
                            'impairment': impairment},
                           ignore_index=True)
    return df
