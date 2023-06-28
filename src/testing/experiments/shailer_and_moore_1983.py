import os
import pandas as pd


def external_results(df: pd.DataFrame, path: str):
    participants = {
        'EL': 'EL.csv',
        'GW': 'GW.csv',
        'MS': 'MS.csv',
    }

    for participant, v in participants.items():
        fp = os.path.join(path, v)
        for i, row in pd.read_csv(fp, sep=',', delimiter=';', header=None).iterrows():
            frequency = round(float(row[0].replace(',', '.')))
            threshold = round(float(row[1].replace(',', '.')), 2)

            df = df.append({'frequency': frequency,
                            'gap threshold': threshold,
                            'participant': participant},
                           ignore_index=True)
    return df
