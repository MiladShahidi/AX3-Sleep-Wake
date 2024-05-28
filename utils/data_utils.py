import pandas as pd
import re
import os
import pickle
import numpy as np


def read_subject_labels(label_file):
    labels_df = pd.read_csv(label_file, skiprows=1, delimiter=';', header=None)
    labels_df = labels_df.rename({0: 'epoch_ts', 1: 'label'}, axis=1)
    labels_df['label'] = labels_df['label'].str.strip()  # remove extra spaces
    
    all_label_values = ['Wake', 'N1', 'N2', 'A', 'N3', 'REM', 'Artefact']
    known_labels = labels_df['label'].isin(all_label_values)
    assert (known_labels.all()), f"Encountered unknown label(s): {pd.unique(labels_df.loc[~known_labels, 'label'])}"

    missing_fltr = labels_df['label'].isin(['A', 'Artefact'])
    missing_pct = missing_fltr.mean()
    # print(f'Dropping missing epochs ({round(missing_pct * 100, 2)}%)')
    labels_df = labels_df[~missing_fltr]
    
    labels_df['epoch_ts'] = labels_df['epoch_ts'].apply(lambda ts: ts.split(',')[0])  # There is a weird ",000" at the end of timestamps
    
    labels_df['epoch_ts'] = labels_df['epoch_ts'].str.strip()
    labels_df['epoch_ts'] = pd.to_datetime(labels_df['epoch_ts'], dayfirst=True)

    labels_df['label'] = labels_df['label'].map(lambda l: 0 if l == 'Wake' else 1)
    labels_df['label'] = labels_df['label'].astype(np.float32)  # TF requires labels to be float

    return labels_df


def read_subject_features(raw_data_file):
    with open(raw_data_file, 'rb') as f:
        subject_data = pickle.load(f)

    subject_data = subject_data[['Label', 'X', 'Y', 'Z', 'Temp']]
    subject_data = subject_data.rename({'Label': 'epoch_ts'}, axis=1)
    subject_data['epoch_ts'] = subject_data['epoch_ts'].str.strip()
    subject_data['epoch_ts'] = pd.to_datetime(subject_data['epoch_ts'])
    subject_data = subject_data.sort_values('epoch_ts')
    
    return subject_data
