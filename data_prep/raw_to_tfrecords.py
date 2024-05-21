import pickle
import pandas as pd
import numpy as np
import os
import re
import sys

# This enables importing modules from the parent directory
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from data_utils import pandas_to_tf_seq_example_list, write_to_tfrecord


def get_subject_id(filename):
    # Assumes raw data files are name [blah]sub[subject_number][blah].pkl
    filename = os.path.basename(filename)  # Remove path
    return int(re.split(r'\D+', filename.split('sub')[1])[0])


def read_subject_features(raw_data_file):
    subject_id = get_subject_id(raw_data_file)
    with open(raw_data_file, 'rb') as f:
        subject_data = pickle.load(f)

    subject_data = subject_data[['Label', 'X', 'Y', 'Z']]
    subject_data = subject_data.rename({'Label': 'epoch_ts'}, axis=1)
    subject_data['epoch_ts'] = subject_data['epoch_ts'].str.strip()
    subject_data['epoch_ts'] = pd.to_datetime(subject_data['epoch_ts'])
    subject_data = subject_data.sort_values('epoch_ts')
    
    cols = list(subject_data.columns)
    subject_data['subject_id'] = subject_id
    subject_data['epoch_id'] = range(len(subject_data))
    # reorder columns
    subject_data = subject_data[['subject_id', 'epoch_id'] + cols]

    return subject_data


def read_subject_labels(label_file):
    labels_df = pd.read_csv(label_file, skiprows=1, delimiter=';', header=None)
    labels_df = labels_df.rename({0: 'epoch_ts', 1: 'label'}, axis=1)
    labels_df['label'] = labels_df['label'].str.strip()  # remove extra spaces
    
    all_label_values = ['Wake', 'N1', 'N2', 'A', 'N3', 'REM', 'Artefact']
    known_labels = labels_df['label'].isin(all_label_values)
    assert (known_labels.all()), f"Encountered unknown label(s): {pd.unique(labels_df.loc[~known_labels, 'label'])}"

    missing_fltr = labels_df['label'].isin(['A', 'Artefact'])
    missing_pct = missing_fltr.mean()
    print(f'Dropping missing epochs ({round(missing_pct * 100, 2)}%)')
    labels_df = labels_df[~missing_fltr]
    
    labels_df['epoch_ts'] = labels_df['epoch_ts'].apply(lambda ts: ts.split(',')[0])  # There is a weird ",000" at the end of timestamps
    
    labels_df['epoch_ts'] = labels_df['epoch_ts'].str.strip()
    labels_df['epoch_ts'] = pd.to_datetime(labels_df['epoch_ts'], dayfirst=True)

    labels_df['label'] = labels_df['label'].map(lambda l: 1 if l == 'Wake' else 0)
    labels_df['label'] = labels_df['label'].astype(np.float32)  # TF requires labels to be float

    return labels_df


def join_features_and_labels(features_df, labels_df):
    
    subject_data = pd.merge(
        left=labels_df,
        right=features_df,
        left_on='epoch_ts',
        right_on='epoch_ts',
        how='left'
        )

    unmatched_pct = subject_data['X'].isna().mean()

    return subject_data, unmatched_pct


if __name__ == '__main__':
    project_root = '/Users/sshahidi/PycharmProjects/Sleep'
    raw_data_path = f'{project_root}/data/raw/Recordings'
    labels_path = f'{project_root}/data/raw/Labels'
    output_path = f'{project_root}/data/processed/test_1/mixed_1-25'

    raw_data_files = [f'{raw_data_path}/{filename}' for filename in os.listdir(raw_data_path) if filename.endswith('pkl')]

    os.makedirs(output_path, exist_ok=True)
    assert len(os.listdir(output_path)) == 0, "Output directory is not empty."  # Avoid writing next to existing data files
    
    for raw_data_file in sorted(raw_data_files):

        subject_id = get_subject_id(raw_data_file)
        
        print('Subject ID: ', subject_id)

        features_df = read_subject_features(raw_data_file)
        print(features_df['epoch_ts'].head())
        break

        labels_filename = f'{labels_path}/SDRI001_PSG_Sleep profile_{subject_id:03d}V4_N1.txt'
        labels_df = read_subject_labels(labels_filename)

        subject_data, unmatched = join_features_and_labels(features_df, labels_df)

        if unmatched > 0:  # Unmatched rows will have missing feature values
            print(f"*** WARNING: Missing data for {round(unmatched * 100)}% of labels ***")
        else:
            print('Complete match between features and labels')
        
        #Shuffle
        subject_data = subject_data.sample(frac=1).reset_index(drop=True)

        print(f"Positive samples: {subject_data['label'].sum()}")
        print(f"Total num. of samples: {len(subject_data)}")

        print('Converting...')
        subject_data['epoch_ts'] = subject_data['epoch_ts'].astype('str')  # TF Example, etc. don't support datetime
        tf_data = pandas_to_tf_seq_example_list(subject_data, ['subject_id', 'epoch_ts'])
        
        print('Writing...')
        write_to_tfrecord(tf_data, output_path, f'sub_{subject_id}', records_per_shard=2000)
        print('*'*80)
