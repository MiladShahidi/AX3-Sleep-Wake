import pickle
import pandas as pd
import numpy as np
import os
import re
import sys

# This enables importing modules from the parent directory
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from utils.tfrecord_utils import pandas_to_tf_seq_example_list, write_to_tfrecord


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
    
    # reorder columns
    subject_data = subject_data[['subject_id'] + cols]

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
        ).sort_values('epoch_ts')

    unmatched_pct = subject_data['X'].isna().mean()

    return subject_data, unmatched_pct


def create_windowed_df(df, window_size):

    def fill_with_centre_value(x):
        assert len(x) % 2 == 1  # Window length must be odd
        mid_value = x.values[(len(x) - 1) // 2]
        x[:] = mid_value
        return x

    # Creates examples containing window_len epochs, where the one in the middle is meant to be 
    # classified.
    # Example: window_len = 3, put together epochs as (t-1, t, t+1) and attaches the label for epoch t
    # This is done by going over (each subject in) the data frame 3 times (window_len):
    # iteration 1 windows: 0-2, 3-5, 6-8, ...
    # iteration 2 windows: 1-3, 4-6, 7-9, ...
    # iteration 3 windows: 2-4, 5-7, 8-10, ...
    # This covers all possible windows of length 3
    df = df.sort_values(['subject_id', 'epoch_ts'])
    df['epoch_id'] = range(len(df))

    windows_df = pd.DataFrame()
    for window_shift in range(window_size):
        win_df = df.copy()
        
        win_df['row_num'] = win_df.groupby('subject_id').cumcount()
        
        # Drop the first window_shift rows per subject. These cannot form a full window
        incomplete_top_rows_fltr = win_df['row_num'] < window_shift
        win_df = win_df[~incomplete_top_rows_fltr]
        
        # Dropping rows changed the row_num. Numbering the rows again
        win_df['row_num'] = win_df.groupby('subject_id').cumcount()

        # Drop the last few rows that cannot form a full window, if num of epochs is not a multiple of window_len
        win_df['window_id'] = win_df['row_num'] // window_size
        residual_epochs_fltr = win_df.groupby(['subject_id', 'window_id'])['row_num'].transform(len).eq(window_size)
        win_df = win_df[residual_epochs_fltr]

        # Each window is intended for classyfying only its central epoch. So, the label for each window is the label of its central epoch
        win_df['central_epoch_id'] = win_df.groupby(['subject_id', 'window_id'])['epoch_id'].transform(fill_with_centre_value)
        win_df['label'] = win_df.groupby(['subject_id', 'window_id'])['label'].transform(fill_with_centre_value)
        # win_df['epoch_ts'] = win_df.groupby(['subject_id', 'window_id'])['epoch_ts'].transform(fill_with_centre_value)

        win_df = win_df.drop(['window_id', 'row_num'], axis=1)
        windows_df = pd.concat([windows_df, win_df])
        
    return windows_df


if __name__ == '__main__':
    WINDOW_SIZE = 1
    
    project_root = '/Users/sshahidi/PycharmProjects/Sleep-Wake'
    raw_data_path = f'{project_root}/data/raw/Recordings'
    labels_path = f'{project_root}/data/raw/Labels'
    output_path = f'{project_root}/data/processed/window_{WINDOW_SIZE}'

    raw_data_files = [f'{raw_data_path}/{filename}' for filename in os.listdir(raw_data_path) if filename.endswith('pkl')]

    os.makedirs(output_path, exist_ok=True)
    assert len(os.listdir(output_path)) == 0, "Output directory is not empty."  # Avoid writing next to existing data files
    
    for raw_data_file in sorted(raw_data_files):

        subject_id = get_subject_id(raw_data_file)
        
        print('Subject ID: ', subject_id)

        features_df = read_subject_features(raw_data_file)

        labels_filename = f'{labels_path}/SDRI001_PSG_Sleep profile_{subject_id:03d}V4_N1.txt'
        labels_df = read_subject_labels(labels_filename)

        subject_data, unmatched = join_features_and_labels(features_df, labels_df)

        if unmatched > 0:  # Unmatched rows will have missing feature values
            print(f"*** WARNING: Missing data for {round(unmatched * 100)}% of labels ***")
        else:
            print('Complete match between features and labels')
        
        print(f"Positive samples: {subject_data['label'].sum()}")
        print(f"Total num. of samples: {len(subject_data)}")

        print('Converting...')
        
        subject_data['epoch_ts'] = subject_data['epoch_ts'].astype('str')  # TF Example, etc. don't support datetime
        
        windowed_df = create_windowed_df(subject_data, window_size=WINDOW_SIZE)
        
        # In princeiple, it's not necessary to group by all of the following
        # But the converter function converts the values of groupby columns as scalars (not lists)
        # So, we group by all constant columns so that they'll be converted into scalars
        tf_data = pandas_to_tf_seq_example_list(windowed_df, ['subject_id', 'central_epoch_id', 'label'])

        print('Writing...')
        write_to_tfrecord(tf_data, output_path, f'sub_{subject_id:02d}', records_per_shard=10000)
        print('*'*80)
        