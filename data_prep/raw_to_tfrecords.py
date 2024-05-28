import pickle
import pandas as pd
import numpy as np
import os
import re
import sys
from functools import reduce
from datetime import datetime

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils.data_utils import (
    read_subject_features,
    read_subject_labels
)

# This enables importing modules from the parent directory
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from utils.tfrecord_utils import pandas_to_tf_seq_example_list, write_to_tfrecord


def get_subject_id(filename):
    # Assumes raw data files are name [blah]sub[subject_number][blah].pkl
    filename = os.path.basename(filename)  # Remove path
    return int(re.split(r'\D+', filename.split('sub')[1])[0])


def normalize_ax3_column(df, col):
    # concatenate all epoch values to create a long list of numbers
    vals = np.array(reduce(lambda x, y: x + y, df[col].values))
    
    m = np.mean(vals)
    s = np.std(vals)
    
    def norm(l):
        np_l = np.array(l)
        np_l = np.round((np_l - m) / (s + 1e-8), 4)  # original data has 4 decimal places too
        return list(np_l)  # each row of data is a list. So this returns a list to preserve that format
    
    return df[col].apply(norm)
    

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


def write_labelled_and_unlabelled(subject_id, features_df, labels_df):
    """
    This is irrelevant to the main purpose of this script
    It splits data into labelled and unlabelled subsets and
    writes the two in separate Pickled files.
    """
    os.makedirs('temp_output/labelled', exist_ok=True)
    os.makedirs('temp_output/unlabelled', exist_ok=True)

    subject_data = pd.merge(
        left=features_df,
        right=labels_df,
        left_on='epoch_ts',
        right_on='epoch_ts',
        how='left'
        ).sort_values('epoch_ts')

    label_fltr = ~subject_data['label'].isna()
    
    labelled_df = subject_data[label_fltr]
    unlabelled_df = subject_data[~label_fltr]

    labelled_df.to_pickle(f'temp_output/labelled/labelled_AX3_sub_{subject_id:02d}.pkl', compression=None)
    unlabelled_df.to_pickle(f'temp_output/unlabelled/unlabelled_AX3_sub_{subject_id:02d}.pkl', compression=None)


def normalize_measurements(df):
    for col in ['X', 'Y', 'Z', 'Temp']:
        df[col] = normalize_ax3_column(df, col)

    return df


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
        win_df['central_epoch_ts'] = win_df.groupby(['subject_id', 'window_id'])['epoch_ts'].transform(fill_with_centre_value)

        win_df = win_df.drop(['window_id', 'row_num'], axis=1)
        windows_df = pd.concat([windows_df, win_df])
        
    return windows_df


if __name__ == '__main__':
    WINDOW_SIZE = 3
    
    project_root = '/Users/sshahidi/PycharmProjects/Sleep-Wake'
    raw_data_path = f'{project_root}/data/raw/Recordings'
    labels_path = f'{project_root}/data/raw/Labels'
    output_path = f'{project_root}/data/Tensorflow/normalised/window_{WINDOW_SIZE}'

    raw_data_files = [f'{raw_data_path}/{filename}' for filename in os.listdir(raw_data_path) if filename.endswith('pkl')]

    os.makedirs(output_path, exist_ok=True)
    # assert len(os.listdir(output_path)) == 0, "Output directory is not empty."  # Avoid writing next to existing data files
    
    for raw_data_file in sorted(raw_data_files):

        subject_id = get_subject_id(raw_data_file)
        
        print('Subject ID: ', subject_id)

        # # # # Reading features
        features_df = read_subject_features(raw_data_file)

        cols = list(features_df.columns)
        features_df['subject_id'] = subject_id        
        # reorder columns
        features_df = features_df[['subject_id'] + cols]

        labels_filename = f'{labels_path}/SDRI001_PSG_Sleep profile_{subject_id:03d}V4_N1.txt'
        
        # # # # Reading PSG labels
        labels_df = read_subject_labels(labels_filename)

        # # # # Joining
        subject_data, unmatched = join_features_and_labels(features_df, labels_df)

        print('Normalizing measurements...')
        subject_data = normalize_measurements(subject_data)

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
        tf_data = pandas_to_tf_seq_example_list(windowed_df, ['subject_id', 'central_epoch_id', 'label', 'central_epoch_ts'])

        print('Writing...')
        write_to_tfrecord(tf_data, output_path, f'sub_{subject_id:02d}', records_per_shard=10000)
        print('*'*80)

        