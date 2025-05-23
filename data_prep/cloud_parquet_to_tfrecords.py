import pandas as pd
import numpy as np
import os
import re
import sys
from functools import reduce
from datetime import datetime
import subprocess

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils.data_utils import (
    read_parquet_AX3_epochs,
    read_PSG_labels
)
from config import project_config as config

# This enables importing modules from the parent directory
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from utils.tfrecord_utils import pandas_to_tf_seq_example_list, write_to_tfrecord
    

def join_features_and_labels(features_df, labels_df):
        
    if labels_df is not None:
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

        return labelled_df, unlabelled_df.drop(columns=['label'])
    else:  # This is a backward-compatible fix
        return None, features_df  # No labels provided


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
        win_df['central_epoch_ts'] = win_df.groupby(['subject_id', 'window_id'])['epoch_ts'].transform(fill_with_centre_value)
        if 'label' in win_df.columns:  # In case we're processing unlabelled data
            win_df['label'] = win_df.groupby(['subject_id', 'window_id'])['label'].transform(fill_with_centre_value)

        win_df = win_df.drop(['window_id', 'row_num'], axis=1)
        windows_df = pd.concat([windows_df, win_df])
        
    return windows_df


if __name__ == '__main__':
    
    WINDOW_SIZE = 1

    project_root = '/Users/sshahidi/PycharmProjects/Sleep-Wake'
    # parquet_epoch_data_path = f'{project_root}/data/Parquet'
    parquet_epoch_data_path = "gs://sleep-wake/data"
    
    # labels_path = f'{project_root}/data/PSG-Labels'
    labels_path = "gs://sleep-wake/data/PSG-Labels"
    local_output_path = "."
    output_path = f"gs://sleep-wake/data/tf"
    
    # Constants for naming stuff
    LABELLED = 'labelled'
    UNLABELLED = 'unlabelled'

    # The follwing controls which type of dataset will be written
    write_flag = {
        LABELLED: False,
        UNLABELLED: True
    }

    assert write_flag[LABELLED] or write_flag[UNLABELLED], "At least one write flag must be set to True"

    output_paths = {
        LABELLED: f'{output_path}/{LABELLED}',
        UNLABELLED: f'{output_path}/{UNLABELLED}',
    }

    for dataset_type in write_flag.keys():
        if write_flag[dataset_type]:
            os.makedirs(output_paths[dataset_type], exist_ok=True)
            # assert len(os.listdir(output_paths[dataset_type])) == 0, f"Output directory is not empty ({dataset_type})."  # Prevents overwriting

    for subject_id in config['subject_ids']:

        start_time = datetime.now()

        print(f'Subject ID: {subject_id}')
        print('-' * 20)

        print('Reading Parquet recordings...')
        features_df = read_parquet_AX3_epochs(parquet_epoch_data_path, subject_id, round_timestamps=True)

        # print('Reading PSG labels...')
        # psg_labels_df = read_PSG_labels(labels_path, subject_id)
        # psg_labels_df = psg_labels_df.rename(columns={'PSG Sleep': 'label'})

        print('Joining features and labels...')
        labelled_data, unlabelled_data = join_features_and_labels(features_df, labels_df=None)
        # labelled_data, unlabelled_data = join_features_and_labels(features_df, psg_labels_df)

        datasets = {
            LABELLED: labelled_data,
            UNLABELLED: unlabelled_data
        }

        for dataset_type in write_flag.keys():
            if write_flag[dataset_type]:
                print(f'Processing {dataset_type} data...')
                
                print('\tConverting to TFRecords...')
                datasets[dataset_type]['epoch_ts'] = datasets[dataset_type]['epoch_ts'].astype('str')  # TF Example, etc. don't support datetime

                datasets[dataset_type] = create_windowed_df(datasets[dataset_type], window_size=WINDOW_SIZE)

                # In principle, it's not necessary to group by all of the following
                # But the converter function converts the values of groupby columns as scalars (not lists)
                # So, we group by all constant columns so that they'll be converted into scalars
                groupby_cols = ['subject_id', 'central_epoch_id', 'central_epoch_ts']
                if dataset_type == LABELLED:
                    groupby_cols += ['label']
                
                datasets[dataset_type] = pandas_to_tf_seq_example_list(datasets[dataset_type], groupby_cols)

                print('\tWriting TFRecords...')
                write_to_tfrecord(datasets[dataset_type],
                                  local_output_path,
                                  f'sub_{subject_id:03d}',
                                  compression='GZIP' if dataset_type==UNLABELLED else None,  # Unlabelled files are large. Compressing
                                  records_per_shard=10000)
                
                upload_cmd = f"gcloud storage mv *.tfrecord.gz {output_paths[dataset_type]}/"
                subprocess.run(upload_cmd.split(" "))

        print('Took ', datetime.now() - start_time)
        print('*'*80)

        break