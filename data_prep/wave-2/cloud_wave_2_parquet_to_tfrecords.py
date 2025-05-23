import pickle
import pandas as pd
import numpy as np
import os
import re
import sys
from functools import reduce
from datetime import datetime
import subprocess
import datetime as dt
import string


here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils.data_utils import (
    read_parquet_AX3_epochs,
    read_PSG_labels
)
from utils.helpers import list_all_subject_ids


# This enables importing modules from the parent directory
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from utils.tfrecord_utils import pandas_to_tf_seq_example_list, write_to_tfrecord
    

def log_to_file(filename, msg):
    f = open(filename, "a")
    f.write(f"{dt.datetime.strftime(dt.datetime.now(), format='%d/%m/%y %H:%M:%S')} {msg}\n ")
    f.close()


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
    
    WINDOW_SIZE = 21

    # project_root = '/Users/sshahidi/PycharmProjects/Sleep-Wake'
    parquet_epoch_data_path = "gs://sleep-wake/data/Wave-2/Parquet"
    
    # labels_path = f'{project_root}/data/PSG-Labels'
    labels_path = "gs://sleep-wake/data/PSG-Labels"
    local_output_path = "."
    output_path = f"gs://sleep-wake/data/Wave-2/Tensorflow/window_{WINDOW_SIZE}"
    
    log_filename = "log.txt"

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
            pass
            # os.makedirs(output_paths[dataset_type], exist_ok=True)
            # assert len(os.listdir(output_paths[dataset_type])) == 0, f"Output directory is not empty ({dataset_type})."  # Prevents overwriting

    subject_ids = list_all_subject_ids(parquet_epoch_data_path, 'parquet')
    done_files = list_all_subject_ids(output_path + '/unlabelled', 'tfrecord.gz')

    for subject_id in subject_ids:
        
        if subject_id in done_files:
            log_to_file(log_filename, f"Skipping {subject_id}")
            continue

        # if subject_id in ["H005", "H006"]:
        #     continue

        start_time = datetime.now()

        # This may seem twisted and unneccessary
        # But it's a backward-compatible way to get this to work with old code
        prefix = subject_id[0]
        id = int(subject_id[1:])
        
        log_to_file(log_filename, f'Subject ID: {subject_id}')
        log_to_file(log_filename, '-' * 20)

        log_to_file(log_filename, 'Reading Parquet recordings...')
        features_df = read_parquet_AX3_epochs(parquet_epoch_data_path, subject_id=id, subject_prefix=prefix, round_timestamps=True)

        try:
            log_to_file(log_filename, 'Reading PSG labels...')  # These only exist for wave where wubsject ids didn't have the prefix letter
            psg_labels_df = read_PSG_labels(labels_path, subject_id=id)
            psg_labels_df = psg_labels_df.rename(columns={'PSG Sleep': 'label'})
        except FileNotFoundError:
            psg_labels_df = None
            log_to_file(log_filename, f"No labels found for {subject_id}")
        except Exception as e:
            raise e

        log_to_file(log_filename, 'Joining features and labels...')
        # labelled_data, unlabelled_data = join_features_and_labels(features_df, labels_df=None)
        log_to_file(log_filename, f"{len(features_df)} rows in features_df")
        labelled_data, unlabelled_data = join_features_and_labels(features_df, psg_labels_df)

        log_to_file(log_filename, f"{len(unlabelled_data)} rows in unlabelled_data")
        if labelled_data is not None:
            log_to_file(log_filename, f"{len(labelled_data)} rows in labelled_data")
        else:
            log_to_file(log_filename, "labelled_data is empty")

        datasets = {
            LABELLED: labelled_data,
            UNLABELLED: unlabelled_data
        }

        for dataset_type in write_flag.keys():
            if write_flag[dataset_type]:
                print(f'Processing {dataset_type} data...')
                
                log_to_file(log_filename, '\tCreating windowed data...')
                datasets[dataset_type]['epoch_ts'] = datasets[dataset_type]['epoch_ts'].astype('str')  # TF Example, etc. don't support datetime

                n_chunks = 3
                breakpoints = np.round(np.linspace(0, len(datasets[dataset_type]) + 1, n_chunks+1))
                for chunk, (from_i, to_i) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
                    log_to_file(log_filename, f"Chunk {chunk+1}/{n_chunks}: {from_i} - {to_i-1}")
                    chunk_df = datasets[dataset_type].loc[from_i:(to_i-1), :].copy(deep=True)
                    
                    chunk_df = create_windowed_df(chunk_df, window_size=WINDOW_SIZE)

                    # In principle, it's not necessary to group by all of the following
                    # But the converter function converts the values of groupby columns as scalars (not lists)
                    # So, we group by all constant columns so that they'll be converted into scalars
                    groupby_cols = ['subject_id', 'central_epoch_id', 'central_epoch_ts']
                    if dataset_type == LABELLED:
                        groupby_cols += ['label']
                    
                    log_to_file(log_filename, '\tConverting to TFRecords...')
                    chunk_df = pandas_to_tf_seq_example_list(chunk_df, groupby_cols)

                    log_to_file(log_filename, f'\tWriting TFRecords...')
                    
                    write_to_tfrecord(
                        chunk_df,
                        local_output_path,
                        f'sub_{prefix}{id:03d}_{string.ascii_lowercase[chunk]}_',
                        compression='GZIP' if dataset_type==UNLABELLED else None,  # Unlabelled files are large. Compressing
                        records_per_shard=10000
                        )

                upload_cmd = f"gcloud storage mv *.tfrecord.gz {output_paths[dataset_type]}/"
                subprocess.run(upload_cmd.split(" "))

        log_to_file(log_filename, f'Took {datetime.now() - start_time}')
        log_to_file(log_filename, '*'*80)
        break
