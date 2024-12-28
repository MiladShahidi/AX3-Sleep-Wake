# Parquets are faster to read. I convert the cwa files into parquet so they can be read faster
# We need to read this data often. This one-off conversion speeds up future reads

import os
import sys
from datetime import datetime
import re


here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../..'))

from utils.data_utils import read_AX3_cwa, process_AX3_raw_data
from config import project_config as config


def list_all_subject_ids(path):
    filenames = [fn.upper() for fn in os.listdir(path) if fn.endswith('.cwa') or fn.endswith('.parquet')]

    subject_ids = []
    for fn in filenames:
        id = re.findall("[PHD]\d{3}", fn)[0]
        if id not in subject_ids:
            subject_ids.append(id)
        
    return subject_ids


if __name__ == '__main__':

    project_root = '/Users/sshahidi/PycharmProjects/Sleep-Wake'
    cwa_data_path = f'{project_root}/data/Wave-2/cwa'
    output_path = f'{project_root}/data/Wave-2/Parquet'

    os.makedirs(output_path, exist_ok=True)
    # assert len(os.listdir(output_path)) == 0, "Output directory is not empty."  # Avoid writing next to existing data files
    
    subject_ids = list_all_subject_ids(cwa_data_path)
    done_files = list_all_subject_ids(output_path)

    for subject_id in subject_ids:
        
        if subject_id in done_files:
            print(f"Skipping {subject_id}")
            continue
            
        # This may seem twiseted and unneccessary
        # But it's a backward-compatible way to get this to work with old code
        prefix = subject_id[0]
        id = int(subject_id[1:])

        t1 = datetime.now()
        print(f'Subject ID: {subject_id}')
        print('-' * 40)

        features_df = read_AX3_cwa(cwa_data_path, subject_id=id, subject_prefix=prefix)

        features_df = features_df.rename(columns={
            'time': 'epoch_ts',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'temperature': 'Temp'
        })

        features_df = process_AX3_raw_data(features_df,
                                           normalise_columns=['X', 'Y', 'Z', 'Temp'],
                                           round_timestamps=False)

        print('Writing to parquet...')
        features_df.to_parquet(f'{output_path}/AX3_sub_{subject_id}.parquet', compression=None)

        print(datetime.now() - t1)
        
        print('-' * 40)
