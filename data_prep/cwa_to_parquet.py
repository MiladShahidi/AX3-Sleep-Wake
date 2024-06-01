# Pickles are faster to read. I convert the cwa files into pickle so they can be read faster
# We need to read this data often. This one-off conversion speeds up future reads

import os
import sys
from datetime import datetime


here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils.data_utils import read_AX3_cwa, process_AX3_raw_data
from config import project_config as config


if __name__ == '__main__':

    project_root = '/Users/sshahidi/PycharmProjects/Sleep-Wake'
    cwa_data_path = f'{project_root}/data/cwa'
    output_path = f'{project_root}/data/Parquet'

    os.makedirs(output_path, exist_ok=True)
    assert len(os.listdir(output_path)) == 0, "Output directory is not empty."  # Avoid writing next to existing data files
    
    for subject_id in config['subject_ids']:

        t1 = datetime.now()
        print(f'Subject ID: {subject_id}')
        print('-' * 40)

        features_df = read_AX3_cwa(cwa_data_path, subject_id)

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
        features_df.to_parquet(f'{output_path}/AX3_sub_{subject_id:02d}.parquet', compression=None)

        print(datetime.now() - t1)
        
        print('-' * 40)
