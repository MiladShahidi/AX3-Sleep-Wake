# Pickles are faster to read. I convert the cwa files into pickle so they can be read faster
# We need to read this data often. This one-off conversion speeds up future reads

# import pandas as pd
import os
import sys

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils.data_utils import read_AX3_cwa
from config import project_config as config


if __name__ == '__main__':

    project_root = '/Users/sshahidi/PycharmProjects/Sleep-Wake'
    cwa_data_path = f'{project_root}/data/cwa'
    output_path = f'{project_root}/data/Pickle'

    os.makedirs(output_path, exist_ok=True)
    assert len(os.listdir(output_path)) == 0, "Output directory is not empty."  # Avoid writing next to existing data files
    
    for subject_id in [id for id in range(23, 36+1) if id != 27]:
    # for subject_id in config['subject_ids']:

        print(f'Subject ID: {subject_id}')
        print('-' * 40)

        features_df = read_AX3_cwa(cwa_data_path, subject_id, round_timestamps=False)
        features_df.to_pickle(f'{output_path}/AX3_sub_{subject_id:02d}.pkl', compression=None)

        print('-' * 40)
