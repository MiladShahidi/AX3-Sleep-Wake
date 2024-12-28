import tensorflow as tf
import numpy as np
import pandas as pd
from train import create_dataset
from utils.metrics import F1Score, PositiveRate, PredictedPositives
from config import project_config as config
import os
import re


def list_all_subject_ids(path):
    filenames = [fn.upper() for fn in os.listdir(path) if fn.find('.tfrecord') >= 0]

    subject_ids = []
    for fn in filenames:
        id = re.findall("[PHD]\d{3}", fn)[0]
        if id not in subject_ids:
            subject_ids.append(id)
        
    return subject_ids


if __name__ == '__main__':

    model_name = 'CNN-Win_3-Freq_10'
    saved_model_path = f'Model Repo/{model_name}'
    datapath = 'data/Wave-2/Tensorflow/window_21/unlabelled'
    pred_output_path = f'Results/Predictions/Wave-2/{model_name}'

    os.makedirs(pred_output_path, exist_ok=True)
    # assert len(os.listdir(pred_output_path)) == 0, f"Output directory is not empty."  # Prevents overwriting

    model = tf.keras.models.load_model(
        saved_model_path,
        custom_objects={  # These are needed to load the model. We're not calculating them here
            'F1Score': F1Score(name='F1Score'),
            'PositiveRate': PositiveRate(name='PositiveRate'),
            'PredictedPositives': PredictedPositives(name='PredictedPositives')
        }
    )

    subject_ids = list_all_subject_ids(datapath)
    
    for subject_id in subject_ids:

        print(f'Making predictions for subject {subject_id}...')
        print('-' * 40)
        
        # This may seem twiseted and unnecessary
        # But it's a backward-compatible way to get this to work with old code
        prefix = subject_id[0]
        id = int(subject_id[1:])

        test_dataset = create_dataset(
            f"{datapath}/sub_{prefix}{id:03d}*",
            compressed=True,  # Unlabelled data is saved with GZIP compression
            has_labels=False,
            repeat=False,
            batch_size=100
        )

        pred_df = model.predict(test_dataset)

        # pred_df['pred'] = np.round(np.squeeze(pred_df['pred']))  # Threshold = 0.5
        
        threshold = 0.5  # Threshold for classifying as 0 or 1
        pred_score = np.squeeze(pred_df['pred'])
        # pred_df['pred_score'] = pred_score
        pred_df['pred'] = np.round(pred_score * (0.5 / threshold))

        pred_df = pd.DataFrame(pred_df)
        pred_df = pred_df[['subject_id', 'epoch_ts', 'pred']]  # Order of columns is not guaranteed. Reordering.

        pred_df['epoch_ts'] = pd.to_datetime(pred_df['epoch_ts'].str.decode("utf-8"))
        pred_df = pred_df.sort_values('epoch_ts')

        print(f'Made predictions for {len(pred_df)} epochs.')
        
        output_filename = f"{pred_output_path}/sub_{subject_id:02d}.csv"
        print(f'Saving to {output_filename}')
        pred_df.to_csv(output_filename, index=False)

        print('-' * 40)
