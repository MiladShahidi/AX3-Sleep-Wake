import tensorflow as tf
import numpy as np
import pandas as pd
from train import create_dataset
from utils.metrics import F1Score, PositiveRate, PredictedPositives
from config import project_config as config
import os


if __name__ == '__main__':

    model_name = 'AWS-CNN'
    saved_model_path = f'Model Repo/{model_name}'
    datapath = 'data/Tensorflow/window_19/unlabelled'
    pred_output_path = f'Results/Predictions/{model_name}'

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

    for subject_id in config['subject_ids']:

        print(f'Making predictions for subject {subject_id}...')
        print('-' * 40)
        
        test_dataset = create_dataset(
            f"{datapath}/sub_{subject_id:02d}*",
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
