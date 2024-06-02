import pandas as pd
from config import project_config as config
import os


if __name__ == '__main__':

    output_path = f'Results/merged_sources'
    aws_labels_path = 'data/AWS-Labels'
    pred_path = 'Predictions/Attn'
    biobank_pred_path = 'data/Toolbox Outputs/Timeseries (predictions)'

    os.makedirs(output_path, exist_ok=True)

    for id in config['subject_ids']:

        # AWS
        aws_df = pd.read_csv(f'{aws_labels_path}/mesa-sleep-{id:04d}_Date_time.csv')

        aws_cols = ['Date_time', 'Sleep/Wake', 'Interval Status', 'Off-Wrist Status']
        aws_df = aws_df[aws_cols]
        aws_df['Interval Status'] = aws_df['Interval Status'].map({'ACTIVE': 0, 'REST': 0, 'REST-S': 1})
        aws_df['Sleep/Wake'] = aws_df['Sleep/Wake'].map({1: 0, 0: 1})  # This one is recorded the "wrong" way
        aws_df = aws_df.rename({
            'Date_time': 'AWS time',
            'Sleep/Wake': 'AWS Sleep',
            'Interval Status': 'Aux AWS Sleep',
            }, axis=1)
        aws_df['AWS time'] = pd.to_datetime(aws_df['AWS time'])

        # Our predictions
        pred_df = pd.read_csv(f'{pred_path}/sub_{id:02d}.csv')
        pred_df['epoch_ts'] = pd.to_datetime(pred_df['epoch_ts'])
        pred_df['time_in_minutes'] = pred_df['epoch_ts'].dt.floor('min')

        # Biobank predictions
        biobank_df = pd.read_csv(f'{biobank_pred_path}/biobank_{id:02d}.csv')

        biobank_cols = ['time', 'sleep']
        biobank_df = biobank_df[biobank_cols]
        biobank_df = biobank_df.rename(columns={'sleep': 'Biobank Sleep'})
        
        # I don't know how to properly parse the timestamps from the predictions
        # This is a quick hack to remove time zone label
        biobank_df['time'] = biobank_df['time'].apply(lambda s: " ".join(s.split(" ")[:2])) # Remove time zone string
        biobank_df['time'] = pd.to_datetime(biobank_df['time']).dt.tz_localize(None)  # Remove time zone
        biobank_df['time'] = biobank_df['time'].dt.floor('30s')
        biobank_df = biobank_df.rename(columns={'time': 'biobank_time'})
        
        # Joining
        biobank_and_us_df = pd.merge(
            left=pred_df,
            right=biobank_df,
            left_on='epoch_ts',
            right_on='biobank_time',
            how='left'
            )
        
        joint_df = pd.merge(
            left=biobank_and_us_df,
            right=aws_df,
            left_on='time_in_minutes',
            right_on='AWS time',
            how='left'
            ).drop(columns=['time_in_minutes'])
        
        
        joint_df.to_csv(f'{output_path}/sub_{id:02d}.csv', index=False)