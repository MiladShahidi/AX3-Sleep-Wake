import pandas as pd
from config import project_config as config
import os
from utils.data_utils import read_AWS_labels, read_PSG_labels
from functools import reduce


if __name__ == '__main__':

    output_path = f'Results/merged_indicators'
    aws_labels_path = 'data/AWS-Labels'
    psg_labels_path = 'data/PSG-Labels'
    models = ['AWS-CNN', 'PSG-CNN']
    predictions_path = 'Results/Predictions'
    biobank_pred_path = 'data/Toolbox Outputs/Timeseries (predictions)'

    os.makedirs(output_path, exist_ok=True)

    for id in config['subject_ids']:
        print(f'Subject {id:02d}', end='\r')

        valid_days = pd.read_csv('data/participation_dates.csv')
        start_timestamp = valid_days.loc[valid_days['subject_id'] == id, 'start_timestamp'].values[0]
        end_timestamp = valid_days.loc[valid_days['subject_id'] == id, 'end_timestamp'].values[0]

        all_epochs = pd.DataFrame({
            'epoch_ts': pd.date_range(start_timestamp, end_timestamp, freq='30s'),
        })
        all_epochs['epoch_ts_minutes'] = all_epochs['epoch_ts'].dt.floor('min')

        # Labels
        aws_df = read_AWS_labels(aws_labels_path, id)
        psg_df = read_PSG_labels(psg_labels_path, id)

        aws_df = aws_df.sort_values('AWS time')

        # Our predictions
        pred_dfs = []
        for model in models:
            pred_df = pd.read_csv(f'{predictions_path}/{model}/sub_{id:02d}.csv')
            pred_df['epoch_ts'] = pd.to_datetime(pred_df['epoch_ts'])
            pred_df = pred_df[['epoch_ts', 'pred']]
            pred_df = pred_df.rename({'pred': f'pred_{model}'}, axis=1)
            pred_dfs.append(pred_df)

        pred_merge_fn = lambda l, r: pd.merge(
            left=l,
            right=r,
            on='epoch_ts',
            how='inner'  # These files all have the exact same set of timestamp. So inner and left will have the same result
        )
        
        model_preds_df = reduce(pred_merge_fn, pred_dfs)

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

        # model_preds_df['epoch_ts_in_minutes'] = model_preds_df['epoch_ts'].dt.floor('min')

        merge_specs = {
            'Models':{'df': model_preds_df, 'right_key': 'epoch_ts', 'left_key': 'epoch_ts'},
            'Biobank': {'df': biobank_df, 'right_key': 'biobank_time', 'left_key': 'epoch_ts'},
            'PSG': {'df': psg_df, 'right_key': 'epoch_ts', 'left_key': 'epoch_ts'},
            'AWS': {'df': aws_df, 'right_key': 'AWS time', 'left_key': 'epoch_ts_minutes'},  # AWS timestamp are one per minute
        }
        
        merged_df = all_epochs.copy()
        for merge_spec in merge_specs.values():
            merged_df = pd.merge(
                left=merged_df,
                right=merge_spec['df'],
                left_on=merge_spec['left_key'],
                right_on=merge_spec['right_key'],
                how='left'
            )
        
        merged_df = merged_df.drop(['biobank_time', 'AWS time'], axis=1)

        merged_df.to_csv(f'{output_path}/sub_{id:02d}.csv', index=False)

    print('\nDone.')