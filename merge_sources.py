import pandas as pd
from config import project_config as config
import os
from utils.data_utils import read_AWS_labels, read_PSG_labels
from functools import reduce


if __name__ == '__main__':

    output_path = f'Results/merged_indicators'
    aws_labels_path = 'data/AWS-Labels'
    psg_labels_path = 'data/PSG-Labels'
    model = 'CNN-Win_3-Freq_10'
    predictions_path = 'Results/Predictions'
    cv_predictions_path = f'Results/Predictions/CV/{model}'
    biobank_pred_path = 'data/Toolbox Outputs/Timeseries (predictions)'

    os.makedirs(output_path, exist_ok=True)

    # CV predictions are bundled together in one file for each fold
    # It's faster to read them only once before the next loop
    cv_preds_df = pd.DataFrame()
    for filename in [f for f in os.listdir(cv_predictions_path) if f.endswith('csv')]:
        fold_preds_df = pd.read_csv(f'{cv_predictions_path}/{filename}')
        cv_preds_df = pd.concat([cv_preds_df, fold_preds_df])
    
    cv_preds_df = cv_preds_df.rename({'pred': f'pred_{model}'}, axis=1)
    cv_preds_df.insert(len(cv_preds_df.columns), 'is_cv_prediction', 1)
    cv_preds_df['epoch_ts'] = pd.to_datetime(cv_preds_df['epoch_ts'])
    # Todo: don't write this column to cv files, and remove this
    if 'PSG Sleep' in cv_preds_df.columns:
        cv_preds_df = cv_preds_df.drop('PSG Sleep', axis=1)

    for id in config['subject_ids']:
        print(f'Subject {id:02d}', end='\r')

        valid_days = pd.read_csv('data/participation_dates.csv')
        start_timestamp = valid_days.loc[valid_days['subject_id'] == id, 'start_timestamp'].values[0]
        end_timestamp = valid_days.loc[valid_days['subject_id'] == id, 'end_timestamp'].values[0]

        all_epochs = pd.DataFrame({
            'epoch_ts': pd.date_range(start_timestamp, end_timestamp, freq='30s'),
        })
        all_epochs['epoch_ts_minutes'] = all_epochs['epoch_ts'].dt.floor('min')

        # # # # # # # # # # # # 
        # # # # Labels: AWS and PSG
        # # # # # # # # # # # # 

        aws_df = read_AWS_labels(aws_labels_path, id)
        psg_df = read_PSG_labels(psg_labels_path, id)

        aws_df = aws_df.sort_values('AWS time')

        # # # # # # # # # # # # 
        # # # # Our predictions
        # # # # # # # # # # # # 
        # pred_dfs = []
        preds_df = pd.read_csv(f'{predictions_path}/{model}/sub_{id:02d}.csv')
        preds_df['epoch_ts'] = pd.to_datetime(preds_df['epoch_ts'])
        preds_df = preds_df[['epoch_ts', 'pred']]
        preds_df = preds_df.rename({'pred': f'pred_{model}'}, axis=1)
        # pred_dfs.append(pred_df)

        # This part was used when I was loading predictions of multiple models
        # pred_merge_fn = lambda l, r: pd.merge(
        #     left=l,
        #     right=r,
        #     on='epoch_ts',
        #     how='inner'  # These files all have the exact same set of timestamp. So inner and left will have the same result
        # )
        # model_preds_df = reduce(pred_merge_fn, pred_dfs)

        # # Our CV Predictions
        # print(cv_preds_df)
        subject_cv_preds_df = cv_preds_df[cv_preds_df['subject_id'] == id]
        subject_cv_preds_df = subject_cv_preds_df[['epoch_ts', f'pred_{model}', 'is_cv_prediction']]
        
        # Stack CV and non-CV predictions on top of each other

        # Check for overlap
        overlap_1 = (preds_df['epoch_ts'].isin(subject_cv_preds_df['epoch_ts'])).any()
        overlap_2 = (subject_cv_preds_df['epoch_ts'].isin(preds_df['epoch_ts'])).any()
        if overlap_1 or overlap_2:
            # CV predictions are done on training data (k-fold cv)
            # The rest of epochs are from test data. There should be no overlap between the two sets
            raise ValueError("Overlapping epochs found between CV and non-CV predictions")
        
        preds_df = pd.concat([preds_df, subject_cv_preds_df])

        # # # # # # # # # # # # 
        # # # # Biobank predictions
        # # # # # # # # # # # # 
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

        # # # # # # # # # # # # 
        # # # # Merge all
        # # # # # # # # # # # # 
        merge_specs = {
            'Models':{'df': preds_df, 'right_key': 'epoch_ts', 'left_key': 'epoch_ts'},
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