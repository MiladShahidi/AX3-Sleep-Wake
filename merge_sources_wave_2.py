import pandas as pd
from config import project_config as config
import os
from utils.data_utils import read_AWS_labels, read_PSG_labels
from functools import reduce
import re
from utils.helpers import list_all_subject_ids


if __name__ == '__main__':

    output_path = f'Results/merged_indicators/Wave-2'
    psg_labels_path = 'data/Wave-2/PSG-Labels'
    model = 'ds_best_on_all'
    predictions_path = 'Results/Predictions/Wave-2'
    biobank_pred_path = 'data/Wave-2/Toolbox Outputs'
    
    os.makedirs(output_path, exist_ok=True)

    subject_ids = list_all_subject_ids(f"{predictions_path}/{model}", "csv")
    
    for subject_id in subject_ids:
                
        if subject_id != 'D022':
            continue
        
        prefix = subject_id[0]
        id = int(subject_id[1:])
        print(f'Subject {subject_id}')

        valid_days = pd.read_csv('data/Wave-2/participation_dates.csv')
        start_timestamp = valid_days.loc[valid_days['subject_id'] == subject_id, 'start_timestamp'].values[0]
        end_timestamp = valid_days.loc[valid_days['subject_id'] == subject_id, 'end_timestamp'].values[0]

        all_epochs = pd.DataFrame({
            'epoch_ts': pd.date_range(start_timestamp, end_timestamp, freq='30s'),
        })
        all_epochs['epoch_ts_minutes'] = all_epochs['epoch_ts'].dt.floor('min')

        # # # # # # # # # # # # 
        # # # # Labels: PSG
        # # # # # # # # # # # # 

        psg_df = read_PSG_labels(psg_labels_path, subject_id=id, subject_prefix=prefix)

        # aws_df = aws_df.sort_values('AWS time')

        # # # # # # # # # # # # 
        # # # # Our predictions
        # # # # # # # # # # # # 
        # pred_dfs = []
        preds_df = pd.read_csv(f'{predictions_path}/{model}/sub_{subject_id}.csv')
        preds_df['epoch_ts'] = pd.to_datetime(preds_df['epoch_ts'])
        preds_df = preds_df[['epoch_ts', 'pred']]
        preds_df = preds_df.rename({'pred': f'pred_{model}'}, axis=1)
        # pred_dfs.append(pred_df)

        # # # # # # # # # # # # 
        # # # # Biobank predictions
        # # # # # # # # # # # # 
        biobank_filename = [fn for fn in os.listdir(biobank_pred_path) if fn.endswith('.csv') and (fn.find(subject_id) >= 0)][0]
        biobank_df = pd.read_csv(f'{biobank_pred_path}/{biobank_filename}')

        biobank_cols = ['time', 'sleep']
        biobank_df = biobank_df[biobank_cols]
        biobank_df = biobank_df.rename(columns={'sleep': 'Biobank Sleep'})
        
        # I don't know how to properly parse the timestamps from the predictions
        # This is a quick hack to remove time zone label
        biobank_df['time'] = biobank_df['time'].apply(lambda s: " ".join(s.split(" ")[:2])) # Remove time zone string
        
        biobank_df['time'] = pd.to_datetime(biobank_df['time'], utc=True)

        biobank_df['time'] = biobank_df['time'].dt.tz_localize(None)  # Remove time zone
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
        
        merged_df = merged_df.drop(['biobank_time'], axis=1)

        merged_df.to_csv(f'{output_path}/sub_{subject_id}.csv', index=False)

    print('\nDone.')