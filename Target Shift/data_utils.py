import pandas as pd
import os
import numpy as np


def read_AWS_labels(path, subject_id):
    aws_df = pd.read_csv(f'{path}/mesa-sleep-{subject_id:04d}_Date_time.csv')

    aws_cols = ['Date_time', 'Sleep/Wake', 'Interval Status']
    aws_df = aws_df[aws_cols]
    aws_df['Interval Status'] = aws_df['Interval Status'].map({'ACTIVE': 0, 'REST': 0, 'REST-S': 1})
    aws_df['Sleep/Wake'] = aws_df['Sleep/Wake'].map({1: 0, 0: 1})  # This one is recorded the "wrong" way
    aws_df = aws_df.rename({
        'Date_time': 'AWS time',
        'Sleep/Wake': 'AWS Sleep',
        'Interval Status': 'AWS Interval Status',
        }, axis=1)
    aws_df['AWS time'] = pd.to_datetime(aws_df['AWS time'])

    return aws_df


def read_PSG_labels(path, subject_id):
    labels_df = pd.read_csv(f'{path}/SDRI001_PSG_Sleep profile_{subject_id:03d}V4_N1.txt', skiprows=1, delimiter=';', header=None)
    labels_df = labels_df.rename({0: 'epoch_ts', 1: 'PSG Sleep'}, axis=1)
    labels_df['PSG Sleep'] = labels_df['PSG Sleep'].str.strip()  # remove extra spaces
    
    all_label_values = ['Wake', 'N1', 'N2', 'A', 'N3', 'REM', 'Artefact']
    known_labels = labels_df['PSG Sleep'].isin(all_label_values)
    assert (known_labels.all()), f"Encountered unknown label(s): {pd.unique(labels_df.loc[~known_labels, 'PSG Sleep'])}"

    missing_fltr = labels_df['PSG Sleep'].isin(['A', 'Artefact'])
    missing_pct = missing_fltr.mean()
    # print(f'Dropping missing epochs ({round(missing_pct * 100, 2)}%)')
    labels_df = labels_df[~missing_fltr]
    
    labels_df['epoch_ts'] = labels_df['epoch_ts'].apply(lambda ts: ts.split(',')[0])  # There is a weird ",000" at the end of timestamps
    
    labels_df['epoch_ts'] = labels_df['epoch_ts'].str.strip()
    labels_df['epoch_ts'] = pd.to_datetime(labels_df['epoch_ts'], dayfirst=True)

    labels_df['PSG Sleep'] = labels_df['PSG Sleep'].map(lambda l: 0 if l == 'Wake' else 1)
    labels_df['PSG Sleep'] = labels_df['PSG Sleep'].astype(np.float32)  # TF requires labels to be float

    return labels_df


def read_sleep_dairies(path):
    sleep_diary_df = pd.DataFrame()
    for filename in [f for f in os.listdir(path) if f.endswith('csv')]:
        if filename.find('nap') >= 0:
            continue
        df = pd.read_csv(f'{path}/{filename}')
        sleep_diary_df = pd.concat([sleep_diary_df, df])

    # reading the extra nap diaries
    nap_df = pd.read_csv(f'{path}/SRCDRI001_Sleep Diary 019-036_nap.csv')
    nap_df = nap_df.rename(columns={
        'date_startnap': 'date_gotosleep',
        'date_endnap': 'date_finalawake',
        'nap_start': 'gotosleep',
        'nap_end': 'finalawake'   
    }).drop(columns=['nap times'])
    sleep_diary_df = pd.concat([sleep_diary_df, nap_df])
    sleep_diary_df = sleep_diary_df.sort_values(['participantNo', 'date_gotosleep']).reset_index(drop=True)

    sleep_diary_df['sleep_start'] = pd.to_datetime(sleep_diary_df['date_gotosleep'] + ' ' + sleep_diary_df['gotosleep'])
    sleep_diary_df['sleep_end'] = pd.to_datetime(sleep_diary_df['date_finalawake'] + ' ' + sleep_diary_df['finalawake'])
    sleep_diary_df = sleep_diary_df[['participantNo', 'sleep_start', 'sleep_end']]
    return sleep_diary_df
