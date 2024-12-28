import tensorflow as tf
import datetime
from utils.metrics import F1Score, PositiveRate, PredictedPositives
import os
from functools import partial
import numpy as np
from utils.data_utils import read_PSG_labels
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    classification_report
)
from sklearn.model_selection import KFold
from utils.training_utils import CustomTensorBoard
from config import project_config as config
from models import CNNModel
from utils.tfrecord_utils import create_dataset
from utils.helpers import keep_subjects


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and debug info mesasages


def train_model(
        datapath,
        train_config,
        train_val_ids,
        output_dir,
        model_nickname,
        save_checkpoints
        ):

    saved_models_dir = f'{output_dir}/savedmodels/{model_nickname}'
    tensorboard_logdir = f"{output_dir}/tb_logs/{model_nickname}"

    train_data = create_dataset(
        datapath,
        filters=[
            partial(keep_subjects, subject_ids=train_val_ids),
            lambda x, y: x['central_epoch_id'] % 10 != 0
            ],
        batch_size=128
        )

    val_data = create_dataset(
        datapath,
        filters=[
            partial(keep_subjects, subject_ids=train_val_ids),
            lambda x, y: x['central_epoch_id'] % 10 == 0
            ],
        batch_size=100
        )
    
    callbacks = [
        CustomTensorBoard(log_dir=f"{tensorboard_logdir}"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=0)
    ]
    if save_checkpoints:
        callbacks += [tf.keras.callbacks.ModelCheckpoint(f'{saved_models_dir}', monitor='val_loss', save_best_only=True)]

    # model = NewCNNModel(down_sample_by=60, window_size=train_config['window_size'])
    model = CNNModel(
        down_sample_by=train_config['down_sample_by'],
        num_conv_filters=train_config['num_conv_filters'],
        num_attention_heads=train_config['num_attention_heads'],
        stride=train_config['stride'],
        window_size=train_config['window_size'],
        )

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),  # Legacy is because the current one runs slow on M1/M2 macs
        loss={'pred': tf.keras.losses.BinaryCrossentropy(name='Loss')},
        # loss={'pred': tf.keras.losses.BinaryFocalCrossentropy(name='Loss')},
        metrics={'pred': [
            # tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision'),
            F1Score(average='macro'),
            F1Score(average='binary'),
            PositiveRate(name='PositiveRate'),
            PredictedPositives(name='PredictedPositives')
            ]}
            )

    model.fit(
        train_data,
        # class_weight={0: 0.6, 1: 0.4},
        epochs=1000,
        steps_per_epoch=100,
        validation_data=val_data,
        validation_steps=100,
        callbacks=callbacks
        )

    return model


def training_main(train_config):
        
    datapath = f"data/Tensorflow/window_{train_config['window_size']}/labelled"
    psg_labels_path = 'data/PSG-Labels'
    output_dir = 'training_output'
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cv_preds_path = f'Results/Predictions/CV/{timestamp}'
    logs_filename = 'training_logs.csv'

    training_log = pd.DataFrame({
        'timestamp': timestamp,
        'Window': train_config['window_size'],
        'Freq': 100 // train_config['down_sample_by'],
    }, index=[0])
    training_log.to_csv(logs_filename, mode='a', header=not os.path.isfile(logs_filename), index=False)
 
    print('*'*20)
    print(f'Model Timestamp: {timestamp}')
    print('*'*20)

    all_subject_ids = np.array(train_config['subject_ids'])

    # # # # # # # # CV stuff only
    #  Read all PSG labels to compute metrics during CV
    psg_labels = pd.DataFrame()
    for id in all_subject_ids:
        subject_labels = read_PSG_labels(psg_labels_path, id)
        subject_labels.insert(0, 'subject_id', id)
        psg_labels = pd.concat([psg_labels, subject_labels])

    metric_fns = {
        'F-1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        'Recall': recall_score,
        'Precision': precision_score,
        "Cohen's Kappa": lambda y_true, y_pred: cohen_kappa_score(y1=y_true, y2=y_pred),
        'Specificity': lambda y_true, y_pred: classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, labels=[0, 1])['0']['precision']
    }

    # test_ids = train_config['test_ids']
    test_ids = []
    train_val_ids = [id for id in train_config['subject_ids'] if id not in test_ids]

    print('*'*80)
    print(f'Test subjects: {test_ids}')
    print('-'*40)

    model_nickname = f"best_on_all_{timestamp}"

    model = train_model(
        datapath=datapath,
        train_config=train_config,
        train_val_ids=train_val_ids,
        output_dir=output_dir,
        model_nickname=model_nickname,
        save_checkpoints=True,
        )

    if len(test_ids) > 0:
        metrics = {metric_name: [] for metric_name in metric_fns.keys()}  # Placeholder for metric values

        test_data = create_dataset(datapath,
                                filters=[partial(keep_subjects, subject_ids=test_ids)],
                                repeat=False, shuffle=False, batch_size=100)

        print('#-'*80)
        print(model.evaluate(test_data))
        print('#-'*80)
        pred = model.predict(test_data)

        pred['pred'] = np.round(np.squeeze(pred['pred']))  # Threshold = 0.5
        pred = pd.DataFrame(pred)
        pred['epoch_ts'] = pd.to_datetime(pred['epoch_ts'].str.decode("utf-8"))

        test_fold_labels = psg_labels[psg_labels['subject_id'].isin(test_ids)]
        labels_pred_df = pd.merge(
            left=test_fold_labels,
            right=pred,
            on=['subject_id', 'epoch_ts'],
            how='left'
        )
        print('^*'*40)
        print(f"Label-Prediction mismatch = {round(labels_pred_df['pred'].isna().mean() * 100, 4)}")
        print('^*'*40)
        labels_pred_df = labels_pred_df.dropna()  # ToDo: Fix windowing for the first and last epochs so we won't have to dropna here

        # labels_pred_df.to_csv(f'{cv_preds_path}/{model_nickname}.csv', index=False)
        
        for metric_name, metric_fn in metric_fns.items():
            metric_value = metric_fn(y_pred=labels_pred_df['pred'], y_true=labels_pred_df['PSG Sleep'])
            metrics[metric_name].append(round(metric_value * 100, 2))
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.insert(1, 'Test Subjects', " - ".join([str(id) for id in test_ids]))

        metrics_df.to_csv(f'metrics.csv', index=False)  # Overwrites to update every time


if __name__ == '__main__':
    # down_sample_list = [10]

    # for ds in down_sample_list:
    # train_config['down_sample_by'] = ds
    training_main(train_config=config)
