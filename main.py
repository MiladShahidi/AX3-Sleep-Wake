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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and debug info mesasages

def parse_tfrecord(example, has_labels):
    seq_features_spec = {
        'X': tf.io.VarLenFeature(dtype=tf.float32),
        'Y': tf.io.VarLenFeature(dtype=tf.float32),
        'Z': tf.io.VarLenFeature(dtype=tf.float32),
        'Temp': tf.io.VarLenFeature(dtype=tf.float32)
        }

    context_features_spec = {
        'subject_id': tf.io.FixedLenFeature([], dtype=tf.int64),
        'epoch_id': tf.io.VarLenFeature(dtype=tf.int64),
        'epoch_ts': tf.io.VarLenFeature(dtype=tf.string),
        'central_epoch_id': tf.io.FixedLenFeature([], dtype=tf.int64),
        'central_epoch_ts': tf.io.VarLenFeature(dtype=tf.string)
        }
    if has_labels:
        context_features_spec['label'] = tf.io.FixedLenFeature([], dtype=tf.float32)

    context, features, _ = tf.io.parse_sequence_example(
        example,
        context_features=context_features_spec,
        sequence_features=seq_features_spec
    )

    for feature_name, tensor in features.items():
        if isinstance(tensor, tf.SparseTensor):
            features[feature_name] = tf.sparse.to_dense(tensor)

        # I don't like this. It's a quick hack to reshape each axis (X, Y, Z) to (T, ) rather than (T, 1)
        features[feature_name] = tf.squeeze(features[feature_name])

    for feature_name, tensor in context.items():
        if isinstance(tensor, tf.SparseTensor):
            context[feature_name] = tf.sparse.to_dense(tensor)

        # I don't like this. It's a quick hack to reshape each axis (X, Y, Z) to (T, ) rather than (T, 1)
        context[feature_name] = tf.squeeze(context[feature_name])

    return context, features


def reshape_features(context, features):
    # Each feature (X, Y, Z, etc.) has shape (window length, epoch length)
    xyz = tf.concat([
        tf.expand_dims(features['X'], axis=-1),  # expand dimensions and insert a new axis for X, Y, Z
        tf.expand_dims(features['Y'], axis=-1),
        tf.expand_dims(features['Z'], axis=-1)
        ], axis=-1)
        
    # axis=-1 means last one. This is the axis we just created above.
    # This finds the L2 norm over the X-Y-Z axis
    triaxial_l2_norm = tf.norm(xyz, ord=2, axis=-1, keepdims=True)  # L2 Norm

    feature_list = [
        tf.expand_dims(features['X'], axis=-1),
        tf.expand_dims(features['Y'], axis=-1),
        tf.expand_dims(features['Z'], axis=-1),
        tf.expand_dims(features['Temp'], axis=-1),
        triaxial_l2_norm,
    ]

    stacked_features = tf.concat(feature_list, axis=-1)
    
    # stacked_features.shape is (win_size, epoch len, n_sequences)
    # Below, we combine the first two dimensions, i.e. concatenate all epochs in the window
    # And make it (win_size * epoch len, n_sequences)
    n_sequences = len(feature_list)
    seq_features = {
        'features': tf.reshape(stacked_features, (-1, n_sequences)),
    }
        
    if 'label' in context:
        label = context.pop('label')
        features = {**seq_features, **context}
        return features, label
    else:
        features = {**seq_features, **context}
        return features


def create_dataset(path, filters=None, has_labels=True, batch_size=None, repeat=True, shuffle=True):
    
    if tf.io.gfile.exists(path):  # This means it's either a single file name or a directory, not a pattern
        
        if tf.io.gfile.isdir(path):
            files = tf.data.Dataset.list_files(f"{path}/*.tfrecord", shuffle=True)
            dataset = tf.data.TFRecordDataset(files)
        else:
            dataset = tf.data.TFRecordDataset(path)
    
    else:  # It must be a pattern like data/sub01_*.tfrecord
        files = tf.data.Dataset.list_files(path, shuffle=True)
        dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(partial(parse_tfrecord, has_labels=has_labels)).map(reshape_features)

    if filters is not None:
      for filter in filters:
        dataset = dataset.filter(filter)
      
    if shuffle:
      shuffle_buffer_size = 10000 if batch_size is None else batch_size * 10
      dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    # Shuffle then batch
    if batch_size:
        dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


class SleepModel(tf.keras.Model):

    def __init__(
            self,
            sample_every_n=None,
            name='SleepModel',
            **kwargs
            ):
        super().__init__(name=name, **kwargs)
        # self.rnn_layer = tf.keras.layers.LSTM(units=16)
        
        self.sample_every_n = sample_every_n
        
        self.input_batch_norm = tf.keras.layers.BatchNormalization()

        # self.lstm_route = [
        #     tf.keras.layers.LSTM(32),
        #     tf.keras.layers.Dropout(0.8)
        #     ]

        # dropout (for attn) etc.
        # BatchNorm axis

        self.cnn_route = [
            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=8,
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv1D(filters=128,
                                   kernel_size=5,
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=3,
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
        ]

        # TODO: Looks like there is an additional matrix multiplication after (at the end of) attention in the paper (W_o)
        self.attn = tf.keras.layers.Attention(use_scale=False)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')


    def call(self, inputs):
        x = inputs['features']

        # Downsampling
        if self.sample_every_n:
            assert (config['AX3_freq'] * config['seconds_per_epoch']) % self.sample_every_n == 0, "Epoch length must be a multiple of downsampling rate."
            # xyz.shape = (Batch Size, Input Length, 3). We're downsampling along axis=1 (input length)
            x = x[:, ::self.sample_every_n, :]  # Sample every n observation, e.g. 10 will downsample from 100 Hz to 10 Hz

        x = self.input_batch_norm(x)

        # Making copies of the input tensor
        cnn_signal = tf.identity(x)
        # lstm_signal = tf.identity(x)
        
        # CNN route
        for layer in self.cnn_route:
            cnn_signal = layer(cnn_signal)
        
        temporal_attn = self.attn([cnn_signal, cnn_signal, cnn_signal], use_causal_mask=True)

        cnn_signal = self.pooling(cnn_signal + temporal_attn)  # TODO: The paper weights attn by a scalar

        # LSTM route
        # for layer in self.lstm_route:
        #     lstm_signal = layer(lstm_signal)
        
        # out = tf.concat([cnn_signal, lstm_signal], axis=-1)

        output = self.output_layer(cnn_signal)

        return {
            'epoch_ts': inputs['central_epoch_ts'],
            'subject_id': inputs['subject_id'],
            'pred': output
        }


def drop_subjects(features, labels=None, subject_ids=[]):
    # This is vectorized and with reduce_all, because subject_ids is a list of possibly multiple ids
    return tf.reduce_all(tf.not_equal(features['subject_id'], subject_ids))


def keep_subjects(features, labels=None, subject_ids=[]):
    # This is vectorized and with reduce_any, because subject_ids is a list of possibly multiple ids
    return tf.reduce_any(tf.equal(features['subject_id'], subject_ids))


def train_model(
        datapath,
        train_val_ids,
        output_dir,
        model_nickname,
        save_checkpoints
        ):

    saved_models_dir = f'{output_dir}/savedmodels/{timestamp}'
    tensorboard_logdir = f"{output_dir}/tb_logs/{timestamp}"

    train_data = create_dataset(
        datapath,
        filters=[
            partial(keep_subjects, subject_ids=train_val_ids),
            lambda x, y: x['central_epoch_id'] % 5 != 0
            ],
        batch_size=128
        )

    val_data = create_dataset(
        datapath,
        filters=[
            partial(keep_subjects, subject_ids=train_val_ids),
            lambda x, y: x['central_epoch_id'] % 5 == 0
            ],
        batch_size=100
        )
    
    callbacks = [
        CustomTensorBoard(log_dir=f"{tensorboard_logdir}/{model_nickname}"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, start_from_epoch=30)
    ]
    if save_checkpoints:
        callbacks += [tf.keras.callbacks.ModelCheckpoint(saved_models_dir, monitor='val_loss', save_best_only=True)]

    model = SleepModel(sample_every_n=5)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),  # Legacy is because the current one runs slow on M1/M2 macs
        loss={'pred': tf.keras.losses.BinaryCrossentropy(name='Loss')},
        metrics={'pred': [
            tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision'),
            F1Score(name='F1Score'),
            # PositiveRate(name='PositiveRate'),
            # PredictedPositives(name='PredictedPositives')
            ]}
            )

    model.fit(
        train_data,
        # class_weight={0: 0.7, 1: 0.3},
        epochs=1000,
        steps_per_epoch=100,
        validation_data=val_data,
        validation_steps=100,
        callbacks=callbacks
        )

    # model.save(f'{saved_models_dir}/{model_nickname}')

    return model


if __name__ == '__main__':

    datapath = f"data/Tensorflow/normalised/window_{config['window_size']}/labelled"
    psg_labes_path = 'data/PSG-Labels'
    output_dir = 'training_output'
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    performance_output_path = f'{output_dir}/Performance/{timestamp}'

    # all_subject_ids = np.array([id for id in range(1, 29) if id != 27])
    # all_subject_ids = config['subject_ids']
    all_subject_ids = np.array(config['subject_ids'])

    #  Read all PSG labels to compute metrics during CV
    psg_labels = pd.DataFrame()
    for id in all_subject_ids:
        subject_labels = read_PSG_labels(psg_labes_path, id)
        subject_labels.insert(0, 'subject_id', id)
        psg_labels = pd.concat([psg_labels, subject_labels])

    metric_fns = {
        'F-1': f1_score,
        'Recall': recall_score,
        'Precision': precision_score,
        "Cohen's Kappa": lambda y_true, y_pred: cohen_kappa_score(y1=y_true, y2=y_pred),
        'Specificity': lambda y_true, y_pred: classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, labels=[0, 1])['0']['precision']
    }

    os.makedirs(performance_output_path)
    
    kfold_splitter = KFold(config['n_cv_folds'])

    metrics_df = pd.DataFrame()

    # for fold_number, (train_index, test_index) in enumerate(kfold_splitter.split(all_subject_ids)):
    for fold_number, (train_index, test_index) in enumerate([(np.arange(0, len(all_subject_ids)), [])]):  # For training on all subjects

        train_val_ids = all_subject_ids[train_index]
        test_ids = all_subject_ids[test_index]

        print('*'*80)
        print(f'Starting Fold {fold_number}')
        print(f'Test subjects: {test_ids}')
        print('-'*40)

        if len(test_ids) > 0:
            model_nickname = f"model_excl_{np.min(test_ids):02d}_to_{np.max(test_ids):02d}"
        else:
            model_nickname = 'AttnSleepModel'

        model = train_model(
            datapath=datapath,
            train_val_ids=train_val_ids,
            output_dir=output_dir,
            model_nickname=model_nickname,
            save_checkpoints=(len(test_ids) == 0)  # Don't save checkpoints during CV
            )

        if len(test_ids) > 0:
            metrics = {metric_name: [] for metric_name in metric_fns.keys()}  # Placeholder for metric values

            test_data = create_dataset(datapath,
                                    filters=[partial(keep_subjects, subject_ids=test_ids)],
                                    repeat=False, shuffle=False, batch_size=100)

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

            labels_pred_df.to_csv(f'{performance_output_path}/preds_fold_{fold_number:02d}_{model_nickname}.csv', index=False)
            
            for metric_name, metric_fn in metric_fns.items():
                metric_value = metric_fn(y_pred=labels_pred_df['pred'], y_true=labels_pred_df['label'])
                metrics[metric_name].append(round(metric_value * 100, 2))
            
            fold_metrics = pd.DataFrame(metrics)
            fold_metrics.insert(0, 'Fold', fold_number)
            fold_metrics.insert(1, 'Test Subjects', f"{np.min(test_ids):02d} to {np.max(test_ids):02d}")

            metrics_df = pd.concat([metrics_df, fold_metrics])

            metrics_df.to_csv(f'{performance_output_path}/cv_metrics.csv', index=False)  # Overwrites to update every time
