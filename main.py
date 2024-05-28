import tensorflow as tf
import datetime
from utils.metrics import F1Score, PositiveRate, PredictedPositives
import os
from functools import partial
import numpy as np
from utils.data_utils import read_subject_labels
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and debug info mesasages

def parse_tfrecord(example):
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
        'central_epoch_ts': tf.io.VarLenFeature(dtype=tf.string),
        'label': tf.io.FixedLenFeature([], dtype=tf.float32)
        }

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
    # Each dimension has shape (window length, epoch length)
    # Here we first flatten (reshape) each one so that the measurements of consequtive epochs are 
    # all placed in a flat tensor
    # Then stack the 3 dimensions
    accel = tf.concat([
        tf.expand_dims(features['X'], axis=-1),  # expand dimensions to insert a new axis for X, Y, Z
        tf.expand_dims(features['Y'], axis=-1),
        tf.expand_dims(features['Z'], axis=-1),
        ], axis=-1)  # (window size, epoch length, n_axes)
    
    temperature = tf.expand_dims(features['Temp'], axis=-1)

    # axis=-1 means last one. This is the axis we just created above.
    # This finds the L2 norm over the X-Y-Z axis
    triaxial = tf.norm(accel, ord=2, axis=-1, keepdims=True)  # L2 Norm

    # Concatenate along the XYZ axis
    # measurements:     (win size, epoch len, 3)
    # triaxial:         (win size, epoch len, 1)
    # stacked_features: (win size, epoch len, 4)
    stacked_features = tf.concat([
        accel,
        triaxial,
        temperature
    ], axis=-1)

    # Note: For some reason, using stack here results in errors when fitting the model
    # and it complains about receiving a tensor with unknown shape
    # The following reshape explicitly sets teh shape and fixes that error, but this is not ideal
    # epoch_length = tf.shape(features['X'])[0]
    # measurements = tf.reshape(measurements, shape=(epoch_length, 3))
    # triaxial = tf.reshape(triaxial, shape=(epoch_length, 1))
    
    seq_features = {
        'measurements': tf.reshape(stacked_features, (-1, 5)),
        # 'temp': temperature
        # 'XYZ': tf.reshape(measurements, (-1, 3))
        # 'XYZ': tf.random.normal(mean=1, stddev=1, shape=(3000, )),
    }
    label = context.pop('label')

    # # # # # #
    # label = tf.squeeze(tf.cast(tf.math.reduce_mean(triaxial, axis=0) <= 1, tf.float32))
    # # # # # #

    features = {**seq_features, **context}

    return  features, label


def create_dataset(path, filter_fn=None, batch_size=None, repeat=True, shuffle=True):
    if tf.io.gfile.isdir(path):
        # filenames = [f'{path}/{filename}' for filename in tf.io.gfile.listdir(path)]
        files = tf.data.Dataset.list_files(f"{path}/*.tfrecord", shuffle=True)
        dataset = tf.data.TFRecordDataset(files)
    else:
        dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.map(parse_tfrecord).map(reshape_features)

    if filter_fn is not None:
      dataset = dataset.filter(filter_fn)
      
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
        # self.attn = tf.keras.layers.Attention(use_scale=False)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')


    def call(self, inputs):
        x = inputs['measurements']

        # Downsampling
        if self.sample_every_n:
            # xyz.shape = (Batch Size, Epoch Length, 3). We're downsampling the epoch length dimension
            x = x[:, ::self.sample_every_n, :]  # Sample every n observation, e.g. 10 will downsample from 100 Hz to 10 Hz

        x = tf.reshape(x, shape=(-1, 3000, 5))
        
        x = self.input_batch_norm(x)

        # Making copies of the input tensor
        cnn_signal = tf.identity(x)
        # lstm_signal = tf.identity(x)
        
        # CNN route
        for layer in self.cnn_route:
            cnn_signal = layer(cnn_signal)
        
        # temporal_attn = self.attn([x, x, x])

        cnn_signal = self.pooling(cnn_signal)  # TODO: The paper weights attn by a scalar

        # LSTM route
        # for layer in self.lstm_route:
        #     lstm_signal = layer(lstm_signal)
        
        # out = tf.concat([cnn_signal, lstm_signal], axis=-1)

        output = self.output_layer(cnn_signal)

        return {
            'epoch_ts': inputs['central_epoch_ts'],
            'subject_id': inputs['subject_id'],
            'prediction': output
        }


class ToyModel(tf.keras.Model):

    def __init__(
            self,
            sample_every_n=None,
            name='SleepModel',
            **kwargs
            ):
        super().__init__(name=name, **kwargs)
        
        self.sample_every_n = sample_every_n        

        self.model_layers = [
            # tf.keras.layers.Dense(128, activation='sigmoid'),
            # tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dense(32, activation='sigmoid')
        ]
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = inputs['XYZ']

        # Downsampling
        if self.sample_every_n:
            # xyz.shape = (Batch Size, Epoch Length, 3). We're downsampling the epoch length dimension
            x = x[:, ::self.sample_every_n, :]  # Sample every n observation, e.g. 10 will downsample from 100 Hz to 10 Hz
        
        x = tf.reshape(x, shape=(-1, 3000))

        x = self.batch_norm(x)

        for layer in self.model_layers:
            x = layer(x)
        
        output = self.output_layer(x)

        return output


def drop_subjects(features, labels, subject_ids):
    return tf.reduce_all(tf.not_equal(features['subject_id'], subject_ids))


def keep_subjects(features, labels, subject_ids):
    # This is vectorized and with reduce_any, because subject_ids is a list of possible multiple ids
    return tf.reduce_any(tf.equal(features['subject_id'], subject_ids))


if __name__ == '__main__':

    datapath = 'data/Tensorflow/normalised/window_3'
    psg_labes_path = 'data/raw/Labels'
    output_dir = 'training_output'
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_models_dir = f'{output_dir}/savedmodels/{timestamp}'
    tensorboard_logdir = f"{output_dir}/tb_logs/{timestamp}"
    performance_output_path = f'{output_dir}/Performance/{timestamp}'

    all_subject_ids = np.array([id for id in range(1, 36) if id != 27])

    # # # # # # Read all PSG labels
    psg_labels = pd.DataFrame()
    for id in all_subject_ids:
        label_file_name = f'{psg_labes_path}/SDRI001_PSG_Sleep profile_{id:03d}V4_N1.txt'
        subject_labels = read_subject_labels(label_file_name)
        cols = list(subject_labels.columns)
        subject_labels['subject_id'] = id
        subject_labels = subject_labels[['subject_id'] + cols]
        psg_labels = pd.concat([psg_labels, subject_labels])

    metric_fns = {
        'F-1': f1_score,
        'Recall': recall_score,
        'Precision': precision_score,
        "Cohen's Kappa": lambda y_true, y_pred: cohen_kappa_score(y1=y_true, y2=y_pred),
        'Specificity': lambda y_true, y_pred: classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, labels=[0, 1])['0']['precision']
    }
    metrics = {metric_name: [] for metric_name in metric_fns.keys()}
    
    os.makedirs(performance_output_path)
    
    kfold_splitter = KFold(7)

    for i, (train_index, test_index) in enumerate(kfold_splitter.split(all_subject_ids)):
        
        train_ids = all_subject_ids[train_index]
        test_ids = all_subject_ids[test_index]

        train_val_data = create_dataset(datapath,
                                        filter_fn=partial(drop_subjects, subject_ids=test_ids),
                                        batch_size=128)

        train_data = create_dataset(datapath,
                                    filter_fn=lambda x, y: x['central_epoch_id'] % 5 != 0,
                                    batch_size=128)
        
        val_data = create_dataset(datapath,
                                  filter_fn=lambda x, y: x['central_epoch_id'] % 5 == 0,
                                  batch_size=128)

        test_data = create_dataset(datapath,
                                   filter_fn=partial(keep_subjects, subject_ids=test_ids),
                                   repeat=False, shuffle=False, batch_size=100)

        model_nickname = f"model_excl_{np.min(test_ids):02d}_to_{np.max(test_ids):02d}"
        
        callbacks = [
            CustomTensorBoard(log_dir=f"{tensorboard_logdir}/{model_nickname}"),
            # tf.keras.callbacks.TensorBoard(log_dir=f"{tensorboard_logdir}/{model_nickname}"),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=7, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=20)
        ]

        model = SleepModel(sample_every_n=3)

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-2),  # Legacy is because the current one runs slow on M1/M2 macs
            loss={'prediction': tf.keras.losses.BinaryCrossentropy()},
            metrics={
                'prediction': [
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.Precision(),
                    F1Score(),
                    PositiveRate(),
                    PredictedPositives()
                    ]}
                    )

        model.fit(
            train_data,
            # class_weight={0: 0.7, 1: 0.3},
            epochs=1000,
            steps_per_epoch=100,
            validation_data=val_data,
            validation_steps=50,
            callbacks=callbacks
            )

        # model.save(f'{saved_models_dir}/{model_nickname}')

        pred = model.predict(test_data)

        pred['prediction'] = np.round(np.squeeze(pred['prediction']))  # Threshold = 0.5
        pred = pd.DataFrame(pred)
        pred['epoch_ts'] = pd.to_datetime(pred['epoch_ts'].str.decode("utf-8"))

        test_fold_labels = psg_labels[psg_labels['subject_id'].isin(test_ids)]
        labels_pred_df = pd.merge(
            left=test_fold_labels,
            right=pred,
            on=['subject_id', 'epoch_ts'],
            how='left'
        ).dropna()  # ToDo: Fix windowing so we won't have to dropna here
        print('^*'*40)
        print(f"Label-Prediction mismatch = {round(labels_pred_df['prediction'].isna().mean() * 100, 4)}")
        print('^*'*40)

        labels_pred_df.to_csv(f'{performance_output_path}/preds_{model_nickname}.csv', index=False)
        for metric_name in metrics.keys():
            metric = metric_fns[metric_name]
            metrics[metric_name].append(round(metric(y_pred=labels_pred_df['prediction'], y_true=labels_pred_df['label']) * 100, 2))

        pd.DataFrame(metrics).to_csv(f'{performance_output_path}/metrics_{model_nickname}.csv', index=False)
