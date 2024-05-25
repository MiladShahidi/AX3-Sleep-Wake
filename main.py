import tensorflow as tf
import datetime
from utils.metrics import F1Score, PositiveRate, PredictedPositives
import os
from functools import partial
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and debug info mesasages

def parse_tfrecord(example):
    seq_features_spec = {
        'X': tf.io.VarLenFeature(dtype=tf.float32),
        'Y': tf.io.VarLenFeature(dtype=tf.float32),
        'Z': tf.io.VarLenFeature(dtype=tf.float32),
        }

    context_features_spec = {
        'subject_id': tf.io.FixedLenFeature([], dtype=tf.int64),
        'epoch_id': tf.io.VarLenFeature(dtype=tf.int64),
        'epoch_ts': tf.io.VarLenFeature(dtype=tf.string),
        'central_epoch_id': tf.io.FixedLenFeature([], dtype=tf.int64),
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
    measurements = tf.concat([
        tf.expand_dims(features['X'], axis=1),
        tf.expand_dims(features['Y'], axis=1),
        tf.expand_dims(features['Z'], axis=1),
        ], axis=-1)

    triaxial = tf.norm(measurements, ord=2, axis=1, keepdims=True)  # L2 Norm

    stacked_features = tf.concat([
        measurements,
        triaxial
    ], axis=-1)

    # Note: For some reason, using stack here results in errors when fitting the model
    # and it complains about receiving a tensor with unknown shape
    # The following reshape explicitly sets teh shape and fixes that error, but this is not ideal
    # epoch_length = tf.shape(features['X'])[0]
    # measurements = tf.reshape(measurements, shape=(epoch_length, 3))
    # triaxial = tf.reshape(triaxial, shape=(epoch_length, 1))
    
    seq_features = {
        'XYZ': stacked_features
        # 'XYZ': tf.reshape(triaxial, shape=(3000, ))
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
        self.rnn_layer = tf.keras.layers.LSTM(units=16)
        
        self.sample_every_n = sample_every_n
        
        self.input_batch_norm = tf.keras.layers.BatchNormalization()

        self.lstm_route = [
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.8)
            ]

        # TODO: BatchNorm after conv
        # dropout (for attn) etc.
        # BatchNorm axis

        self.cnn_route = [
            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=8,
                                #    padding='same',
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv1D(filters=128,
                                   kernel_size=5,
                                #    padding='same',
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=3,
                                #    padding='same',
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
        ]

        # TODO: Looks like there is an additional matrix multiplication after (at the end of) attention in the paper (W_o)
        self.attn = tf.keras.layers.Attention(use_scale=False)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')


    def call(self, inputs):
        x = inputs['XYZ']

        # Downsampling
        if self.sample_every_n:
            # xyz.shape = (Batch Size, Epoch Length, 3). We're downsampling the epoch length dimension
            x = x[:, ::self.sample_every_n, :]  # Sample every n observation, e.g. 10 will downsample from 100 Hz to 10 Hz

        x = tf.reshape(x, shape=(-1, 1000, 4))
        
        x = self.input_batch_norm(x)

        # Making copies of the input tensor
        cnn_signal = tf.identity(x)
        lstm_signal = tf.identity(x)
        
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

        return output


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


def data_split_filter(features, labels, ids):
    return tf.reduce_any(tf.equal(features['subject_id'], ids))


if __name__ == '__main__':
    datapath = 'data/processed/normalised/window_1'

    train_subject_ids = tf.constant(range(1, 30 + 1), dtype=tf.int64)
    val_subject_ids = tf.constant(range(31, 35 + 1), dtype=tf.int64)
    test_subject_ids = tf.constant(range(36, 36 + 1), dtype=tf.int64)

    train_data = create_dataset(datapath,
                                # filter_fn=lambda x, y: x['central_epoch_id'] % 5 == 0,
                                filter_fn=partial(data_split_filter, ids=train_subject_ids),
                                batch_size=128)

    val_data = create_dataset(datapath,
                              filter_fn=partial(data_split_filter, ids=val_subject_ids),
                              batch_size=128)
    
    test_data = create_dataset(datapath,
                               filter_fn=partial(data_split_filter, ids=test_subject_ids),
                               repeat=False,
                               shuffle=False)

    output_dir = 'training_output'
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    saved_model_dir = f'{output_dir}/savedmodels/{timestamp}'

    tensorboard_logs = f"{output_dir}/tb_logs/{timestamp}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs, histogram_freq=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=7, min_lr=1e-6)
    model = SleepModel(sample_every_n=3)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),  # Legacy is because the current one runs slow on M1/M2 macs
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            F1Score(),
            PositiveRate(),
            PredictedPositives()
            ]
    )

    model.fit(
        train_data,
        epochs=1000,
        steps_per_epoch=100,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tensorboard_callback, reduce_lr]
        )

    # model.save(saved_model_dir)

    # eval_res = model.evaluate(test_data.filter(lambda x, y: tf.equal(x['central_epoch_id'] % 100, 0)).batch(1), return_dict=True)
    # print(eval_res)

    # pred = model.predict(test_data.filter(lambda x, y: tf.equal(x['central_epoch_id'] % 10, 0)).batch(1))
    # print(pred)
    
    # l = []
    # for i, (features, label) in enumerate(train_data.take(100)):
    #     # print('Features:\n')
    #     # print(features['XYZ'])
    #     # print('- '*20)
    #     # print('Label:\n')
    #     # print(label)
    #     # print('*' * 80)
    #     print(i, end='\r')
    #     l.append(tf.math.reduce_std(features['XYZ'].numpy()))

    # l = np.array(l)
    # print(l)
    # print(np.mean(l))

