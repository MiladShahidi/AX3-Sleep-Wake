import tensorflow as tf
import datetime
from utils.metrics import F1Score
import keras_tuner
import os


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
    stacked_features = tf.stack([
        tf.reshape(features['X'], shape=(-1,)),
        tf.reshape(features['Y'], shape=(-1,)),
        tf.reshape(features['Z'], shape=(-1,))
        ], axis=-1)

    # Note: For some reason, using stack here results in errors when fitting the model
    # and it complains about receiving a tensor with unknown shape
    # reshaping here fixes that error but this is not ideal
    epoch_length = tf.shape(features['X'])[0]
    stacked_features = tf.reshape(stacked_features, shape=(epoch_length, 3))
    
    seq_features = {
        'XYZ': stacked_features,
    }
    labels = context.pop('label')

    features = {**seq_features, **context}

    return  features, labels


def create_dataset(path, batch_size=None, repeat=True):
    if tf.io.gfile.isdir(path):
        # filenames = [f'{path}/{filename}' for filename in tf.io.gfile.listdir(path)]
        files = tf.data.Dataset.list_files(f"{path}/*.tfrecord", shuffle=True)
        dataset = tf.data.TFRecordDataset(files)
    else:
        dataset = tf.data.TFRecordDataset(path)
    
    dataset = dataset.map(parse_tfrecord).map(reshape_features).shuffle(buffer_size=1000, reshuffle_each_iteration=True)

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
            name='ToyModel',
            **kwargs
            ):
        super().__init__(name=name, **kwargs)
        self.rnn_layer = tf.keras.layers.LSTM(units=16)
        
        self.sample_every_n = sample_every_n
        
        self.model_layers = [
            tf.keras.layers.Conv1D(filters=16, kernel_size=1024, activation='ReLU', strides=100),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(),

            # tf.keras.layers.Conv1D(filters=16, kernel_size=256, activation='ReLU', strides=2),
            # tf.keras.layers.Dropout(rate=0.25),
            # tf.keras.layers.BatchNormalization(),

            # tf.keras.layers.Conv1D(filters=16, kernel_size=256, activation='ReLU', strides=2),
            # tf.keras.layers.Dropout(rate=0.25),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.MaxPool1D(),

            # tf.keras.layers.Conv1D(filters=16, kernel_size=32, activation='ReLU', strides=2),
            # tf.keras.layers.Dropout(rate=0.25),
            # tf.keras.layers.BatchNormalization(),

            tf.keras.layers.LSTM(units=128),

            tf.keras.layers.Dense(units=128, activation='ReLU'),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ]

    def call(self, inputs):
        xyz = inputs['XYZ']

        if self.sample_every_n:  # Downsampling
            # xyz.shape = (Batch Size, Epoch Length, 3). We're downsampling the epoch length dimension
            xyz = xyz[:, ::self.sample_every_n, :]  # Sample every n observation, e.g. 10 will downsample from 100 Hz to 10 Hz
        
        #####

        for layer in self.model_layers:
            xyz = layer(xyz)
        
        return xyz
    

# def build_model(hp):
#     lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
#     activation = hp.Choice("activation", ["relu", "tanh"])

#     model = ToyModel(activation=activation)

#     model.compile(
#         optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),  # Legacy is because the current one runs slow on M1/M2 macs
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=[
#             tf.keras.metrics.BinaryAccuracy(),
#             tf.keras.metrics.Recall(),
#             tf.keras.metrics.Precision(),
#             F1Score()
#             ]
#     )

#     return model


def train_data_filter(features, labels):
    return features['central_epoch_id'] % 5 != 0


if __name__ == '__main__':
    datapath = 'data/processed/window_1'

    train_data = create_dataset(datapath).filter(train_data_filter).batch(128)
    val_data = create_dataset(datapath).filter(lambda x, y: not train_data_filter(x, y)).batch(128)
    
    # hp_tuner = keras_tuner.Hyperband(
    #     hypermodel=build_model,
    #     objective=keras_tuner.Objective("val_F1Score", direction="max"),
    #     max_epochs=3,
    #     # executions_per_trial=2,
    #     # overwrite=True,
    #     # directory="hp_search",
    #     # project_name="sleep",
    # )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = ToyModel(sample_every_n=1)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),  # Legacy is because the current one runs slow on M1/M2 macs
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            F1Score()
            ]
    )

    model.fit(
        train_data,
        epochs=1000,
        steps_per_epoch=100,
        validation_data=val_data,
        validation_steps=100,
        callbacks=[tensorboard_callback]
        )

    # model.save('logs')