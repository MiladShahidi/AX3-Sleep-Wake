import tensorflow as tf
from config import project_config as config
import numpy as np

class CNNModel(tf.keras.Model):

    def __init__(
            self,
            down_sample_by=None,
            name='CNNModel',
            **kwargs
            ):
        super().__init__(name=name, **kwargs)
        # self.rnn_layer = tf.keras.layers.LSTM(units=16)
        self.down_sample_by = down_sample_by
        
        if self.down_sample_by:
            self.down_sampler = tf.keras.layers.MaxPool1D(pool_size=self.down_sample_by, strides=self.down_sample_by)
            
        self.input_batch_norm = tf.keras.layers.BatchNormalization()

        # self.lstm_route = [
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.LSTM(128),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.3)
        # ]
        
        # ToDo: dropout (for attn) etc.

        self.cnn_route = [
            tf.keras.layers.Conv1D(filters=128,
                                   kernel_size=8,
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=256,
                                   kernel_size=5,
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=128,
                                   kernel_size=3,
                                   kernel_initializer='he_uniform', strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Dropout(0.3),
        ]

        # TODO: Looks like there is an additional matrix multiplication after (at the end of) attention in the paper (W_o)
        self.multi_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,  # head size
            dropout=0.1
        )

        self.variable_multihead_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,  # head size
            dropout=0.1
        )

        # self.attn = tf.keras.layers.Attention()
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')


    def call(self, inputs):
        x = inputs['features']

        # Downsampling
        if self.down_sample_by:
            assert (config['AX3_freq'] * config['seconds_per_epoch']) % self.down_sample_by == 0, "Epoch length must be a multiple of downsampling rate."

            # x.shape = (Batch Size, Input Length, 3). We're downsampling along axis=1 (input length)

            # x = x[:, ::self.sample_every_n, :]  # Sample every n observation, e.g. 10 will downsample from 100 Hz to 10 Hz

            # Down-sample by averging
            # This changes x from (batch size, length, n_features)
            # to (batch size, length // down_sample_by, n_features)
            # by averaging over a moving
            x = self.down_sampler(x)

        x = tf.reshape(x, (-1, config['window_size'] * 3000 // self.down_sample_by, 5))
        
        x = self.input_batch_norm(x)

        # Making copies of the input tensor
        cnn_signal = tf.identity(x)
        # lstm_signal = tf.identity(x)
        
        # CNN route
        for layer in self.cnn_route:
            cnn_signal = layer(cnn_signal)
        
        temporal_attn = self.multi_attn(cnn_signal, cnn_signal, cnn_signal)

        cnn_signal = cnn_signal + temporal_attn

        # cnn_signal_t = tf.transpose(cnn_signal, perm=[0, 2, 1])  # Keep batch dimension in place

        # var_attention = self.variable_multihead_attn(cnn_signal_t, cnn_signal_t, cnn_signal_t)

        # cnn_signal = cnn_signal + tf.transpose(var_attention, perm=[0, 2, 1])

        cnn_signal = self.pooling(cnn_signal)  # TODO: The paper weights attn by a scalar

        # lstm_signal = tf.identity(cnn_signal)
        # # LSTM route
        # for layer in self.lstm_route:
        #     lstm_signal = layer(lstm_signal)
        
        # cnn_lstm = tf.concat([cnn_signal, lstm_signal], axis=-1)

        output = self.output_layer(cnn_signal)

        return {
            'epoch_ts': inputs['central_epoch_ts'],
            'subject_id': inputs['subject_id'],
            'pred': output
        }

