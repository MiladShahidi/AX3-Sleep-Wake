import json
from datetime import datetime
import numpy as np
from tensorflow import keras
import keras_tuner
from models import CNNModel
from keras_tuner_cv.inner_cv import inner_cv
from keras_tuner_cv.utils import pd_inner_cv_get_result
from utils.training_utils import CustomTensorBoard
from sklearn.model_selection import KFold
from keras_tuner.tuners import Hyperband, RandomSearch
from keras_tuner import HyperParameters
from keras_tuner import Objective
from config import project_config as config
import tensorflow as tf
from utils.metrics import F1Score, PositiveRate, PredictedPositives
from utils.tfrecord_utils import create_dataset
from utils.helpers import keep_subjects
from functools import partial
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and debug info mesasages


class HyperModel(keras_tuner.HyperModel):

    def _save_hps(self, path, filename, hp):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{filename}", 'w') as f:
            json.dump(hp.values, f)
        
    def build(self, hp):
        # down_sample_by = hp.Choice("down_sample_by", [10, 100, 1000])
        # num_conv_filters = hp.Choice("num_conv_filters", [32, 64, 128])
        # window_size=hp.Choice("window_size", [3])
        down_sample_by = hp.get("down_sample_by")
        num_conv_filters = hp.get("num_conv_filters")
        window_size = hp.get("window_size")
        num_attention_heads = hp.get("num_attention_heads")
        stride = hp.get("stride")
        
        model = CNNModel(
            down_sample_by=down_sample_by,
            num_conv_filters=num_conv_filters,
            window_size=window_size,
            num_attention_heads=num_attention_heads,
            stride=stride,
            eval_datapath='data/Tensorflow',
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
                tf.keras.metrics.AUC(name='ROC-AUC'),
                # F1Score(average='binary'),
                # PositiveRate(name='PositiveRate'),
                # PredictedPositives(name='PredictedPositives')
                ]}
                )

        return model
    
    def fit(self, hp, model, x, y, **kwargs):
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

        window_size = hp.get("window_size")
        # down_sample_by = hp.get("down_sample_by")

        datapath = f"{kwargs['datapath']}/window_{window_size}/labelled"
        output_dir = kwargs['output_dir']
        model_nickname = f'model-{timestamp}'
        save_checkpoints = kwargs['save_checkpoints']
        
        saved_models_dir = f'{output_dir}/savedmodels/{model_nickname}'
        tensorboard_logdir = f"{output_dir}/tb_logs/{model_nickname}"
        self._save_hps(f"{output_dir}/hps", f"{model_nickname}.json", hp)

        train_val_ids = x
        
        train_data = create_dataset(
            datapath,
            filters=[
                partial(keep_subjects, subject_ids=train_val_ids),
                lambda x, y: x['central_epoch_id'] % 8 != 0
                ],
            batch_size=128
            )

        val_data = create_dataset(
            datapath,
            filters=[
                partial(keep_subjects, subject_ids=train_val_ids),
                lambda x, y: x['central_epoch_id'] % 8 == 0
                ],
            batch_size=100
            )
        
        callbacks = [
            CustomTensorBoard(log_dir=f"{tensorboard_logdir}"),  # probably redundant
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=0)
        ]
        
        callbacks += kwargs["callbacks"]  # Callbacks from Tuner
        
        if save_checkpoints:
            callbacks += [tf.keras.callbacks.ModelCheckpoint(f'{saved_models_dir}', monitor='val_loss', save_best_only=True)]

        return model.fit(
            train_data,
            # class_weight={0: 0.6, 1: 0.4},
            epochs=1000,
            steps_per_epoch=100,
            validation_data=val_data,
            validation_steps=50,
            callbacks=callbacks
            )
        
        
if __name__ == '__main__':

    output_dir = 'training_output'
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    search_name = f"search-{timestamp}"
    hyper_model = HyperModel()
    
    hp = HyperParameters()

    hp.Choice("down_sample_by", [10, 100, 1000])
    hp.Choice("num_conv_filters", [32, 64, 128])
    hp.Choice("window_size", [3, 11, 21])
    hp.Choice("num_attention_heads", [1, 2, 4])
    hp.Choice("stride", [1, 2, 3])

    tuner = inner_cv(Hyperband)(
        hypermodel=hyper_model,
        inner_cv=KFold(n_splits=3, random_state=12345, shuffle=True),
        save_output=False,
        save_history=True,
        restore_best=False,
        # # # Tuner parameters
        objective=Objective("val_pred_macro-F1", direction="max"),
        directory="tuning",
        project_name=f"./{search_name}/",
        max_epochs=250,
        factor=3,
        hyperband_iterations=1,  # as high as is affordable
        seed=None,
        hyperparameters=hp,
        # tune_new_entries=True,
        # allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        overwrite=False,  # False resumes previous search
        )

    x_train = np.array(config['subject_ids'])
    y_train = np.array(config['subject_ids'])

    tuner.search(
        x_train,
        y_train,
        datapath=f"data/Tensorflow",
        output_dir=f"{output_dir}/{search_name}",
        save_checkpoints=True,
        # validation_split=0.2,
        batch_size="full-batch",
        validation_batch_size="full-batch",
        epochs=250,
        verbose=True,
        callbacks=[keras.callbacks.TensorBoard(f"{output_dir}/{search_name}/sreach_tb_logs")],
    )
    
    # df = pd_inner_cv_get_result(tuner)
