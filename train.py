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
from models import CNNModel, Transformer


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
    
    norm_diff = triaxial_l2_norm[:, 1:, :] - triaxial_l2_norm[:, :-1, :]
    # [[0, 0], [1, 0], [0, 0]] means only insert 1 pad in axis=1 and before the values
    norm_diff = tf.pad(norm_diff, [[0, 0], [1, 0], [0, 0]])  # pad the diff at t=0 to make it the same shape again

    feature_list = [
        tf.expand_dims(features['X'], axis=-1),
        tf.expand_dims(features['Y'], axis=-1),
        tf.expand_dims(features['Z'], axis=-1),
        tf.expand_dims(features['Temp'], axis=-1),
        triaxial_l2_norm,
        # tf.abs(norm_diff)
    ]

    stacked_features = tf.concat(feature_list, axis=-1)
    
    # stacked_features.shape is (window_size, epoch len, n_sequences)
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


def create_dataset(path, compressed=False, filters=None, has_labels=True, batch_size=None, repeat=True, shuffle=True):
    
    if tf.io.gfile.exists(path):  # This means it's either a single file name or a directory, not a pattern
        
        if tf.io.gfile.isdir(path):
            print("Note: When passing a directory to `create_dataset`, compression will determine which files are included (.gz or not)")
            extenstion = "tfrecord" + (".gz" if compressed else "")
            files = tf.data.Dataset.list_files(f"{path}/*.{extenstion}", shuffle=True)
        else:
            files = path
    
    else:  # It must be a pattern like data/sub01_*.tfrecord
        files = tf.data.Dataset.list_files(path, shuffle=True)

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP' if compressed else None)

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
        CustomTensorBoard(log_dir=f"{tensorboard_logdir}/{model_nickname}"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=8, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=16, start_from_epoch=0)
    ]
    if save_checkpoints:
        callbacks += [tf.keras.callbacks.ModelCheckpoint(f'{saved_models_dir}/{model_nickname}', monitor='val_loss', save_best_only=True)]

    model = CNNModel(down_sample_by=3)
    # model = Transformer(
    #     head_size=32,
    #     d_model=5,  # num of variables
    #     num_heads=4,
    #     ff_dim=4,
    #     num_transformer_blocks=1,
    #     mlp_units=[128],
    #     mlp_dropout=0.4,
    #     dropout=0.25,
    #     down_sample_by=3
    # )

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),  # Legacy is because the current one runs slow on M1/M2 macs
        loss={'pred': tf.keras.losses.BinaryCrossentropy(name='Loss')},
        # loss={'pred': tf.keras.losses.BinaryFocalCrossentropy(name='Loss')},
        metrics={'pred': [
            tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision'),
            F1Score(name='F1Score'),
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
        validation_steps=50,
        callbacks=callbacks
        )

    # model.save(f'{saved_models_dir}/{model_nickname}')

    return model


if __name__ == '__main__':

    datapath = f"data/Tensorflow/window_{config['window_size']}/labelled"
    psg_labes_path = 'data/PSG-Labels'
    output_dir = 'training_output'
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    performance_output_path = f'{output_dir}/Performance/{timestamp}'

    print('*'*20)
    print(f'Model Timestamp: {timestamp}')
    print('*'*20)

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

        # if len(test_ids) > 0:
        if False:
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
