import tensorflow as tf
import re
from google.cloud import storage
import os


def drop_subjects(features, labels=None, subject_ids=[]):
    # This is vectorized and with reduce_all, because subject_ids is a list of possibly multiple ids
    return tf.reduce_all(tf.not_equal(features['subject_id'], subject_ids))


def keep_subjects(features, labels=None, subject_ids=[]):
    # This is vectorized and with reduce_any, because subject_ids is a list of possibly multiple ids
    return tf.reduce_any(tf.equal(features['subject_id'], subject_ids))


def list_blobs(bucket_path, delimiter=None):
    bucket_name = bucket_path.split('/')[2]
    prefix = "/".join(bucket_path.split('/')[3:]) + "/"
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    return [blob.name for blob in blobs]


def list_all_subject_ids(path, file_type):
    if path.startswith('gs://'):
        filenames = [fn.upper().split('/')[-1] for fn in list_blobs(path) if fn.endswith(file_type)]
    else:
        filenames = [fn.upper() for fn in os.listdir(path) if fn.endswith(file_type)]

    subject_ids = []
    for fn in filenames:
        id = re.findall("[PHD]\d{3}", fn)[0]
        if id not in subject_ids:
            subject_ids.append(id)
        
    return subject_ids