import tensorflow as tf


def drop_subjects(features, labels=None, subject_ids=[]):
    # This is vectorized and with reduce_all, because subject_ids is a list of possibly multiple ids
    return tf.reduce_all(tf.not_equal(features['subject_id'], subject_ids))


def keep_subjects(features, labels=None, subject_ids=[]):
    # This is vectorized and with reduce_any, because subject_ids is a list of possibly multiple ids
    return tf.reduce_any(tf.equal(features['subject_id'], subject_ids))
