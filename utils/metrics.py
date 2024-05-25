import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    # ToDo: TF doesn't have an F1 metric (tfa does, but didn't want to use that).
    #  This used to be MaskedF1. After we created the wrapper MaskerMetric class I changed this to be a normal F1,
    #  so that it can be wrapped by that class but didn't test it.
    #  Test this before using it.
    def __init__(self, name='F1Score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.condition_true = self.add_weight(name='condition_true', initializer='zeros')
        self.predicted_true = self.add_weight(name='pred_true', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5

        tp = tf.logical_and(tf.cast(y_true, tf.int32) == 1, tf.cast(y_pred, tf.int32) == 1)
        tp = tf.cast(tp, dtype=tf.float32)

        condition_true = (tf.cast(y_true, tf.int32) == 1)
        condition_true = tf.cast(condition_true, dtype=tf.float32)

        predicted_true = (tf.cast(y_pred, tf.int32) == 1)
        predicted_true = tf.cast(predicted_true, dtype=tf.float32)
        self.tp.assign_add(tf.reduce_sum(tp))
        self.condition_true.assign_add(tf.reduce_sum(condition_true))
        self.predicted_true.assign_add(tf.reduce_sum(predicted_true))

    def result(self):
        return 2 * self.tp / (self.condition_true+self.predicted_true)

    def reset_state(self):
        self.tp.assign(0.)
        self.condition_true.assign(0.)
        self.predicted_true.assign(0.)


class PositiveRate(tf.keras.metrics.Metric):

    def __init__(self, name='positive_rate', **kwargs):
        super(PositiveRate, self).__init__(name=name, **kwargs)
        self.positive_samples = self.add_weight(name='positive_rate', initializer='zeros')
        self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        # mask = tf.cast(mask, dtype=y_true.dtype)

        # masked_y_true = y_true * mask

        self.positive_samples.assign_add(tf.reduce_sum(y_true))
        self.n_items.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.positive_samples / self.n_items

    def reset_state(self):
        self.positive_samples.assign(0.)
        self.n_items.assign(0.)


class PredictedPositives(tf.keras.metrics.Metric):

    def __init__(self, name='pred_positives', **kwargs):
        super(PredictedPositives, self).__init__(name=name, **kwargs)
        self.n_pred_positives = self.add_weight(name='pred_returned', initializer='zeros')
        self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5
        # mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        # mask = tf.cast(mask, dtype=y_true.dtype)

        # masked_y_pred = y_pred * mask

        self.n_pred_positives.assign_add(tf.reduce_sum(y_pred))
        self.n_items.assign_add(tf.cast(tf.size(y_pred), tf.float32))

    def result(self):
        return self.n_pred_positives / self.n_items

    def reset_state(self):
        self.n_pred_positives.assign(0.)
        self.n_items.assign(0.)

if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import f1_score

    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.uniform(0, 1, size=100)

    f1 = F1Score()
    f1.reset_states()
    f1.update_state(y_true=y_true, y_pred=y_pred)
    print(f1.result())

    print(f1_score(y_true, np.round(y_pred)))