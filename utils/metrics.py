import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    # ToDo: TF doesn't have an F1 metric (tfa does, but didn't want to use that).
    #  This used to be MaskedF1. After we created the wrapper MaskerMetric class I changed this to be a normal F1,
    #  so that it can be wrapped by that class but didn't test it.
    #  Test this before using it.
    def __init__(self, name='F1', average='binary', **kwargs):
        super(F1Score, self).__init__(name=f'{average}-{name}', **kwargs)

        self.average = average
        
        self.classes = [0, 1]  # only binary classification supported for now
        # Apparently TF doesn't allow saving dicts with non-string keys. And these dicts will have to be serialized and
        # saved as part of the model. So, in addition to numerical class labels I use these string class labels.
        self.class_labels = [str(cls) for cls in self.classes]

        self.label_positive = {cls: self.add_weight(name='label_positive', initializer='zeros') for cls in self.class_labels}
        self.predicted_positive = {cls: self.add_weight(name='predicted_positive', initializer='zeros') for cls in self.class_labels}
        self.tp = {cls: self.add_weight(name='tp', initializer='zeros') for cls in self.class_labels}

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5

        # For various averaging methods (other than binary) we need to calculate F1 for each class separately
        for cls, cls_label in zip(self.classes, self.class_labels):

            # positive here means belonging to that class. not label = 1 necessarily
            cls_label_positive = (tf.cast(y_true, tf.int32) == cls)
            cls_predicted_positive = (tf.cast(y_pred, tf.int32) == cls)
            cls_tp = tf.logical_and(cls_label_positive, cls_predicted_positive)

            cls_label_positive = tf.cast(cls_label_positive, dtype=tf.float32)
            cls_predicted_positive = tf.cast(cls_predicted_positive, dtype=tf.float32)
            cls_tp = tf.cast(cls_tp, dtype=tf.float32)

            self.tp[cls_label].assign_add(tf.reduce_sum(cls_tp))
            self.label_positive[cls_label].assign_add(tf.reduce_sum(cls_label_positive))
            self.predicted_positive[cls_label].assign_add(tf.reduce_sum(cls_predicted_positive))

    def result(self):
        # The below formula is equivalent to: F1 = (2*tp) / (2*tp + fp + fn)
        # Note that tp + fn = label_positive and tp + fp = predicted_positive
        class_wise_f1 = {
            cls: (2*self.tp[cls]) / (self.label_positive[cls] + self.predicted_positive[cls]) for cls in self.class_labels
        }
        
        if self.average.lower() == 'binary':
            return class_wise_f1["1"]  # for TF serializability, keys of this dict have to be strings
        elif self.average.lower() == 'macro':
            return sum(class_wise_f1.values()) / len(self.classes)
        else:
            raise ValueError("F1: average should be one of ['binary', 'macro']")

    def reset_state(self):
        for cls in self.class_labels:
            self.tp[cls].assign(0.)
            self.label_positive[cls].assign(0.)
            self.predicted_positive[cls].assign(0.)


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

    average_method = 'macro'
    for _ in range(1000):
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.random.uniform(0, 1, size=100)

        f1 = F1Score(average=average_method)
        f1.reset_states()
        f1.update_state(y_true=y_true, y_pred=y_pred)

        a = f1.result().numpy()
        b = f1_score(y_true, np.round(y_pred), average=average_method)

        if not np.isclose(a, b):
            print('Error')
            print(a)
            print(b)
            break