import tensorflow as tf
import keras

class SparseRecall(keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), num_classes)
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)
        return super().update_state(y_true, y_pred, sample_weight)


class SparsePrecision(keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), num_classes)
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)
        return super().update_state(y_true, y_pred, sample_weight)
