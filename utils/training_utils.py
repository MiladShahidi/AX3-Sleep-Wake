import tensorflow as tf


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    This class inherits from and modifies TensorBoard callback
    to add learning rate to TensorBoard logs
    """
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'learning_rate': self.model.optimizer.learning_rate})
        super().on_epoch_end(epoch, logs)
