import tensorflow as tf


# loss function
def weighted_cross_entropy(beta):
    """ Weighted cross entropy loss function with weighting beta """

    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            y_true, y_pred, pos_weight=beta)
        return tf.reduce_mean(loss)

    return loss
