import tensorflow as tf
# performance metric
def dice_coef(y_true, y_pred, smooth_bias=1):
  itsct = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
  union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
  return (2. * itsct + smooth_bias) / (union + smooth_bias)