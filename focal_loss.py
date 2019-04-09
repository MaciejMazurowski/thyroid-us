import tensorflow as tf
from keras import backend as K


def focal_loss(gamma=2, alpha=2):

    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt + 1e-6), axis=-1)

    return focal_loss_fixed
