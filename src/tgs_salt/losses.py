import keras.backend as K
import tensorflow as tf


def binary_crossentropy(y_true, y_pred, alpha=.75):
    _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)

    pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
    pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))

    return -K.mean(alpha * K.log(pt_1) + (1 - alpha) * K.log(1. - pt_0))

def focal_loss(y_true, y_pred, gamma=2, alpha=.75):
    _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
    pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))

    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) + (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def bce_dice_loss(y_true, y_pred, dice=0.2, bce=0.8):
    return bce * binary_crossentropy(y_true, y_pred) + dice * dice_loss(y_true, y_pred)

def bce_jaccard_loss(y_true, y_pred, jaccard=0.2, bce=0.8):
    return bce * binary_crossentropy(y_true, y_pred) + jaccard * dice_loss(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return 1 - soft_dice_coef(y_true, y_pred)

def jaccard_loss(y_true, y_pred):
    return 1 - soft_jaccard_coef(y_true, y_pred)

def soft_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_flattened = K.flatten(y_true)
    y_pred_flatenned = K.flatten(y_pred)
    intersection = K.sum(y_true_flattened * y_pred_flatenned)

    return K.mean((2. * intersection + smooth) / (K.sum(y_true_flattened) + K.sum(y_pred_flatenned) + smooth))

def soft_jaccard_coef(y_true, y_pred, smooth=1e-3):
    y_true_flattened = K.flatten(y_true)
    y_pred_flattened = K.flatten(y_pred)
    intersection = K.sum(y_true_flattened * y_pred_flattened)
    union = (K.sum(y_true_flattened) + K.sum(y_pred_flattened) - intersection)

    return K.mean((intersection + smooth) / (union + smooth))

def make_loss(loss_name):

    if loss_name == 'binary_crossentropy':
        return binary_crossentropy

    elif loss_name == 'focal_loss':
        return focal_loss

    elif loss_name == 'bce_dice_loss':
        return bce_dice_loss

    elif loss_name == 'bce_jaccard_loss':
        return bce_jaccard_loss

    elif loss_name == 'dice_loss':
        return dice_loss

    elif loss_name == 'jaccard_loss':
        return jaccard_loss

    else:
        ValueError('Unknown loss')
