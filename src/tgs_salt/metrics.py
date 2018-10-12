import keras.backend as K
import tensorflow as tf
import numpy as np

def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_flattened = K.flatten(y_true)
    y_pred_flattened = K.round(K.flatten(y_pred))
    intersection = K.sum(y_true_flattened * y_pred_flattened)

    return K.mean((2. * intersection + smooth) / (K.sum(y_true_flattened) + K.sum(y_pred_flattened) + smooth))

def hard_jaccard_coef(y_true, y_pred, smooth=1e-3):
    y_true_flattened = K.flatten(y_true)
    y_pred_flattened = K.round(K.flatten(y_pred))
    intersection = K.sum(y_true_flattened * y_pred_flattened)
    union = (K.sum(y_true_flattened) + K.sum(y_pred_flattened) - intersection)

    return K.mean((intersection + smooth) / (union + smooth))

def map_at_different_iou(y_true, y_pred):
    return tf.py_func(map_at_different_iou_numpy, [y_true, y_pred], tf.float64)

def map_at_different_iou_numpy(y_true, y_pred, smooth=1e-3):
    number_of_masks = y_true.shape[0]
    map_at_different_ious = list()

    for i in range(number_of_masks):
        y_true_flattened = y_true[i].ravel()
        y_pred_flattened = np.round(y_pred[i].ravel())

        intersection = np.sum(y_true_flattened * y_pred_flattened)
        union = np.sum(y_true_flattened) + np.sum(y_pred_flattened) - intersection
        iou = (intersection + smooth) / (union + smooth)

        thresholds = np.arange(0.5, 1, 0.05)
        true_positives = [iou > threshold for threshold in thresholds]
        map_at_different_ious.append(np.mean(true_positives))

    return np.mean(map_at_different_ious)
