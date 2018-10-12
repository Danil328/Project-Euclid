import keras.backend as K

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))

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