from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation
from keras.layers import multiply, add


def attention_gating_block(shortcut, gating_signal, inter_channels):
    theta = Conv2D(inter_channels, (1, 1), use_bias=True, padding='same') (shortcut)
    phi = Conv2D(inter_channels, (1, 1), use_bias=True, padding='same') (gating_signal)

    concat_theta_phi = add([theta, phi])
    psi = Activation('relu') (concat_theta_phi)
    compatibility_score = Conv2D(1, (1, 1), use_bias=True, padding='same') (psi)
    alpha = Activation('sigmoid') (compatibility_score)

    return multiply([alpha, shortcut])

def cSE_block(input_tensor, ratio=2):
    feature_maps = input_tensor._keras_shape[3]

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, feature_maps))(se)
    se = Dense(feature_maps // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(feature_maps, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return multiply([input_tensor, se])

def sSE_block(input_tensor):
    se = Conv2D(1, (1, 1), strides=(1, 1), activation='sigmoid', padding='same', use_bias=False) (input_tensor)

    return multiply([input_tensor, se])

def scSE_block(input_tensor):
    channel_se = cSE_block(input_tensor)
    spatial_se = sSE_block(input_tensor)

    return add([channel_se, spatial_se])