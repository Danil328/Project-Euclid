import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation, BatchNormalization, Lambda
from keras.layers import multiply, add, concatenate


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

def deep_supervision(input_tensor, in_channels, base_name, activation, initializer):

    conv_a = Conv2D(
        in_channels, (1, 1),
        padding='same',
        activation=activation,
        name='conv_{0}_a'.format(base_name),
        kernel_initializer=initializer) (input_tensor)
    bn_a = BatchNormalization(name='bn_{0}_a'.format(base_name))(conv_a)
    conv_b = Conv2D(
        in_channels // 2, (1, 1),
        padding='same',
        activation=activation,
        name='conv_{0}_b'.format(base_name),
        kernel_initializer=initializer) (bn_a)
    bn_b = BatchNormalization(name='bn_{0}_b'.format(base_name))(conv_b)
    conv_c = Conv2D(
        1, (1, 1),
        padding='same',
        name='conv_{0}_c'.format(base_name),
        kernel_initializer=initializer) (bn_b)

    conv_score = Activation('sigmoid', name='conv_{0}_score'.format(base_name)) (conv_c)

    return conv_score


def hypercolumn(last_layer, *args):
    layers = list()
    layers.append(last_layer)

    for layer in args:
        layers.append(Lambda(resize_bilinear, arguments={'target_tensor': last_layer}) (layer))

    return add(layers)

def resize_bilinear(input_tensor, target_tensor):
    target_height = target_tensor.get_shape()[1]
    target_width = target_tensor.get_shape()[2]

    return tf.image.resize_bilinear(input_tensor, [target_height.value, target_width.value])