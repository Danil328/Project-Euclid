from keras.models import Input, Model
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers.merge import concatenate
from keras.initializers import he_normal
from .model_utils import scSE_block, attention_gating_block

def vanilla_unet(input_shape, random_state):
    initializer = he_normal(random_state)
    activation = 'relu'

    inputs            = Input(input_shape, name='input')

    # 1
    bn_1              = BatchNormalization(name='bn_1') (inputs)
    conv_d0a_b        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_d0a-b', kernel_initializer=initializer) (bn_1)
    bn_2              = BatchNormalization(name='bn_2') (conv_d0a_b)
    conv_d0b_c        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_d0b-c', kernel_initializer=initializer) (bn_2)
    conv_d0b_c        = scSE_block(conv_d0b_c)
    bn_3              = BatchNormalization(name='bn_3') (conv_d0b_c)
    pool_d0c_1a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d0c-1a') (bn_3)

    # 2
    conv_d1a_b        = Conv2D(128, (3, 3), padding='valid', activation=activation, name='conv_d1a-b', kernel_initializer=initializer) (pool_d0c_1a)
    bn_4              = BatchNormalization(name='bn_4') (conv_d1a_b)
    conv_d1b_c        = Conv2D(128, (3, 3), padding='valid', activation=activation, name='conv_d1b-c', kernel_initializer=initializer) (bn_4)
    conv_d1b_c        = scSE_block(conv_d1b_c)
    bn_5              = BatchNormalization(name='bn_5') (conv_d1b_c)
    pool_d1c_2a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d1c-2a') (bn_5)

    # 3
    conv_d2a_b        = Conv2D(256, (3, 3), padding='valid', activation=activation, name='conv_d2a-b', kernel_initializer=initializer) (pool_d1c_2a)
    bn_6              = BatchNormalization(name='bn_6') (conv_d2a_b)
    conv_d2b_c        = Conv2D(256, (3, 3), padding='valid', activation=activation, name='conv_d2b-c', kernel_initializer=initializer) (bn_6)
    conv_d2b_c        = scSE_block(conv_d2b_c)
    bn_7              = BatchNormalization(name='bn_7') (conv_d2b_c)
    pool_d2c_3a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d2c-3a') (bn_7)

    # 4
    conv_d3a_b        = Conv2D(512, (3, 3), padding='valid', activation=activation, name='conv_d3a-b', kernel_initializer=initializer) (pool_d2c_3a)
    bn_8              = BatchNormalization(name='bn_8') (conv_d3a_b)
    conv_d3b_c        = Conv2D(512, (3, 3), padding='valid', activation=activation, name='conv_d3b-c', kernel_initializer=initializer) (bn_8)
    conv_d3b_c        = scSE_block(conv_d3b_c)
    bn_9              = BatchNormalization(name='bn_9') (conv_d3b_c)
    pool_d3c_4a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d3c-4a') (bn_9)

    # bottleneck
    conv_d4a_b        = Conv2D(1024, (3, 3), padding='valid', activation=activation, name='conv_d4a-b', kernel_initializer=initializer) (pool_d3c_4a)
    bn_10             = BatchNormalization(name='bn_10') (conv_d4a_b)
    conv_d4b_c        = Conv2D(1024, (3, 3), padding='valid', activation=activation, name='conv_d4a-c', kernel_initializer=initializer) (bn_10)
    bn_11             = BatchNormalization(name='bn_11') (conv_d4b_c)
    upconv_d4c_u3a    = Conv2DTranspose(512, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_d4c_u3a', kernel_initializer=initializer) (bn_11)
    bn_12             = BatchNormalization(name='bn_12') (upconv_d4c_u3a)

    # 4
    crop_d3c_d3cc     = Cropping2D(4, name='crop_d3c-d3cc') (conv_d3b_c)
    crop_d3c_d3cc     = attention_gating_block(shortcut=crop_d3c_d3cc, gating_signal=bn_12, inter_channels=256)
    concat_d3cc_u3a_b = concatenate([crop_d3c_d3cc, bn_12], axis=3, name='concat_d3cc_u3a-b')
    concat_d3cc_u3a_b = scSE_block(concat_d3cc_u3a_b)
    conv_u3b_c        = Conv2D(512, (3, 3), padding='valid', activation=activation, name='conv_u3b-c', kernel_initializer=initializer) (concat_d3cc_u3a_b)
    bn_13             = BatchNormalization(name='bn_13') (conv_u3b_c)
    conv_u3c_d        = Conv2D(512, (3, 3), padding='valid', activation=activation, name='conv_u3c-d', kernel_initializer=initializer) (bn_13)
    bn_14             = BatchNormalization(name='bn_14') (conv_u3c_d)
    upconv_u3d_u2a    = Conv2DTranspose(256, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_u3d_u2a', kernel_initializer=initializer) (bn_14)
    bn_15             = BatchNormalization(name='bn_15') (upconv_u3d_u2a)

    # 3
    crop_d2c_d2cc     = Cropping2D(16, name='crop_d2c-d2cc') (conv_d2b_c)
    crop_d2c_d2cc     = attention_gating_block(shortcut=crop_d2c_d2cc, gating_signal=bn_15, inter_channels=128)
    concat_d2cc_u2a_b = concatenate([crop_d2c_d2cc, bn_15], axis=3, name='concat_d2cc_u2a-b')
    concat_d2cc_u2a_b = scSE_block(concat_d2cc_u2a_b)
    conv_u2b_c        = Conv2D(256, (3, 3), padding='valid', activation=activation, name='conv_u2b-c', kernel_initializer=initializer) (concat_d2cc_u2a_b)
    bn_16             = BatchNormalization(name='bn_16') (conv_u2b_c)
    conv_u2c_d        = Conv2D(256, (3, 3), padding='valid', activation=activation, name='conv_u2c-d', kernel_initializer=initializer) (bn_16)
    bn_17             = BatchNormalization(name='bn_17') (conv_u2c_d)
    upconv_u2d_u1a    = Conv2DTranspose(128, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_u2d_u1a', kernel_initializer=initializer) (bn_17)
    bn_18             = BatchNormalization(name='bn_18') (upconv_u2d_u1a)

    # 2
    crop_d1c_d1cc     = Cropping2D(40, name='crop_d1c-d1cc') (conv_d1b_c)
    crop_d1c_d1cc     = attention_gating_block(shortcut=crop_d1c_d1cc, gating_signal=bn_18, inter_channels=64)
    concat_d1cc_u1a_b = concatenate([crop_d1c_d1cc, bn_18], axis=3, name='concat_d1cc_u1a-b')
    concat_d1cc_u1a_b = scSE_block(concat_d1cc_u1a_b)
    conv_u1b_c        = Conv2D(128, (3, 3), padding='valid', activation=activation, name='conv_u1b-c', kernel_initializer=initializer) (concat_d1cc_u1a_b)
    bn_19             = BatchNormalization(name='bn_19') (conv_u1b_c)
    conv_u1c_d        = Conv2D(128, (3, 3), padding='valid', activation=activation, name='conv_u1c-d', kernel_initializer=initializer) (bn_19)
    bn_20             = BatchNormalization(name='bn_20') (conv_u1c_d)
    upconv_u1d_u0a    = Conv2DTranspose(64, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_u1d_u0a', kernel_initializer=initializer) (bn_20)
    bn_21             = BatchNormalization(name='bn_21') (upconv_u1d_u0a)

    # 1
    crop_d0c_d0cc     = Cropping2D(88, name='crop_d0c-d0cc') (conv_d0b_c)
    crop_d0c_d0cc     = attention_gating_block(shortcut=crop_d0c_d0cc, gating_signal=bn_21, inter_channels=32)
    concat_d0cc_u0a_b = concatenate([crop_d0c_d0cc, bn_21], axis=3, name='concat_d0cc_u0a-b')
    concat_d0cc_u0a_b = scSE_block(concat_d0cc_u0a_b)
    conv_u0b_c        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_u0b-c', kernel_initializer=initializer) (concat_d0cc_u0a_b)
    bn_22             = BatchNormalization(name='bn_22') (conv_u0b_c)
    conv_u0c_d        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_u0c-d', kernel_initializer=initializer) (bn_22)
    bn_24             = BatchNormalization(name='bn_24') (conv_u0c_d)
    conv_u0d_score    = Conv2D(1, (1, 1), padding='valid', activation='sigmoid', name='conv_u0d-score', kernel_initializer=initializer) (bn_24)

    model = Model(inputs=inputs, outputs=conv_u0d_score, name='vanilla_unet')
    model.summary()

    return model