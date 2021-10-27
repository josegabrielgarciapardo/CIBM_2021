# When problems with CUDA libraries
# sudo ldconfig /usr/local/cuda-9.0/lib64/

import os
# USE THE SELECTED GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.layers import GlobalMaxPool2D, GlobalAvgPool2D, concatenate, Conv2D, Multiply, BatchNormalization, \
     Dense, MaxPooling2D, TimeDistributed, Lambda, Input, Flatten, Dropout, AveragePooling2D, UpSampling2D, Conv2DTranspose, Reshape
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
#import tensorflow_probability as tfp

# GETTING REPRODUCIBLE THE RESULTS WITH KERAS
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
#--------------------------------------------
warnings.filterwarnings('ignore')


# Define Attention blocks
def attention_block(x, flag_SE):

    x = BatchNormalization(name='att_bn')(x)
    if flag_SE['activate']==True:
        # Implement attention block and Squeeze-Expectation blocks
        x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = SE_block(x, flag_SE['factor'])
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = SE_block(x, flag_SE['factor'])
        #x = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        #x = SE_block(x, flag_SE['factor'])
        out = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)
        out = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(out)
    else:
        # Implement attention block
        x = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', name='att_1')(x)
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='att_2')(x)
        #x = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        out = Conv2D(filters=1, kernel_size=(1,1), strides = (1,1), padding='same', activation='sigmoid', name='att_sig')(x)
        out = Conv2D(filters=256, kernel_size=(1,1),strides = (1,1), padding='same', activation='relu', name='att_3')(out)

    return out

# Define Squeeze-Excitation blocks
def SE_block(inputs, factor):

    # name of filters of the input
    C = inputs.get_shape().as_list()[-1]

    x = GlobalAvgPool2D()(inputs)
    x = Dense(units=int(round(C/factor)), activation='relu')(x)
    out = Dense(units=C, activation='sigmoid')(x)

    output = Multiply()([inputs, out])

    return output

# Build CAE
def build_CAE(shape_images, params_dc):
    input_layer = Input(shape=(shape_images['rows'], shape_images['cols'], shape_images['channel']), name='new_input')
    x = input_layer

    filt = int(params_dc['first_filter'])

    # ----------------------------------------  ENCODER  ------------------------------------- #
    x = Conv2D(filters=filt, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    for i in range(params_dc['num_convs']-2):
        filt = filt * 2
        x = Conv2D(filters=filt, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(x)

    filt = filt*2
    if params_dc['feat_mode']['option'] == 'None':
        x = Conv2D(filters=filt, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='flatten')(x)
        encoder_output = x
    else:
        x = Conv2D(filters=filt, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='flatten_0')(x)

        # ------------------------------------ BOTTLE NECK ------------------------------------ #
        h,w,d = x.shape[1], x.shape[2], x.shape[3]

        if params_dc['feat_mode']['option']=='GMP':
            f = GlobalMaxPool2D(name='bottle_neck')(x)
        elif params_dc['feat_mode']['option']=='GAP':
            f = GlobalAvgPool2D(name='bottle_neck')(x)
        elif params_dc['feat_mode']['option']=='Flatten':
            f = Flatten(name='bottle_neck')(x)
        elif params_dc['feat_mode']['option']=='Conv_1_1':
            custom_filt = int(x.shape[-1]/params_dc['feat_mode']['temp'])
            x = Conv2D(filters=custom_filt, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
            f = Flatten(name='bottle_neck')(x)
        elif params_dc['feat_mode']['option']=='GMP_Z':
            x = tf.reduce_max(x, axis=3)
            f = Flatten(name='bottle_neck')(x)
        elif params_dc['feat_mode']['option']=='GAP_Z':
            x = tf.reduce_sum(x, axis=3)
            f = Flatten(name='bottle_neck')(x)
        elif params_dc['feat_mode']['option']=='ATT':
            x_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='att_1')(x)
            x_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='att_2')(x_2)
            x_2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid', name='att_sig')(x_2)
            x_2 = Conv2D(filters=filt, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='att_3')(x_2)
            f = concatenate([x,x_2], axis=3)
            f = Conv2D(filters=filt, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='bottle_neck')(f)

        encoder_output = f

        # ----------------------------------------  DECODER  ------------------------------------- #
        if params_dc['feat_mode']['option']=='GMP' or params_dc['feat_mode']=='GAP':
            x = tf.expand_dims(f, axis=1)
            x = tf.expand_dims(x, axis=1)
            x = UpSampling2D(size=(h,w))(x)
        elif params_dc['feat_mode']['option']=='Flatten':
            x = Reshape((h, w, d))(f)
        elif params_dc['feat_mode']['option']=='Conv_1_1':
            x = Reshape((h,w,custom_filt))(f)
        elif params_dc['feat_mode']['option']=='ATT':
            x = f
        else:
            x = Reshape((h, w, 1))(f)

    for i in range(params_dc['num_convs']-1):
        if params_dc['reconstruction_type']=='Upsampling':
            filt = filt / 2
            x = Conv2D(filters=filt, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(x)
            x = UpSampling2D(size=(2,2))(x)
        else:
            filt = filt / 2
            x = Conv2DTranspose(filters=filt, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)

    # Final layer
    f = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='sigmoid')(x)

    decoder_output = f

    encoder = Model(inputs=input_layer, outputs=encoder_output, name='encoder')
    CAE = Model(inputs=input_layer, outputs=decoder_output, name='CAE')

    print(CAE.summary())

    return encoder, CAE







