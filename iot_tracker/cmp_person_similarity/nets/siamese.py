import os

import numpy as np
import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.models import Model

from .vgg import VGG16


#-------------------------#
#   siamese network
#-------------------------#
def siamese(input_shape):
    vgg_model = VGG16()
    
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)

    #------------------------------------------#
    #   extract feature
    #------------------------------------------#
    encoded_image_1 = vgg_model.call(input_image_1)
    encoded_image_2 = vgg_model.call(input_image_2)

    #-------------------------#
    #   minus and get absolute value
    #-------------------------#
    l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_image_1, encoded_image_2])

    #-------------------------#
    #   connect dense layer
    #-------------------------#
    out = Dense(512,activation='relu')(l1_distance)
    #---------------------------------------------#
    #   use sigmoid activation fix value from 0~1。
    #---------------------------------------------#
    out = Dense(1,activation='sigmoid')(out)

    model = Model([input_image_1, input_image_2], out)
    return model
