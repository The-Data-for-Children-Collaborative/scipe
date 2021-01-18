import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dropout, Activation, Convolution2D, MaxPool2D, concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical

def get_vgg16(n_classes, input_shape, batch_norm=False, dropout=False, dropout_rate=0.2, seed=42, frozen=True):
    from tensorflow.keras.applications import VGG16
    # Pretrained network - NOTE: input shape must have depth 3 for vgg16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.layers.pop() # remove MaxPooling layer
    if frozen:
        for layer in base_model.layers:
            layer.trainable = False

    # Expanding section from U-net, with input and skip connections from pretrained network
    u6 = concatenate([base_model.layers[13].output, Conv2DTranspose(512,2,strides=(2,2), padding='same')(base_model.layers[17].output)])
    u6 = Dropout(rate=dropout_rate,seed=seed)(u6) if dropout else u6
    c6_1 = Convolution2D(512,3,activation='relu', padding='same')(u6)
    c6_1 = BatchNormalization()(c6_1) if batch_norm else c6_1
    c6_2 = Convolution2D(512,3,activation='relu', padding='same')(c6_1)
    c6_2 = BatchNormalization()(c6_2) if batch_norm else c6_2

    u7 = concatenate([base_model.layers[9].output, Conv2DTranspose(256,2,strides=(2,2))(c6_2)])
    u7 = Dropout(rate=dropout_rate,seed=seed)(u7) if dropout else u7
    c7_1 = Convolution2D(256,3,activation='relu', padding='same')(u7)
    c7_1 = BatchNormalization()(c7_1) if batch_norm else c7_1
    c7_2 = Convolution2D(256,3,activation='relu', padding='same')(c7_1)
    c7_2 = BatchNormalization()(c7_2) if batch_norm else c7_2

    u8 = concatenate([base_model.layers[5].output, Conv2DTranspose(128,2,strides=(2,2))(c7_2)])
    u8 = Dropout(rate=dropout_rate,seed=seed)(u8) if dropout else u8
    c8_1 = Convolution2D(128,3,activation='relu', padding='same')(u8)
    c8_1 = BatchNormalization()(c8_1) if batch_norm else c8_1
    c8_2 = Convolution2D(128,3,activation='relu', padding='same')(c8_1)
    c8_2 = BatchNormalization()(c8_2) if batch_norm else c8_2

    u9 = concatenate([base_model.layers[2].output, Conv2DTranspose(64,2,strides=(2,2))(c8_2)])
    u9 = Dropout(rate=dropout_rate,seed=seed)(u9) if dropout else u9
    c9_1 = Convolution2D(64,3,activation='relu', padding='same')(u9)
    c9_1 = BatchNormalization()(c9_1) if batch_norm else c9_1
    c9_2 = Convolution2D(64,3,activation='relu', padding='same')(c9_1)
    c9_2 = BatchNormalization()(c9_2) if batch_norm else c9_2

    c10 = Convolution2D(n_classes,(1, 1),activation='sigmoid')(c9_2)
    model = Model(inputs=base_model.layers[0].input,outputs=c10)
    return model
