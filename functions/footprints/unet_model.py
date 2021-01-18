# -*- coding: utf-8 -*-
from keras.layers import Input, Dropout, Activation, Convolution2D, MaxPool2D, concatenate, Conv2DTranspose, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
 
import tensorflow as tf

def get_unet(n_classes, input_shape, batch_norm=False, dropout=False, dropout_rate=0.2, seed=42):
  # Contracting
  inputs = Input(input_shape)
  c1_1 = BatchNormalization()(inputs) if batch_norm else inputs
  c1_1 = Convolution2D(64,3,activation='relu', padding='same')(c1_1)
  c1_1 = BatchNormalization()(c1_1) if batch_norm else c1_1
  c1_2 = Convolution2D(64,3,activation='relu', padding='same')(c1_1)
  c1_2 = BatchNormalization()(c1_2) if batch_norm else c1_2
  p_1 = MaxPool2D(pool_size=(2, 2), strides=2)(c1_2)
  p_1 = Dropout(rate=dropout_rate,seed=seed)(p_1) if dropout else p_1

  c2_1 = Convolution2D(128,3,activation='relu', padding='same')(p_1)
  c2_1 = BatchNormalization()(c2_1) if batch_norm else c2_1
  c2_2 = Convolution2D(128,3,activation='relu', padding='same')(c2_1)
  c2_2 = BatchNormalization()(c2_2) if batch_norm else c2_2
  p_2 = MaxPool2D(pool_size=(2, 2), strides=2)(c2_2)
  p_2 = Dropout(rate=dropout_rate,seed=seed)(p_2) if dropout else p_2

  c3_1 = Convolution2D(256,3,activation='relu', padding='same')(p_2)
  c3_1 = BatchNormalization()(c3_1) if batch_norm else c3_1
  c3_2 = Convolution2D(256,3,activation='relu', padding='same')(c3_1)
  c3_2 = BatchNormalization()(c3_2) if batch_norm else c3_2
  p_3 = MaxPool2D(pool_size=(2, 2), strides=2)(c3_2)
  p_3 = Dropout(rate=dropout_rate,seed=seed)(p_3) if dropout else p_3

  c4_1 = Convolution2D(512,3,activation='relu', padding='same')(p_3)
  c4_1 = BatchNormalization()(c4_1) if batch_norm else c4_1
  c4_2 = Convolution2D(512,3,activation='relu', padding='same')(c4_1)
  c4_2 = BatchNormalization()(c4_2) if batch_norm else c4_2
  p_4 = MaxPool2D(pool_size=(2, 2), strides=2)(c4_2)
  p_4 = Dropout(rate=dropout_rate,seed=seed)(p_4) if dropout else p_4

  c5_1 = Convolution2D(1024,3,activation='relu', padding='same')(p_4)
  c5_1 = BatchNormalization()(c5_1) if batch_norm else c5_1
  c5_2 = Convolution2D(1024,3,activation='relu', padding='same')(c5_1)
  c5_2 = BatchNormalization()(c5_2) if batch_norm else c5_2

  # Expanding
  u6 = concatenate([c4_2, Conv2DTranspose(512,2,strides=(2,2), padding='same')(c5_2)])
  u6 = Dropout(rate=dropout_rate,seed=seed)(u6) if dropout else u6
  c6_1 = Convolution2D(512,3,activation='relu', padding='same')(u6)
  c6_1 = BatchNormalization()(c6_1) if batch_norm else c6_1
  c6_2 = Convolution2D(512,3,activation='relu', padding='same')(c6_1)
  c6_2 = BatchNormalization()(c6_2) if batch_norm else c6_2

  u7 = concatenate([c3_2, Conv2DTranspose(256,2,strides=(2,2))(c6_2)])
  u7 = Dropout(rate=dropout_rate,seed=seed)(u7) if dropout else u7
  c7_1 = Convolution2D(256,3,activation='relu', padding='same')(u7)
  c7_1 = BatchNormalization()(c7_1) if batch_norm else c7_1
  c7_2 = Convolution2D(256,3,activation='relu', padding='same')(c7_1)
  c7_2 = BatchNormalization()(c7_2) if batch_norm else c7_2

  u8 = concatenate([c2_2, Conv2DTranspose(128,2,strides=(2,2))(c7_2)])
  u8 = Dropout(rate=dropout_rate,seed=seed)(u8) if dropout else u8
  c8_1 = Convolution2D(128,3,activation='relu', padding='same')(u8)
  c8_1 = BatchNormalization()(c8_1) if batch_norm else c8_1
  c8_2 = Convolution2D(128,3,activation='relu', padding='same')(c8_1)
  c8_2 = BatchNormalization()(c8_2) if batch_norm else c8_2

  u9 = concatenate([c1_2, Conv2DTranspose(64,2,strides=(2,2))(c8_2)])
  u9 = Dropout(rate=dropout_rate,seed=seed)(u9) if dropout else u9
  c9_1 = Convolution2D(64,3,activation='relu', padding='same')(u9)
  c9_1 = BatchNormalization()(c9_1) if batch_norm else c9_1
  c9_2 = Convolution2D(64,3,activation='relu', padding='same')(c9_1)
  c9_2 = BatchNormalization()(c9_2) if batch_norm else c9_2

  # Output
  c10 = Convolution2D(n_classes,(1, 1),activation='sigmoid')(c9_2)

  return Model(inputs=inputs,outputs=c10)