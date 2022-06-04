import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import argparse
import sys
import os

class UNet():
    def __init__(self):
        pass
        
    def get_model(self, img_size):
        DIMS = img_size
        BC = 32 # Base channels: how many channels (kernels) do we have at our base level?  Ronneberger used 64, but 32 is sufficient
        pool_K = (2,2) # For MaxPooling2D, the extent we downsampled--here we collapse every 2x2 to a 1x1
        Kernel = (3,3) # Size of our convolutional kernel, a 3x3 square of weights!
        # First we define an input layer; necessary for all models!  It requires a name and a shape
        inputs = Input((DIMS,DIMS,3), name = 'inputs')
        # Our basic 2D conv layer, with kernel shape (3,3).  Some differences to note with the original:
        # 1) We use same padding to retain image shape--basically, convolutions "nibble" away at edges sequentially when using 'valid'
        # padding; with 'same' padding, TF zero-pads the edges to prevent dimensional loss with minimal effects on results
        # 2) We use SeLU instead of ReLU.  It functions like a hybrid of sigmoid on the negative side and linear on the right.  
        # Read up on it here: https://mlfromscratch.com/activation-functions-explained/#/
        # As far as why I use it, it permits negative values (which is potentialy important for certain problems), is simple to differnetiate
        # and apparently prevents vanishing/exploding gradients through normalization
        # 3) We use the loecun normal as our kernel initializer, as it is always paired with SeLU
        conv1 = Conv2D(BC, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(inputs)
        conv1 = Conv2D(BC, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv1)
        # MaxPool here to summarize most impactful data and ruduce sensitivity to object scale (size)
        pool1 = MaxPooling2D(pool_size=pool_K)(conv1)
        conv2 = Conv2D(BC*2, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(pool1)
        conv2 = Conv2D(BC*2, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=pool_K)(conv2)
        conv3 = Conv2D(BC*4, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(pool2)
        conv3 = Conv2D(BC*4, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=pool_K)(conv3)

        conv4 = Conv2D(BC*8, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(pool3)
        conv4 = Conv2D(BC*8, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=pool_K)(conv4)
        conv5 = Conv2D(BC*16, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(pool4)
        conv5 = Conv2D(BC*16, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv5)
        # These are the up-conv layers, where we first use nearest neighbor upsampling to regenerate dimensionality for pixel-wise classification
        # NearestNeighbors is a pretty bad way of upsampling, so immediately performing a Conv2D on it is somewhat similar to polishing/super-resolving it
        up6 = Conv2D(BC*8, 2, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(UpSampling2D(size = pool_K)(conv5))
        # To combat vanishing gradient (too many chain rule ops can lead to very small updates), \
        # we short circuit features across the UNet with concatenations
        merge6 = Concatenate(axis = -1)([conv4,up6])
        conv6 = Conv2D(BC*8, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(merge6)
        conv6 = Conv2D(BC*8, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv6)
        up7 = Conv2D(BC*4, 2, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(UpSampling2D(size = pool_K)(conv6))
        merge7 = Concatenate(axis = -1)([conv3,up7])
        conv7 = Conv2D(BC*4, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(merge7)
        conv7 = Conv2D(BC*4, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv7)
        up8 = Conv2D(BC*2, 2, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(UpSampling2D(size = pool_K)(conv7))
        merge8 = Concatenate(axis = -1)([conv2,up8])
        conv8 = Conv2D(BC*2, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(merge8)
        conv8 = Conv2D(BC*2, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv8)
        up9 = Conv2D(BC, 2, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(UpSampling2D(size = pool_K)(conv8))
        merge9 = Concatenate(axis = -1)([conv1,up9])
        conv9 = Conv2D(BC, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(merge9)
        conv9 = Conv2D(BC, Kernel, activation = 'selu', padding = 'same', kernel_initializer = 'lecun_normal')(conv9)
        # We set our output activation to be sigmoid, since we know our label image exists in range [0,1]
        # Note the name!
        flat = Flatten()(conv9)
        # d1 = Dense(8, activation='relu')(flat)
        # d2 = Dense(len(diseases), activation='sigmoid')(d1)
        return Model(inputs=[inputs], outputs=[flat])