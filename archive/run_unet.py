# %%
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
from keras.models import Model
import argparse
import sys
import os
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print(tf.__version__)
# %%

# set args manually for notebook
args = argparse.ArgumentParser()
args.dataset_dir = "/groups/CS156b/data"
args.preprocessed_dir = "../preprocessed_data/basic_preprocessed"
args.results_dir = "../results"
args.batch_size = 32
args.learning_rate = 0.001
args.epochs = 10
args.num_train_samples = None
args.num_test_samples = None

diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
dataset_dir = args.dataset_dir
train_valid_df = pd.read_csv("/home/jyhuang/preprocessed_data/train_frontal.csv")
test_df = pd.read_csv("/home/jyhuang/preprocessed_data/test_frontal.csv")

# splitting to train and validation
train_valid_pids = train_valid_df["patient_id"].unique()
train_pids = train_valid_pids[:int(len(train_valid_pids) * 0.8)]
valid_pids = train_valid_pids[int(len(train_valid_pids) * 0.8):]
train_df = train_valid_df[train_valid_df["patient_id"].isin(train_pids)]
valid_df = train_valid_df[train_valid_df["patient_id"].isin(valid_pids)]
# %%
train_valid_generator = ImageDataGenerator(rescale=1. / 255)
test_generator = ImageDataGenerator(rescale=1. / 255)

print("Setting up generators")
train_gen = train_valid_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=dataset_dir,
    x_col="Path",
    y_col=diseases,
    class_mode="raw",
    target_size=(256, 256),
    batch_size=args.batch_size,
    shuffle=True,
    color_mode='grayscale'
)
# %%
valid_gen = train_valid_generator.flow_from_dataframe(
    dataframe=valid_df,
    directory=dataset_dir,
    x_col="Path",
    y_col=diseases,
    class_mode="raw",
    target_size=(256, 256),
    batch_size=args.batch_size,
    shuffle=True,
    color_mode='grayscale'
)
# %%
test_gen = ImageDataGenerator().flow_from_dataframe(
    dataframe=test_df,
    directory=dataset_dir,
    x_col="Path",
    target_size = (256, 256),
    class_mode=None,
    color_mode='grayscale'
)

DIMS = 256
BC = 16 # Base channels: how many channels (kernels) do we have at our base level?  Ronneberger used 64, but 32 is sufficient
pool_K = (2,2) # For MaxPooling2D, the extent we downsampled--here we collapse every 2x2 to a 1x1
Kernel = (3,3) # Size of our convolutional kernel, a 3x3 square of weights!

# First we define an input layer; necessary for all models!  It requires a name and a shape
inputs = Input((DIMS,DIMS,1), name = 'inputs')
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
d1 = Dense(8, activation='relu')(flat)
d2 = Dense(len(diseases), activation='sigmoid')(d1)

model = Model(inputs=[inputs], outputs=[d2])
opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
model.compile(loss='binary_crossentropy', optimizer=opt)
print(model.summary())

model.fit(
    train_gen,
    steps_per_epoch=len(train_pids) // args.batch_size,
    epochs=args.epochs,
    validation_data=valid_gen,
    validation_steps=len(valid_pids) // args.batch_size,
    # verbose=1
)