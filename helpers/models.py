from numpy import double
import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v3 import MobileNetV3Large, MobileNetV3Small
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
from keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
import tensorflow as tf
from .unet import UNet
import os
from icecream import ic

# %%
models = {
    "UNet": UNet,
    "Xception": Xception,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNet50V2": ResNet50V2,
    "ResNet101V2": ResNet101V2,
    "ResNet152V2": ResNet152V2,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2,
    "MobileNetV3Large": MobileNetV3Large,
    "MobileNetV3Small": MobileNetV3Small,
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "NASNetMobile": NASNetMobile,
    "NASNetLarge": NASNetLarge,
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB5": EfficientNetB5,
    "EfficientNetB6": EfficientNetB6,
    "EfficientNetB7": EfficientNetB7,
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B1": EfficientNetV2B1,
    "EfficientNetV2B2": EfficientNetV2B2,
    "EfficientNetV2B3": EfficientNetV2B3,
    "EfficientNetV2S": EfficientNetV2S,
    "EfficientNetV2M": EfficientNetV2M,
    "EfficientNetV2L": EfficientNetV2L
}

preprocessing_fns = {
    "UNet": None,
    "Xception": tf.keras.applications.xception.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
    "VGG19": tf.keras.applications.vgg19.preprocess_input,
    "ResNet50": tf.keras.applications.resnet.preprocess_input,
    "ResNet101": tf.keras.applications.resnet.preprocess_input,
    "ResNet152": tf.keras.applications.resnet.preprocess_input,
    "ResNet50V2": tf.keras.applications.resnet_v2.preprocess_input,
    "ResNet101V2": tf.keras.applications.resnet_v2.preprocess_input,
    "ResNet152V2": tf.keras.applications.resnet_v2.preprocess_input,
    "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
    "InceptionResNetV2": tf.keras.applications.inception_resnet_v2.preprocess_input,
    "MobileNetV3Large": tf.keras.applications.mobilenet_v3.preprocess_input,
    "MobileNetV3Small": tf.keras.applications.mobilenet_v3.preprocess_input,
    "DenseNet121": tf.keras.applications.densenet.preprocess_input,
    "DenseNet169": tf.keras.applications.densenet.preprocess_input,
    "DenseNet201": tf.keras.applications.densenet.preprocess_input,
    "NASNetMobile": tf.keras.applications.nasnet.preprocess_input,
    "NASNetLarge": tf.keras.applications.nasnet.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB1": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB2": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB3": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB4": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB5": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB6": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB7": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetV2B0": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2B1": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2B2": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2B3": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2S": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2M": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2L": tf.keras.applications.efficientnet_v2.preprocess_input
}

default_img_size = {
    "UNet": 224,
    "Xception": 299,
    "VGG16": 224,
    "VGG19": 224,
    "ResNet50": 224,
    "ResNet101": 224,
    "ResNet152": 224,
    "ResNet50V2": 224,
    "ResNet101V2": 224,
    "ResNet152V2": 224,
    "InceptionV3": 299,
    "InceptionResNetV2": 299,
    "MobileNetV3Large": 224,
    "MobileNetV3Small": 224,
    "DenseNet121": 224,
    "DenseNet169": 224,
    "DenseNet201": 224,
    "NASNetMobile": 224,
    "NASNetLarge": 331,
    "EfficientNetB0": 224,
    "EfficientNetB1": 240,
    "EfficientNetB2": 260,
    "EfficientNetB3": 300,
    "EfficientNetB4": 380,
    "EfficientNetB5": 456,
    "EfficientNetB6": 528,
    "EfficientNetB7": 600,
    "EfficientNetV2B0": 224,
    "EfficientNetV2B1": 240,
    "EfficientNetV2B2": 260,
    "EfficientNetV2B3": 300,
    "EfficientNetV2S": 384,
    "EfficientNetV2M": 480,
    "EfficientNetV2L": 576,
}

preprocessing_fns = {
    "UNet": None,
    "Xception": tf.keras.applications.xception.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
    "VGG19": tf.keras.applications.vgg19.preprocess_input,
    "ResNet50": tf.keras.applications.resnet.preprocess_input,
    "ResNet101": tf.keras.applications.resnet.preprocess_input,
    "ResNet152": tf.keras.applications.resnet.preprocess_input,
    "ResNet50V2": tf.keras.applications.resnet_v2.preprocess_input,
    "ResNet101V2": tf.keras.applications.resnet_v2.preprocess_input,
    "ResNet152V2": tf.keras.applications.resnet_v2.preprocess_input,
    "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
    "InceptionResNetV2": tf.keras.applications.inception_resnet_v2.preprocess_input,
    "MobileNetV3Large": tf.keras.applications.mobilenet_v3.preprocess_input,
    "MobileNetV3Small": tf.keras.applications.mobilenet_v3.preprocess_input,
    "DenseNet121": tf.keras.applications.densenet.preprocess_input,
    "DenseNet169": tf.keras.applications.densenet.preprocess_input,
    "DenseNet201": tf.keras.applications.densenet.preprocess_input,
    "NASNetMobile": tf.keras.applications.nasnet.preprocess_input,
    "NASNetLarge": tf.keras.applications.nasnet.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB1": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB2": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB3": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB4": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB5": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB6": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetB7": tf.keras.applications.efficientnet.preprocess_input,
    "EfficientNetV2B0": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2B1": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2B2": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2B3": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2S": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2M": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2L": tf.keras.applications.efficientnet_v2.preprocess_input
}
def get_models():
    # return the list of models
    return list(models.keys())
def get_model_default_img_size(model_type):
    if model_type not in default_img_size:
        assert False, f"Unknown model type: {model_type}"
    return default_img_size[model_type]

def get_model_default_preprocessing(model_type):
    if model_type not in preprocessing_fns:
        assert False, f"Unknown model type: {model_type}"
    return preprocessing_fns[model_type]

def get_model(model_type, num_classes, img_size = None, freeze_base=False, feature_extraction=False, double_pop=False):
    if model_type not in models:
        assert False, f"Unknown model type: {model_type}"
    if not img_size:
        img_size = default_img_size[model_type]
    default_model_args = {
        "include_top": False,
        "weights": "imagenet",
        "input_shape": (img_size, img_size, 3),
        "pooling": "avg"
    }
    if model_type == "UNet":
        base_model = UNet().get_model(img_size)
    else:
        base_model = models[model_type](**default_model_args)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    if feature_extraction:
        d1 = Dense(256, activation='relu')(base_model.output)
        d2 = Dense(128, activation='relu')(d1)
        preds = Dense(num_classes, activation='sigmoid')(d2)
    elif double_pop:
        preds = Dense(num_classes, activation='sigmoid')(base_model.layers[-11].output)
    else:
        preds = Dense(num_classes, activation='sigmoid')(base_model.output)
    model = Model(inputs=base_model.input, outputs=preds)
    return model

def load_model(model_name, remove_top=False, top_config="1", freeze_base = False, num_classes=14):
    model_path = "../models/" + model_name + ".h5"
    if not os.path.exists(model_path):
        assert False, f"Model {model_name} does not exist in ../models directory"
    base_model = tf.keras.models.load_model(model_path)
    base_model.trainable = not freeze_base
    if remove_top:
        # https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
        if top_config == "1":
            preds = Dense(num_classes, activation='sigmoid', name="k1")(base_model.layers[-2].output)
        elif top_config == "2":
            d1 = Dense(256, activation='relu', name="k1")(base_model.layers[-2].output)
            d2 = Dense(128, activation='relu', name="k2")(d1)
            preds = Dense(num_classes, activation='sigmoid', name="k3")(d2)
        base_model = Model(inputs=base_model.input, outputs=preds)
    return base_model
