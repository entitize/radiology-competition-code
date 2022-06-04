import pandas as pd
import tensorflow as tf
import datetime
import keras.backend as K
import argparse
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from functools import partial, update_wrapper
from icecream import ic

def weighted_binary_crossentropy(self):
        """A weighted binary crossentropy loss function
        that works for multilabel classification
        Modified from https://github.com/akensert/keras-weighted-multilabel-binary-crossentropy/blob/master/wbce.py
        """
        # create a 2 by N array with weights for 0's and 1's
        i = 0
        weights = np.zeros((2, len(self.diseases)))
        # calculates weights for each label in a for loop
        for disease in self.diseases:
            total = (self.train_df[disease] == -1.).sum() + (self.train_df[disease] == 1.).sum()
            weights_n, weights_p = (total/(2 * (self.train_df[disease] == -1.).sum())), (total/(2 * (self.train_df[disease] == 1.).sum()))
            # weights could be log-dampened to avoid extreme weights for extremly unbalanced data.
            weights[1, i], weights[0, i] = weights_p, weights_n
            i += 1

        # The below is needed to be able to work with keras' model.compile()
        def wrapped_partial(func, *args, **kwargs):
            partial_func = partial(func, *args, **kwargs)
            update_wrapper(partial_func, func)
            return partial_func

        def wrapped_weighted_binary_crossentropy(y_true, y_pred, class_weights):
            y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
            # cross-entropy loss with weighting
            out = -(y_true * K.log(y_pred)*class_weights[1] + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights[0])

            return K.mean(out, axis=-1)

        return wrapped_partial(wrapped_weighted_binary_crossentropy, class_weights=weights)
"""
def weighted_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

# Define the 14 weight matrices  in list called weights

# Define the weighted loss functions
from functools import partial
for i, w in enumerate(weights):
    loss = partial(weighted_categorical_crossentropy, weights=w)
    loss.__name__ = 'loss' + i


# Finally, apply the loss functions to the outputs
model.compile(loss={'output1': loss1, 'output2': loss2}, optimizer='adam')
"""

def compute_class_weight_one_class(df, diseases):
    if len(diseases) != 1:
        assert ValueError('df must have one column')
    tmp_df = df.copy()
    tmp_df = tmp_df[diseases[0]]
    tmp_df = tmp_df.dropna()
    tmp_df = tmp_df[tmp_df != 0]
    w = compute_class_weight(class_weight='balanced', classes=[-1.0, 1.0], y=tmp_df.values)
    w = dict(enumerate(w))
    ic(w)
    return w