import os
import pathlib
import math, random
import numpy as np
import pandas as pd

from sklearn import preprocessing


# Available basic Scaler types
# 'caffe' option also allowed for image feature
BASIC_SCALERS = {
    'minmax' : preprocessing.MinMaxScaler,
    'standard' : preprocessing.StandardScaler,
    'robust' : preprocessing.RobustScaler,
    'power' : preprocessing.PowerTransformer,
}

def convert_to_caffe(x):
    """
    Callable to convert RGB->BGR and center w.r.t. ImageNet dataset. Recommended for pretrained keras.applications.
    """
    assert x.shape[-1] == 3, 'Can only use `caffe` scaling on RGB images (3 channels).'

    # Scale to range [0.0, 255.0]
    if x.max() <= 1.0:
        multiplier = 127.5 if x.min() < 0.0 else 255.0
        x *= multiplier

    # RGB -> BGR
    x = x[..., ::-1]

    # subtract ImageNet mean from nonzero pixels, set zeros to -255.
    x[x.sum(axis=-1).nonzero()] -= np.array([103.939, 116.779, 123.68])
    x[np.where(x.sum(axis=-1) == 0.0)] = np.array([-255.,-255.,-255.])

    return x

"""
TODO: having inverse function (w/ zero_val arg) might be nice for viz functions.
"""

class FeatureScaler():
    """
    Wrapper for sklearn.preprocessing Scalers that takes care of appropriate reshaping and NaN filling.
    """
    def __init__(self, feature, feature_args):

        if feature_args['scaler'] in BASIC_SCALERS.keys():
            self.scaler = BASIC_SCALERS[feature_args['scaler']](**feature_args.get('scaler_args', {}))

        elif feature is 'image' and feature_args['scaler'] is 'caffe':
            self.scaler = preprocessing.FunctionTransformer(func=convert_to_caffe)

        else:
            raise ValueError(f'Invalid arguments to {feature} `SmartScaler`: {feature_args}')

        self.feature = feature
        self.scale_by = feature_args['scale_by']          # every feature should have 'scale_by' value
        self.fill_mode = feature_args.get('fill_mode')    # but fill_mode can be None for 'image'


    def scale_wells(self, X, fit=True):
        """
        `X` is assumed to be a list of feature arrays from one or more WellLoader `X` attributes.
        `train` is a bool indicating whether to fit the Scaler, ignored when `self.scale_by` = 'well'.
        """
        if self.scale_by == 'well':
            X = np.concatenate([self.transform(x, fit=True) for x in X])

        elif self.scale_by == 'dataset':
            X = self.transform(np.concatenate(X), fit=fit)

        return X


    def transform(self, X, fit=True):
        """
        `X` is assumed to be an array of appropriate shape for the given feature.
        """
        if self.feature is 'image':
            orig_shape = X.shape
            return self.scaler.transform(X.reshape((-1, 3))).reshape(orig_shape)

        elif self.feature is 'pseudoGR':
            orig_shape = X.shape
            if fit:
                return self.scaler.fit_transform(self.fill(X.reshape(-1, X.shape[-1]))).reshape(orig_shape)
            else:
                return self.scaler.transform(self.fill(X.reshape(-1, X.shape[-1]))).reshape(orig_shape)

        elif self.feature is 'logs':
            if fit:
                return self.scaler.fit_transform(self.fill(X))
            else:
                return self.scaler.transform(self.fill(X))


    def fill(self, X):
        """
        `X` is an array. Use pandas/numpy to fill any missing values.
        """
        if self.fill_mode == 'mean':
            df = pd.DataFrame(X)
            X = df.fillna(df.mean()).values

        elif self.fill_mode in ['bfill', 'ffill']:
            # fill in the opposite direction afterward
            mode1 = self.fill_mode
            mode2 = 'ffill' if mode1 is 'bfill' else 'bfill'
            X = pd.DataFrame(X).fillna(method=mode1).fillna(method=mode2).values

        elif self.fill_mode is not None:
            print(f'Unknown `fill_mode` <{self.fill_mode}> for feature `{self.feature}`. NaN will just be set to 0.0.')

        return np.nan_to_num(X, copy=False)
