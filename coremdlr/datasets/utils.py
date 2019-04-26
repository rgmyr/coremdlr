import os
import pathlib
import math, random
import numpy as np
import pandas as pd

from sklearn import preprocessing
from skimage import transform
from skimage.color import rgb2gray

from coremdlr.config.defaults import DEFAULT_TRAIN_PATH


def available_wells(path=DEFAULT_TRAIN_PATH):
    return set([p.name.split('_')[0] for p in pathlib.Path(path).glob('*.npy')])


############################
### PseudoGR Computation ###
############################

# minimum threshold for pixel values considered
EPSILON = 1e-2


class PseudoExtractor():
    """
    Callable for extracting row-wise "pseudo log" signals from images.

    **kwargs
    --------
    features : list(str)
        Values should be one of {'mean', 'var', 'pN', 'histN'}, where 'N' is a numeric
        string: a percentile (float/int) or number of histogram bins (int), respectively.
    hist_range : tuple(float)
        Lower and upper bounds for histogram binning (image scaled to [0.0, 1.0] range).
    per_channel : bool
        Whether to extract feature from each channel (U, R, G, B), or just grayscale (U).
    """
    def __init__(self, **kwargs):

        self.features = kwargs.pop('features', ['mean', 'var', 'p10', 'p90'])
        self.hist_range = kwargs.pop('hist_range', (0.1, 0.9))
        self.per_channel = kwargs.pop('per_channel', True)

        feat_count = 0
        for feat in self.features:
            is_mv = feat in ['mean', 'var']
            is_pN = (feat[0] == 'p' and feat[1:].isdigit())
            if is_mv or is_pN:
                feat_count += 1
            elif feat[:4] == 'hist' and feat[4:].isdigit():
                feat_count += int(feat[4:])
            else:
                raise ValueError(f'Invalid PseudoGR feature: {feat}')

        self.num_columns = 4*feat_count if self.per_channel else feat_count


    def __call__(self, image_arr):

        img = image_arr / 255.0 if image_arr.max() > 1.0 else image_arr

        if self.per_channel:
            # Do we even want the U channel here?
            img = np.dstack((rgb2gray(img), img))
            channels = ['U', 'R', 'G', 'B']
        else:
            img = np.expand_dims(rgb2gray(img), -1)
            channels = ['U']

        img[np.where(img < img.min()+EPSILON)] = np.nan

        output_features = []
        output_names = []

        for feat in self.features:
            if feat is 'mean':
                output_features.append(np.nanmean(img, axis=1))
                output_names += [C+'mean' for C in channels]

            if feat is 'var':
                output_features.append(np.nanvar(img, axis=1))
                output_names += [C+'var' for C in channels]

            if feat.startswith('p'):
                percent = float(feat[1:])
                output_features.append(np.nanpercentile(img, percent, axis=1))
                output_names += [C+'p'+str(int(percent)) for C in channels]

            if feat.startswith('hist'):
                bins = int(feat[4:])
                hist_fn = lambda x: np.histogram(x, bins=bins, range=self.hist_range)[0]
                hist = np.apply_along_axis(hist_fn, 1, img)
                output_features.append(hist.reshape((-1, bins*len(channels))))
                output_names += [C+'hist'+str(i) for i in range(10) for C in channels]

        return np.hstack(tuple(output_features)), output_names


###########################
#### LOADING & SCALING ####
###########################

def load_logs_clean(logs_df, which_logs):
    """
    Load `which_logs` into new DataFrame, after cleaning up column names.
    """
    logs_df.columns = logs_df.columns.str.replace('_', '')   # DTS1 vs DTS_1, etc.
    logs_df.columns = logs_df.columns.str.replace('AIT', 'AT')   # AIT vs AT
    swap = {
        'AT10' : 'RSHAL',
        'AT30' : 'RMED',
        'AT90' : 'RDEP'
    }
    logs_df.rename(columns=lambda x: swap[x] if x in swap.keys() else x, inplace=True)

    # fill any missing columns with NaN
    for log_name in which_logs:
        if log_name not in logs_df.columns:
            print('Adding NaN log: ', log_name)
            logs_df[log_name] = np.nan

    return logs_df[which_logs]


def crop_or_pad_image(img, width, downsample=1):
    """
    Make image_width = `width` by center cropping or padding as necessary.
    """
    # Crop/pad image to width
    width_diff = width - img.shape[1]
    if width_diff == 0:
        return img
    else:
        l = abs(width_diff) // 2
        r = abs(width_diff) // 2 + abs(width_diff) % 2
        if width_diff < 0:
            # center crop if img too wide
            return img[:,l:-r,:]
        else:
            # zero pad if img too narrow
            lhs = np.zeros((img.shape[0],l,3), dtype=img.dtype)
            rhs = np.zeros((img.shape[0],r,3), dtype=img.dtype)
            return np.hstack([lhs, img, rhs])


def downsample(data, factor, fn=None):
    """
    Downsample data by integer factor using `fn` to aggregate.
    """
    if factor is None or factor <= 1.0:
        return data

    elif data.ndim == 1:
        new_shape = (data.size // factor,)
        new_data = np.zeros(new_shape, dtype=data.dtype)
        for i in np.arange(new_data.size):
            t, b = i*factor, i*factor + factor
            new_data[i] = fn(data[t:b])
        return new_data

    else:
        new_shape = (data.shape[0]//factor, data.shape[1]//factor, *data.shape[2:])
        if data.max() > 1.0:
            data /= 255.0
        return transform.resize(data, new_shape, mode='constant', anti_aliasing=True)



############################
##### SLICING & DICING #####
############################

def get_idx_pairs(num_samples, boundary_idxs, sequence_size=32, step_size=16):
    """
    Generate an array of (start, end) idx pairs for sequence creation.

    Parameters
    ----------
    num_samples : int
        Number of instances in dataset to be split (e.g., dataset.y_train.size).
    boundary_idxs : list(int), optional
        Indices of first instances in a subset (e.g., an individual well).
        Sequences that would contain a boundary index are not added to map.
    sequence_size : int, optional
        Length of sequences in map. (Multiples of 32 are recommended for image patches).
    step_size : int, optional
        Step size (in instances) between successive start_idx locations.
    """
    starts = np.append(np.arange(num_samples-sequence_size, step=step_size), [num_samples-sequence_size])
    stops = np.append(np.arange(sequence_size, num_samples, step=step_size), [num_samples])
    idx_pairs = np.vstack([starts,stops]).T

    # remove any rows that straddle a boundary index
    for idx in boundary_idxs:
        good_idxs = ~np.logical_and(idx_pairs[:,0] < idx, idx_pairs[:,1] >= idx)
        idx_pairs = idx_pairs[good_idxs]

    return idx_pairs


def get_concrete_sequences(X, y, idx_pairs, concatenate_X=True):
    """
    Given X, y, and idx_pairs, generate concrete X_seq and y_seq arrays.

    Parameters
    ----------
    X : array, shape (num_samples, ...)
    y : array, shape (num_samples,)
    idx_pairs : array, shape (N, 2)
        Like returned from `get_sequence_map`.
    concatenate_X : boolean, optional
        Whether to concatenate X instances along steps axis (i.e., depth) into single instance.
        If False, returns X_new with shape (N, sequence_length, ...).
        If True, returns X_new with shape (N, ...), default=True.
    """
    get_slice = lambda x, i : x[idx_pairs[i,0]:idx_pairs[i,1]]
    Xs, ys = [], []
    for i in range(idx_pairs.shape[0]):
        X_i = np.concatenate(get_slice(X, i)) if concatenate_X else get_slice(X, i)
        Xs.append(X_i)
        ys.append(get_slice(y, i))

    return np.stack(Xs), np.stack(ys)
