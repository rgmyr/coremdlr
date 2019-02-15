#import h5py
import os
import math
import numpy as np
import pandas as pd
from skimage import io
from scipy.stats import mode

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from coremdlr.datasets import WellLoader
from coremdlr.datasets import FeatureScaler
from coremdlr.datasets import utils as datasets_utils

from coremdlr.config import defaults
from coremdlr.config.strip_config import lithologies_dict


class FaciesDataset():
    """
    Class for modeling of depth-labeled facies in processed core image + log datasets.
    Each well should have an image file, a depths file, an LAS file, and a labels file.
    Each file should have the same prefix (name of the well) and appropriate suffixes.
    (See WellLoader.__doc__ and __init__ for details.)

    Parameters
    ----------
    wells : list(str)
        Names of wells to use, should have all required data in data_dir.
    test_wells : list(str)
        Names of wells to use for validation.
    data_dir : str, optional
        Directory of well data files, default ('') resolves to '/home/$USER/Dropbox/core_data/facies/train_data'.
    features : list(str), optional
        Which patch features to aggregate and load, default=['image'].
        All elements must be one of {'image', 'pseudoGR', 'logs'}.
    label_resolution : int, optional
        Number of image/depth rows to aggregated into individual labeled sections of core, default=32.
        The resolution should be chosen thoughtfully w.r.t. image and pseudoGR network strides.
        Default corresponds to effective stride of local features in pretrained VGG/ResNet50.
    collapse_missing : bool, optional
        Whether to collapse data by dropping depths/instances with a label in [0, 1].
        With default lithology_classes, these correspond to 'no core' (missing) and 'bad-sandstone' labels.
        TODO: flexibility to set different number of non-training classes.
    downsample : int, optional
        If `downsample` > 1, data and labels will be downsampled by a factor of `downsample`, default=None.
        Note that `label_resolution` will be employed w/r/t the downsampled data & labels, not the original.
    lithology_classes : dict, optional
        Dict of {label: class_name}, where label is csv abbreviation and class name is full lithology name.

    Any `logs_args`, `pseudoGR_args`, and `image_args` will override default params for loading/preproc.
    Defaults are sensible, but see `config/defaults.py` for values.
    See `WellLoader` docs for details on field values and their effects.
    """
    def __init__(self, wells, test_wells=[],
                data_dir='',
                features=['image'],
                label_resolution=32,
                collapse_missing=True,
                downsample=None,
                lithology_classes=lithologies_dict,
                image_args={},
                pseudoGR_args={},
                logs_args={}):

        self.data_dir = data_dir
        self.features = list(set(features))
        self.meta_keys = ['depth', 'top', 'base']
        self.data_keys = self.features + self.meta_keys
        #self.seed = random_seed

        # classes info
        self.labels = np.array(list(lithology_classes.keys()), dtype='a2').tolist()
        self.label_resolution = label_resolution
        self.classes = list(lithology_classes.values())
        self.num_classes = len(self.classes)-2 if collapse_missing else len(self.classes)
        self.output_shape = (self.num_classes,)

        # feature options
        self.image_args    = {**defaults.DEFAULT_IMAGE_ARGS, **image_args}
        self.pseudoGR_args = {**defaults.DEFAULT_PGR_ARGS  , **pseudoGR_args}
        self.logs_args     = {**defaults.DEFAULT_LOGS_ARGS , **logs_args}

        # feature scalers
        self.scalers = {f : FeatureScaler(f, getattr(self, f+'_args')) for f in self.features}

        # WellLoader objects
        self.well_names = wells
        self.wells = [WellLoader(os.path.join(data_dir, well_name),
                                label_resolution=label_resolution,
                                collapse_missing=collapse_missing,
                                use_dummy_labels=False if len(test_wells) > 0 else True,
                                use_image='image' in features,
                                use_pseudoGR='pseudoGR' in features,
                                use_logs='logs' in features,
                                downsample=downsample,
                                image_args=self.image_args,
                                pseudoGR_args=self.pseudoGR_args,
                                logs_args=self.logs_args) for well_name in self.well_names]

        self.test_well_names = test_wells
        self.test_wells = [WellLoader(os.path.join(data_dir, test_well_name),
                                     label_resolution=label_resolution,
                                     collapse_missing=collapse_missing,
                                     use_image='image' in features,
                                     use_pseudoGR='pseudoGR' in features,
                                     use_logs='logs' in features,
                                     downsample=downsample,
                                     image_args=self.image_args,
                                     pseudoGR_args=self.pseudoGR_args,
                                     logs_args=self.logs_args) for test_well_name in self.test_well_names]


    def load_or_generate_data(self):
        """
        Define X/y train/test.
        """
        self.wells = [w.load_data(self.labels, random_shift=False) for w in self.wells]
        if self.wells[0].collapse_missing:
            self.classes = self.classes[2:]

        self.X_train, self.y_train = self._aggregate_wells(self.wells, train=True)

        if len(self.test_wells) > 0:
            self.test_wells = [tw.load_data(self.labels) for tw in self.test_wells]
            self.X_test, self.y_test = self._aggregate_wells(self.test_wells, train=False)
        else:
            self.X, self.y = self.X_train, self.y_train


    def _aggregate_wells(self, wells, train=True):
        """
        Join well.X dict and well.y into single dict with added 'boundary_idxs' key.
        Scaling/preproc done here via FeatureScalers. `train` option specifies whether to
        fit scalers, or just transform data (only meaningful where 'scale_by' == 'dataset').
        """
        y, boundary_idxs = [], [0]

        d = {k : [] for k in self.data_keys}

        for well in wells:
            y.append(well.y)
            boundary_idxs.append(well.y.size + boundary_idxs[-1])
            for k in self.data_keys:
                d[k].append(well.X[k])

        y = np.concatenate(y)

        X = {**{f: self.scalers[f].scale_wells(d[f], fit=train) for f in self.features},
             **{m: np.concatenate(d[m]) for m in self.meta_keys}}

        X['boundary_idxs'] = boundary_idxs

        return X, y


    def get_well(self, well_name):
        """
        Return WellLoader object by name.
        """
        if well_name in self.well_names:
            return self.wells[self.well_names.index(well_name)]
        elif well_name in self.test_well_names:
            return self.test_wells[self.test_well_names.index(well_name)]
        else:
            raise ValueError('Well {} is not in `wells` or `test_wells` of dataset'.format(well_name))


    def get_well_data(self, well_name, include_meta=False, top=None, base=None):
        """
        Get X (feature dict) and y (array) for a well by name.

        `include_meta` controls whether dataset.meta_keys ['depth', 'top', 'base'] are included in X.
        """
        if well_name in self.well_names:
            idx = self.well_names.index(well_name)
            flag = 'train'

        elif well_name in self.test_well_names:
            idx = self.test_well_names.index(well_name)
            flag = 'test'

        else:
            raise ValueError(f'Name {well_name} not in {self.well_names} or {self.test_well_names}')

        X, y = getattr(self, 'X_'+flag), getattr(self, 'y_'+flag)
        t, b = X['boundary_idxs'][idx], X['boundary_idxs'][idx+1]

        # optional slicing by depth
        depths = X['depth'][t:b]

        top = top if top is not None else depths[0]
        below_top = depths >= top

        base = base if base is not None else depths[-1]
        above_base = depths <= base

        good_depths = np.where(np.logical_and(below_top, above_base))

        if include_meta:
            X_data = {k : X[k][t:b][good_depths] for k in self.data_keys}
        else:
            X_data = {f : X[f][t:b][good_depths] for f in self.features}

        return X_data, y[t:b][good_depths]


    def __repr__(self):
        return (
            f'\nFacies Patch Dataset loaded from {self.data_dir}\n'
            f'Features: {self.features}\n'
            f'Classes: {self.classes}\n'
            f'Train Wells: {self.well_names}\n'
            f'Test Wells: {self.test_well_names}\n'
        )


    def _to_class(self, s):
        return self.classes.index(s.split('/')[0])
