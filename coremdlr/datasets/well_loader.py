"""
Data loader for individual wells.
"""
import os
import pathlib
import random
import lasio
from io import StringIO
import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.interpolate import interp1d

from striplog import Interval, Striplog
from striplog.striplog import StriplogError

from coremdlr.config import defaults, strip_config
from coremdlr.datasets import utils as datasets_utils


class WellLoader:
    """
    Load/combine multiple X/y input data sources from an individual well. Pass `use_dummy_labels=True`
    to generate dummy labels for inference on unlabeled wells (still must have _depth.npy file)

    Parameters
    ----------
    well_name_or_path : str
        Name of well (if in canonical train_data location), otherwise full path to well directory.
        It is assumed that any string containing a forward slash is meant as an absolute path.
    label_resolution : int, optional
        Number of image/depth rows to aggregated into individual labeled sections of core, default=32.
        The resolution should be chosen thoughtfully w.r.t. image and pseudoGR feature scales.
        Default corresponds to effective stride of local features in pretrained VGG/ResNet variants.
    collapse_missing : bool, optional
        Whether to drop any [0,1] labels before setting X, y attributes, default=True.
    use_dummy_labels : bool, optional
        Whether to generate dummy labels for X data, default=False. Useful for performing inference on
        unlabeled wells. Fills empty rows with 0's, non-empty rows with 2's (for collapse_missing).
    use_image : bool, optional
        Whether to load and generate data from image file, default=True.
        If true, numpy file should be uniquely identified by *_image.npy
    use_psuedoGR : bool, optional
        Whether to compute and generate psuedo-gamma ray (luminance) log, defualt=False.
    use_logs : bool, optional
        Whether to load logs from LAS file, default=False.
        If True, LAS file should be uniquely identified by *.las
    downsample : int, optional
        If `downsample` > 1, data and labels will be downsampled by a factor of `downsample`, default=None.
        Note that `label_resolution` will be employed w/r/t the downsampled data & labels, not the original.


    feature_args [in common]
    ---------------------
    scaler : str, optional
        One of {'standard', 'minmax', 'robust', 'power'}. May also be 'caffe' for `image_args`
    scale_by : str, optional
        One of {'well', 'dataset'}. Whether to scale by

    image_args
    ----------
    image_width : int, optional
        Width of center cropped / padded image, default=600.
    image_scaler : one of {'standard', 'minmax', 'caffe', None}, optional

    pseudoGR_args
    -------------
    features : list(str)
        Which row-wise pseudo-logs to extract from image. See PseudoExtractor docs.
    per_channel : bool
        If True, extract features from RGB+gray channels. If False, only use grayscale.


    logs_args
    ---------
    which_logs : list(str), optional
        Which logs to load and consider as features if use_logs=True, default=['GR'].
    interp_kind : str, optional
        Which kind of interpolation to use for log resampling if use_logs=True, default='linear'.
        Must be a valid `kind` argument to scipy.interpolate.interp1d constructor.
    """

    def __init__(self, well_name_or_path,
                 label_resolution=32,
                 collapse_missing=True,
                 use_dummy_labels=False,
                 use_image=True,
                 use_pseudoGR=False,
                 use_logs=False,
                 downsample=None,
                 image_args={},
                 pseudoGR_args={},
                 logs_args={}):

        if '/' in well_name_or_path:
            self.data_path = pathlib.PosixPath(well_name_or_path)
            self.well_name = well_name_or_path.split('/')[-1]
        else:
            self.data_path = defaults.DEFAULT_TRAIN_PATH
            self.well_name = well_name_or_path

        # label/aggregation options
        self.label_resolution = label_resolution
        self.collapse_missing = collapse_missing
        self.use_dummy_labels = use_dummy_labels
        self.downsample = downsample

        # check for depth file
        self.depth_path = self._get_data_path('_depth.npy', assert_exists=True)

        # check for labels
        if not self.use_dummy_labels:
            self.labels_path = self._get_data_path('_labels.npy', assert_exists=True)
        else:
            print('Loading well with dummy labels. DO NOT TRAIN ON THIS WELL!')
            assert self.collapse_missing, 'Always collapse the dummy labels.'
            self.labels_path = None
            if not use_pseudoGR:
                print('Need pseudoGR to generate dummy labels... setting `use_pseudoGR=True`.')
                use_pseudoGR = True

        # check for image/pGR
        self.use_image = use_image
        self.use_pseudoGR = use_pseudoGR
        if use_image or use_pseudoGR:
            self.image_path = self._get_data_path('_image.npy', assert_exists=True)
            self.image_args    = {**defaults.DEFAULT_IMAGE_ARGS, **image_args}
            self.pseudoGR_args = {**defaults.DEFAULT_PGR_ARGS  , **pseudoGR_args}

        # check for logs
        self.use_logs = use_logs
        if use_logs:
            self.logs_path = self._get_data_path('_logs.las', assert_exists=True)
            self.logs_args = {**defaults.DEFAULT_LOGS_ARGS, **logs_args}


    def load_data(self, label_names, aggregate=True, random_shift=False):
        """
        Load the data into X/y attributes. Note: `X` is a dict.
        """
        print('Loading Well: ', self.well_name, ' from ', self.data_path)

        if self.use_dummy_labels:
            print('Loading well with dummy labels. DO NOT TRAIN ON THIS WELL!')

        self._depth = np.load(str(self.depth_path))
        self.top = self._depth[0]
        self.base = self._depth[-1]
        self._depth = datasets_utils.downsample(self._depth, self.downsample, fn=np.median)

        if self.use_image or self.use_pseudoGR:
            image = np.load(str(self.image_path)).astype(np.float32)
            self._image = datasets_utils.crop_or_pad_image(image,
                                                          self.image_args['image_width'],
                                                          self.image_args['crop_method'])
            self._image = datasets_utils.downsample(self._image, self.downsample)

        if self.use_pseudoGR:
            pGR_extractor = datasets_utils.PseudoExtractor(**self.pseudoGR_args)
            self._pseudoGR, self.pGR_feat_names = pGR_extractor(self._image)
            print('Extracted pGR features: ', self.pGR_feat_names)

        if self.use_logs:
            logs_df = lasio.read(str(self.logs_path)).df()
            self.logs_df = datasets_utils.load_logs_clean(logs_df, self.logs_args['which_logs'])
            # NOTE: _logs is now a function, callable on one or more depths to get log values
            self._logs = interp1d(self.logs_df.index.values, self.logs_df.values, axis=0,
                                  kind=self.logs_args['interp_kind'], bounds_error=True)

        self._load_labels(label_names)

        if aggregate:
            self._aggregate_instances(random_shift=random_shift)

        return self


    def _load_labels(self, label_names):
        """
        Load labels (or dummy labels based on pseudoGR).
        """
        if self.use_dummy_labels:
            self.row_labels = (~np.isnan(self._pseudoGR)).astype(np.int)*2
        else:
            row_labels = np.load(str(self.labels_path))
            self.row_labels = np.vectorize(lambda l: label_names.index(l))(row_labels)

        self.row_labels = datasets_utils.downsample(self.row_labels, self.downsample, fn=lambda x: mode(x)[0][0])


    def _aggregate_instances(self, random_shift=False):
        """
        Collapse data into single-label instances.
        `random_shift` controls whether aggregation starts strictly from 0-th row,
            or from a random idx in range [0, label_resolution).
        """
        # compute size and top/bottom indices
        start_idx = random.randint(0, self.label_resolution-1) if random_shift else 0
        num_samples = (self._depth.size - start_idx - 1) // self.label_resolution
        label_tops = [start_idx + self.label_resolution*i for i in range(num_samples)]
        label_bottoms = [top + self.label_resolution for top in label_tops]

        # set up 'patch' arrays
        depth = np.zeros((num_samples,), dtype=float)
        self._y = np.zeros((num_samples,), dtype=int)
        self._tops = np.zeros((num_samples,), dtype=float)
        self._bottoms = np.zeros((num_samples,), dtype=float)
        image_patches = np.zeros((num_samples,self.label_resolution,self._image.shape[1],3)) if self.use_image else None
        pseudoGR_sections = np.zeros((num_samples,self.label_resolution,self._pseudoGR.shape[-1])) if self.use_pseudoGR else None

        # split labels + data into 'patches'
        for i, (t, b) in enumerate(zip(label_tops, label_bottoms)):
            self._y[i] = mode(self.row_labels[t:b])[0][0]
            depth[i] = np.median(self._depth[t:b])
            self._tops[i] = self._depth[t]
            self._bottoms[i] = self._depth[b]
            if self.use_image:
                image_patches[i] = self._image[t:b]
            if self.use_pseudoGR:
                pseudoGR_sections[i] = self._pseudoGR[t:b]


        # Remove 0's and 1's from data if `collapse_missing`
        collapse_fn = lambda x: x[np.where(self._y > 1)] if self.collapse_missing else lambda x: x
        self.X = {'depth': collapse_fn(depth),
                  'top'  : collapse_fn(self._tops),
                  'base' : collapse_fn(self._bottoms)}

        if self.use_image:
            self.X['image'] = collapse_fn(image_patches)

        if self.use_pseudoGR:
            self.X['pseudoGR'] = collapse_fn(pseudoGR_sections)

        if self.use_logs:
            self.X['logs'] = np.array([(self._logs(d)).flatten() for d in self.X['depth']])

        self.y = collapse_fn(self._y)
        self.y = self.y - 2 if self.collapse_missing else self.y

        print('Feature shapes: ', [(k, v.shape) for k, v in self.X.items()])

    def _get_data_path(self, ext, assert_exists=False):
        """
        Return the path of a (`well_name` + `ext`) data file.
        If `assert_exists`, assert that this points to a file that exists.
        """
        file_path = self.data_path / (self.well_name + ext)

        if assert_exists:
            assert file_path.is_file(), f'{str(file_path)} does not exist!'

        return file_path

    def slice_data(self, top=None, base=None):
        """
        Slice `self.X` features in place, return self.
        """
        good_idxs = np.logical_and(self.X['top'] >= top)


    def del_unnecessary_data(self):
        """
        Deallocate all significant data attributes that aren't strictly required for
        generating a striplog or
        """
        del self.X

    ###++++++++++++++++++###
    ### Make Output Data ###
    ###++++++++++++++++++###

    def make_striplog(self, labels=None, save_csv=None):
        """
        Make and return striplog. Optionally save as csv to `save_csv`.
        If labels is None, uses uncollapsed self._y as labels.
        """
        if labels is None:
            try:
                return self.striplog
            except AttributeError:
                labels = self._y
                using_true_labels = True
        elif not np.array_equal(labels, self._y):
            labels = self._expand_preds(labels)
            using_true_labels = False
        else:
            raise ValueError('Something is fishy...')

        intervals = []
        facies_keys = list(strip_config.facies.keys())

        current_label = labels[0]
        current_top, current_bottom = self._tops[0], self._bottoms[0]

        for i in range(1, labels.size):
            if labels[i] == current_label or i == (labels.size-1):
                current_bottom = self._bottoms[i]
            if labels[i] != current_label or i == (labels.size-1):
                # close previous interval
                descrip = facies_keys[current_label]
                component = strip_config.facies[descrip]
                interval = Interval(top=current_top,
                                    base=current_bottom,
                                    description = descrip,
                                    components = [component])
                intervals.append(interval)

                # open new interval
                current_label = labels[i]
                current_top, current_bottom = self._tops[i], self._bottoms[i]
        try:
            striplog = Striplog(intervals)
        except StriplogError as e:
            print(e)
            for iv in intervals:
                print(f'top: {iv.top}, base: {iv.base}, order: {iv.order}')


        if save_csv is not None:
            df = pd.read_csv(StringIO(striplog.to_csv()))
            df.Top = df.Top.apply(lambda x: '{0:.3f}'.format(x))
            df.Base = df.Base.apply(lambda x: '{0:.3f}'.format(x))
            df.Component = df.Component.apply(strip_config.lithology_to_key)
            df.columns = ['top', 'base', 'lithology']
            df.to_csv(save_csv, index=False)

        if using_true_labels:
            self.striplog = striplog

        return striplog


    def _expand_preds(self, preds):
        """
        Make new preds array by 'uncollapsing' preds like self._y labels.
        In other words, keep [0,1] labels from self._y + fill in new labels from preds.
        """
        if not self.collapse_missing:
            return preds
        else:
            preds = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
            labels = np.copy(self._y)
            labels[np.where(labels > 1)] = (preds + 2)
            print(np.bincount(self._y))
            print(np.bincount(labels))
            return labels
