import numpy as np

from keras.applications import imagenet_utils
from tensorflow.keras.utils import Sequence, to_categorical

from coremdlr.datasets import utils as datasets_utils


class DepthSequenceGenerator(Sequence):
    """
    Subclassing of Sequence for use with fit_generator. Generates pairs of (features, labels)
    sequences, optionally concatenating timesteps along first feature axis.

    Parameters
    ----------
    X : array or dict
        Input array of training data. Shape should be (N, <feature_shape>).
        If dict, must specify `feature` mapping to appropriate array, and dict
        should have a 'boundary_idxs' key mapping to array of indices of boundaries b/t wells.
    y : array or None
        Array of corresponding labels, one per sample. Shape should be (N,). If None, generate dummy labels.
    sequence_size : int
        Number of instances to join into single training example.
    step_size : int, optional
        Step size between sequences. Default=None results in step_size=sequence_size.
    concatenate_X : bool, optional
        Whether to concatenate instances within a sequence, or to leave them as individual timesteps.
        E.g., for pGR: True -> (sequence_size*label_resolution,), False -> (sequence_size, label_resolution).
    batch_size : int, optional
        Batch size, default=1self.
    feature : dict key, optional
        Key mapping to feature array, if `X` is a dict.
    augment_fn : func, optional
    format_fn : func, optional
        Functions to apply to every (batch_X, batch_y) pair.

    """
    def __init__(self, X, y,
                 sequence_size,
                 step_size=None,
                 concatenate_X=False,
                 batch_size=1,
                 feature=None,
                 augment_fn=None,
                 format_fn=None,
                 shuffle=True,
                 preprocess_mode='caffe'):

        if type(X) is dict:
            assert feature is not None, 'Must specify `feature` key if `X` is a dict'
            try:
                _X = X[feature]
            except KeyError:
                raise ValueError('Invalid `feature` key for given dict `X`')
        else:
            _X = X

        self.num_samples = _X.shape[0]

        if y is None:
            y = np.zeros((_X.shape[0], 1), dtype=np.int8)

        y = to_categorical(y) if y.ndim == 1 else y

        boundary_idxs = X['boundary_idxs'] if type(X) is dict else []

        self.sequence_size = sequence_size
        self.step_size = sequence_size if step_size is None else step_size
        self.idx_pairs = datasets_utils.get_idx_pairs(self.num_samples, boundary_idxs, self.sequence_size, self.step_size)

        self.X, self.y = datasets_utils.get_concrete_sequences(_X, y, self.idx_pairs, concatenate_X=concatenate_X)
        if sequence_size == 1:
            self.X = self.X if concatenate_X else np.squeeze(self.X, axis=1)
            self.y = np.squeeze(self.y, axis=1)

        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.format_fn = format_fn
        self.preprocess_mode = preprocess_mode
        self.shuffle = shuffle


    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))


    def __getitem__(self, idx):
        # idx = 0 -- uncomment to overfit a single batch
        begin = idx * self.batch_size
        end = (idx+1) * self.batch_size

        batch_X = self.X[begin:end]
        batch_y = self.y[begin:end]

        if self.augment_fn:
            batch_X, batch_y = self.augment_fn(batch_X, batch_y)

        if self.format_fn:
            batch_X, batch_y = self.format_fn(batch_X, batch_y)

        return batch_X, batch_y


    def on_epoch_end(self):
        """Shuffle the examples, not just batches via fit_generator."""
        if self.shuffle:
            p = np.random.permutation(len(self.X))
            self.X = self.X[p]
            self.y = self.y[p]
