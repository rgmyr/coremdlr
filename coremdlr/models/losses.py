import numpy as np

from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops


_EPSILON = 1e-7

def categorical_ordered(num_classes, scaling='linear'):
    """
    Return a loss function which penalizes more 'distant' categories more heavily.

    `scaling` determines the type of scaling applied to the difference in class indices,
        and may be one of {'linear', 'sqrt', 'square'}. 'sqrt' is the least severe, and
        'square' is the most severe, in terms of increased penalty for increased distance.
    """
    idx_distance = lambda row: np.abs(np.arange(-np.argmax(row),-np.argmax(row)+num_classes))
    if scaling == 'sqrt':
        scale_fn = lambda row: np.sqrt(row)
    elif scaling == 'square':
        scale_fn = lambda row: np.square(row)
    else:
        scale_fn = lambda row: row

    epsilon_ = ops.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)

    def loss_func(y_true, y_pred):
        _y_true = tf.
        scaled_distances = np.array([scale_fn(idx_distance(row)) for row in y_true]) / (num_classes-1)

        epsilon_ = ops.convert_to_tensor(_EPSILON, dtype=y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1 - epsilon_)

        # from tf.categorical_crossentropy:
        
