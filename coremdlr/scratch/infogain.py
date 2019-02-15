from keras import backend as K

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops


def init_infogain(label_bincount, grouping=None, intra_inter_ratio=None)
    '''Initial implementation assumes target will contain only one non-zero value (of 1).'''
    H = 5 * 5 * 5

    # tf/keras implementation of categorical crossentropy below ...
    # def categorical_crossentropy(target, output):

    output = output / math_ops.reduce_sum(output, len(output.get_shape()) - 1, True)

    # manual computation of crossentropy
    epsilon_ = _to_tensor(K.epsilon(), output.dtype.base_dtype)

    # makes all values between [eps, 1-eps]
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

    return -math_ops.reduce_sum(target * math_ops.log(output), axis=len(output.get_shape()) - 1)



