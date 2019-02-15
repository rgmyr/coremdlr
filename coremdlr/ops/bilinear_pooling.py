"""The bilinear pooling operation (see networks/bilinear_cnn.py)

TODO: - support for matrix square root described in "Improved Bilinear Pooling with CNNs"
        (claimed to add 2-3% accuracy on fine-grained benchmark datasets)
"""
import tensorflow as tf
from tensorflow.keras.backend import ndim
from tensorflow.keras.layers import Flatten

def bilinear_pooling(inputs, epsilon=1e-12):
    '''Pool outer products of local features. Returns tf Function usable with keras.layers.Lambda.

    Parameters
    ----------
    inputs : (tf.Tensor, tf.Tensor)
        Both tensors should be 4D (channels last), with same shape in all but channels dimension.

    Returns
    -------
    phi : tensorflow Function
        Result of outer product pooling and normalization operations.
    '''
    iA, iB = inputs
    ndimA, ndimB = ndim(iA), ndim(iB)
    assert ndimA == ndimB, 'Inputs must have the same dimensions'

    # sum pooled outer product
    if ndimA == 3:
        ein_string = 'bim,bin->bmn'
    elif ndimA == 4:
        ein_string = 'bijm,bijn->bmn'
    else:
        raise ValueError('Bad input dimensions (must be 3 or 4).')

    phi = tf.einsum(ein_string, iA, iB)

    # sum --> avg (is this necessary?)
    #n_feat = tf.reduce_prod(tf.gather(tf.shape(iA), tf.constant([1,2])))
    #phi_I  = tf.divide(phi_I, tf.to_float(n_feat))

    phi = Flatten()(phi)
    phi = tf.multiply(tf.sign(phi), tf.sqrt(tf.maximum(phi, epsilon)))    # signed square root
    phi = tf.nn.l2_normalize(phi, axis=-1)

    return phi
