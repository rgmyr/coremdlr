"""Bilinear Pooling CNN (BCNN) network as described in:

@inproceedings{lin2015bilinear,
    Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
    Title = {Bilinear CNNs for Fine-grained Visual Recognition},
    Booktitle = {International Conference on Computer Vision (ICCV)},
    Year = {2015}
}
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Flatten, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model as KerasModel

from coremdlr.ops import bilinear_pooling
from coremdlr.networks.util import make_backbone, make_dense_layers, unstack_rows, stack_rows



def bilinear_cnn(num_classes,
                 input_shape,
                 backbone_cnn=None,
                 backbone_layer=None,
                 fB=None,
                 fB_layer=None,
                 conv1x1=None,
                 dense_layers=[],
                 dropout_rate=None,
                 apply_rowwise=False,
                 lstm_features=None):
    '''Combine two feature extracting CNNs into single Model with bilinear_pooling + FC layers.
       fA and fB should output 4D tensors of equal shape, except (optionally) in # of channels.

    Parameters
    ----------
    num_classes : int
            Number of classes for softmax output layer
    input_shape : tuple of int
        Shape of input images. Must be compatible with fA.input & fB.input.
    backbone_cnn : KerasModel or str
        Feature network A. Should output features (N, H, W, cA).
        If str, loads the corresponding ImageNet model from `keras.applications`.
    backbone_layer : str, int, or None, optional
        If given, will try to build fA with output_layer having this name or index.
    fB : KerasModel or str, optional
        Feature network B. Should output features (N, H, W, cB).
        If str, loads the corresponding ImageNet model from `keras.applications`.
        If `None`, will return symmetric BCNN using fA.
    fB_layer, str, int, or None, optional
        Same as `backbone_layer`.
    conv1x1 : int or iterable(int), optional
        Add a 1x1 conv to reduce number of channels in (fA, fB) to some value(s).
        If iterable, must be length 2; values then mapped to (fA, fB).
    dense_layers : iterable of int, optional
        Sizes for additional Dense layers between bilinear vector and softmax. Default=[].
    dropout_rate : float, optional
        Specify a dropout rate for Dense layers
    apply_rowwise : bool, optional
        If True, apply pooling to each row in conv output volume, default=False.
    lstm_features : int or None, optional
        If not None & apply_rowwise=True, then apply bidirectional LSTM to row features.
        Int specifies the number of output features for each (of two) LSTM layers.

    Returns
    -------
    B-CNN : KerasModel
        Single bilinear CNN composed from fA & fB (asymmetric) or fA with itself (symmetric)
    '''
    assert backbone_cnn is not None
    fA = make_backbone(backbone_cnn, input_shape)
    fB = make_backbone(fB, input_shape)

    input_image = Input(shape=input_shape)

    outA = fA(input_image)
    if fB is None:
        outB = outA             # symmetric B-CNN
    else:
        outB = fB(input_image)  # asymmetric B-CNN

    if isinstance(conv1x1, int):
        outA = Conv2D(conv1x1, (1,1), name='reduce_A')(outA)
        outB = Conv2D(conv1x1, (1,1), name='reduce_B')(outB)
    elif hasattr(conv1x1, '__iter__'):
        assert len(conv1x1) == 2, 'if iterable, conv1x1 must have length of 2'
        outA = Conv2D(conv1x1[0], (1,1), name='reduce_A')(outA)
        outB = Conv2D(conv1x1[1], (1,1), name='reduce_B')(outB)

    bilinear_pooling_layer = Lambda(bilinear_pooling, name='bilinear_pooling')

    if apply_rowwise:
        rowsA = unstack_rows(outA)
        rowsB = unstack_rows(outB)
        x = stack_rows([bilinear_pooling_layer([rA, rB]) for rA, rB in zip(rowsA, rowsB)])
        if lstm_features is not None:
            lstm_dropout = dropout_rate if dropout_rate is not None else 0.0
            x = Bidirectional(LSTM(lstm_features, return_sequences=True, dropout=lstm_dropout))(x)
    else:
        x = bilinear_pooling_layer([outA, outB])

    x = make_dense_layers(dense_layers, dropout=dropout_rate)(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = KerasModel(inputs=input_image, outputs=pred)

    return model
