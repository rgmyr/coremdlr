import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout
from tensorflow.keras.models import Model as KerasModel

from coremdlr.layers import KernelPooling
from coremdlr.networks.util import make_backbone, make_dense_layers, unstack_rows, stack_rows


def kernel_pooling_cnn(num_classes,
                       input_shape,
                       backbone_cnn,
                       kernel_p=4,
                       kernel_d=4096,
                       conv1x1=None,
                       dense_layers=[],
                       dropout_rate=None,
                       apply_rowwise=False,
                       lstm_features=None):
    '''Combine a backbone CNN + Kernel Pooling layer + Dense layers.

    Parameters
    ----------
    backbone_cnn : KerasModel or str
        Feature extraction network. If KerasModel, should output features (N, H, W, C).
        If str, loads the corresponding ImageNet model from `keras.applications`.
    n_classes : int
        Number of classes for softmax output layer
    input_shape : tuple of int, optional
        Shape of input image. Can be None, since Encoding layer allows variable input sizes.
    encode_K : int, optional
        Number of codewords to learn, default=64.
    conv1x1 : int, optional
        Add a 1x1 conv to reduce number of filters in backbone_cnn.output before Encoding layer.
    dense_layers : iterable of int, optional
        Sizes for additional Dense layers between Encoding.output and softmax, default=[].
    dropout_rate: float, optional
        Specify a dropout rate for Dense layers

    Returns
    -------
    KernelPoolingCNN : KerasModel
        Kernel Pooling Encoding Network
    '''
    backbone_model = make_backbone(backbone_cnn, input_shape)
    conv_output = backbone_model.output
    if conv1x1 is not None:
        conv_output = Conv2D(conv1x1, (1,1))(conv_output)

    kp_layer = KernelPooling(p=kernel_p, d=kernel_d)

    if apply_rowwise:
        add_h_dim = Lambda(lambda x: tf.expand_dims(x, 1), name='add_h_dim')
        x = stack_rows([kp_layer(add_h_dim(row)) for row in unstack_rows(conv_output)])
        if lstm_features is not None:
            lstm_dropout = dropout_rate if dropout_rate is not None else 0.0
            x = Bidirectional(LSTM(lstm_features, return_sequences=True, dropout=lstm_dropout))(x)
    else:
        x = kp_layer(conv_output)

    x = make_dense_layers(dense_layers, dropout=dropout_rate)(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = KerasModel(inputs=backbone_model.input, outputs=pred)

    return model
