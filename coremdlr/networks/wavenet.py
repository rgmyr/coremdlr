"""
My own experimental dilation-block networks for texture problems.
"""
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, Activation, Multiply, Add, Reshape, Dropout


def wavenetBlock(filters, residual_filters, kernel_size, dilation_rate):
    def f(input_):
        residual = input_

        tanh_out = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                          padding='same', activation='tanh')(input_)
        sigm_out = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                          padding='same', activation='sigmoid')(input_)

        merged = Multiply()([tanh_out, sigm_out])

        skip_out = Conv1D(residual_filters, 1, activation='relu', padding='same')(merged)

        out = Add()([skip_out, residual])

        return out, skip_out

    return f


def wavenet(num_classes,
            input_shape,
            num_blocks=20,
            block_filters=64,
            residual_filters=16,
            output_resolution=32,
            dropout_rate=None):

    input_ = Input(shape=input_shape)

    if len(input_shape) == 1:
        x = Reshape((input_shape[0], 1))(input_)
    else:
        x = input_

    x = Conv1D(residual_filters, 7, padding='same', activation='relu')(x)

    A, B = wavenetBlock(block_filters, residual_filters, 3, 2)(x)
    skip_connections = [B]

    for i in range(num_blocks):
        A, B = wavenetBlock(block_filters, residual_filters, 3, 2**((i+2)%9))(A)
        skip_connections.append(B)

    net = Add()(skip_connections)
    net = Activation('relu')(net)

    net = Conv1D(residual_filters, 1, activation='relu')(net)
    net = Conv1D(residual_filters, output_resolution, strides=output_resolution, padding='valid')(net)
    #net = Reshape((-1, output_resolution))(net)

    if dropout_rate is not None:
        net = Dropout(dropout_rate)(net)

    net = Dense(num_classes, activation='softmax')(net)

    model = KerasModel(inputs=input_, outputs=net)

    return model
