from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, TimeDistributed, Bidirectional

from coremdlr.networks.util import make_dense_layers


def rmlp(num_classes,
         input_shape,
         output_mode='classify',
         layer_type='lstm',
         layer_args={},
         hidden1=None,
         hidden2=None,
         dropout_rate=0.0):
    '''Recurrent multilayer perceptron network (presumably for logs).'''

    _layer_fn = GRU if 'gru' in layer_type.lower() else LSTM

    if 'regress' in output_mode:
        output_layer = Dense(1, kernel_initializer='normal')
    else:
        output_layer = Dense(num_classes, activation='softmax')

    input_seq = Input(shape=input_shape)
    hidden1 = hidden1 if hidden1 is not None else input_shape[0]
    x = Bidirectional(_layer_fn(hidden1, return_sequences=True, dropout=dropout_rate, **layer_args))(input_seq)
    if hidden2 is not None:
        x = Bidirectional(_layer_fn(hidden2, return_sequences=True, dropout=dropout_rate, **layer_args))(x)

    preds = output_layer(x) # TimeDistributed(output_layer)

    return KerasModel(inputs=input_seq, outputs=preds)
