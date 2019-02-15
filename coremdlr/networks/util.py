import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout, Lambda
from tensorflow.keras.models import Model as KerasModel

# Lambdas for applying row-wise pooling in image CNNs
unstack_rows = Lambda(lambda x: tf.unstack(x, axis=-3), name='unstack_rows')
stack_rows = Lambda(lambda x: tf.stack(x, axis=-2), name='stack_rows')

# Callables for ImageNet-pretrained model fetching
keras_apps = {'vgg16'               : applications.vgg16.VGG16,
              'vgg19'               : applications.vgg19.VGG19,
              'resnet50'            : applications.resnet50.ResNet50,
              'xception'            : applications.xception.Xception,
              'mobilenet'           : applications.mobilenet.MobileNet,
              'densenet121'         : applications.densenet.DenseNet121,
              'densenet169'         : applications.densenet.DenseNet169,
              'densenet201'         : applications.densenet.DenseNet201,
              'nasnet_large'        : applications.nasnet.NASNetLarge,
              'nasnet_mobile'       : applications.nasnet.NASNetMobile,
              'inception_v3'        : applications.inception_v3.InceptionV3,
              'inception_resnet_v2' : applications.inception_resnet_v2.InceptionResNetV2}


def make_backbone(backbone_cnn, input_shape):
    '''Check an existing backbone Model or grab ImageNet pretrained from keras_apps.'''
    if backbone_cnn is None:
        return None
    elif isinstance(backbone_cnn, KerasModel):
        assert len(backbone_cnn.output_shape)==4, 'backbone_cnn.output must output a 4D Tensor'
        return backbone_cnn
    elif isinstance(backbone_cnn, str):
        assert backbone_cnn in keras_apps.keys(), 'Invalid keras.applications string'
        model = keras_apps[backbone_cnn](include_top=False, input_shape=input_shape)
        # resnet50 ends with a 7x7 pooling, which collapses conv to 1x1 for 224x224 input
        if backbone_cnn == 'resnet50':
            model = KerasModel(inputs=model.input, outputs=model.layers[-2].output)
        return model
    else:
        raise ValueError('input to make_backbone() has invalid type')


def make_dense_layers(dense_layers, dropout=None):
    '''Instantiate a series of Dense layers, optionally with Dropout.'''
    if len(dense_layers) == 0:
        if dropout is not None:
            return lambda x: Dropout(rate=dropout)(x)
        else:
            return lambda x: x
    else:
        def dense_layers_fn(x):
            for N in dense_layers:
                x = Dense(N, activation='relu')(x)
                if dropout is not None:
                    x = Dropout(rate=dropout)(x)
            return x
        return dense_layers_fn


def compute_depth_compression_factor(network):
    '''Get reduction factor (input_depth_size / ouput_depth_size) for a network.'''
    in_depth = network.input_shape[1]
    out_depth = network.output_shape[1] if len(network_output_shape) > 2 else 1

    if in_depth is not None and out_depth is not None:
        factor = in_depth / out_depth
        if not factor.is_integer():
            raise UserWarning('Computed `depth_compression_factor` is not a whole number!')
        else:
            return factor
    else:
        raise NotImplementedError('Havent implemented compression factor check for variable input_depth networks')
