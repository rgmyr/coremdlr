import numpy as np
from getpass import getuser

from keras import backend as K
from keras import utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping
from keras.applications import mobilenetv2, xception
from keras.preprocessing.image import ImageDataGenerator

#from keras.utils.training_utils import multi_gpu_model

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def get_layerwise_responses(model, test_input, pool=None):
    '''Get responses of each layer in model to test_input

    Parameters
    ----------
    model : keras.models.Model
    test_input : np.array
        Input used to generate outputs
    pool : None, or one of {'max', 'avg', 'random'} 
        None returns all channels
        Collapse each layer by 'max' or 'avg' across channels
        Anything else will result in random channel sampling

    Returns
    -------
    responses : dict of {layer.name : np.array}
    '''
    test = test_input[np.newaxis,...]

    inp = model.input
    outputs = [layer.output for layer in model.layers[1:-3]]
    functor = K.function([inp] + [K.learning_phase()], ouputs)

    layer_outs = functor([test, 1.])
    responses = {}
    for i, out in enumerate(layer_outs):
        if pool is None:
            response = out
        elif pool == 'max':
            response = np.max(out[0,...],axis=-1)
        elif pool == 'avg':
            response = np.mean(out[0,...],axis=-1)
        else: # will assume pool == 'random'
            c_idx = np.random.randint(0, out.shape[-1])
            response = out[0,...,c_idx]
        responses[model.layers[i].name] = response

    return responses

# Currently allowed base_models from keras.applications
supported_models = ['mobilenet', 'xception']

# Allowed out_layers for xception-based model, value is layer index
mobilenet_out_layers = {}

# Allowed out_layers for xception-based model, value is layer index
xception_out_layers = {
        'block2_sepconv2_act' : 9,
        'block3_sepconv2_act' : 19,
        'block4_sepconv2_act' : 29,
        'block5_sepconv3_act' : 42,
        'block6_sepconv3_act' : 52,
        'block7_sepconv3_act' : 62,
        'block8_sepconv3_act' : 72,
        'block9_sepconv3_act' : 82,
        'block10_sepconv3_act' : 92,
        'block11_sepconv3_act' : 102,
        'block12_sepconv3_act' : 112,
        'block13_sepconv2_act' : 119
        }

supported_layers = {
        'mobilenet' : mobilenet_out_layers,
        'xception' : xception_out_layers
        }

def build_conv_model(input_shape, n_classes, base_model_name, pretrained=True,
                    pooling='avg', out_layer=None, n_features=256):
    '''Build a conv classifier model based on Xception or MobileNetV2.
    
    Parameters
    ----------
    input_shape : tuple(h, w, channels)
        Input shape for model
    n_classes : int
        Number of unique training labels
    base_model_name : string, one of {'mobilenet', 'xception'}
        Which base model to use
    pretrained : bool, optional
        Whether to use imagenet weights (default=True). Otherwise, use random initialization.
    pooling : string, optional
        One of {'avg', 'max'}. Pooling type to use before dense layers. (default='avg')
    out_layer : string, optional
        Name of base model layer to use as output to dense layers (default=last activation layer)
    n_features : int, optional
        Size of pre-softmax dense layer output (default=256)
    
    Returns
    -------
    model : keras.models.Model
        Uncompiled model instance
    '''
    assert base_model_name in supported_models, 'base_model_name must be supported'
    assert pooling in ['avg', 'max'], 'pooling argument must be avg or max'
    weights = 'imagenet' if pretrained else None
    if out_layer is not None:
        assert out_layer in supported_layers[base_model_name].keys(), 'out_layer must be supported'

    # Model Setup
    if base_model_name == 'mobilenet':
        model_call = mobilenetv2.MobileNetV2
    elif base_model_name == 'xception':
        model_call = xception.Xception

    base_model = model_call(include_top=False, weights=weights, input_shape=input_shape)
    print('Setup base_model: ', base_model_name)
    if out_layer is None:
        x = base_model.output
        print('Using base_model.output as output layer')
    else:
        # Note: will throw exception if out_layer not in supported_layers
        idx = supported_layers[base_model_name][out_layer]
        x = base_model.layers[idx].output
        print('Using layer ', base_model.layers[idx].name, ' as output layer')

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)
    
    x = Dense(n_features, activation='relu')(x)
    preds = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    return model
    

