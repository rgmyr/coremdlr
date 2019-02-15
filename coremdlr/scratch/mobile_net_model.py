'''
Working on model construction function with option for using multiple adjacent mini-patches that are width-centered in a vertical core section.

Also working on InfoGain loss implementation w/ parameter for tunable between-category vs within-category loss weighting.
'''
import numpy as np
import argparse, pickle

from keras import backend as K
from keras.models import Model
from keras.layers import (Dense, concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D)
from keras.applications import mobilenetv2

from keras import utils
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from coremdlr.utils.facies_utils import make_labeled_frames
from coremdlr.utils.models_utils import mnet_idxs_above, save_keras_model, print_layers


param_string = 'alpha'+str(args.alpha)+'_depth'+str(args.depth_multiplier)+'_'+args.label_type

# Create Training Data, set Metrics
img = np.load('./data/FULL_CORE_image_arr.npy')
depths = np.load('./data/FULL_CORE_depth_arr.npy')

def make_infogain_loss(y):
    '''Make a symbolic tensorflow infogain loss function.

    Parameters
    ----------
    y: tf.tensor
        labels to calculate frequencies from

    TODO: add controlable between-category vs within-category weighting
    '''
    pass


def subpatches(n_patches, patch_size):
    '''Map input patch to mobileNet sized subpatches.'''
    def slice_func(x):
        start = x.shape[2]
    pass
        

def multi_mobile_net(pretrained=True, label_type='facies', patch_size=96, n_patches=1, n_features=256,
                     pooling='average', alpha=1.0, depth_multiplier=1.0)
    '''Make a (multi-)MobileNetV2 model. Input must have width of at least patch_size*n_patches.'''

    # handle arguments
    assert(n_patches in [1,2,3])
    input_shape = (patch_size, patch_size, 3)
    weights = 'imagenet' if pretrained else None
    GlobalPooling = GlobalAveragePooling2D() if pooling=='average' else GlobalMaxPooling2D()

    # base mini-patch classifier
    mNet_base = mobilenetv2.MobileNetV2(include_top=False, weights=weights, input_shape=input_shape,
                                        alpha=alpha, depth_multiplier=depth_multiplier)
    
    # handle multi-patch input 
    
    inputs = [mNet_base for i in range(n_patches)  
    
    # to concat before or after final feature mapping?
    x = GlobalPooling(x)
    x = Dense(n_features, activation='relu')(x)

    x = concatenate([mnet_base.output]*n_patches)
    if n_patches

    predictions = Dense(n_classes, activation='softmax')(x)

    model_build = Model(inputs=mnet_base.input, outputs=predictions)

    try: # Use multiple GPUs if available
        model = multi_gpu_model(model_build)
        print("Model Built. Using multiple (data parallel) GPUs.")
    except:
        model = model_build
        print("Model Built. Using single GPU.")
        pass

class CoreNetContainer():

    '''Make a (multi-)MobileNetV2 model. Input must have width of at least patch_size*n_patches.'''
    def __init__(self, id_string=None, pretrained=True, n_classes=9,
                    patch_size=96, n_patches=1, n_features=256,
                    pooling='average', alpha=1.0, depth_multiplier=1.0)
        # handle arguments
        assert(n_patches in [1,2,3])
        self.patch_shape = (patch_size, patch_size, 3)
        weights = 'imagenet' if pretrained else None
        GlobalPoolingLayer = GlobalAveragePooling2D() if pooling=='average' else GlobalMaxPooling2D()
        #if id_string is None:
            #self.id_string = 'nPatch%d_nClass%d_ '
        #else:
        self.id_string = id_string

        # base mini-patch classifier
        mNet_base = mobilenetv2.MobileNetV2(include_top=False, weights=weights, input_shape=self.patch_shape,
                                            alpha=alpha, depth_multiplier=depth_multiplier)
        # handle multi-patch input 
        # concat before or after final feature mapping? additional dense layer? or width-wise convolution?
        inputs = [mNet_base.input for i in range(n_patches)]
        x = [mNet_base.output for i in range(n_patches)] 
        x = [GlobalPoolingLayer(xi) for xi in x]
        x = [Dense(n_features*2, activation='relu')(xi) for xi in x]
        x = concatenate(x)
        x = Dense(n_features, activation='relu', name='Feature_Layer')(x)
        predictions = Dense(n_classes, activation='softmax')(x)
        self.model_build = Model(inputs=inputs, outputs=predictions)

        try: # Use multiple GPUs if available
            self.model = multi_gpu_model(self.model_build)
            print("Model Built. Using multiple (data parallel) GPUs.")
        except:
            self.model = self.model_build
            print("Model Built. Using single GPU.")
            pass

        return self

    def add_biLSTM(n_layers=1, merge_mode='concat')
        input_features = self.model
        pass

    def train(self, X, y, custom_loss=None):
        # forward pass execution --> return output layer
        pass

    def save_model(self):


if args.label_type == 'facies':
    class_labels = np.load('./data/FULL_CORE_facies_labels.npy')
    n_classes = 9
    '''
    import keras.backend as K
    def implicit_map(facies_num):
        if facies_num <= 2:
            return 1
        elif facies_num <= 5:
            return 2
        else:
            return 3

    def implicit_category_accuracy(y_true, y_pred):
        mapped_true = K.map_fn(implicit_map, y_true)
        mapped_pred = K.map_fn(implicit_map, y_pred)
        return K.accuracy(mapped_true, mapped_pred)

    metrics = ['accuracy', implicit_category_accuracy]
    '''
    metrics=['accuracy']
        
else:
    class_labels = np.load('./data/FULL_CORE_class_labels.npy')
    n_classes = 3
    metrics = ['accuracy']

X_train, y_train = make_labeled_frames(img[:,152:248], class_labels, 96, stride=96, shuffle=True)

datagen = ImageDataGenerator(width_shift_range=0.1,
                            shear_range=-5.0,
                            zoom_range=0.05,
                            fill_mode='constant', cval=0.0,
                            horizontal_flip=True,
                            validation_split=0.2)

y_train = utils.to_categorical(y_train, n_classes)
datagen.fit(X_train)
print("Image Data Generator Complete")

# Build Model
mnet_base = mobilenetv2.MobileNetV2(include_top=False, weights='imagenet', input_shape=(96,96,3),
                                    alpha=args.alpha, depth_multiplier=args.depth_multiplier)
x = mnet_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)

model_build = Model(inputs=mnet_base.input, outputs=predictions)

try:
    model = multi_gpu_model(model_build)
    print("Model Built. Using multiple (data parallel) GPUs.")
except:
    model = model_build
    print("Model Built. Using single GPU.")
    pass

# first train new top layers
for layer in mnet_base.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy', 
            metrics=metrics)
print("CURRENT TRAINABLE LAYERS:")
print_layers(model, only_trainable=True)

epochs = 500
batch_size = 64 * args.gpus
early_stop = EarlyStopping(monitor='loss', patience=25)
hist_1 = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size,
                    epochs=epochs,
                    callbacks=[early_stop], verbose=1)

save_keras_model(model_build, './saved_models/mobileNet_classLayer_'+param_string)

with open('./keras_logs/mobileNet_classHist_'+param_string, 'wb') as hist_pickle:
        pickle.dump(hist_1.history, hist_pickle)

print("Saved MobileNet classification-tuned network.")

# train the top two sepconv blocks now
#mnet_unfreeze_above(model, block=15) # This isn't having an effect --> maybe get indices and set manually
for i in mnet_idxs_above(model, block=15):
    model.layers[i].trainable = True

# need to recompile
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy', 
            metrics=metrics)
print("CURRENT TRAINABLE LAYERS:")
print_layers(model, only_trainable=True)

epochs = 500
batch_size = 64 * args.gpus
early_stop = EarlyStopping(monitor='loss', patience=25)
hist_2 = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size,
                    epochs=epochs,
                    callbacks=[early_stop], verbose=1)

save_keras_model(model_build, './saved_models/mobileNet_top2blocks_'+param_string)

with open('./keras_logs/mobileNet_top2Hist_'+param_string, 'wb') as hist_pickle:
        pickle.dump(hist_2.history, hist_pickle)

print("Saved MobileNet network with top two conv modules tuned.")

K.clear_session()

