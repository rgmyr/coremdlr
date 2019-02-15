import numpy as np
import argparse, pickle

from keras import backend as K
from keras import utils
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import mobilenetv2
from keras.callbacks import EarlyStopping

from coremdlr.utils.facies_utils import make_labeled_frames
from coremdlr.utils.models_utils import mnet_idxs_above, save_keras_model, print_layers


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', dest='alpha', type=float, 
            help='Alpha (n_filters) multiplier for MobileNetV2.', default=1.0)  

    parser.add_argument('-d', '--depth_multiplier', dest='depth_multiplier', type=float, default=1.0,
            help='Depth (or resolution) multiplier for depthwise convolution.')

    parser.add_argument('-l', '--label_type', dest='label_type', default='class',
            help='One of \'category\' (default) or \'facies\'. Accuracy on categories is computed implicitly when '+
                 'training on facies.')

    parser.add_argument("-g", "--gpus", dest='gpus', type=int, default=1,
            help="# of GPUs to use for training")

    return parser

parser = make_parser()
args = parser.parse_args()
param_string = 'alpha'+str(args.alpha)+'_depth'+str(args.depth_multiplier)+'_'+args.label_type

# Create Training Data, set Metrics
img = np.load('./data/FULL_CORE_image_arr.npy')
depths = np.load('./data/FULL_CORE_depth_arr.npy')

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
for layer in model.layers:
    layer.trainable = True

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

