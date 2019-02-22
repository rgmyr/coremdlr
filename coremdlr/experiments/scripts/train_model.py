import os
import pathlib
from time import time
import numpy as np

#from clr_callback import CyclicLR  (doesn't actually set lr?)
from tensorflow.keras.backend import eval
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback


TB_SCRIPT_NAME = 'tensorboard_view_experiment.sh'
TB_INIT_STRING = 'tensorboard --logdir='

'''
def add_to_tensorboard_script(save_dir):
    # Append to tensorboard script (or create if none exists yet).
    p = pathlib.PosixPath(save_dir)
    experiment_dir = p.parent
    experiment_name = p.name
    if not experiment_dir.exists():
        experiment_dir.mkdir()
    tensorboard_file = experiment_dir / TB_SCRIPT_NAME
    tensorboard_experiment_string = str(experiment_name)+':'+str(p)+','
    if tensorboard_file.is_file():
        with tensorboard_file.open(mode='a') as tb_file:
            tb_file.write(tensorboard_experiment_string)
    else:
        with tensorboard_file.open(mode='w') as tb_file:
            tb_file.write(TB_INIT_STRING+tensorboard_experiment_string)
'''

def train_model(model,
                dataset,
                epochs,
                batch_size,
                flags,
                gpu_ind=None,
                save_dir=None,
                reporter=None):

    callbacks = []

    if 'EARLY_STOPPING' in flags:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=25, verbose=1)
        callbacks.append(early_stopping)
    # GPU_UTIL_SAMPLER

    if 'TENSORBOARD' in flags:
        #add_to_tensorboard_script(save_dir)
        tensorboard = TensorBoard(log_dir=save_dir)
        print('Logging TENSORBOARD at ', save_dir)
        callbacks.append(tensorboard)

    if 'CHECKPOINT' in flags:
        model_checkpoint = ModelCheckpoint(model.weights_filename(save_dir), monitor='val_loss', verbose=1,
                                           save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)

    t = time()
    model.fit(dataset, batch_size, epochs, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    return model, save_dir
