import os
import pathlib


###############################
######## DATASET INFO #########
###############################

WELL_NAMES = [
    '204-19-2',
    '204-19-3A',
    '204-19-6',
    '204-19-7',
    '204-20-1',
    '204-20-1Z',
    '204-20-2',
    '204-20-3',
    '204-20-6a',
    '204-20a-7',
    '204-24a-6',
    '204-24a-7',
    '205-21b-3'
]

DEFAULT_TRAIN_PATH = pathlib.PosixPath('/home/'+os.environ['USER']+'/Dropbox/core_data/facies/train_data')


#################################
######## DATASET PARAMS #########
#################################

DEFAULT_IMAGE_ARGS = {
    'image_width' : 600,
    'scale_by': 'dataset',
    'scaler' : 'caffe'
}

DEFAULT_PGR_ARGS = {
    'features' : ['mean', 'var'],
    'per_channel' : True,
    'fill_mode' : 'ffill',
    'scale_by' : 'dataset',
    'scaler' : 'robust',
}

DEFAULT_LOGS_ARGS = {
    'which_logs' : ['GR','SP','DENS','NEUT','PEF','RDEP','RSHAL','DTC','DTS','DTS1','DTS2'],
    'interp_kind' : 'linear',
    'fill_mode' : 'mean',
    'scale_by' : 'well',
    'scaler' : 'standard'
}
