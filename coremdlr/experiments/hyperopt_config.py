"""
Hyperparameter optimization configuration for experiments with hyperopt.

Any valid experiment_config["network"] name should have a corresponding
search space defined here to be usable with ./hyperopt_experiment.py
"""
import hyperopt
from hyperopt import hp
from hyperopt.pyll.base import scope

########################
### NAMING UTILITIES ###
########################

# substitute some param names for prettier ID strings
modified_param_names = {
    'backbone_cnn': '',
    'optimizer': '',
    'conv1x1': 'conv',
    'dropout_rate' : 'dropout',
    'encode_K': 'K'
}

def strlike(value):
    '''Like str(), except truncate floats.'''
    if isinstance(value, str):
        return value
    elif isinstance(value, float):
        return "%2.2f" % value
    else:
        return str(value)

def kv_string(k, v):
    '''Make modified string with (k, v).'''
    k_str = modified_param_names.get(k, k)
    v_str = strlike(v)
    return k_str + v_str

def make_unique_name_function(space):
    '''Return a function that generates unique names via tunable parameter values.'''
    net_tunables = [k for k, v in space['network_args'].items() if isinstance(v, hyperopt.pyll.base.Apply)]
    opt_tunables = [k for k, v in space['optimizer_args'].items() if isinstance(v, hyperopt.pyll.base.Apply)]

    def unique_name_function(params):
        net_strs = [kv_string(k,v) for k, v in params['network_args'].items() if k in net_tunables]
        opt_strs = [kv_string(k,v) for k, v in params['optimizer_args'].items() if k in opt_tunables]
        return '_'.join(net_strs + opt_strs)

    return unique_name_function


###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++###
###------------------------  SEARCH SPACE DEFINITIONS --------------------------------###
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++###


##############################
##### FEATURE PREDICTORS #####
##############################

xgb_classifier = {
    'model_type' : 'XGB',
    'max_depth' : scope.int(hp.quniform('max_depth', 3, 10, 1)),
    'learning_rate' : hp.uniform('learning_rate', 0.01, 0.2),
    'n_estimators' : scope.int(hp.quniform('n_estimators', 10, 1000, 1)),
    'objective' : 'multi:softprob',
    'n_jobs' : 2,
    'gamma' : hp.uniform('gamma', 0, 0.5),
    'subsample' : hp.uniform('subsample', 0.3, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.3, 1.0),
    'colsample_bylevel' : 1,
    'reg_alpha' : 0,                                    # L1 penalty
    'reg_lambda' : hp.uniform('reg_lambda', 0.1, 10),   # L2 penalty
    'tree_method' : 'gpu_exact',
}
# XGBClassifier documentation:
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn



##############################
####### IMAGE NETWORKS #######
##############################

deepten = {
    "network_args" : {
        "backbone_cnn": "resnet50",
        "encode_K": scope.int(hp.quniform('encode_K', 4, 64, 1)),
        "conv1x1": scope.int(hp.quniform('conv1x1', 32, 256, 1)),
        "dropout_rate": hp.uniform('dropout_rate', 0.05, 0.95)
    },
    "optimizer_args" : {
        "optimizer": "Adam"
    }
}

########################
### SPACES CONTAINER ###
########################

search_spaces = {
    'deepten': deepten
}
