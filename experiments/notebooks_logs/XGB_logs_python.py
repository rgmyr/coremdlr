
# coding: utf-8

# In[4]:
import lasio

import numpy as np
from coremdlr.datasets import WellLoader, FaciesDataset
from coremdlr.models import FeaturePredictor, IdentityModel


# In[5]:


fdset = FaciesDataset(["205-21b-3", "204-19-6"],
                    test_wells=["204-20-6a"],
                    features=["logs"])


# In[7]:


fdset.load_or_generate_data()


# In[8]:


import hyperopt
from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.metrics import f1_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight

# for balanced log_loss computation
sample_weights = compute_sample_weight('balanced', fdset.y_test) 

fmodel_args = {
    'logs': {
        'model': 'IdentityModel',
        'model_args': {}
    }
}

XGB_SEARCH_SPACE = {
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

def train_xgb_model(model_config):
    xgb_predictor = FeaturePredictor(fdset, model_args=model_config, feature_model_args=fmodel_args)
    test_acc = xgb_predictor.fit(fdset, verbose=False)
    y_pred = xgb_predictor.predict(fdset.X_test)
    print('F1 score:', f1_score(fdset.y_test, y_pred, average='macro'))
    return log_loss(fdset.y_test, xgb_predictor.predict_proba(fdset.X_test), sample_weight=sample_weights)


# In[9]:


best_params = hyperopt.fmin(
    fn=train_xgb_model,
    space=XGB_SEARCH_SPACE,
    algo=hyperopt.rand.suggest,
    max_evals=250
)


# In[10]:


best_params


# In[12]:


params = {**XGB_SEARCH_SPACE, **best_params, **{'max_depth':5, 'n_estimators':705}}
xgb_predictor = FeaturePredictor(fdset, model_args=params, feature_model_args=fmodel_args)
xgb_predictor.fit(fdset, verbose=True)
list(zip(fdset.wells[0].which_logs, xgb_predictor.model.feature_importances_))


# In[13]:


f1_score(fdset.y_test, xgb_predictor.predict(fdset.X_test), average='macro')


