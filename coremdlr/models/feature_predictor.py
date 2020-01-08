import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import dataset classes here
from coremdlr import models
from coremdlr.models import PredictorModel
#from coremdlr.datasets.generator import DatasetGenerator

DEFAULT_XBG_ARGS = {
    'max_depth' : 3,
    'learning_rate' : 0.1,
    'n_estimators' : 1000,
    'objective' : 'multi:softprob',
    'n_jobs' : 2,
    'gamma' : 0,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'reg_alpha' : 0,                # L1 penalty
    'reg_lambda' : 1,               # L2 penalty
    'tree_method' : 'gpu_exact'
}

DEFAULT_SVC_ARGS = {
    'probability' : True
}


def cls_report_to_csv(report):
    """
    Turn a classification report into a pd.DataFrame
    """
    report_data = []
    lines = report.split('\n')

    for line in lines[2:-3]:
        row = {}
        row_data = list(filter(None, line.split()))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)

    return pd.DataFrame.from_dict(report_data)


class FeaturePredictor(PredictorModel):
    """Class for top level classifiers (operating on arbitrary 1D feature vectors).

    Parameters
    ----------

    model_args : dict, optional
        Parameters for constuctor & fit methods of chosen predictor type.
    """
    def __init__(self, dataset_cls, dataset_args={}, model_args={}, feature_model_args={}):

        PredictorModel.__init__(self, dataset_cls, dataset_args, model_args)

        model_type = self.model_args.pop('model_type', 'XGB')
        if 'XGB' in model_type:
            self.model_type = 'XGB'
            self.model_args = {**DEFAULT_XBG_ARGS, **model_args}
            self.model = XGBClassifier(**self.model_args)
        elif 'SVC' in model_type:
            self.model_type = 'SVC'
            self.model_args = {**DEFAULT_SVC_ARGS, **model_args}
            self.model = LinearSVC(**self.model_args)
        else:
            raise ValueError('`model_name` must contain one of {`XGB`, `SVC`}')

        self.features = list(feature_model_args.keys())
        self.feature_models = {}
        self.feature_model_fit_args = {}

        # Instantiate subordinate FeatureModel(s)
        for feature in self.features:
            assert feature in self.data.features, 'FeatureModels must operate on features present in Dataset.features'
            fmodel_cls = getattr(models, feature_model_args[feature]['model'])
            fmodel_args = {**feature_model_args[feature].get('model_args', {}), **{'feature': feature}}

            self.feature_models[feature] = fmodel_cls(self.data, {}, fmodel_args)
            self.feature_model_fit_args[feature] = feature_model_args[feature].get('fit_args', {})


    def fit(self, dataset, **fit_args):
        '''Must implement the fit method using only feature_models, dataset, & model_args.'''
        self.classes = dataset.classes
        for feature in self.features:
            print("Training model for feature: ", feature)
            self.feature_models[feature].fit(dataset, **self.feature_model_fit_args[feature])

        X_train = self.collect_features(dataset.X_train)
        y_train = dataset.y_train.argmax(-1) if dataset.y_train.ndim > 1 else dataset.y_train

        self.model.fit(X_train, y_train, **fit_args)

        return self.evaluate(dataset.X_test, dataset.y_test,
                            print_report=fit_args.pop('verbose', True),
                            save_report=fit_args.get('save_report'))


    def predict(self, X):
        '''Must implement a predict method for input data X.'''
        X_features = self.collect_features(X)
        return self.model.predict(X_features)


    def predict_proba(self, X):
        '''Class-wise probability predictions.'''
        X_features = self.collect_features(X)
        return self.model.predict_proba(X_features)


    def evaluate(self, X, y, print_report=False, save_report=None):
        '''
        Return mean accuracy of predict(X).
        '''
        X_features = self.collect_features(X)
        y_pred = self.model.predict(X_features)
        y_true = y.argmax(-1) if y.ndim > 1 else y
        acc = accuracy_score(y_true, y_pred)

        if print_report or save_report:
            cls_report = classification_report(y_true, y_pred, target_names=self.classes)
            if print_report:
                print(cls_report, "\nTotal accuracy Score : ", acc)
                print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))
            if save_report:
                cls_report_to_csv(cls_report).to_csv(save_report+f'_{round(acc*100, 2)}.csv')

        return acc


    def feature_importances(self):
        pass


    def collect_features(self, X):
        '''Get concatenated feature vectors for input data X.'''
        X_features = []
        for feature in self.features:
            X_features.append(self.feature_models[feature].extract_features(X))

        if not hasattr(self, 'features_info'):
            feature_sizes = [feats.shape[-1] for feats in X_features]
            self.features_info = {n : d for n, d in zip(self.features, feature_sizes)}

        return np.concatenate(X_features, axis=-1).squeeze()


    def save(self, path):
        pass

    def load(self, path):
        pass
