from abc import ABC, abstractmethod
import dill as pickle

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all <>Model subclasses.

    Defines concrete `save` and `load` methods for pickling/unpickling obj.__dict__ values.
    Defines `from_file(path)` @classmethod for initializing class instances using `load` method.
    More complex save/load steps (e.g., for NetworkModel) can be added in subclass methods.

    Parameters
    ----------
    dataset_cls : Dataset type or instance
        If type, instantiated with dataset_args, otherwise treated as ref to existing Dataset.
    dataset_args : dict
        Arguments for Dataset constructor, if necessary.
    model_args : dict
        Arguments for the Model subclass, saved as an attribute.
    """
    def __init__(self, dataset_cls, dataset_args, model_args):

        if 'from_file' in model_args.keys():
            self.load(model_args['from_file'])
            self.was_loaded_from_file = True

        else:
            self.was_loaded_from_file = False
            self.model_args = model_args
            if isinstance(dataset_cls, type):
                self.data = dataset_cls(**dataset_args)
            else:
                self.data = dataset_cls

    @classmethod
    def from_file(cls, path):
        """
        Construct the Model instance from a save path.
        """
        return cls(None, None, {'from_file': path})

    @abstractmethod
    def fit(self, dataset, **fit_args):
        pass


    def save(self, path, save_data=False):
        path = str(path)
        if path.endswith('.pkl'):
            pkl_file = open(path, 'wb')
        elif '.' in path.split('/')[-1]:
            raise ValueError('model.save/load path must be a stem or a `.pkl` file')
        else:
            pkl_file = open(path + '.pkl', 'wb')

        if not save_data:
            del self.data

        pickle.dump(self.__dict__, pkl_file)
        pkl_file.close()


    def load(self, path, set_data=None):
        path = str(path)
        if path.endswith('.pkl'):
            pkl_file = open(path, 'rb')
        elif '.' in path.split('/')[-1]:
            raise ValueError('model.save/load path must be a stem or a `.pkl` file')
        else:
            pkl_file = open(path + '.pkl', 'rb')

        self.__dict__.update(pickle.load(pkl_file))
        pkl_file.close()

        if set_data is not None:
            self.data = set_data


class FeatureModel(BaseModel):
    """
    Base class for Models that can provide features from raw data to Predictors.
    A `feature` must be given in `model_args`, and the feature must be present in the data object.
    """
    def __init__(self, dataset_cls, dataset_args, model_args):
        BaseModel.__init__(self, dataset_cls, dataset_args, model_args)
        if self.was_loaded_from_file:
            return
        try:
            self.feature = self.model_args['feature']
            assert self.feature in self.data.features, '`feature` provided to FeatureModel must be present in `data`'
        except KeyError:
            raise UserWarning('User must provide a `feature` argument for FeatureModels')

    @abstractmethod
    def extract_features(self, X):
        pass

    def _get_X(self, X):
        return X[self.feature] if type(X) is dict else X


class PredictorModel(BaseModel):
    """
    Base class for Models that can perform (probabalistic) class prediction.
    """
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    def proba2regress(self, probas):
        """
        Probabilities to regression values.
        Essentially weighted probability summed and scaled to [-0.5, k+0.5]
        """
        k = probas.shape[-1]
        return ((k+1)/k)*np.matmul(probas, np.arange(k)) - 0.5


    def plot_preds(self, well, **kwargs):
        """
        Plot image, labels, and predictions for a given well. `well` can be str or WellLoader.
        If it's a str, must be present in self.data.wells or self.data.test_wells.
        """
        if type(well) is str:
            well = self.data.get_well(well)
        preds = self.predict(well.X)
        return well.plot_strip(preds=preds, **kwargs)


    def preds_striplog(self, well_name, save_csv=None):
        """
        Generate a Striplog from predictions on `well_name`.

        Parameters
        ----------
        well : str
            Name of well (must be present in self.data)
        save_csv : str or Path, optional
            Destination of output csv. If None, just returns StripLog instance.
        """
        X, _ = self.data.get_well_data(well_name)
        preds = self.predict(X)

        well = self.data.get_well(well_name)

        return well.make_striplog(labels=preds, save_csv=save_csv)


    def preds_dataframe(self, well_name, logs=['GR','RDEP'], save_csv=None):
        """
        Generate a DataFrame from predictions (and probas) on `well_name`.

        Parameters
        ----------
        well_name : str
            Name of well (must be present in self.data)
        logs : list(str)
            Names of logs to include in the DataFrame.
            If 'logs' is not in self.data.features, this will be ignored.
        save_csv : str or Path, optional
            Destination of output csv. If None, just returns DataFrame instance.
        """
        X, y = self.data.get_well_data(well_name, include_meta=True)

        probas = self.predict_proba(X)
        preds = np.argmax(probas, axis=1)

        df = pd.DataFrame({
            'y_true' : y,
            'y_pred' : preds,
            'confidence' : np.max(probas, axis=1),
            'regression' : self.proba2regress(probas),
            **{'proba_'+str(i) : probas[:,i] for i in range(probas.shape[1])},
            **{d : X[d] for d in self.data.meta_keys}
        })

        df = df.astype({'y_pred': int, 'y_true': int})

        if 'pseudoGR' in self.data.features:
            df[self.data.wells[0].pGR_feat_names] = pd.DataFrame(np.mean(X['pseudoGR'], axis=1), index=df.index)

        if 'logs' in self.data.features and len(logs) > 0:
            log_idxs = {log: self.data.logs_args['which_logs'].index(log) for log in logs}
            for log in logs:
                df[log] = X['logs'][:, log_idxs[log]]

        if save_csv is not None:
            df.to_csv(save_csv, index=False)

        return df
