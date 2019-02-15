from coremdlr.models import FeatureModel


class LambdaModel(FeatureModel):
    """
    LambdaModel returns feature vectors of arbitrary `func`(x[`feature`]).

    Parameters
    ----------
    model_args['feature'] : str
        Name of valid feature from dataset to be modeled
    model_args['lambda_fn'] : function, optional
        Function to apply to feature vector instances, default=`lambda x: x`.
    """
    def __init__(self, dataset_cls, dataset_args={}, model_args={}):
        FeatureModel.__init__(self, dataset_cls, dataset_args, model_args)

        self.lambda_fn = self.model_args.get('lambda_fn', lambda x: x)

        try:
            self.feature = self.model_args['feature']
        except KeyError:
            raise ValueError('LambdaModel.model_args must have a `feature` key')

    def fit(self, dataset, **fit_args):
        return

    def extract_features(self, X):
        return self.lambda_fn(self._get_X(X))
