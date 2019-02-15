# `Models` Docs

Facies prediction models have a unified interface enforced by the abstract base classes in `base.py`.

Concrete model classes are constructed with `__init__(dataset_cls, dataset_args={}, model_args={})`. The one exception is `FeaturePredictor`, which also requires `feature_model_args` to specify the models & args for feature extraction sub-models.

In general, the required and optional `model_args` vary between different kinds of models.

# Base Classes

### `BaseModel`

Abstract base class for all `<>Model` subclasses.

Enforces `@abstractmethod` `fit(self, dataset **fit_args)`.

Defines concrete `save` and `load` methods for pickling/unpickling `obj.__dict__` values.

Defines `@classmethod` `from_file(path)` for initializing class instances using `load` method.

More complex save/load steps (e.g., for `NetworkModel`) can be added in subclass methods.

```
Parameters
----------
dataset_cls : Dataset type or instance
    If type, instantiated with `dataset_args`, otherwise treated as reference to existing Dataset.
dataset_args : dict
    Arguments for Dataset constructor, if necessary.
model_args : dict
    Arguments for the Model subclass, saved as `model_args` attribute.
```

### `FeatureModel( BaseModel )`

Adds `'feature'` as a required key in `model_args`.

Adds `@abstractmethod` `extract_features(self, X)`.

Adds concrete method `_get_X(self, X)`, which just returns `feature` from `X` iff `type(X) is dict`.


### `PredictorModel( BaseModel )`

Adds `@abstractmethod`s:

- `predict(self, X)`
- `predict_proba(self, X)`
- `evaluate(self, X, y)`

Adds concrete methods:

- `proba2regress(self, probas)`
  - Map predicted class probabilities `probas` to regression value in range `[-0.5, k+0.5]`
- `preds_striplog(self, well_name, save_csv=None)`
  - Predict on `well_name` and return predictions as a `striplog.Striplog` instance
- `preds_dataframe(self, well_name, logs=['GR','RDEP'], save_csv=None)`
  - Predict on `well_name` and return features + labels + predictions in a `pandas.DataFrame`


# Derived Classes

## `LambdaModel( FeatureModel )`

The simplest and most flexible feature extractor model. It applies an arbitrary function to a given feature from the dataset. The `'feature'` argument must be given in `model_args`.

The function to apply may be specified with the `'lambda_fn'` argument. It defaults to `lambda x: x`, although this will usually only work for sampled logs, since typically you need to return a single 1D vector per labeled instance in the dataset.


## `NetworkModel( FeatureModel, PredictorModel )`

Wrapper for models built on `tensorflow.keras` networks, regardless of feature type.

The `model_args` should include a `'network'` (name of a valid network-returning function from `facies.networks`), and `'network_args'` (any required parameters for the network function). For sequence models, a `'sequence_size'` (labels per instance) can be specified (`default = 1`).

Optionally, an `'optimizer'` and `'optimizer_args'` may be specified. The default is to just use `Adam()`.

The `'loss'` parameter defaults to `'categorical_crossentropy'`, but `'ordinal_squared_error'` is also valid. The latter converts class `probas` to regression values and optimizes `MSE` w.r.t. class indices. Using a weighted combination of these two losses is also allowed, and has been beneficial in some cases:

```
'loss' : {'categorical_crossentropy' : 0.25,
          'ordinal_squared_error'    : 0.75}
```

Additional `fit_args` may be specified in the constructor or in the `fit()` call. These include `'batches'`, `'epochs'`, `'step_size'`, and `'class_weighted'`. A list of callbacks can also be given in the `'callbacks'` field.

See implementation for details and experiments for examples.

## `FeaturePredictor( PredictorModel )`

This model uses `XGBClassifier` or `sklearn`'s `SVC` to build a classifier one top of one or more `FeatureModel`s.

The `feature_model_args` should be a dict of `feature -> {feature_model, feature_model_args}` sub-dicts:

```
{
'featureA' : {
    'model' : 'SomeModel_A',
    'model_args' : { <SomeModel_A args> }
}
'featureB' : {
    'model' : 'OtherModel_B',
    'model_args' : { <OtherModel_B args> }
}
...  
}  
```
