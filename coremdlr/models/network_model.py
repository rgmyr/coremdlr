import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import LearningRateScheduler

import coremdlr.networks as facies_networks
from coremdlr.models import FeatureModel, PredictorModel
from coremdlr.datasets.generators import DepthSequenceGenerator

from imgaug import augmenters as iaa


DEFAULT_FIT_ARGS = {
    'epochs' : 1,
    'batch_size' : 1,
    'callbacks' : []
}

image_aug_fn = iaa.Sequential([
    iaa.Fliplr(0.5),
    #iaa.Sharpen(alpha=(0.0,0.25), lightness=(0.9, 1.1)),
    #iaa.Invert(0.5)
    #iaa.CoarseDropout((0.0, 0.25), size_percent=(0.02, 0.2))
])

#image_nets = ['bilinear_cnn', 'deepten', 'kernel_pooling_cnn']
#pseudoGR_nets = ['conv1d_net']
#logs_nets = ['rmlp']


class NetworkModel(FeatureModel, PredictorModel):
    """
    Network predictors / feature extractors.
    """
    def __init__(self, dataset_cls, dataset_args={}, model_args={}):

        FeatureModel.__init__(self, dataset_cls, dataset_args, model_args)
        if self.was_loaded_from_file:
            return

        self.sequence_size = self.model_args.get('sequence_size', 1)
        self.input_shape = self._compute_input_shape()

        network_fn = getattr(facies_networks, self.model_args['network'])
        network_args = self.model_args.get('network_args', {})
        self.network = network_fn(self.data.num_classes, self.input_shape, **network_args)

        if self.model_args.get('summary', True):
            self.network.summary()

        self.optimizer_args = self.model_args.get('optimizer_args', {})
        self.optimizer_name = self.optimizer_args.pop('optimizer', 'Adam')

        self.fit_args = {**DEFAULT_FIT_ARGS, **self.model_args.get('fit_args', {})}

        self.batch_augment_fn = lambda x, y : (image_aug_fn(x), y) if self.feature == 'image' else None
        self.batch_format_fn = None

    """
    @property
    def weights_filename(self):
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')
    """

    def fit(self, dataset, **fit_args):

        self.fit_args.update(fit_args)
        self.train_generator_args = {
            'step_size' : self.fit_args.get('step_size', max(1, self.sequence_size // 2)),
            'concatenate_X' : self.fit_args.get('concatenate_X', False if self.feature == 'logs' else True),
            'batch_size' : self.fit_args.get('batch_size', 1),
            'feature' : self.feature,
            'augment_fn' : None,
            'format_fn' : None
        }
        callbacks = self.fit_args.pop('callbacks', [])

        if self.fit_args.get('class_weighted', False):
            class_weights = 1 - (np.bincount(dataset.y_train) / dataset.y_train.size)
        else:
            class_weights = None

        self.network.compile(loss=self.loss(class_weights),
                             optimizer=self.optimizer(callbacks),
                             metrics=self.metrics())

        train_gen = DepthSequenceGenerator(dataset.X_train, dataset.y_train, self.sequence_size, **self.train_generator_args)

        _batch_X, _batch_y = train_gen[0]
        print('Shapes of `(batch_X, batch_y)`: {}, {}'.format(_batch_X.shape, _batch_y.shape))

        self.val_generator_args = {**self.train_generator_args, **{'batch_size': 1}}
        val_gen = DepthSequenceGenerator(dataset.X_test, dataset.y_test, self.sequence_size, **self.val_generator_args)

        hist = self.network.fit_generator(
            train_gen,
            epochs=self.fit_args['epochs'],
            callbacks=callbacks,
            validation_data=val_gen,
            use_multiprocessing=False,
            workers=1,
            shuffle=False
        )

        return min(hist.history['val_loss'])


    def predict(self, X, **kwargs):
        probas = self.predict_proba(X, **kwargs)
        preds = np.argmax(probas, axis=1)
        return preds


    def predict_proba(self, X, **kwargs):
        _X = self._get_X(X)
        self.val_generator_args.update(kwargs)
        X_gen = DepthSequenceGenerator(_X, None, self.sequence_size, **self.val_generator_args)
        probas = self.network.predict_generator(X_gen)
        probas = self.align_probas(probas, X_gen)
        return probas


    def align_probas(self, probas, generator):
        """
        Align pred_probas of shape (n_seq, seq_len, classes) --> (n_sample, classes).
        Used to generate 1-to-1 pred->instance mapping when predicting on overlapping sequences.
        """
        if probas.ndim <= 2:
            return probas

        num_class = probas.shape[-1]
        output = np.zeros((generator.num_samples, num_class), dtype=probas.dtype)

        for idxs, p in zip(generator.idx_pairs, probas):
            output[idxs[0]:idxs[1],:] += p

        output /= np.linalg.norm(output, axis=1, keepdims=True)

        return output


    def evaluate(self, X, y, **generator_args):
        '''Return mean accuracy of predict(X).'''
        preds = self.predict(X)
        return np.mean(np.argmax(preds,-1) == np.argmax(y, -1))


    def extract_features(self, X, **kwargs):
        '''Return features for inputs X (assumed that penultimate layer = features).'''
        _X = self._get_X(X)

        try:
            feat_network = self.feature_network
        except AttributeError:
            self.feature_network = KerasModel(inputs=self.network.input, outputs=self.network.layers[-2].output)
            feat_network = self.feature_network

        self.val_generator_args.update(kwargs)
        X_gen = DepthSequenceGenerator(_X, None, self.sequence_size, **self.val_generator_args)
        feats = feat_network.predict_generator(X_gen)
        return self.align_probas(feats, X_gen)


    def loss(self, class_weights):
        """
        Returns a loss function based on the value of `model_args['loss']`.
        Loss can be one of {'categorical_crossentropy', 'ordinal_squared_error'}.
        It may also be a dict mapping each of these names to a weight for combined loss.
        Default is 'categorical_crossentropy'.

        `class_weights` is a vector with len == num_classes, or None (default).
        If given, will be used to weight training losses by classes of examples.
        May be specified as boolean, e.g.: fit_args['class_weighted'] = True.
        """
        losses = self.model_args.get('loss', 'categorical_crossentropy')

        if type(losses) is str:
            multi_loss = False
            losses = {losses: 1.0}
        elif type(losses) is dict:
            multi_loss = True

        if class_weights is not None:
            class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)

        # custom 'ordinal' loss option
        if 'ordinal_squared_error' in losses.keys():
            k = float(self.data.num_classes)
            a = tf.expand_dims(tf.range(0, k, dtype=tf.float32), axis=-1)
            k_factor = tf.constant((k+1)/k, shape=[1,1], name='k_factor')
            min_regr = tf.constant(-0.5, shape=[1,1], name='min_regression_value')

            def ordinal_loss(y_true, y_pred):
                y_estimate = tf.tensordot(y_pred, a, [[-1], [0]])
                y_estimate = k_factor * y_estimate + min_regr    # scale to range [-0.5, k+0.5]
                y_values = tf.cast(tf.argmax(y_true, -1), dtype=y_estimate.dtype)

                min_class = tf.convert_to_tensor(0.0, dtype=y_estimate.dtype)
                max_class = tf.convert_to_tensor(  k, dtype=y_estimate.dtype)
                sqr_error = tf.square(y_values - tf.squeeze(tf.clip_by_value(y_estimate, min_class, max_class)))

                if class_weights is not None:
                    weight_vec = tf.gather(class_weights, tf.argmax(y_true, -1))
                    sqr_error *= weight_vec

                return tf.reduce_mean(sqr_error)

            if not multi_loss:
                return ordinal_loss

        if 'categorical_crossentropy' in losses.keys():
            # TODO: option for clipping?
            def categorical_loss(y_true, y_pred):
                epsilon_ = tf.convert_to_tensor(1e-5, dtype=y_pred.dtype)
                y_pred = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

                cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

                if class_weights is not None:
                    weight_vec = tf.gather(class_weights, tf.argmax(y_true, -1))
                    cross_entropy *= weight_vec

                return cross_entropy

            if not multi_loss:
                return categorical_loss

        # weighted multi-loss option
        if multi_loss:
            def weighted_loss(y_true, y_pred):
                ord_weight = tf.constant(losses['ordinal_squared_error'], shape=[1,1])
                cat_weight = tf.constant(losses['categorical_crossentropy'], shape=[1,1])
                loss = ord_weight * ordinal_loss(y_true, y_pred) \
                     + cat_weight * categorical_loss(y_true, y_pred)
                return loss
            return weighted_loss


    def optimizer(self, callbacks):
        # Using SGD if LRSchedule present, since it plays nicer than stateful opts
        if any([isinstance(callback, LearningRateScheduler) for callback in callbacks]):
            print("Found LearningRateScheduler callback --> using SGD Optimizer")
            return getattr(optimizers, 'SGD')(**self.optimizer_args)
        else:
            return getattr(optimizers, self.optimizer_name)(**self.optimizer_args)


    def metrics(self):
        return ['accuracy']


    def save(self, path):
        self.network.save(path+'.h5')
        del self.network
        PredictorModel.save(self, path)

    def load(self, path):
        PredictorModel.load(self, path)
        self.network = load_model(path+'.h5')


    def _compute_input_shape(self):
        rows_per_instance = self.data.label_resolution*self.sequence_size
        if self.feature == 'image':
            return (rows_per_instance, self.data.X_train['image'].shape[-2], 3)
        elif self.feature == 'pseudoGR':
            # might need to change this for variable pGR "channels"
            return (rows_per_instance, self.data.X_train['pseudoGR'].shape[-1])
        elif self.feature == 'logs':
            return (self.sequence_size, len(self.data.log_args['which_logs']))
