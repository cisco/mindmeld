"""This module contains base classes for models defined in the models subpackage."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object, super, str

import copy
import itertools
import logging
import math
import random

import numpy
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import (KFold, GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit)
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder, MaxAbsScaler, StandardScaler

from .helpers import get_feature_extractor

logger = logging.getLogger(__name__)

# model scoring types
ACCURACY_SCORING = "accuracy"
LIKELIHOOD_SCORING = "log_loss"

_NEG_INF = -1e10


class ModelConfig(object):
    """A simple named tuple containing a model configuration

    Deleted Attributes:
        model_type (str): The name of the model type. Will be used to find the
            model class to instantiate
        example_type (str): The type of the examples which will be passed into
            `predict()`
        label_type (str): The type of the labels which `predict()` will return
        model_settings (dict)
        params (dict): Params to pass to the underlying model when
        param_selection (dict): Configuration for param selection

            {'type': 'shuffle',
             'n': 3,
             'k': 10,
             'n_jobs': 2,
             'scoring': '',
             'grid': {}
            }

        features (dict): A mapping from feature extractor names, as given in
            FEATURE_NAME_MAP, to a kwargs dict, which will be passed into the
            associated feature extractor function.
        cv (dict): A dict that contains "type", which specifies the name of the
            cross-validation strategy, such as "k-folds" or "shuffle". The
            remaining keys are parameters specific to the cross-validation type,
            such as "k" when the type is "k-folds".

    """
    __slots__ = ['model_type', 'example_type', 'label_type', 'features', 'model_settings', 'params',
                 'param_selection']

    def __init__(self, model_type=None, example_type=None, label_type=None, features=None,
                 model_settings=None, params=None, param_selection=None):
        for arg, val in {'model_type': model_type, 'example_type': example_type,
                         'label_type': label_type, 'features': features}.items():
            if val is None:
                raise TypeError('__init__() missing required argument {!r}'.format(arg))
        if params is None and (param_selection is None or param_selection.get('grid') is None):
            raise ValueError("__init__() One of 'params' and 'param_selection' is required")
        self.model_type = model_type
        self.example_type = example_type
        self.label_type = label_type
        self.features = features
        self.model_settings = model_settings
        self.params = params
        self.param_selection = param_selection

    def __repr__(self):
        args_str = ', '.join("{}={!r}".format(key, getattr(self, key)) for key in self.__slots__)
        return "{}({})".format(self.__class__.__name__, args_str)


class Model(object):
    """An abstract class upon which all models are based.

    Attributes:
        config (ModelConfig): The configuration for the model
    """
    DEFAULT_CV_SCORING = ACCURACY_SCORING

    def __init__(self, config):
        self.config = config
        self._label_encoder = self._get_label_encoder()
        self._current_params = None
        self._resources = {}
        self._clf = None

    def fit(self, examples, labels, params=None):
        raise NotImplementedError

    def select_params(self, examples, labels, selection_settings=None):
        raise NotImplementedError

    def _convert_params(self, param_grid, y):
        """Convert the params from the style given by the config to the style
        passed in to the actual classifier.

        Args:
            params_grid (dict): lists of classifier parameter values, keyed by
                parameter name

        Returns:
            (dict): revised params_grid
        """
        raise NotImplementedError

    def predict(self, examples):
        raise NotImplementedError

    def predict_proba(self, examples):
        raise NotImplementedError

    def predict_log_proba(self, examples):
        raise NotImplementedError

    def register_resources(self, **kwargs):
        """Registers resources which are accessible to feature extractors

        Args:
            **kwargs: dictionary of resources to register

        """
        self._resources.update(kwargs)

    def get_feature_matrix(self, examples):
        raise NotImplementedError

    def _get_label_encoder(self):
        # TODO: support other label encoders
        return LabelEncoder()

    def _iter_settings(self, params_grid):
        """Iterates through all model settings.

        Yields:
            (dict): A kwargs dict to be passed to the classifier object. Each
                item yielded is a unique combination of values from with
                a choice of settings from the hyper params grid.
        """
        if params_grid:
            for config in self.settings_for_params_grid({}, params_grid):
                yield config

    @staticmethod
    def settings_for_params_grid(base, params_grid):
        """Iterates through all model settings.

        Args:
            base (dict): A dictionary containing the base settings which all
                permutations contain
            params_grid (dict): A kwargs dict of parameters that will be used to
                initialize the classifier object. The value for each key is a
                list of candidate parameters. The training process will grid
                search over the Cartesian product of these parameter lists and
                select the best via cross-validation.

        Yields:
            (dict): A kwargs dict to be passed to an underlying model object.
                Each item yielded is a unique combination of values from with
                a choice of settings from the hyper params grid.
        """
        base = copy.deepcopy(base)
        keys, values = list(zip(*list(params_grid.items())))
        for settings in itertools.product(*values):
            base.update(dict(list(zip(keys, settings))))
            yield copy.deepcopy(base)

    def _extract_features(self, example):
        """Gets all features from an example.

        Args:
            example: An example object.

        Returns:
            (dict of str: number): A dict of feature names to their values.
        """
        example_type = self.config.example_type
        feat_set = {}
        for name, kwargs in self.config.features.items():
            if callable(kwargs):
                # a feature extractor function was passed in directly
                feat_extractor = kwargs
            else:
                feat_extractor = get_feature_extractor(example_type, name)(**kwargs)
            feat_set.update(feat_extractor(example, self._resources))
        return feat_set

    def _get_cv_iterator(self, settings):
        if not settings:
            return None
        cv_type = settings['type']
        try:
            cv_iterator = {"k-fold": self._k_fold_iterator,
                           "shuffle": self._shuffle_iterator,
                           "group-k-fold": self._groups_k_fold_iterator,
                           "group-shuffle": self._groups_shuffle_iterator,
                           "stratified-k-fold": self._stratified_k_fold_iterator,
                           "stratified-shuffle": self._stratified_shuffle_iterator,
                           }.get(cv_type)(settings)
        except KeyError:
            raise ValueError('Unknown param selection type: {!r}'.format(cv_type))

        return cv_iterator

    @staticmethod
    def _k_fold_iterator(settings):
        k = settings['k']
        return KFold(n_splits=k)

    @staticmethod
    def _shuffle_iterator(settings):
        k = settings['k']
        n = settings.get('n', k)
        test_size = 1.0 / k
        return ShuffleSplit(n_splits=n, test_size=test_size)

    @staticmethod
    def _groups_k_fold_iterator(settings):
        k = settings['k']
        return GroupKFold(n_splits=k)

    @staticmethod
    def _groups_shuffle_iterator(settings):
        k = settings['k']
        n = settings.get('n', k)
        test_size = 1.0 / k
        return GroupShuffleSplit(n_splits=n, test_size=test_size)

    @staticmethod
    def _stratified_k_fold_iterator(settings):
        k = settings['k']
        return StratifiedKFold(n_splits=k)

    @staticmethod
    def _stratified_shuffle_iterator(settings):
        k = settings['k']
        n = settings.get('n', k)
        test_size = 1.0 / k
        return StratifiedShuffleSplit(n_splits=n, test_size=test_size)

    def requires_resource(self, resource):
        example_type = self.config.example_type
        for name, kwargs in self.config.features.items():
            if callable(kwargs):
                # a feature extractor function was passed in directly
                feature_extractor = kwargs
            else:
                feature_extractor = get_feature_extractor(example_type, name)
            if ('requirements' in feature_extractor.__dict__ and
                    resource in feature_extractor.requirements):
                return True
        return False



class SkLearnModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self._class_encoder = SKLabelEncoder()
        self._feat_vectorizer = DictVectorizer()
        self._feat_selector = self._get_feature_selector()
        self._feat_scaler = self._get_feature_scaler()

    def fit(self, examples, labels, params=None):
        """Trains this model.

        This method inspects instance attributes to determine the classifier
        object and cross-validation strategy, and then fits the model to the
        training examples passed in.

        Args:
            examples (list): A list of examples.
            labels (list): A parallel list to examples. The gold labels
                for each example.
            params (dict, optional): Parameters to use when training. Parameter
                selection will be bypassed if this is provided

        Returns:
            (SkLearnModel): Returns self to match classifier scikit-learn
                interfaces.
        """
        skip_param_selection = params is not None or self.config.param_selection is None

        # Prepare resources

        # Shuffle to prevent order effects
        indices = list(range(len(labels)))
        random.shuffle(indices)
        examples = [examples[i] for i in indices]
        labels = [labels[i] for i in indices]

        # TODO: add this code back in
        # distinct_labels = set(labels)
        # if len(set(distinct_labels)) <= 1:
        #     return None

        # Extract features and classes
        X, groups = self.get_feature_matrix(examples, fit=True)
        y = self._label_encoder.encode(labels)
        y = self._class_encoder.fit_transform(labels)

        if skip_param_selection:
            self._clf = self._fit(X, y, self.config.params)
            self._current_params = params
        else:
            # run cross validation to select params
            best_clf, best_params = self._fit_cv(X, y, groups)
            self._clf = best_clf
            self._current_params = best_params

        return self

    def select_params(self, examples, labels, selection_settings=None):
        X, groups = self.get_feature_matrix(examples, fit=True)
        y = self._label_encoder.encode(labels)
        y = self._class_encoder.fit_transform(labels)
        clf, params = self._fit_cv(X, y, groups, selection_settings)
        self._clf = clf
        return params

    def _fit(self, X, y, params):
        """Trains a classifier without cross-validation.

        Args:
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.
            params (dict): Parameters of the classifier

        Returns:
            SkLearnModel
        """
        params = self._convert_params(params, y)
        model_class = self._get_model_class()
        return model_class(**params).fit(X, y)

    def _fit_cv(self, X, y, groups=None, selection_settings=None):
        """Summary

        Args:
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.
            selection_settings (None, optional): Description

        Returns:
            SkLearnModel
        """
        selection_settings = selection_settings or self.config.param_selection
        cv_iterator = self._get_cv_iterator(selection_settings)

        if selection_settings is None:
            return self.config.params

        scoring = selection_settings.get('scoring', self.DEFAULT_CV_SCORING)
        n_jobs = selection_settings.get('n_jobs', -1)

        param_grid = self._convert_params(selection_settings['grid'], y)
        model_class = self._get_model_class()

        grid_cv = GridSearchCV(estimator=model_class(), scoring=scoring, param_grid=param_grid,
                               cv=cv_iterator, n_jobs=n_jobs, verbose=1)
        model = grid_cv.fit(X, y, groups)

        for candidate in model.grid_scores_:
            logger.debug('Candidate parameters: {}'.format(candidate.parameters))
            std_err = (2 * numpy.std(candidate.cv_validation_scores) /
                       math.sqrt(len(candidate.cv_validation_scores)))
            if scoring == ACCURACY_SCORING:
                msg = 'Candidate average accuracy: {:.2%} ± {:.2%}'
                logger.debug(msg.format(candidate.mean_validation_score, std_err))
            elif scoring == LIKELIHOOD_SCORING:
                msg = 'Candidate average log likelihood: {:.4} ± {:.4}'
                logger.debug(msg.format(candidate.mean_validation_score, std_err))
        if scoring == ACCURACY_SCORING:
            logger.info('Best accuracy: {:.2%}, settings: {}'.format(model.best_score_,
                                                                     model.best_params_))
            cv_loss_ = 1 - model.best_score_
        elif scoring == LIKELIHOOD_SCORING:
            logger.info('Best log likelihood: {:.4}, settings: {}'.format(model.best_score_,
                                                                          model.best_params_))
            cv_loss_ = - model.best_score_

        return model.best_estimator_, model.best_params_

    def predict(self, examples):
        X, _ = self.get_feature_matrix(examples)

        y = self._clf.predict(X)
        predictions = self._class_encoder.inverse_transform(y)

        return self._label_encoder.decode(predictions)

    def predict_proba(self, examples):
        X, _ = self.get_feature_matrix(examples)
        return self._predict_proba(X, self._clf.predict_proba)

    def predict_log_proba(self, examples):
        X, _ = self.get_feature_matrix(examples)
        predictions, log_proba = self._predict_proba(X, self._clf.predict_log_proba)

        # JSON can't reliably encode infinity, so replace it with large number
        for row in log_proba:
            for label in row:
                if row[label] == -numpy.Infinity:
                    row[label] = _NEG_INF
        return predictions, log_proba

    def _predict_proba(self, X, predictor):
        predictions = []
        proba = []
        for row in predictor(X):
            class_index = row.argmax()
            prediction = self._class_encoder.inverse_transform([class_index])[0]
            predictions.append(self._label_encoder.decode([prediction])[0])
            proba.append(dict(
                (self._label_encoder.decode(self._class_encoder.inverse_transform([j]))[0], row[j])
                for j in range(len(row))))

        return predictions, proba

    def _get_model_class(self):
        """Returns the class of the actual underlying model"""
        raise NotImplementedError

    def _get_feature_selector(self):
        """Get a feature selector instance based on the feature_selector model
        parameter.

        Returns:
            (Object): a feature selector which returns a reduced feature matrix,
                given the full feature matrix (X) and the class labels (y)
        """
        raise NotImplementedError

    def _get_feature_scaler(self):
        """Get a feature value scaler based on the model settings"""
        if self.config.model_settings is None:
            scale_type = None
        else:
            scale_type = self.config.model_settings.get('feature_scaler')
        scaler = {'std-dev': StandardScaler(with_mean=False),
                  'max-abs': MaxAbsScaler()}.get(scale_type)
        return scaler

    def get_feature_matrix(self, examples, fit=False, y=None):
        """Transforms a list of examples into a feature matrix.

        Args:
            examples (list): The examples.
            fit (bool): Whether to (re)fit vectorizer with examples

        Returns:
            (numpy.matrix): The feature matrix.
            (numpy.array): The group labels for examples.
        """
        feats = [self._extract_features(e) for e in examples]
        if fit:
            X = self._feat_vectorizer.fit_transform(feats)
            if self._feat_scaler is not None:
                X = self._feat_scaler.fit_transform(X)
            if self._feat_selector is not None:
                X = self._feat_selector.fit_transform(X, y)
        else:
            X = self._feat_vectorizer.transform(feats)
            if self._feat_scaler is not None:
                X = self._feat_scaler.transform(X)
            if self._feat_selector is not None:
                X = self._feat_selector.transform(X)

        return X, None


class LabelEncoder(object):
    """The label encoder is responsible for converting between rich label
    objects such as a ProcessedQuery and basic formats a model can interpret.

    A workbench model use its label encoder at fit time to encode labels into a
    form it can deal with, and at predict time to decode predictions into
    objects
    """
    def encode(self, labels):
        """Transforms a list of label objects into a vector of classes.


        Args:
            labels (list): A list of labels to encoder
        """
        return labels

    def decode(self, classes):
        """Decodes a vector of classes into a list of labels

        Args:
            classes (list): A list of classes

        Returns:
            list: The decoded labels
        """
        return classes
