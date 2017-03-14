# coding=utf-8
"""
This module contains the domain classifier component.
"""

from __future__ import unicode_literals
from builtins import object, zip

import copy
import logging
import os

from sklearn.externals import joblib

from ..exceptions import ClassifierLoadError, FileNotFoundError
from ..core import Query

from ..models import create_model, ModelConfig

logger = logging.getLogger(__name__)


class ClassifierConfig(object):
    """A value object representing a classifier configuration

    Attributes:
        model_type (str): The name of the model type. Will be used to find the
            model class to instantiate
        model_settings (dict): Settings specific to the model type specified
        params (dict): Params to pass to the underlying classifier
        param_selection (dict): Configuration for param selection (using cross
            validation)
            {'type': 'shuffle',
             'n': 3,
             'k': 10,
             'n_jobs': 2,
             'scoring': '',
             'grid': {}
            }
        features (dict): The keys are the names of feature extractors and the
            values are either a kwargs dict which will be passed into the
            feature extractor function, or a callable which will be used as to
            extract features

    """
    __slots__ = ['model_type', 'features', 'model_settings', 'params', 'param_selection']

    def __init__(self, model_type=None, features=None, model_settings=None, params=None,
                 param_selection=None):
        for arg, val in {'model_type': model_type, 'features': features}.items():
            if val is None:
                raise TypeError('__init__() missing required argument {!r}'.format(arg))
        if params is None and (param_selection is None or param_selection.get('grid') is None):
            raise ValueError("__init__() One of 'params' and 'param_selection' is required")
        self.model_type = model_type
        self.features = features
        self.model_settings = model_settings
        self.params = params
        self.param_selection = param_selection

    def to_dict(self):
        """Converts the model config object into a dict

        Returns:
            dict: A dict version of the config
        """
        result = {}
        for attr in self.__slots__:
            result[attr] = getattr(self, attr)
        return result

    def __repr__(self):
        args_str = ', '.join("{}={!r}".format(key, getattr(self, key)) for key in self.__slots__)
        return "{}({})".format(self.__class__.__name__, args_str)

    @classmethod
    def from_model_config(cls, model_config):
        config = model_config.to_dict()
        config.pop('example_type')
        config.pop('label_type')
        return cls(**config)


class Classifier(object):
    DEFAULT_CONFIG = None

    def __init__(self, resource_loader):
        """Initializes a classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
        """
        self._resource_loader = resource_loader
        self._model = None  # will be set when model is fit or loaded
        self.ready = False
        self.dirty = False
        self.config = None

    def fit(self, queries=None, config_name=None, label_set='train', **kwargs):
        """Trains the model

        Args:
            model_type (str): The type of model to use. If omitted, the default model type will
                be used.
            features (dict): If omitted, the default features for the model type will be used.
            params_grid (dict): If omitted the default params will be used
            cv (None, optional): Description
            queries (list of ProcessedQuery): The labeled queries to use as training data

        """
        """Trains the model

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as training data
            config_name (str): The type of model to use. If omitted, the default model type will
                be used.

        """
        queries, classes = self._get_queries_and_labels(queries, label_set)
        model_config = self._get_model_config(config_name, **kwargs)
        model = create_model(model_config)
        gazetteers = self._resource_loader.get_gazetteers()
        model.register_resources(gazetteers=gazetteers)
        model.fit(queries, classes)
        self._model = model
        self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = True

    def predict(self, query):
        """Predicts a domain for the specified query

        Args:
            query (Query or str): The input query

        Returns:
            str: the predicted domain
        """
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(query)
        gazetteers = self._resource_loader.get_gazetteers()
        self._model.register_resources(gazetteers=gazetteers)
        return self._model.predict([query])[0]

    def predict_proba(self, query):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query

        Returns:
            list: a list of tuples of the form (str, float) grouping predictions and their
                probabilities
        """
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(query)
        gazetteers = self._resource_loader.get_gazetteers()
        self._model.register_resources(gazetteers=gazetteers)
        return list(zip(*self._model.predict_proba([query])))[0]

    def evaluate(self, use_blind=False):
        """Evaluates the model on the specified data

        Returns:
            TYPE: Description
        """
        raise NotImplementedError('Subclasses must implement this method')

    def _get_model_config(self, config_name, **kwargs):
        config_name = config_name or self.DEFAULT_CONFIG['default_model']
        model_config = copy.copy(self.DEFAULT_CONFIG['models'][config_name])
        model_config.update(kwargs)
        return ModelConfig(**model_config)

    def dump(self, model_path):
        """Persists the model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored

        """
        # make directory if necessary
        folder = os.path.dirname(model_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        joblib.dump(self._model, model_path)

        self.dirty = False

    def load(self, model_path):
        """Loads the model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        try:
            self._model = joblib.load(model_path)
        except FileNotFoundError:
            msg = 'Unable to load {}. Pickle file not found at {!r}'
            raise ClassifierLoadError(msg.format(self.__class__.__name__, model_path))
        if self._model is not None:
            gazetteers = self._resource_loader.get_gazetteers()
            self._model.register_resources(gazetteers=gazetteers)
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = False

    def _get_queries_and_labels(self, queries=None, label_set='train'):
        """Returns the set of queries and their labels to train on

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
        """
        raise NotImplementedError('Subclasses must implement this method')

    def __repr__(self):
        msg = '<{} ready: {!r}, dirty: {!r}>'
        return msg.format(self.__class__.__name__, self.ready, self.dirty)
