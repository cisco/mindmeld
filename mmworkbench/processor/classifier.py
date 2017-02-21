# coding=utf-8
"""
This module contains the domain classifier component.
"""

from __future__ import unicode_literals

from builtins import object
import logging
import os

from sklearn.externals import joblib

logger = logging.getLogger(__name__)


class Classifier(object):
    DEFAULT_CONFIG = None
    MODEL_CLASS = None

    def __init__(self, resource_loader):
        """Initializes a classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
        """
        self._resource_loader = resource_loader
        self._model = None  # will be set when model is fit or loaded

    def get_fit_config(self, model_type=None, features=None, params_grid=None, cv=None,
                       model_name=None):
        model_name = model_name or self.DEFAULT_CONFIG['default_model']
        model_config = self.DEFAULT_CONFIG['models'][model_name]
        model_type = model_type or model_config['model_type']
        features = features or model_config['features']
        params_grid = params_grid or model_config['params_grid']
        cv = cv or model_config['cv']
        return {'classifier_type': model_type, 'features': features, 'params_grid': params_grid,
                'cv': cv}

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

    def load(self, model_path):
        """Loads the model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        self._model = joblib.load(model_path)


class MultinomialClassifier(Classifier):
    DEFAULT_CONFIG = None
    MODEL_CLASS = None

    def fit(self, model_type=None, features=None, params_grid=None, cv=None, model_name=None,
            queries=None):
        """Trains the model

        Args:
            model_type (str): The type of model to use. If omitted, the default model type will
                be used.
            features (dict): If omitted, the default features for the model type will be used.
            params_grid (dict): If omitted the default params will be used
            cv (None, optional): Description

        """
        queries, classes = self._get_queries_and_classes(queries)
        gazetteers = self._get_gazetteers()

        params = self.get_fit_config(model_type, features, params_grid, cv)

        model = self.MODEL_CLASS(**params)
        model.register_resources(gazetteers)
        model.fit(queries, classes)
        self._model = model

    def predict(self, query):
        """Predicts a domain for the specified query

        Args:
            query (Query): The input query

        Returns:
            str: the predicted domain
        """
        return self._model.predict([query])[0]

    def predict_proba(self, query):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query

        Returns:
            list: a list of tuples of the form (str, float) grouping predictions and their
                probabilities
        """
        return self._model.predict_proba([query])[0]

    def evaluate(self, use_blind=False):
        """Evaluates the model on the specified data

        Returns:
            TYPE: Description
        """
        pass

    def _get_queries_and_classes(self, queries=None):
        """Returns the set of queries and their classes to train on

        Args:
            queries (list): A list of ProcessedQuery objects to train. If not passed, the default
                training set will be loaded.

        """
        raise NotImplementedError

    def _get_gazetteers(self):
        """Returns the gazetteers needed by this classifier

        Returns:
            TYPE: Description

        """
        raise NotImplementedError
