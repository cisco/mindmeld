# -*- coding: utf-8 -*-
"""
This module contains the domain classifier component.
"""

import logging
import os

from ..classifiers.text_classifier import TextClassifier

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'default_model': 'main',
    'models': {
        "main": {
            "model_type": "logreg",
            "params_grid": {
                "fit_intercept": [True, False],
                "C": [10, 100, 1000, 10000, 100000]
            },
            "cv": {
                "type": "k-fold",
                "k": 10
            },
            "features": {
                "bag-of-words": {
                    "lengths": [1]
                },
                "freq": {"bins": 5},
                "in-gaz": {}
            }
        }
    }
}


class DomainClassifier(object):
    """A domain classifier is used to determine the target domain for a given query. It is trained
    using all of the labeled queries across all intents for all domains in an application. The
    labels for the training data are the domain names associated with each query. The classifier
    takes in a query whose normalized text is sent to a text classifier.
    """
    def __init__(self, resource_loader):
        """Initializes a domain classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
        """
        self._resource_loader = resource_loader
        self._model = None  # will be set when model is fit or loaded

    def fit(self, model_type=None, features=None, params_grid=None, cv=None):
        """Trains the model

        Args:
            model_type (str): The type of model to use. If omitted, the default model type will
                be used.
            features (dict): If omitted, the default features for the model type will be used.
            params_grid (dict): If omitted the default params will be used
            cv (None, optional): Description

        """

        query_tree = self._resource_loader.get_labeled_queries()
        domains, _, queries = self._resource_loader.flatten_query_tree(query_tree)
        queries = [q.query for q in queries]
        # gazetteers = self._resource_loader.get_gazetteers()

        logger.info('Training domain classifier')

        default_config = DEFAULT_CONFIG['models'][DEFAULT_CONFIG['default_model']]
        model_type = model_type or default_config['model_type']
        features = features or default_config['features']
        params_grid = params_grid or default_config['params_grid']
        cv = cv or default_config['cv']

        domain_model = TextClassifier(model_type, features, params_grid, cv)

        # domain_model.register_resources(gazetteers)
        domain_model.fit(queries, domains)
        self._model = domain_model

    def predict(self, query):
        """Predicts a role for the specified query

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
            list: a list of tuples of the form (str, float) grouping domains and their probabilities
        """
        pass

    def evaluate(self, use_blind=False):
        """Evaluates the model on the specified data

        Returns:
            TYPE: Description
        """
        pass

    def dump(self, model_path):
        """Persists the model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored

        """
        # make directory if necessary
        folder, filename = os.split(model_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # joblib.dump(self._model, model_path)
        pass

    def load(self, model_path):
        """Loads the model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        # self._model = joblib.load(model_path)
        pass
