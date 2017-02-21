# -*- coding: utf-8 -*-
"""
This module contains the named entity linker component.
"""

from __future__ import unicode_literals
from builtins import object


class EntityLinker(object):
    """A named entity linker which is used to link entities to their specific values in a given
    query.

    """

    def __init__(self, resource_loader, domain):
        self._resource_loader = resource_loader
        self.domain = domain
        pass

    def fit(self):
        """Trains the model"""
        # self._model = something
        pass

    def predict(self, query, entities):
        """Predicts linked values for the entities provided

        Args:
            query (Query): The input query
            entities (list): The entities in the query

        Returns:
            list: a list containing the corresponding values for the entities passed in
        """
        pass

    def predict_proba(self, query, entities):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query

        Returns:
            list: a list of tuples of the form (str, float) grouping roles and their probabilities
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
        # joblib.dump(self._model, model_path)
        pass

    def load(self, model_path):
        """Loads the model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        # self._model = joblib.load(model_path)
        pass
