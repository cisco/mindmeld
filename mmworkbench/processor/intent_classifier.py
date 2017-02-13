# -*- coding: utf-8 -*-
"""
This module contains the intent classifier component.
"""

import os

# from sklearn.externals import joblib


class IntentClassifier(object):
    """An intent classifier is used to determine the target intent for a given query. It is trained
    using all of the labeled queries across all intents for a domain in an application. The
    labels for the training data are the intent names associated with each query. The classifier
    takes in a query whose normalized text is sent to a text classifier.

    Attributes:
        domain (str): The domain of this intent classifier

    """
    def __init__(self, resource_loader, domain):
        """Initializes an intent classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain of the intent classifier

        """
        self._resource_loader = resource_loader
        self.domain = domain
        self._model = None  # will be set when model is fit or loaded

    def fit(self, model_type=None, features=None, params_grid=None, cv=None):
        """Trains the model

        Args:
            model_type (str): The type of model to use
            features (None, optional): Description
            params_grid (None, optional): Description
            cv (None, optional): Description

        """
        query_tree = self._resource_loader.get_labeled_queries(domain=self.domain)
        # self._model = something
        pass

    def predict(self, query):
        """Predicts a role for the specified query

        Args:
            query (Query): The input query

        Returns:
            str: the predicted intent
        """
        pass

    def predict_proba(self, query):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query

        Returns:
            list: a list of tuples of the form (str, float) grouping intents and their probabilities
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
