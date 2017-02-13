# -*- coding: utf-8 -*-
"""
This module contains the domain classifier component.
"""

import os

# from sklearn.externals import joblib


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
        # self._model = something
        pass

    def predict(self, query):
        """Predicts a role for the specified query

        Args:
            query (Query): The input query

        Returns:
            str: the predicted domain
        """
        pass

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
