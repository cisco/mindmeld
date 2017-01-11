# -*- coding: utf-8 -*-
"""
This module contains the intent classifier component.
"""
# from sklearn.externals import joblib


class IntentClassifier(object):
    """An intent classifier is used to determine the target intent for a given query. It is trained
    using all of the labeled queries across all intents for a domain in an application. The
    labels for the training data are the intent names associated with each query. The classifier
    takes in a query whose normalized text is sent to a text classifier.

    Attributes:
        model_type (str): the name of the underlying model to use
        features (dict): the features which the classifier will use for predictions
        gazetteers (dict): the gazetteers used by the classifier

    """
    def __init__(self, model_type, features, gazetteers):
        """Initializes a intent classifier

        Args:
            model_type (str): the name of the underlying model to use
            features (dict): the features which the classifier will use for predictions
            gazetteers (dict): the gazetteers used by the classifier
        """
        self.model_type = model_type
        self.features = features
        self.gazetteers = gazetteers
        self._model = None  # will be set when model is fit or loaded

    def fit(self, data, params_grid=None, cv=None):
        """Trains the model

        Args:
            data (TYPE): Description
            params_grid (None, optional): Description
            cv (None, optional): Description

        Returns:
            TYPE: Description
        """
        # self._model = something
        pass

    def predict(self, query):
        """Predicts a intent for the specified query

        Args:
            query (mmworkbench.parser.Query): The input query

        Returns:
            str: the predicted intent
        """
        pass

    def predict_proba(self, query):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (mmworkbench.parser.Query): The input query

        Returns:
            list: a list of tuples of the form (str, float) grouping intents and their probabilities
        """
        pass

    def evaluate(self, data):
        """Evaluates the model on the specified data

        Args:
            data (list): A list of ParsedQuery objects

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
