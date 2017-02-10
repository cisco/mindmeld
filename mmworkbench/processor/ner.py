# -*- coding: utf-8 -*-
"""
This module contains the named entity recognizer component.
"""
# from sklearn.externals import joblib


class NamedEntityRecognizer(object):
    """A named entity recognizer which is used to identify the entities for a given query. It is
    trained using all the labeled queries for a particular intent. The labels are the entity
    annotations for each query.

    Attributes:
        domain (str): The domain of this named entity recognizer
        intent (str): The intent of this named entity recognizer
    """
    def __init__(self, resource_loader, domain, intent):
        """Initializes a named entity recognizer

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain of this named entity recognizer
            intent (str): The intent of this named entity recognizer
        """
        self._resource_loader = resource_loader
        self.domain = domain
        self.intent = intent
        self._model = None  # will be set when model is fit or loaded

    def fit(self, model_type=None, features=None, params_grid=None, cv=None):
        """Trains the model

        Args:
            model_type (str): The type of model to use
            features (None, optional): Description
            params_grid (None, optional): Description
            cv (None, optional): Description

        """
        # self._model = something
        pass

    def predict(self, query):
        """Predicts a role for the specified query

        Args:
            query (Query): The input query

        Returns:
            list: the predicted entities
        """
        pass

    def predict_proba(self, query):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query

        Returns:
            list: a list of tuples of the form (Entity, float) grouping potential entities and their
            probabilities
        """
        # Note(jj): Not sure we can support this with all classifiers (MEMM?)
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
