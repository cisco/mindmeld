# -*- coding: utf-8 -*-
"""
This module contains the named entity recognizer component.
"""
from __future__ import unicode_literals

import logging

from .classifier import Classifier
from ..models import MemmModel
from ..core import Query

# from sklearn.externals import joblib

logger = logging.getLogger(__name__)


class EntityRecognizer(Classifier):
    """A named entity recognizer which is used to identify the entities for a given query. It is
    trained using all the labeled queries for a particular intent. The labels are the entity
    annotations for each query.

    Attributes:
        domain (str): The domain of this named entity recognizer
        intent (str): The intent of this named entity recognizer
        entity_types (set): A set containing the entities which can be recognized
    """

    DEFAULT_CONFIG = {
        "default_model": "main",
        "models": {
            "main": {
                "model_type": "memm",
                "params_grid": {
                    "penalty": ["l2"],
                    "C": [100]
                },
                "model_settings": {
                    "tag-scheme": "IOB",
                    "feature-scaler": "none"
                }
            },
            "sparse": {
                "model_type": "memm",
                "params_grid": {
                    "penalty": ["l2"],
                    "C": [100]
                },
                "model_settings": {
                    "tag-scheme": "IOB",
                    "feature-scaler": "max-abs",
                    "feature-selector": "l1"
                }
            },
            "memm-cv": {
                "model_type": "memm",
                "params_grid": {
                    "penalty": ["l1", "l2"],
                    "C": [0.01, 1, 100, 10000, 1000000, 100000000]
                },
                "cv": {
                    "type": "k-fold",
                    "k": 5,
                    "metric": "accuracy"
                },
                "model_settings": {
                    "tag-scheme": "IOB",
                    "feature-scaler": "max-abs"
                }
            },
            "ngram": {
                "model_type": "ngram",
                "params_grid": {
                    "C": [100]
                }
            }
        }
    }
    MODEL_CLASS = MemmModel

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

    def get_fit_config(self, model_type=None, features=None, params_grid=None, cv=None,
                       model_settings=None, model_name=None):
        model_name = model_name or self.DEFAULT_CONFIG['default_model']
        model_config = self.DEFAULT_CONFIG['models'][model_name]
        model_settings = model_settings or model_config['model_settings']
        features = features or model_config.get('features')
        params_grid = params_grid or model_config['params_grid']
        cv = cv or model_config.get('cv')
        return {'model_settings': model_settings, 'features': features, 'params_grid': params_grid,
                'cv': cv}

    def fit(self, model_type=None, features=None, params_grid=None, cv=None, model_settings=None):
        """Trains the model

        Args:
            model_type (str): The type of model to use
            features (None, optional): Description
            params_grid (None, optional): Description
            cv (None, optional): Description

        """
        query_tree = self._resource_loader.get_labeled_queries(domain=self.domain,
                                                               intent=self.intent)
        queries = query_tree[self.domain][self.intent]

        params = self.get_fit_config(model_type, features, params_grid, cv, model_settings)
        model_class = self._get_model_class(model_type)
        model = model_class(**params)
        gazetteers = self._resource_loader.get_gazetteers()
        model.register_resources(gazetteers=gazetteers)

        model.fit(queries)
        self._model = model

    def predict(self, query):
        """Predicts a role for the specified query

        Args:
            query (Query): The input query

        Returns:
            list: the predicted entities
        """
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(query)
        return self._model.predict(query)

    def predict_proba(self, query):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query
`
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

    def load(self, model_path):
        super().load(model_path)
        if self._model:
            gazetteers = self._resource_loader.get_gazetteers()
            self._model.register_resources(gazetteers=gazetteers)

    @property
    def entity_types(self):
        return self._model.entity_types if self._model else set()
