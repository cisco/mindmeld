# -*- coding: utf-8 -*-
"""
This module contains the intent classifier component.
"""
from __future__ import unicode_literals

import logging

from .classifier import MultinomialClassifier
from ..learners.text_classifier import TextClassifier

logger = logging.getLogger(__name__)


class IntentClassifier(MultinomialClassifier):
    """An intent classifier is used to determine the target intent for a given query. It is trained
    using all of the labeled queries across all intents for a domain in an application. The
    labels for the training data are the intent names associated with each query. The classifier
    takes in a query whose normalized text is sent to a text classifier.

    Attributes:
        domain (str): The domain of this intent classifier

    """
    DEFAULT_CONFIG = {
        'default_model': 'main',
        'models': {
            "main": {
                "model_type": "logreg",
                "params_grid": {
                    "fit_intercept": [True, False],
                    "C": [0.01, 1, 100, 10000, 1000000],
                    "class_bias": [1, 0.7, 0.3, 0]
                },
                "cv": {
                    "type": "k-fold",
                    "k": 10
                },
                "features": {
                    "bag-of-words": {
                        "lengths": [1]
                    },
                    "in-gaz": {},
                    "freq": {"bins": 5},
                    "length": {}
                }
            },
            "rforest": {
                "model_type": "rforest",
                "params_grid": {
                    "n_estimators": [10],
                    "max_features": ["auto"],
                    "n_jobs": [-1]
                },
                "cv": {
                    "type": "k-fold",
                    "k": 10
                },
                "features": {
                    "bag-of-words": {
                        "lengths": [1, 2, 3]
                    },
                    "edge-ngrams": {"lengths": [1, 2, 3]},
                    "in-gaz": {},
                    "freq": {"bins": 5},
                    "length": {}
                }
            }
        }
    }
    MODEL_CLASS = TextClassifier

    def __init__(self, resource_loader, domain):
        """Initializes an intent classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain of the intent classifier

        """
        super().__init__(resource_loader)
        self.domain = domain

    def _get_queries_and_classes(self, queries=None):
        if not queries:
            query_tree = self._resource_loader.get_labeled_queries()
            queries = self._resource_loader.flatten_query_tree(query_tree)
        return list(zip(*[(q.query, q.intent) for q in queries]))
