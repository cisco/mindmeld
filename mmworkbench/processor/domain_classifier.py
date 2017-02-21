# -*- coding: utf-8 -*-
"""
This module contains the domain classifier component.
"""
from __future__ import unicode_literals

import logging

from .classifier import MultinomialClassifier
from ..learners.text_classifier import TextClassifier

logger = logging.getLogger(__name__)


class DomainClassifier(MultinomialClassifier):
    """A domain classifier is used to determine the target domain for a given query. It is trained
    using all of the labeled queries across all intents for all domains in an application. The
    labels for the training data are the domain names associated with each query. The classifier
    takes in a query whose normalized text is sent to a text classifier.
    """

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
    MODEL_CLASS = TextClassifier

    def _get_queries_and_classes(self, queries=None):
        if not queries:
            query_tree = self._resource_loader.get_labeled_queries()
            queries = self._resource_loader.flatten_query_tree(query_tree)
        return list(zip(*[(q.query, q.domain) for q in queries]))
