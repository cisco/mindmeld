# -*- coding: utf-8 -*-
"""
This module contains the domain classifier component.
"""
from __future__ import unicode_literals
from builtins import super

import logging

from .classifier import Classifier

logger = logging.getLogger(__name__)


class DomainClassifier(Classifier):
    """A domain classifier is used to determine the target domain for a given query. It is trained
    using all of the labeled queries across all intents for all domains in an application. The
    labels for the training data are the domain names associated with each query. The classifier
    takes in a query whose normalized text is sent to a text classifier.
    """

    DEFAULT_CONFIG = {
        'default_model': 'main',
        'models': {
            'main': {
                'model_type': 'text',
                'example_type': 'query',
                'model_settings': {
                    'classifier_type': 'logreg',
                },
                'param_selection': {
                    'type': 'k-fold',
                    'k': 10,
                    'grid': {
                        'fit_intercept': [True, False],
                        'C': [10, 100, 1000, 10000, 100000]
                    },
                },
                'features': {
                    'bag-of-words': {
                        'lengths': [1]
                    },
                    'freq': {'bins': 5},
                    'in-gaz': {}
                }
            }
        }
    }

    def fit(self, *args, **kwargs):
        logger.info('Fitting domain classifier')
        super().fit(*args, **kwargs)

    def _get_examples_and_labels(self, examples=None, label_set=None):
        if not examples:
            query_tree = self._resource_loader.get_labeled_queries(label_set=label_set)
            queries = self._resource_loader.flatten_query_tree(query_tree)
        else:
            queries = examples
        return list(zip(*[(q.query, q.domain) for q in queries]))
