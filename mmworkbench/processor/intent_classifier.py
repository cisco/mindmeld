"""
This module contains the intent classifier component.
"""
from __future__ import unicode_literals
from builtins import super

import logging

from ..models import QUERY_EXAMPLE_TYPE, CLASS_LABEL_TYPE

from .classifier import Classifier

logger = logging.getLogger(__name__)


class IntentClassifier(Classifier):
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
            'main': {
                'model_type': 'text',
                'model_settings': {
                    'classifier_type': 'logreg'
                },
                'param_selection': {
                    'type': 'k-fold',
                    'k': 10,
                    'grid': {
                        'fit_intercept': [True, False],
                        'C': [0.01, 1, 100, 10000, 1000000],
                        'class_bias': [1, 0.7, 0.3, 0]
                    }
                },
                'features': {
                    'bag-of-words': {
                        'lengths': [1]
                    },
                    'in-gaz': {},
                    'freq': {'bins': 5},
                    'length': {}
                }
            },
            'rforest': {
                'model_type': 'text',
                'model_settings': {
                    'classifier_type': 'rforest'
                },
                'param_selection': {
                    'type': 'k-fold',
                    'k': 10,
                    'grid': {
                        'n_estimators': [10],
                        'max_features': ['auto'],
                        'n_jobs': [-1]
                    },
                },
                'features': {
                    'bag-of-words': {
                        'lengths': [1, 2, 3]
                    },
                    'edge-ngrams': {'lengths': [1, 2, 3]},
                    'in-gaz': {},
                    'freq': {'bins': 5},
                    'length': {}
                }
            }
        }
    }

    def __init__(self, resource_loader, domain):
        """Initializes an intent classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain of the intent classifier

        """
        super().__init__(resource_loader)
        self.domain = domain

    def get_model_config(self, config_name, **kwargs):
        kwargs['example_type'] = QUERY_EXAMPLE_TYPE
        kwargs['label_type'] = CLASS_LABEL_TYPE
        return super().get_model_config(config_name, **kwargs)

    def fit(self, *args, **kwargs):
        logger.info('Fitting intent classifier: domain=%r', self.domain)
        super().fit(*args, **kwargs)

    def dump(self, *args, **kwargs):
        logger.info('Saving intent classifier: domain=%r', self.domain)
        super().dump(*args, **kwargs)

    def load(self, *args, **kwargs):
        logger.info('Loading intent classifier: domain=%r', self.domain)
        super().load(*args, **kwargs)

    def _get_queries_and_labels(self, queries=None, label_set=None):
        """Returns the set of queries and their labels to train on

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
        """
        if not queries:
            query_tree = self._resource_loader.get_labeled_queries()
            queries = self._resource_loader.flatten_query_tree(query_tree)
        return list(zip(*[(q.query, q.intent) for q in queries]))
