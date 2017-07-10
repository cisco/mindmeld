"""
This module contains the intent classifier component of the Workbench natural language processor.
"""
from __future__ import absolute_import, unicode_literals
from builtins import super

import logging

from ..models import QUERY_EXAMPLE_TYPE, CLASS_LABEL_TYPE

from .classifier import Classifier
from ._config import get_classifier_config

logger = logging.getLogger(__name__)


class IntentClassifier(Classifier):
    """An intent classifier is used to determine the target intent for a given query. It is trained
    using all of the labeled queries across all intents for a domain in an application. The
    labels for the training data are the intent names associated with each query.

    Attributes:
        domain (str): The domain that this intent classifier belongs to

    """
    CLF_TYPE = 'intent'

    def __init__(self, resource_loader, domain):
        """Initializes an intent classifier

        Args:
        resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain that this intent classifier belongs to

        """
        super().__init__(resource_loader)
        self.domain = domain

    def _get_model_config(self, **kwargs):
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs['example_type'] = QUERY_EXAMPLE_TYPE
        kwargs['label_type'] = CLASS_LABEL_TYPE
        default_config = get_classifier_config(self.CLF_TYPE, self._resource_loader.app_path,
                                               domain=self.domain)
        return super()._get_model_config(default_config, **kwargs)

    def fit(self, *args, **kwargs):
        """Trains the intent classification model using the provided training queries

        Args:
            model_type (str): The type of machine learning model to use. If omitted, the default
                model type will be used.
            features (dict): Features to extract from each example instance to form the feature
                vector used for model training. If omitted, the default feature set for the model
                type will be used.
            params_grid (dict): The grid of hyper-parameters to search, for finding the optimal
                hyper-parameter settings for the model. If omitted, the default hyper-parameter
                search grid will be used.
            cv (None, optional): Cross-validation settings
            queries (list of ProcessedQuery): The labeled queries to use as training data

        """
        logger.info('Fitting intent classifier: domain=%r', self.domain)
        super().fit(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """Persists the trained intent classification model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored
        """
        logger.info('Saving intent classifier: domain=%r', self.domain)
        super().dump(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Loads the trained intent classification model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        logger.info('Loading intent classifier: domain=%r', self.domain)
        super().load(*args, **kwargs)

    def evaluate(self, queries=None):
        """Evaluates the trained intent classification model on the given test data

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as training data. If none
                are provided, the test label set will be used.

        Returns:
            ModelEvaluation: A ModelEvaluation object that contains evaluation results
        """
        if not self._model:
            logger.error('You must fit or load the model before running evaluate.')
            return
        gazetteers = self._resource_loader.get_gazetteers()
        self._model.register_resources(gazetteers=gazetteers)
        queries, classes = self._get_queries_and_labels(queries, label_set='test')
        evaluation = self._model.evaluate(queries, classes)
        return evaluation

    def _get_queries_and_labels(self, queries=None, label_set='train'):
        """Returns a set of queries and their labels based on the label set

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
        """
        if not queries:
            query_tree = self._resource_loader.get_labeled_queries(
                label_set=label_set, domain=self.domain)
            queries = self._resource_loader.flatten_query_tree(query_tree)
        return list(zip(*[(q.query, q.intent) for q in queries]))
