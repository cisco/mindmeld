# -*- coding: utf-8 -*-
"""
This module contains the domain classifier component of the Workbench natural language processor.
"""
import logging

from ..markup import mark_down
from ..models import QUERY_EXAMPLE_TYPE, CLASS_LABEL_TYPE

from .classifier import Classifier
from ..constants import DEFAULT_TRAIN_SET_REGEX

from ._config import get_classifier_config


logger = logging.getLogger(__name__)


class DomainClassifier(Classifier):
    """A domain classifier is used to determine the target domain for a given query. It is trained
    using all of the labeled queries across all intents for all domains in an application. The
    labels for the training data are the domain names associated with each query.
    """

    CLF_TYPE = 'domain'

    def _get_model_config(self, **kwargs):
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs['example_type'] = QUERY_EXAMPLE_TYPE
        kwargs['label_type'] = CLASS_LABEL_TYPE
        loaded_config = get_classifier_config(self.CLF_TYPE, self._resource_loader.app_path)
        return super()._get_model_config(loaded_config, **kwargs)

    def fit(self, *args, **kwargs):
        """Trains the domain classification model using the provided training queries

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
        logger.info('Fitting domain classifier')
        super().fit(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """Persists the trained domain classification model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored
        """
        logger.info('Saving domain classifier')
        super().dump(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Loads the trained domain classification model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        logger.info('Loading domain classifier')
        super().load(*args, **kwargs)

    def inspect(self, query, domain=None, dynamic_resource=None):
        return self._model.inspect(
            example=query, gold_label=domain, dynamic_resource=dynamic_resource)

    def _get_query_tree(self, queries=None, label_set=DEFAULT_TRAIN_SET_REGEX, raw=False):
        """Returns the set of queries to train on

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
            raw (bool, optional): When True, raw query strings will be returned

        Returns:
            List: list of queries
        """
        if queries:
            return self._build_query_tree(queries, raw=raw)

        return self._resource_loader.get_labeled_queries(label_set=label_set, raw=raw)

    def _get_queries_and_labels(self, queries=None, label_set=DEFAULT_TRAIN_SET_REGEX):
        """Returns a set of queries and their labels based on the label set

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
        """
        query_tree = self._get_query_tree(queries, label_set=label_set)
        queries = self._resource_loader.flatten_query_tree(query_tree)

        if len(queries) < 1:
            return [None, None]
        return list(zip(*[(q.query, q.domain) for q in queries]))

    def _get_queries_and_labels_hash(self, queries=None, label_set=DEFAULT_TRAIN_SET_REGEX):
        query_tree = self._get_query_tree(queries, label_set=label_set, raw=True)
        queries = []
        for domain in query_tree:
            for intent in query_tree[domain]:
                for query_text in query_tree[domain][intent]:
                    queries.append(domain + "###" + mark_down(query_text))

        queries.sort()
        return self._resource_loader.hash_list(queries)
