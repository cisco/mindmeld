# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the intent classifier component of the MindMeld natural language processor.
"""
import logging

from ..constants import DEFAULT_TRAIN_SET_REGEX
from ..markup import mark_down
from ..models import CLASS_LABEL_TYPE, QUERY_EXAMPLE_TYPE
from ._config import get_classifier_config
from .classifier import Classifier

logger = logging.getLogger(__name__)


class IntentClassifier(Classifier):
    """An intent classifier is used to determine the target intent for a given query. It is trained
    using all of the labeled queries across all intents for a domain in an application. The
    labels for the training data are the intent names associated with each query.

    Attributes:
        domain (str): The domain that this intent classifier belongs to.
    """

    CLF_TYPE = "intent"
    """The classifier type."""

    def __init__(self, resource_loader, domain):
        """Initializes an intent classifier.

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the \
                classifier.
            domain (str): The domain that this intent classifier belongs to.
        """
        super().__init__(resource_loader)
        self.domain = domain

    def _get_model_config(self, **kwargs):
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs["example_type"] = QUERY_EXAMPLE_TYPE
        kwargs["label_type"] = CLASS_LABEL_TYPE
        loaded_config = get_classifier_config(
            self.CLF_TYPE, self._resource_loader.app_path, domain=self.domain
        )
        return super()._get_model_config(loaded_config, **kwargs)

    def fit(self, *args, **kwargs):  # pylint: disable=signature-differs
        """Trains the intent classification model using the provided training queries.

        Args:
            model_type (str): The type of machine learning model to use. If omitted, the default
                model type will be used.
            features (dict): Features to extract from each example instance to form the feature
                vector used for model training. If omitted, the default feature set for the model
                type will be used.
            params_grid (dict): The grid of hyper-parameters to search, for finding the optimal
                hyper-parameter settings for the model. If omitted, the default hyper-parameter
                search grid will be used.
            queries (list[ProcessedQuery]): The labeled queries to use as training data.
            cv (optional): Cross-validation settings.
        """
        logger.info("Fitting intent classifier: domain=%r", self.domain)
        super().fit(*args, **kwargs)

    def dump(self, *args, **kwargs):  # pylint: disable=signature-differs
        """Persists the trained intent classification model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored.
        """
        logger.info("Saving intent classifier: domain=%r", self.domain)
        super().dump(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Loads the trained intent classification model from disk.

        Args:
            model_path (str): The location on disk where the model is stored.
        """
        logger.info("Loading intent classifier: domain=%r", self.domain)
        super().load(*args, **kwargs)

    def inspect(self, query, intent=None, dynamic_resource=None):
        """Inspects the query.

        Args:
            query (Query): The query to be predicted.
            intent (str): The expected intent label for this query.
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference.

        Returns:
            (list of lists): 2D list that includes every feature, their value, weight and \
                probability.
        """
        return self._model.inspect(
            example=query, gold_label=intent, dynamic_resource=dynamic_resource
        )

    def _get_query_tree(
        self, queries=None, label_set=DEFAULT_TRAIN_SET_REGEX, raw=False
    ):
        """Returns the set of queries to train on

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
            raw (bool, optional): When True, raw query strings will be returned

        Returns:
            (list): list of queries
        """
        if queries:
            return self._build_query_tree(queries, domain=self.domain, raw=raw)

        return self._resource_loader.get_labeled_queries(
            domain=self.domain, label_set=label_set, raw=raw
        )

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
        return list(zip(*[(q.query, q.intent) for q in queries]))

    def _get_queries_and_labels_hash(
        self, queries=None, label_set=DEFAULT_TRAIN_SET_REGEX
    ):
        query_tree = self._get_query_tree(queries, label_set=label_set, raw=True)
        queries = []

        for intent in query_tree.get(self.domain, []):
            for query_text in query_tree[self.domain][intent]:
                queries.append(
                    self.domain + "###" + intent + "###" + mark_down(query_text)
                )

        queries.sort()
        return self._resource_loader.hash_list(queries)
