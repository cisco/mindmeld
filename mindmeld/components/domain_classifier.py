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
This module contains the domain classifier component of the MindMeld natural language processor.
"""
import logging

from ..constants import DEFAULT_TRAIN_SET_REGEX
from ..markup import mark_down
from ..models import CLASS_LABEL_TYPE, QUERY_EXAMPLE_TYPE
from ._config import get_classifier_config
from .classifier import Classifier

logger = logging.getLogger(__name__)


class DomainClassifier(Classifier):
    """A domain classifier is used to determine the target domain for a given query. It is trained
    using all of the labeled queries across all intents for all domains in an application. The
    labels for the training data are the domain names associated with each query.
    """

    CLF_TYPE = "domain"
    """The classifier type."""

    def _get_model_config(self, **kwargs):
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs["example_type"] = QUERY_EXAMPLE_TYPE
        kwargs["label_type"] = CLASS_LABEL_TYPE
        loaded_config = get_classifier_config(
            self.CLF_TYPE, self._resource_loader.app_path
        )
        return super()._get_model_config(loaded_config, **kwargs)

    def fit(self, *args, **kwargs):  # pylint: disable=signature-differs
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
        logger.info("Fitting domain classifier")
        return super().fit(*args, **kwargs)

    def dump(self, *args, **kwargs):  # pylint: disable=signature-differs
        """Persists the trained domain classification model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored
        """
        logger.info("Saving domain classifier")
        super().dump(*args, **kwargs)

    def unload(self):
        logger.info("Unloading domain classifier")
        super().unload()

    def load(self, *args, **kwargs):
        """Loads the trained domain classification model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        logger.info("Loading domain classifier")
        super().load(*args, **kwargs)

    def inspect(self, query, domain=None, dynamic_resource=None):
        """Inspects the query.

        Args:
            query (Query): The query to be predicted.
            domain (str): The expected domain label for this query.
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference.

        Returns:
            (list of lists): 2D list that includes every feature, their value, weight and \
                probability.
        """
        return self._model.inspect(
            example=query, gold_label=domain, dynamic_resource=dynamic_resource
        )

    def _get_query_tree(
        self, label_set=DEFAULT_TRAIN_SET_REGEX
    ):
        """Returns the set of queries to train on

        Args:
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.

        Returns:
            ProcessedQueryList: list of queries
        """
        return self._resource_loader.get_labeled_queries(label_set=label_set)

    def _get_queries_and_labels(self, label_set=DEFAULT_TRAIN_SET_REGEX):
        """
        Returns a set of queries and their labels based on the label set.
        Because the query set is large for domain classification, it uses a
        lazy loading iterator, the labels are relatively small and iterated
        over multiple times so they are resolved to an in-memory list.

        Args:
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.

        Returns:
            tuple(ProcessedQueryList.QueryIterator,list[str])

        """
        query_tree = self._get_query_tree(label_set=label_set)
        queries = self._resource_loader.flatten_query_tree(query_tree)

        if not queries:
            return [None, None]
        return (queries.queries(),
                list(queries.domains()))

    def _get_queries_and_labels_hash(
        self, label_set=DEFAULT_TRAIN_SET_REGEX
    ):
        query_tree = self._get_query_tree(label_set=label_set)
        queries = []
        for domain in query_tree:
            for intent in query_tree[domain]:
                for query_text in query_tree[domain][intent].raw_queries():
                    queries.append(domain + "###" + mark_down(query_text))

        queries.sort()
        return self._resource_loader.hash_list(queries)
