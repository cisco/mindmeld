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
This module contains the entity recognizer component of the MindMeld natural language processor.
"""
import logging

from sklearn.externals import joblib

from ..constants import DEFAULT_TRAIN_SET_REGEX
from ..core import Entity, Query
from ..models import ENTITIES_LABEL_TYPE, QUERY_EXAMPLE_TYPE, create_model
from ._config import get_classifier_config
from .classifier import Classifier, ClassifierConfig, ClassifierLoadError

logger = logging.getLogger(__name__)


class EntityRecognizer(Classifier):
    """An entity recognizer which is used to identify the entities for a given query. It is
    trained using all the labeled queries for a particular intent. The labels are the entity
    annotations for each query.

    Attributes:
        domain (str): The domain that this entity recognizer belongs to
        intent (str): The intent that this entity recognizer belongs to
        entity_types (set): A set containing the entity types which can be recognized
    """

    CLF_TYPE = "entity"
    """The classifier type."""

    def __init__(self, resource_loader, domain, intent):
        """Initializes an entity recognizer

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain that this entity recognizer belongs to
            intent (str): The intent that this entity recognizer belongs to
        """
        super().__init__(resource_loader)
        self.domain = domain
        self.intent = intent
        self.entity_types = set()
        self._model_config = None

    def _get_model_config(self, **kwargs):  # pylint: disable=arguments-differ
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs["example_type"] = QUERY_EXAMPLE_TYPE
        kwargs["label_type"] = ENTITIES_LABEL_TYPE
        loaded_config = get_classifier_config(
            self.CLF_TYPE,
            self._resource_loader.app_path,
            domain=self.domain,
            intent=self.intent,
        )
        return super()._get_model_config(loaded_config, **kwargs)

    def fit(self, queries=None, label_set=None, incremental_timestamp=None, **kwargs):
        """Trains the entity recognition model using the provided training queries.

        Args:
            queries (list[ProcessedQuery]): The labeled queries to use as training data.
            label_set (list, optional): A label set to load. If not specified, use the default.
            incremental_timestamp (str, optional): The timestamp folder to cache models in.
        """
        logger.info(
            "Fitting entity recognizer: domain=%r, intent=%r", self.domain, self.intent
        )

        # create model with given params
        self._model_config = self._get_model_config(**kwargs)
        model = create_model(self._model_config)

        if not label_set:
            label_set = self._model_config.train_label_set
            label_set = label_set if label_set else DEFAULT_TRAIN_SET_REGEX

        new_hash = self._get_model_hash(self._model_config, queries, label_set)
        cached_model = self._resource_loader.hash_to_model_path.get(new_hash)

        if incremental_timestamp and cached_model:
            logger.info("No need to fit. Loading previous model.")
            self.load(cached_model)
            return

        # Load labeled data
        queries, labels = self._get_queries_and_labels(queries, label_set=label_set)

        # Build entity types set
        self.entity_types = set()
        for label in labels:
            for entity in label:
                self.entity_types.add(entity.entity.type)

        model.initialize_resources(self._resource_loader, queries, labels)
        model.fit(queries, labels)
        self._model = model
        self.config = ClassifierConfig.from_model_config(self._model.config)
        self.hash = new_hash

        self.ready = True
        self.dirty = True

    def _data_dump_payload(self):
        return {
            "entity_types": self.entity_types,
            "w_ngram_freq": self._model.get_resource("w_ngram_freq"),
            "c_ngram_freq": self._model.get_resource("c_ngram_freq"),
            "model_config": self._model_config,
        }

    def _create_and_dump_payload(self, path):
        self._model.dump(path, self._data_dump_payload())

    def dump(self, model_path, incremental_model_path=None):
        """Save the model.

        Args:
            model_path (str): The model path.
            incremental_model_path (str, Optional): The timestamped folder where the cached \
                models are stored.
        """
        logger.info(
            "Saving entity classifier: domain=%r, intent=%r", self.domain, self.intent
        )
        super().dump(model_path, incremental_model_path)

    def load(self, model_path):
        """Loads the trained entity recognition model from disk.

        Args:
            model_path (str): The location on disk where the model is stored.
        """
        logger.info(
            "Loading entity recognizer: domain=%r, intent=%r", self.domain, self.intent
        )
        try:
            er_data = joblib.load(model_path)

            self.entity_types = er_data["entity_types"]
            self._model_config = er_data.get("model_config")

            # The default is True since < MM 3.2.0 models are serializable by default
            is_serializable = er_data.get("serializable", True)

            if is_serializable:
                # Load the model in directly from the dictionary since its serializable
                self._model = er_data["model"]
            else:
                self._model = create_model(self._model_config)
                self._model.load(model_path, er_data)
        except (OSError, IOError) as e:
            msg = "Unable to load {}. Pickle file cannot be read from {!r}"
            raise ClassifierLoadError(msg.format(self.__class__.__name__, model_path)) from e

        if self._model is not None:
            if not hasattr(self._model, "mindmeld_version"):
                msg = (
                    "Your trained models are incompatible with this version of MindMeld. "
                    "Please run a clean build to retrain models"
                )
                raise ClassifierLoadError(msg)

            try:
                self._model.config.to_dict()
            except AttributeError:
                # Loaded model config is incompatible with app config.
                self._model.config.resolve_config(self._get_model_config())

            gazetteers = self._resource_loader.get_gazetteers()
            tokenizer = self._resource_loader.get_tokenizer()
            sys_types = set(
                (t for t in self.entity_types if Entity.is_system_entity(t))
            )

            w_ngram_freq = er_data.get("w_ngram_freq")
            c_ngram_freq = er_data.get("c_ngram_freq")

            self._model.register_resources(
                gazetteers=gazetteers,
                sys_types=sys_types,
                w_ngram_freq=w_ngram_freq,
                c_ngram_freq=c_ngram_freq,
                tokenizer=tokenizer,
            )
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.hash = self._load_hash(model_path)

        self.ready = True
        self.dirty = False

    def predict(self, query, time_zone=None, timestamp=None, dynamic_resource=None):
        """Predicts entities for the given query using the trained recognition model.

        Args:
            query (Query, str): The input query.
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference.

        Returns:
            (str): The predicted class label.
        """
        prediction = (
            super().predict(
                query,
                time_zone=time_zone,
                timestamp=timestamp,
                dynamic_resource=dynamic_resource,
            )
            or ()
        )
        return tuple(sorted(prediction, key=lambda e: e.span.start))

    def predict_proba(
        self, query, time_zone=None, timestamp=None, dynamic_resource=None
    ):
        """Runs prediction on a given query and generates multiple entity tagging hypotheses with
        their associated probabilities using the trained entity recognition model

        Args:
            query (Query, str): The input query.
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).
            dynamic_resource (optional): Dynamic resource, unused.

        Returns:
            (list): A list of tuples of the form (Entity list, float) grouping potential entity \
                tagging hypotheses and their probabilities.
        """
        del dynamic_resource

        if not self._model:
            logger.error("You must fit or load the model before running predict_proba")
            return []
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(
                query, time_zone=time_zone, timestamp=timestamp
            )
        predict_proba_result = self._model.predict_proba([query])
        return predict_proba_result

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
            List: list of queries
        """
        if queries:
            return self._build_query_tree(
                queries, domain=self.domain, intent=self.intent, raw=raw
            )

        return self._resource_loader.get_labeled_queries(
            domain=self.domain, intent=self.intent, label_set=label_set, raw=raw
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
        raw_queries = [q.query for q in queries]
        labels = [q.entities for q in queries]
        return raw_queries, labels

    def _get_queries_and_labels_hash(
        self, queries=None, label_set=DEFAULT_TRAIN_SET_REGEX
    ):
        query_tree = self._get_query_tree(queries, label_set=label_set, raw=True)
        queries = self._resource_loader.flatten_query_tree(query_tree)
        hashable_queries = [
            self.domain + "###" + self.intent + "###entity###"
        ] + sorted(queries)
        return self._resource_loader.hash_list(hashable_queries)

    def inspect(self, query, gold_label=None, dynamic_resource=None):
        del query
        del gold_label
        del dynamic_resource
        logger.warning("method not implemented")
