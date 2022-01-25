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
This module contains the role classifier component of the MindMeld natural language processor.
"""
import logging
import pickle

from sklearn.externals import joblib

from ._config import get_classifier_config
from .classifier import Classifier, ClassifierConfig, ClassifierLoadError
from ..constants import DEFAULT_TRAIN_SET_REGEX
from ..core import Query
from ..models import CLASS_LABEL_TYPE, ENTITY_EXAMPLE_TYPE, create_model, load_model
from ..resource_loader import ProcessedQueryList

logger = logging.getLogger(__name__)


class RoleClassifier(Classifier):
    """A role classifier is used to determine the target role for entities in a given query. It is
    trained using all the labeled queries for a particular intent. The labels are the role names
    associated with each entity within each query.

    Attributes:
        domain (str): The domain that this role classifier belongs to
        intent (str): The intent that this role classifier belongs to
        entity_type (str): The entity type that this role classifier is for
        roles (set): A set containing the roles which can be classified
    """

    CLF_TYPE = "role"

    def __init__(self, resource_loader, domain, intent, entity_type):
        """Initializes a role classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain that this role classifier belongs to
            intent (str): The intent that this role classifier belongs to
            entity_type (str): The entity type that this role classifier is for
        """
        super().__init__(resource_loader)
        self.domain = domain
        self.intent = intent
        self.entity_type = entity_type
        self.roles = set()

    # pylint: disable=arguments-differ
    def _get_model_config(self, **kwargs):
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs["example_type"] = ENTITY_EXAMPLE_TYPE
        kwargs["label_type"] = CLASS_LABEL_TYPE
        loaded_config = get_classifier_config(
            self.CLF_TYPE,
            self._resource_loader.app_path,
            domain=self.domain,
            intent=self.intent,
            entity=self.entity_type,
        )
        return super()._get_model_config(loaded_config, **kwargs)

    def fit(self,
            queries=None,
            label_set=None,
            incremental_timestamp=None,
            load_cached=True, **kwargs):
        """Trains a statistical model for role classification using the provided training examples.

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as training data
            label_set (list, optional): A label set to load. If not specified, the default
                training set will be loaded.
            incremental_timestamp (str, optional): The timestamp folder to cache models in
        """
        logger.info(
            "Fitting role classifier: domain=%r, intent=%r, entity_type=%r",
            self.domain,
            self.intent,
            self.entity_type,
        )

        # create model with given params
        model_config = self._get_model_config(**kwargs)

        label_set = label_set or model_config.train_label_set or DEFAULT_TRAIN_SET_REGEX
        queries = self._resolve_queries(queries, label_set)

        new_hash = self._get_model_hash(model_config, queries)
        cached_model_path = self._resource_loader.hash_to_model_path.get(new_hash)

        # These examples and labels are flat lists, not
        # a ProcessedQueryList.Iterator
        examples, labels = self._get_examples_and_labels(queries)

        if examples:
            # Build roles set
            self.roles.update(labels)

        if incremental_timestamp and cached_model_path:
            logger.info("No need to fit. Previous model is cached.")
            if load_cached:
                # load() sets self.ready = True
                self.load(cached_model_path)
                return True
            return False

        if self.roles:
            model = create_model(model_config)
            model.initialize_resources(self._resource_loader, examples, labels)
            model.fit(examples, labels)
            self._model = model
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.hash = new_hash

        self.ready = True
        self.dirty = True
        return True

    def dump(self, model_path, incremental_model_path=None):
        """Persists the trained role classification model to disk.

        Args:
            model_path (str): The model path.
            incremental_model_path (str, Optional): The timestamped folder where the cached \
                models are stored.
        """
        logger.info(
            "Saving role classifier: domain=%r, intent=%r, entity=%r",
            self.domain,
            self.intent,
            self.entity_type,
        )
        super().dump(model_path, incremental_model_path)

    def _dump(self, path):
        rc_data = {"roles": self.roles}
        pickle.dump(rc_data, open(self._get_classifier_resources_save_path(path), "wb"))

    def unload(self):
        self._model = None
        self.roles = set()
        self.ready = False

    def load(self, model_path):
        """Loads the trained role classification model from disk.

        Args:
            model_path (str): The location on disk where the model is stored
        """
        logger.info(
            "Loading role classifier: domain=%r, intent=%r, entity_type=%r",
            self.domain,
            self.intent,
            self.entity_type,
        )

        # underlying model specific load
        self._model = load_model(model_path)

        # classifier specific load
        try:
            rc_data = pickle.load(open(self._get_classifier_resources_save_path(model_path), "rb"))
        except FileNotFoundError:  # backwards compatability for previous version's saved models
            rc_data = joblib.load(model_path)
        self.roles = rc_data["roles"]

        # validate and register resources
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
            text_preparation_pipeline = self._resource_loader.get_text_preparation_pipeline()
            self._model.register_resources(
                gazetteers=gazetteers,
                text_preparation_pipeline=text_preparation_pipeline
            )
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.hash = self._load_hash(model_path)

        self.ready = True
        self.dirty = False

    def predict(
        self, query, entities, entity_index
    ):  # pylint: disable=arguments-differ
        """Predicts a role for the given entity using the trained role classification model.

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity_index (int): The index of the entity whose role should be classified

        Returns:
            str: The predicted role for the provided entity
        """
        if not self._model:
            logger.error("You must fit or load the model before running predict")
            return
        if len(self.roles) == 1:
            return list(self.roles)[0]
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(query)
        gazetteers = self._resource_loader.get_gazetteers()
        text_preparation_pipeline = self._resource_loader.get_text_preparation_pipeline()
        self._model.register_resources(
            gazetteers=gazetteers,
            text_preparation_pipeline=text_preparation_pipeline
        )
        return self._model.predict([(query, entities, entity_index)])[0]

    def predict_proba(
        self, query, entities, entity_index
    ):  # pylint: disable=arguments-differ
        """Runs prediction on a given entity and generates multiple role hypotheses and
        associated probabilities using the trained role classification model.

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity_index (int): The index of the entity whose role should be classified

        Returns:
            list: a list of tuples of the form (str, float) grouping roles and their probabilities
        """
        if not self._model:
            logger.error("You must fit or load the model before running predict")
            return
        if len(self.roles) == 1:
            return [(list(self.roles)[0], 1.0)]
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(query)
        gazetteers = self._resource_loader.get_gazetteers()
        text_preparation_pipeline = self._resource_loader.get_text_preparation_pipeline()
        self._model.register_resources(
            gazetteers=gazetteers,
            text_preparation_pipeline=text_preparation_pipeline
        )
        predict_proba_result = self._model.predict_proba(
            [(query, entities, entity_index)]
        )
        class_proba_tuples = list(predict_proba_result[0][1].items())
        return sorted(class_proba_tuples, key=lambda x: x[1], reverse=True)

    # pylint: disable=arguments-differ
    def view_extracted_features(self, query, entities, entity_index):
        """
        Extracts features for a given entity for role classification.

        Args:
            query (Query or str): The input query
            entities (list): The entities in the query
            entity_index (int): The index of the entity whose role should be classified

        Returns:
            dict: The extracted features from the given input
        """
        if not self._model:
            logger.error("You must fit or load the model to initialize resources")
            return
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(query)
        gazetteers = self._resource_loader.get_gazetteers()
        text_preparation_pipeline = self._resource_loader.get_text_preparation_pipeline()
        self._model.register_resources(
            gazetteers=gazetteers,
            text_preparation_pipeline=text_preparation_pipeline
        )
        return self._model._extract_features((query, entities, entity_index))

    def _get_queries_from_label_set(self, label_set=DEFAULT_TRAIN_SET_REGEX):
        return self._resource_loader.get_flattened_label_set(
            domain=self.domain,
            intent=self.intent,
            label_set=label_set
        )

    def _get_examples_and_labels(self, queries):
        """Returns a set of queries and their labels based on the label set

        Args:
            queries (list): A list of ProcessedQuery objects, to
                train on. If not specified, a label set will be loaded.
        Returns:
            tuple(list(Any), list(Any))
        """
        # build list of examples -- entities of this role classifier's type
        examples = []
        labels = []
        for query in queries.processed_queries():
            for idx, entity in enumerate(query.entities):
                if entity.entity.type == self.entity_type and entity.entity.role:
                    examples.append((query.query, query.entities, idx))
                    labels.append(entity.entity.role)

        unique_labels = set(labels)
        if len(unique_labels) == 0:
            # No roles
            return (), ()
        if None in unique_labels:
            bad_examples = [e for i, e in enumerate(examples) if labels[i] is None]
            for example in bad_examples:
                logger.error(
                    "Invalid entity annotation, expecting role in query %r", example[0]
                )
            raise ValueError("One or more invalid entity annotations, expecting role")

        return (ProcessedQueryList.ListIterator(examples),
                ProcessedQueryList.ListIterator(labels))

    def _get_examples_and_labels_hash(self, queries):
        hashable_queries = (
            [self.domain + "###" + self.intent + "###" + self.entity_type + "###"]
            + sorted(list(queries.raw_queries()))
        )
        return self._resource_loader.hash_list(hashable_queries)

    def inspect(self, query, gold_label=None, dynamic_resource=None):
        del gold_label
        del dynamic_resource
        del query
        logger.warning("method not implemented")
