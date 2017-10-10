# -*- coding: utf-8 -*-
"""
This module contains the entity recognizer component of the Workbench natural language processor.
"""
from __future__ import absolute_import, unicode_literals
from builtins import super

import os
import logging

from sklearn.externals import joblib

from ..core import Entity
from ..models import create_model, QUERY_EXAMPLE_TYPE, ENTITIES_LABEL_TYPE

from .classifier import Classifier, ClassifierConfig, ClassifierLoadError
from ._config import get_classifier_config

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

    CLF_TYPE = 'entity'

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

    def _get_model_config(self, **kwargs):
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs['example_type'] = QUERY_EXAMPLE_TYPE
        kwargs['label_type'] = ENTITIES_LABEL_TYPE
        loaded_config = get_classifier_config(self.CLF_TYPE, self._resource_loader.app_path,
                                              domain=self.domain, intent=self.intent)
        return super()._get_model_config(loaded_config, **kwargs)

    def fit(self, queries=None, label_set='train', **kwargs):
        """Trains the entity recognition model using the provided training queries

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as training data
            label_set (list, optional): A label set to load. If not specified, the default
                training set will be loaded.
        """
        logger.info('Fitting entity recognizer: domain=%r, intent=%r', self.domain, self.intent)

        # create model with given params
        self._model_config = self._get_model_config(**kwargs)
        model = create_model(self._model_config)

        # Load labeled data
        queries, labels = self._get_queries_and_labels(queries, label_set=label_set)

        # initialize resources
        model.initialize_resources(self._resource_loader, queries, labels)

        # Build entity types set
        self.entity_types = set()
        for label in labels:
            for entity in label:
                self.entity_types.add(entity.entity.type)

        model.fit(queries, labels)
        self._model = model
        self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = True

    def dump(self, model_path):
        """Persists the trained entity recognition model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored

        """
        logger.info('Saving entity recognizer: domain=%r, intent=%r', self.domain, self.intent)
        # make directory if necessary
        folder = os.path.dirname(model_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        er_data = {'entity_types': self.entity_types, 'model_config': self._model_config}

        self._model.dump(model_path, er_data)
        self.dirty = False

    def load(self, model_path):
        """Loads the trained entity recognition model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        logger.info('Loading entity recognizer: domain=%r, intent=%r', self.domain, self.intent)
        try:
            er_data = joblib.load(model_path)

            self.entity_types = er_data['entity_types']
            self._model_config = er_data.get('model_config')
            self._is_not_serializable = er_data.get('is_not_serializable')

            if self._is_not_serializable:
                # If model config is given, this means it is a non-SKLearn model,
                # which is easily serialized using joblib, but tensorflow models
                # are not, so we have to create and load them in.
                self._model = create_model(self._model_config)
                self._model.load(model_path, er_data)
            else:
                self._model = er_data['model']
        except (OSError, IOError):
            msg = 'Unable to load {}. Pickle file cannot be read from {!r}'
            raise ClassifierLoadError(msg.format(self.__class__.__name__, model_path))

        if self._model is not None:
            gazetteers = self._resource_loader.get_gazetteers()
            sys_types = set((t for t in self.entity_types if Entity.is_system_entity(t)))
            self._model.register_resources(gazetteers=gazetteers, sys_types=sys_types)
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = False

    def predict(self, query):
        """Predicts entities for the given query using the trained recognition model

        Args:
            query (Query or str): The input query

        Returns:
            str: The predicted class label

        """
        prediction = super().predict(query) or ()
        return tuple(sorted(prediction, key=lambda e: e.span.start))

    def predict_proba(self, query):
        """Runs prediction on a given query and generates multiple entity tagging hypotheses with
        their associated probabilities using the trained entity recognition model

        Args:
            query (Query): The input query

        Returns:
            list: a list of tuples of the form (Entity list, float) grouping potential entity
                tagging hypotheses and their probabilities
        """
        raise NotImplementedError

    def _get_queries_and_labels(self, queries=None, label_set='train'):
        """Returns a set of queries and their labels based on the label set

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
        """
        if not queries:
            query_tree = self._resource_loader.get_labeled_queries(domain=self.domain,
                                                                   intent=self.intent,
                                                                   label_set=label_set)
            queries = query_tree.get(self.domain, {}).get(self.intent, {})
        raw_queries = [q.query for q in queries]
        labels = [q.entities for q in queries]
        return raw_queries, labels
