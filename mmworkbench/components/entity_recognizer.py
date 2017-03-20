# -*- coding: utf-8 -*-
"""
This module contains the named entity recognizer component.
"""
from __future__ import unicode_literals
from builtins import super

import os
import logging

from sklearn.externals import joblib

from ..exceptions import FileNotFoundError
from ..models import create_model, QUERY_EXAMPLE_TYPE, ENTITIES_LABEL_TYPE

from .classifier import Classifier, ClassifierConfig, ClassifierLoadError

logger = logging.getLogger(__name__)


class EntityRecognizer(Classifier):
    """A named entity recognizer which is used to identify the entities for a given query. It is
    trained using all the labeled queries for a particular intent. The labels are the entity
    annotations for each query.

    Attributes:
        domain (str): The domain of this named entity recognizer
        intent (str): The intent of this named entity recognizer
        entity_types (set): A set containing the entities which can be recognized
    """

    DEFAULT_CONFIG = {
        'default_model': 'memm-cv',
        'models': {
            'main': {
                'model_type': 'memm',
                'model_settings': {
                    'tag_scheme': 'IOB',
                    'feature_scaler': 'none'
                },
                'params': {
                    'penalty': 'l2',
                    'C': 100
                },
                'features': {}  # use default
            },
            'sparse': {
                'model_type': 'memm',
                'model_settings': {
                    'tag_scheme': 'IOB',
                    'feature_scaler': 'max-abs',
                    'feature_selector': 'l1'
                },
                'params': {
                    'penalty': 'l2',
                    'C': 100
                },
                'features': {}  # use default
            },
            'memm-cv': {
                'model_type': 'memm',
                'model_settings': {
                    'tag_scheme': 'IOB',
                    'feature_scaler': 'max-abs'
                },
                'param_selection': {
                    'type': 'k-fold',
                    'k': 5,
                    'scoring': 'accuracy',
                    'grid': {
                        'penalty': ['l1', 'l2'],
                        'C': [0.01, 1, 100, 10000, 1000000, 100000000]
                    },
                },
                'features': {}  # use default
            }
        }
    }

    def __init__(self, resource_loader, domain, intent):
        """Initializes a named entity recognizer

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain of this named entity recognizer
            intent (str): The intent of this named entity recognizer
        """
        super().__init__(resource_loader)
        self.domain = domain
        self.intent = intent
        self.entity_types = set()

    def _get_model_config(self, config_name, **kwargs):
        kwargs['example_type'] = QUERY_EXAMPLE_TYPE
        kwargs['label_type'] = ENTITIES_LABEL_TYPE
        return super()._get_model_config(config_name, **kwargs)

    def fit(self, queries=None, config_name=None, label_set='train', **kwargs):
        """Trains the model

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as training data
            config_name (str): The type of model to use. If omitted, the default model type will
                be used.

        """
        logger.info('Fitting entity recognizer: domain=%r, intent=%r', self.domain, self.intent)
        queries, labels = self._get_queries_and_labels(queries, label_set=label_set)
        model_config = self._get_model_config(config_name, **kwargs)
        model = create_model(model_config)
        gazetteers = self._resource_loader.get_gazetteers()

        # build  entity types set
        self.entity_types = set()
        for label in labels:
            for entity in label:
                self.entity_types.add(entity.entity.type)

        model.register_resources(gazetteers=gazetteers)
        model.fit(queries, labels)
        self._model = model
        self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = True

    def dump(self, model_path):
        """Persists the model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored

        """
        logger.info('Saving entity recognizer: domain=%r, intent=%r', self.domain, self.intent)
        # make directory if necessary
        folder = os.path.dirname(model_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        er_data = {'model': self._model, 'entity_types': self.entity_types}
        joblib.dump(er_data, model_path)

        self.dirty = False

    def load(self, model_path):
        """Loads the model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        logger.info('Loading entity recognizer: domain=%r, intent=%r', self.domain, self.intent)
        try:
            er_data = joblib.load(model_path)
            self._model = er_data['model']
            self.entity_types = er_data['entity_types']
        except FileNotFoundError:
            msg = 'Unable to load {}. Pickle file not found at {!r}'
            raise ClassifierLoadError(msg.format(self.__class__.__name__, model_path))
        if self._model is not None:
            gazetteers = self._resource_loader.get_gazetteers()
            self._model.register_resources(gazetteers=gazetteers)
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = False

    def predict_proba(self, query):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query
`
        Returns:
            list: a list of tuples of the form (Entity, float) grouping potential entities and their
            probabilities
        """
        raise NotImplementedError

    def evaluate(self, queries=None):
        """Evaluates the classifier

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as training data. If none
                are provided, the heldout label set will be used.

        Returns:
            ModelEvaluation object

        """
        gazetteers = self._resource_loader.get_gazetteers()
        self._model.register_resources(gazetteers=gazetteers)
        queries, labels = self._get_queries_and_labels(queries, label_set='heldout')
        evaluation = self._model.evaluate(queries, labels)
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
            query_tree = self._resource_loader.get_labeled_queries(domain=self.domain,
                                                                   intent=self.intent,
                                                                   label_set=label_set)
            queries = query_tree[self.domain][self.intent]
        raw_queries = [q.query for q in queries]
        labels = [q.entities for q in queries]
        return raw_queries, labels
