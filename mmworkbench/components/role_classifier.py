# -*- coding: utf-8 -*-
"""
This module contains the role classifier component of the Workbench natural language processor.
"""
from __future__ import absolute_import, unicode_literals
from builtins import super

import logging
import os

from sklearn.externals import joblib

from ..models import create_model, ENTITY_EXAMPLE_TYPE, CLASS_LABEL_TYPE

from .classifier import Classifier, ClassifierConfig
from ._config import get_classifier_config

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

    CLF_TYPE = 'role'

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

    def _get_model_config(self, **kwargs):
        """Gets a machine learning model configuration

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        kwargs['example_type'] = ENTITY_EXAMPLE_TYPE
        kwargs['label_type'] = CLASS_LABEL_TYPE
        default_config = get_classifier_config(self.CLF_TYPE, self._resource_loader.app_path,
                                               domain=self.domain, intent=self.intent,
                                               entity=self.entity_type)
        return super()._get_model_config(default_config, **kwargs)

    def fit(self, queries=None, label_set='train', **kwargs):
        """Trains a statistical model for role classification using the provided training examples

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as training data
            label_set (list, optional): A label set to load. If not specified, the default
                training set will be loaded.
        """

        logger.info('Fitting role classifier: domain=%r, intent=%r, entity_type=%r',
                    self.domain, self.intent, self.entity_type)

        # create model with given params
        model_config = self._get_model_config(**kwargs)
        model = create_model(model_config)

        # Load labeled data
        examples, labels = self._get_queries_and_labels(queries, label_set=label_set)

        if examples:
            # Build roles set
            self.roles = set()
            for label in labels:
                self.roles.add(label)

            # Get gazetteers (they will be built if necessary)
            gazetteers = self._resource_loader.get_gazetteers()

            model.register_resources(gazetteers=gazetteers)
            model.fit(examples, labels)
            self._model = model
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = True

    def dump(self, model_path):
        """Persists the trained role classification model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored
        """
        logger.info('Saving role classifier: domain=%r, intent=%r, entity_type=%r',
                    self.domain, self.intent, self.entity_type)
        # make directory if necessary
        folder = os.path.dirname(model_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        rc_data = {'model': self._model, 'roles': self.roles}
        joblib.dump(rc_data, model_path)

        self.dirty = False

    def load(self, model_path):
        """Loads the trained role classification model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        logger.info('Loading role classifier: domain=%r, intent=%r, entity_type=%r',
                    self.domain, self.intent, self.entity_type)
        try:
            rc_data = joblib.load(model_path)
            self._model = rc_data['model']
            self.roles = rc_data['roles']
        except (OSError, IOError):
            logger.error('Unable to load %s. Pickle file cannot be read from %r',
                         self.__class__.__name__, model_path)
            return
            # msg = 'Unable to load {}. Pickle file cannot be read from {!r}'
            # raise ClassifierLoadError(msg.format(self.__class__.__name__, model_path))
        if self._model is not None:
            gazetteers = self._resource_loader.get_gazetteers()
            self._model.register_resources(gazetteers=gazetteers)
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.ready = True
        self.dirty = False

    def predict(self, query, entities, entity_index):
        """Predicts a role for the given entity using the trained role classification model

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity_index (int): The index of the entity whose role should be classified

        Returns:
            str: The predicted role for the provided entity
        """
        if not self._model:
            logger.error('You must fit or load the model before running predict')
            return
        gazetteers = self._resource_loader.get_gazetteers()
        self._model.register_resources(gazetteers=gazetteers)
        return self._model.predict([(query, entities, entity_index)])[0]

    def predict_proba(self, query, entities, entity_index):
        """Runs prediction on a given entity and generates multiple role hypotheses with their
        associated probabilities using the trained role classification model

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity_index (int): The index of the entity whose role should be classified

        Returns:
            list: a list of tuples of the form (str, float) grouping roles and their probabilities
        """
        raise NotImplementedError

    def evaluate(self, queries=None):
        """Evaluates the trained entity recognition model on the given test data

        Args:
            queries (list of ProcessedQuery): The labeled queries to use as test data. If none
                are provided, the heldout label set will be used.

        Returns:
            ModelEvaluation: A ModelEvaluation object that contains evaluation results
        """
        if not self._model:
            logger.error('You must fit or load the model before running evaluate.')
            return

        gazetteers = self._resource_loader.get_gazetteers()
        self._model.register_resources(gazetteers=gazetteers)
        queries, labels = self._get_queries_and_labels(queries, label_set='heldout')

        if not queries:
            logger.info('Could not evaluate model. No relevant examples in evaluation set.')
            return

        evaluation = self._model.evaluate(queries, labels)
        return evaluation

    def _get_queries_and_labels(self, queries=None, label_set='train'):
        """Returns a set of queries and their labels based on the label set

        Args:
            queries (list, optional): A list of ProcessedQuery objects, to
                train on. If not specified, a label set will be loaded.
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
        """
        if not queries:
            query_tree = self._resource_loader.get_labeled_queries(domain=self.domain,
                                                                   intent=self.intent,
                                                                   label_set=label_set)
            queries = query_tree[self.domain][self.intent]

        # build list of examples -- entities of this role classifier's type
        examples = []
        labels = []
        for query in queries:
            for idx, entity in enumerate(query.entities):
                if entity.entity.type == self.entity_type and entity.entity.role:
                    examples.append((query.query, query.entities, idx))
                    labels.append(entity.entity.role)

        unique_labels = set(labels)
        if len(unique_labels) == 1:
            # No roles
            return (), ()
        if None in unique_labels:
            bad_examples = [e for i, e in enumerate(examples) if labels[i] is None]
            for example in bad_examples:
                logger.error('Invalid entity annotation, expecting role in query %r', example[0])
            raise ValueError('One or more invalid entity annotations, expecting role')

        return examples, labels
