# -*- coding: utf-8 -*-
"""
This module contains all code required to perform tagging.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import logging

from sklearn_crfsuite import CRF

from .helpers import extract_sequence_features
from .tagging import get_tags_from_entities, get_entities_from_tags


logger = logging.getLogger(__name__)


class Tagger(object):
    def __init__(self, config):
        """
        Args:
            config (ModelConfig): model configuration
        """
        self.config = config
        # Default tag scheme to IOB
        self._tag_scheme = self.config.model_settings.get('tag_scheme', 'IOB').upper()
        # Placeholders
        self._resources = {}
        self._clf = None
        self._current_params = {}

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior. For the _resources field,
        we save the resources that are memory intensive
        """
        attributes = self.__dict__.copy()
        resources_to_persist = set(['sys_types'])
        for key in list(attributes['_resources'].keys()):
            if key not in resources_to_persist:
                del attributes['_resources'][key]
        return attributes

    def fit(self, examples, labels, resources=None):
        """Trains the model

        Args:
            labeled_queries (list of mmworkbench.core.Query): a list of queries to train on
            labels (list of tuples of mmworkbench.core.QueryEntity): a list of predicted labels
        """
        raise NotImplementedError

    def predict(self, examples):
        """Predicts for a list of examples
        Args:
            examples (list of mmworkbench.core.Query): a list of queries to be predicted
        Returns:
            (list of tuples of mmworkbench.core.QueryEntity): a list of predicted labels
        """
        raise NotImplementedError

    def _get_model_constructor(self):
        """Returns the python class of the actual underlying model"""
        raise NotImplementedError


class ConditionalRandomFields(Tagger):
    """A Conditional Random Fields model."""

    def fit(self, examples, labels, resources=None):
        self._resources = resources
        # Extract features and classes
        all_tags = []
        for idx, label in enumerate(labels):
            all_tags.append(get_tags_from_entities(examples[idx], label, self._tag_scheme))

        X = self._get_features(examples)
        self._clf = self._fit(X, all_tags, self.config.params)
        self._current_params = self.config.params
        return self

    def predict(self, examples):
        X = self._get_features(examples)
        tags_by_example = self._clf.predict(X)
        labels = [get_entities_from_tags(examples[idx], tags, self._tag_scheme)
                  for idx, tags in enumerate(tags_by_example)]
        return labels

    def _get_model_constructor(self):
        """Returns the python class of the actual underlying model"""
        return CRF

    def _get_features(self, examples):
        """Transforms a list of examples into a feature matrix.

        Args:
            examples (list of mmworkbench.core.Query): a list of queries
        Returns:
            (list of list of str): features in CRF suite format
        """
        feats = []
        for idx, example in enumerate(examples):
            feats.append(self._extract_features(example))
        X = self._preprocess_data(feats)
        return X

    def _extract_features(self, example):
        """Extracts feature dicts for each token in an example.

        Args:
            example (mmworkbench.core.Query): an query
        Returns:
            (list dict): features
        """
        return extract_sequence_features(example, self.config.example_type,
                                         self.config.features, self._resources)

    def _preprocess_data(self, X):
        """Converts data into formats of CRF suite.

        Args:
            X (list of dict): features of an example
        Returns:
            (list of list of str): features in CRF suite format
        """
        new_X = []
        for feat_seq in X:
            feat_list = []
            for feature in feat_seq:
                temp_list = []
                for elem in sorted(feature.keys()):
                    temp_list.append(elem + '=' + str(feature[elem]))
                feat_list.append(temp_list)
            new_X.append(feat_list)
        return new_X

    def _fit(self, X, y, params):
        """Trains a classifier without cross-validation.

        Args:
            X (list of list of list of str): a list of queries to train on
            y (list of list of str): a list of expected labels
            params (dict): Parameters of the classifier
        """
        model_class = self._get_model_constructor()
        return model_class(**params).fit(X, y)
