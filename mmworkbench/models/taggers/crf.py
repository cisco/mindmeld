# -*- coding: utf-8 -*-
"""
This module contains the Memm entity recognizer.
"""
from __future__ import print_function, absolute_import, unicode_literals, division
from sklearn_crfsuite import CRF

from ..helpers import extract_sequence_features
from .taggers import Tagger, get_tags_from_entities, get_entities_from_tags

import logging
import numpy as np

logger = logging.getLogger(__name__)

ZERO = 1e-20


class ConditionalRandomFields(Tagger):
    """A Conditional Random Fields model."""
    def __init__(self, **parameters):
        self.set_params(**parameters)

    def fit(self, examples, labels):
        self._config = self._passed_params.get('config', None)
        self._feat_binner = self._get_feature_binner()
        self._tag_scheme = self._config.model_settings.get('tag_scheme', 'IOB').upper()
        # Extract features and classes
        all_tags = []
        for idx, label in enumerate(labels):
            all_tags.append(get_tags_from_entities(examples[idx], label, self._tag_scheme))

        X = self._get_features(examples, fit=True)
        self._clf = self._fit(X, all_tags, self._current_params)
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

    def _get_features(self, examples, fit=False):
        """Transforms a list of examples into a feature matrix.

        Args:
            examples (list of mmworkbench.core.Query): a list of queries
        Returns:
            (list of list of str): features in CRF suite format
        """
        feats = []
        for idx, example in enumerate(examples):
            feats.append(self._extract_features(example))
        X = self._preprocess_data(feats, fit)
        return X

    def _extract_features(self, example):
        """Extracts feature dicts for each token in an example.

        Args:
            example (mmworkbench.core.Query): an query
        Returns:
            (list dict): features
        """
        return extract_sequence_features(example, self._config.example_type,
                                         self._config.features, self._resources)

    def _preprocess_data(self, X, fit=False):
        """Converts data into formats of CRF suite.

        Args:
            X (list of dict): features of an example
        Returns:
            (list of list of str): features in CRF suite format
        """
        if fit:
            self._feat_binner.fit(X)

        new_X = []
        for feat_seq in self._feat_binner.transform(X):
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

    def _get_feature_binner(self):
        return FeatureBinner()


# Feature extraction for CRF

class FeatureMapper(object):
    def __init__(self):
        self.feat_name = None
        self.values = []
        self.std = None
        self.mean = None

        self.std_bins = []

        self._num_std = 2
        self._size_std = 0.5

    def add_value(self, value):
        self.values.append(value)

    def fit(self):
        self.std = np.std(self.values)
        self.mean = np.mean(self.values)

        range_start = self.mean - self.std * self._num_std
        num_bin = 2 * int(self._num_std / self._size_std)
        bins = [range_start]

        while num_bin > 0 and self.std > ZERO:
            range_start += self.std * self._size_std
            bins.append(range_start)
            num_bin -= 1
        self.std_bins = np.array(bins)

    def map_bucket(self, value):
        return np.searchsorted(self.std_bins, value)


class FeatureBinner(object):
    def __init__(self):
        self.features = {}

    def fit(self, X_train):
        """
        Get necessary information for each bucket

        Args:
            X_train (list of list of list): training data
        """
        for sentence in X_train:
            for word in sentence:
                for feat_name, feat_value in word.items():
                    self._collect_feature(feat_name, feat_value)

        for feat, mapper in self.features.items():
            mapper.fit()

    def transform(self, X_train):
        """
        Get necessary information for each bucket

        Args:
            X_train (list of list of dict): training data
        """
        new_X_train = []
        for sentence in X_train:
            new_sentence = []
            for word in sentence:
                new_word = {}
                for feat_name, feat_value in word.items():
                    new_feats = self._map_feature(feat_name, feat_value)
                    if new_feats:
                        new_word.update(new_feats)
                new_sentence.append(new_word)
            new_X_train.append(new_sentence)
        return new_X_train

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)

    def _collect_feature(self, feat_name, feat_value):
        try:
            feat_value = float(feat_value)
        except Exception:
            return
        mapper = self.features.get(feat_name, FeatureMapper())
        mapper.feat_name = feat_name
        mapper.add_value(feat_value)

        self.features[feat_name] = mapper

    def _map_feature(self, feat_name, feat_value):
        try:
            feat_value = float(feat_value)
        except Exception:
            return {feat_name: feat_value}
        if feat_name not in self.features:
            return {feat_name: feat_value}

        mapper = self.features[feat_name]
        new_feat_value = mapper.map_bucket(feat_value)
        return {feat_name: new_feat_value}
