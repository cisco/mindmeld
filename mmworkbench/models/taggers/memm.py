# -*- coding: utf-8 -*-
"""
This module contains the Memm entity recognizer.
"""
from __future__ import print_function, absolute_import, unicode_literals, division

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder, MaxAbsScaler, StandardScaler

from .taggers import Tagger, START_TAG
from ..helpers import extract_sequence_features, get_label_encoder

import logging
logger = logging.getLogger(__name__)


class MemmModel(Tagger):
    """A maximum-entropy Markov model."""
    def __init__(self, **parameters):
        self.set_params(**parameters)

    def fit(self, examples, labels):
        parameters = self._passed_params

        # TODO: add error when config is not passed in
        self.config = parameters['config']

        self._label_encoder = get_label_encoder(self.config)
        self._feat_selector = self._get_feature_selector()
        self._feat_scaler = self._get_feature_scaler()
        # Default tag scheme to IOB
        self._tag_scheme = self.config.model_settings.get('tag_scheme', 'IOB').upper()

        # Extract features and classes
        y = self._label_encoder.encode(labels, examples=examples)
        X, y, groups = self.get_feature_matrix(examples, y, fit=True)

        # Fit the underlying classifier
        model_class = self._get_model_constructor()
        self._clf = model_class(**self._current_params).fit(X, y)
        return self

    def get_params(self, deep=True):
        return self._clf.get_params()

    def set_params(self, **parameters):
        """Sets the parameters
        """
        self._class_encoder = SKLabelEncoder()
        self._feat_vectorizer = DictVectorizer()
        self._passed_params = parameters
        self._current_params = {}
        self._resources = parameters.get('resources', {})

        model_class = self._get_model_constructor()
        self._clf = model_class()

        for parameter, value in parameters.items():
            if parameter == 'config' or parameter == 'resources':
                continue
            self._current_params[parameter] = value
        self._clf.set_params(**self._current_params)
        return self

    def predict(self, examples):
        return [self._predict_example(example) for example in examples]

    def _predict_example(self, example):
        features_by_segment = self._extract_features(example)
        if len(features_by_segment) == 0:
            return self._label_encoder.decode([], examples=[example])[0]

        predicted_tags = []
        prev_tag = START_TAG
        for features in features_by_segment:
            features['prev_tag'] = prev_tag
            X, _ = self._preprocess_data([features])
            prediction = self._clf.predict(X)
            predicted_tag = self._class_encoder.inverse_transform(prediction)[0]
            predicted_tags.append(predicted_tag)
            prev_tag = predicted_tag

        return self._label_encoder.decode([predicted_tags], examples=[example])[0]

    def _extract_features(self, example):
        """Extracts feature dicts for each token in an example.

        Args:
            example (mmworkbench.core.Query): an query
        Returns:
            (list dict): features
        """
        return extract_sequence_features(example, self.config.example_type,
                                         self.config.features, self._resources)

    def get_feature_matrix(self, examples, y=None, fit=True):
        """Transforms a list of examples into a feature matrix.

        Args:
            examples (list): The examples.

        Returns:
            (numpy.matrix): The feature matrix.
            (numpy.array): The group labels for examples.
        """
        groups = []
        feats = []
        y_offset = 0
        for i, example in enumerate(examples):
            features_by_segment = self._extract_features(example)
            feats.extend(features_by_segment)
            groups.extend([i for _ in features_by_segment])
            for j, segment in enumerate(features_by_segment):
                if j == 0:
                    segment['prev_tag'] = START_TAG
                elif fit:
                    segment['prev_tag'] = y[y_offset + j - 1]

            y_offset += len(features_by_segment)
        X, y = self._preprocess_data(feats, y, fit=fit)
        return X, y, groups

    def _get_feature_selector(self):
        """Get a feature selector instance based on the feature_selector model
        parameter

        Returns:
            (Object): a feature selector which returns a reduced feature matrix,
                given the full feature matrix, X and the class labels, y
        """
        if self.config.model_settings is None:
            selector_type = None
        else:
            selector_type = self.config.model_settings.get('feature_selector')
        selector = {'l1': SelectFromModel(LogisticRegression(penalty='l1', C=1)),
                    'f': SelectPercentile()}.get(selector_type)
        return selector

    def _get_feature_scaler(self):
        """Get a feature value scaler based on the model settings"""
        if self.config.model_settings is None:
            scale_type = None
        else:
            scale_type = self.config.model_settings.get('feature_scaler')
        scaler = {'std-dev': StandardScaler(with_mean=False),
                  'max-abs': MaxAbsScaler()}.get(scale_type)
        return scaler

    def _preprocess_data(self, X, y=None, fit=False):

        if fit:
            y = self._class_encoder.fit_transform(y)
            X = self._feat_vectorizer.fit_transform(X)
            if self._feat_scaler is not None:
                X = self._feat_scaler.fit_transform(X)
            if self._feat_selector is not None:
                X = self._feat_selector.fit_transform(X, y)
        else:
            X = self._feat_vectorizer.transform(X)
            if self._feat_scaler is not None:
                X = self._feat_scaler.transform(X)
            if self._feat_selector is not None:
                X = self._feat_selector.transform(X)

        return X, y

    def _get_model_constructor(self):
        """Returns the python class of the actual underlying model"""
        return LogisticRegression
