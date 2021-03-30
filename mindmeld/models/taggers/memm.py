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
This module contains the Memm entity recognizer.
"""
import logging

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from .taggers import START_TAG, Tagger, extract_sequence_features

logger = logging.getLogger(__name__)


class MemmModel(Tagger):
    """A maximum-entropy Markov model."""

    @staticmethod
    def _predict_proba(X):
        del X
        pass

    @staticmethod
    def load(model_path):
        del model_path
        pass

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def set_params(self, **parameters):
        self._clf = LogisticRegression()
        self._clf.set_params(**parameters)
        return self

    def get_params(self, deep=True):
        return self._clf.get_params()

    def predict(self, X, dynamic_resource=None):
        return self._clf.predict(X)

    @staticmethod
    def extract_example_features(example, config, resources):
        """Extracts feature dicts for each token in an example.

        Args:
            example (mindmeld.core.Query): A query.
            config (ModelConfig): The ModelConfig which may contain information used for feature \
                                  extraction.
            resources (dict): Resources which may be used for this model's feature extraction.

        Returns:
            (list[dict]): Features.
        """
        return extract_sequence_features(
            example, config.example_type, config.features, resources
        )

    def extract_features(self, examples, config, resources, y=None, fit=True):
        """Transforms a list of examples into a feature matrix. Use extract_and_predict if you are
        extracting features for an example at test time, since the previous tag prediction is needed
        as a feature of the next tag.

        Args:
            examples (list of core.Query): The examples.
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction

        Returns:
            (tuple): tuple containing:

                * (numpy.matrix): The feature matrix.
                * (numpy.array): The group labels for examples.
        """
        groups = []
        X = []
        y_flat = [tag for example in y for tag in example]
        y_offset = 0
        for i, example in enumerate(examples):
            features_by_segment = self.extract_example_features(
                example, config, resources
            )
            X.extend(features_by_segment)
            groups.extend([i for _ in features_by_segment])
            for j, segment in enumerate(features_by_segment):
                if j == 0:
                    segment["prev_tag"] = START_TAG
                elif fit:
                    segment["prev_tag"] = y_flat[y_offset + j - 1]

            y_offset += len(features_by_segment)
        X, y = self._preprocess_data(X, y_flat, fit)
        return X, y, groups

    def extract_and_predict(self, examples, config, resources):
        return [
            self._predict_example(example, config, resources) for example in examples
        ]

    def _predict_example(self, example, config, resources):
        features_by_segment = self.extract_example_features(example, config, resources)
        if len(features_by_segment) == 0:
            return []

        predicted_tags = []
        prev_tag = START_TAG
        for features in features_by_segment:
            features["prev_tag"] = prev_tag
            X, _ = self._preprocess_data([features])
            prediction = self.predict(X)
            predicted_tag = self.class_encoder.inverse_transform(prediction)[0]
            predicted_tags.append(predicted_tag)
            prev_tag = predicted_tag

        return predicted_tags

    def predict_proba(self, examples, config, resources):
        return [
            self._predict_proba_example(example, config, resources)
            for example in examples
        ]

    def _predict_proba_example(self, example, config, resources):
        features_by_segment = self.extract_example_features(example, config, resources)
        if len(features_by_segment) == 0:
            return []

        prev_tag = START_TAG
        seq_log_probs = []
        for features in features_by_segment:
            features["prev_tag"] = prev_tag
            X, _ = self._preprocess_data([features])
            prediction = self._clf.predict_proba(X)[0]
            predicted_tag = np.argmax(prediction)
            prev_tag = self.class_encoder.inverse_transform(predicted_tag)
            seq_log_probs.append([prev_tag, prediction[predicted_tag]])
        return seq_log_probs

    @staticmethod
    def _get_feature_selector(selector_type):
        """Get a feature selector instance based on the feature_selector model
        parameter.

        Returns:
            (Object): A feature selector which returns a reduced feature matrix, \
                given the full feature matrix, X and the class labels, y.
        """
        selector = {
            "l1": SelectFromModel(LogisticRegression(penalty="l1", C=1)),
            "f": SelectPercentile(),
        }.get(selector_type)
        return selector

    @staticmethod
    def _get_feature_scaler(scale_type):
        """Get a feature value scaler based on the model settings"""
        scaler = {
            "std-dev": StandardScaler(with_mean=False),
            "max-abs": MaxAbsScaler(),
        }.get(scale_type)
        return scaler

    def setup_model(self, config):
        if config.model_settings is None:
            selector_type = None
            scale_type = None
        else:
            selector_type = config.model_settings.get("feature_selector")
            scale_type = config.model_settings.get("feature_scaler")
        self.class_encoder = SKLabelEncoder()
        self.feat_vectorizer = DictVectorizer()
        self._feat_selector = self._get_feature_selector(selector_type)
        self._feat_scaler = self._get_feature_scaler(scale_type)

    def _preprocess_data(self, X, y=None, fit=False):
        if fit:
            y = self.class_encoder.fit_transform(y)
            X = self.feat_vectorizer.fit_transform(X)
            if self._feat_scaler is not None:
                X = self._feat_scaler.fit_transform(X)
            if self._feat_selector is not None:
                X = self._feat_selector.fit_transform(X, y)
        else:
            X = self.feat_vectorizer.transform(X)
            if self._feat_scaler is not None:
                X = self._feat_scaler.transform(X)
            if self._feat_selector is not None:
                X = self._feat_selector.transform(X)

        return X, y
