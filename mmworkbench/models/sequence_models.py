# -*- coding: utf-8 -*-
"""This module contains the Memm entity recognizer."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from builtins import range, super

import logging
import random

from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression

from . import tagging
from .helpers import get_feature_extractor, register_model
from .model import ModelConfig, SkLearnModel

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = {
    'bag-of-words-seq': {
        'ngram_lengths_to_start_positions': {
            1: [-2, -1, 0, 1, 2],
            2: [-2, -1, 0, 1]
        }
    },
    'in-gaz-span-seq': {},
    'sys-candidates-seq': {
        'start_positions': [-1, 0, 1]
    }
}


class MemmModel(SkLearnModel):
    """A maximum-entropy Markov model."""

    def __init__(self, config):
        if not config.features:
            config_dict = config.to_dict()
            config_dict['features'] = DEFAULT_FEATURES
            config = ModelConfig(**config_dict)

        super().__init__(config)

        # Default tag scheme to IOB
        self._tag_scheme = self.config.model_settings.get('tag_scheme', 'IOB').upper()

        self._no_entities = False

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

    def _extract_features(self, example):
        """Extracts feature dicts for each token in an example"""

        feat_seq = []
        example_type = self.config.example_type
        for name, kwargs in self.config.features.items():
            if callable(kwargs):
                # a feature extractor function was passed in directly
                feat_extractor = kwargs
            else:
                feat_extractor = get_feature_extractor(example_type, name)(**kwargs)

            update_feat_seq = feat_extractor(example, self._resources)
            if not feat_seq:
                feat_seq = update_feat_seq
            else:
                for idx, features in enumerate(update_feat_seq):
                    feat_seq[idx].update(features)

        return feat_seq

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
                    segment['prev_tag'] = tagging.START_TAG
                elif fit:
                    segment['prev_tag'] = y[y_offset + j - 1]

            y_offset += len(features_by_segment)
        X, y = self._preprocess_data(feats, y, fit=fit)
        return X, y, groups

    def fit(self, examples, labels, params=None):
        """Trains the model

        Args:
            labeled_queries (list of ProcessedQuery): a list of queries to train on
            entity_types (list): entity types as a filter (defaults to all)
            verbose (boolean): show more debug/diagnostic output

        """
        system_types = self._get_system_types()
        self.register_resources(sys_types=system_types)

        skip_param_selection = params is not None or self.config.param_selection is None
        params = params or self.config.params

        # Shuffle to prevent order effects
        indices = list(range(len(labels)))
        random.shuffle(indices)
        examples = [examples[i] for i in indices]
        labels = [labels[i] for i in indices]

        # TODO: add this code back in
        # distinct_labels = set(labels)
        # if len(set(distinct_labels)) <= 1:
        #     return None

        # Extract features and classes
        y = self._label_encoder.encode(labels, examples=examples)

        if len(set(y)) == 1:
            self._no_entities = True
            return self

        X, y, groups = self.get_feature_matrix(examples, y, fit=True)

        if skip_param_selection:
            self._clf = self._fit(X, y, params)
            self._current_params = params
        else:
            # run cross validation to select params
            best_clf, best_params = self._fit_cv(X, y, groups)
            self._clf = best_clf
            self._current_params = best_params

        return self

    def predict(self, examples):
        labels = []
        if self._no_entities:
            return self._label_encoder.decode([[] for e in examples], examples=examples)
        return [self._predict_example(example) for example in examples]

    def _predict_example(self, example):
        features_by_segment = self._extract_features(example)
        if len(features_by_segment) == 0:
            return self._label_encoder.decode([], examples=[example])[0]

        predicted_tags = []
        prev_tag = tagging.START_TAG
        for features in features_by_segment:
            features['prev_tag'] = prev_tag
            X, _ = self._preprocess_data([features])
            prediction = self._clf.predict(X)
            predicted_tag = self._class_encoder.inverse_transform(prediction)[0]
            predicted_tags.append(predicted_tag)
            prev_tag = predicted_tag

        return self._label_encoder.decode([predicted_tags], examples=[example])[0]

    def predict_proba(self, examples):
        # TODO: implement this
        raise NotImplementedError

    def predict_log_proba(self, examples):
        # TODO: implement this
        raise NotImplementedError

    def _convert_params(self, param_grid, y):
        return param_grid

    def _get_system_types(self):
        sys_types = set()
        for gaz in self._resources['gazetteers'].values():
            sys_types.update(gaz['sys_types'])
        return sys_types

    def _get_feature_selector(self):
        """Get a feature selector instnace based on the feature_selector model
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

    def _get_model_class(self):
        """Returns the class of the actual underlying model"""
        return LogisticRegression


register_model('memm', MemmModel)
