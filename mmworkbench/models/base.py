# -*- coding: utf-8 -*-
"""
This module defines the interface for entity recognition.
"""
from __future__ import division, unicode_literals
from builtins import object, zip
from past.utils import old_div

import itertools
import logging

from sklearn import cross_validation as cross_val
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

logger = logging.getLogger(__name__)


class BaseEntityRecognizer(object):
    """A machine learning recognizer for entities.

    This BaseEntityRecognizer manages feature extraction, training, cross-validation,
    and prediction.
    """

    def __init__(self, params_grid=None, cv=None, model_settings=None, features=None):

        self._resources = {}
        self.model_parameter_choices = params_grid
        self._model_parameters = {}
        self.cross_validation_settings = cv
        self.classifier_settings = model_settings or {}
        self.feat_specs = features

    @classmethod
    def from_config(cls, config, model_configuration):
        """Initializes a BaseEntityRecognizer instance from config file data.

        Args:
            config (dict): The Python representation of a parsed config file.
            model_configuration (str): One of the keys in config['models'].

        Returns:
            (BaseEntityRecognizer): A BaseEntityRecognizer instance initialized with the
                settings from the config entry given by model_configuration.
        """
        model_config_entry = config['models'].get(model_configuration)
        if not model_config_entry:
            error_msg = "Model config does not contain a model named '{}'"
            raise ValueError(error_msg.format(model_configuration))

        params = model_config_entry.get('params_grid')
        cv_settings = model_config_entry.get('cv')
        clf_settings = model_config_entry.get('model_settings')
        features = model_config_entry.get('features')

        model_type = model_config_entry.get('model_type',
                                            model_config_entry.get('classifier_type'))
        if model_type == 'memm':
            from . import memm
            return memm.MemmModel(params, cv_settings, clf_settings, features)
        else:
            raise ValueError

    def register_resources(self, **kwargs):
        """

        Args:
            **kwargs: dictionary of resources to load

        """
        self._resources.update(kwargs)

    def fit(self, labeled_queries, entity_types=None, verbose=False):
        """

        Args:
            labeled_queries (list): Query objects with annotations
            entity_types (list): entity types as a filter (defaults to all)
            verbose (boolean): show more debug/diagnostic output
        """
        raise NotImplementedError

    def predict(self, query, entity_types=None, verbose=False):
        """

        Args:
            query (Query): Query object to apply model to
            entity_types (list): entity typeas as a filter (defaults to all)
            verbose (boolean): show more debug/diagnostic output

        Returns:
            (tuple): (non-numeric_entities, numeric_entities)
        """
        raise NotImplementedError

    def _iter_settings(self):
        """Iterates through all classifier settings.

        Yields:
            (dict): A kwargs dict to be passed to the classifier object.
                Each item yielded is a unique choice of settings from
                self.model_parameter_choices.
        """
        if self.model_parameter_choices is not None:
            gsh_keys, gsh_values = list(zip(*list(self.model_parameter_choices.items())))
            for settings in itertools.product(*gsh_values):
                yield dict(zip(gsh_keys, settings))
        else:
            yield {}

    def get_cv_iterator(self):
        """Get a cv_iterator instance based on the cross-validation-settings,
        defaulting to None if the 'type' parameter is not found or unknown.

        Returns:
            (Object): a cross_validation iterator that takes a label list as a
            single parameter. In the case of the by-query iterators, the labels
            are interpreted as query group labels; otherwise they don't matter.
        """
        if self.cross_validation_settings is None:
            cv_type = None
        else:
            cv_type = self.cross_validation_settings.get('type')
        cv_iterator = {"k-fold": self._groups_k_fold_iterator,
                       "shuffle": self._groups_shuffle_iterator,
                       "stratified-k-fold": self._stratified_k_fold_iterator,
                       "stratified-shuffle": self._stratified_shuffle_iterator,
                       "total-shuffle": self._shuffle_iterator}.get(cv_type)
        return cv_iterator

    def _shuffle_iterator(self, groups):
        k = self.cross_validation_settings['k']
        n = self.cross_validation_settings.get('n', k)
        return cross_val.ShuffleSplit(len(groups), n_iter=n, test_size=old_div(1.0, k))

    def _groups_k_fold_iterator(self, groups):
        k = self.cross_validation_settings['k']
        return cross_val.LabelKFold(groups, n_folds=k)

    def _groups_shuffle_iterator(self, groups):
        k = self.cross_validation_settings['k']
        n = self.cross_validation_settings.get('n', k)
        return cross_val.LabelShuffleSplit(groups, n_iter=n, test_size=old_div(1.0, k))

    def _stratified_k_fold_iterator(self, groups):
        k = self.cross_validation_settings['k']
        return cross_val.StratifiedKFold(groups, n_folds=k)

    def _stratified_shuffle_iterator(self, groups):
        k = self.cross_validation_settings['k']
        n = self.cross_validation_settings.get('n', k)
        return cross_val.StratifiedShuffleSplit(groups, n_iter=n, test_size=old_div(1.0, k))

    def get_feature_selector(self):
        """Get a feature selector instance based on the feature-selector model
        parameter.

        Returns:
            (Object): a feature selector which returns a reduced feature matrix,
                given the full feature matrix (X) and the class labels (y)
        """
        if self.classifier_settings is None:
            fs_type = None
        else:
            fs_type = self.classifier_settings.get('feature-selector')
        selector = {'l1': SelectFromModel(LogisticRegression(penalty='l1', C=1)),
                    'f': SelectPercentile()
                    }.get(fs_type)

        return selector

    def get_feature_scaler(self):
        """Get a feature value scaler based on the classifier settings

        """
        if self.classifier_settings is None:
            scale_type = None
        else:
            scale_type = self.classifier_settings.get('feature-scaler')
        scaler = {'std-dev': StandardScaler(with_mean=False),
                  'max-abs': MaxAbsScaler()}.get(scale_type)
        return scaler
