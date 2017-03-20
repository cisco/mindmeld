# -*- coding: utf-8 -*-
"""This module contains base classes for models defined in the models subpackage."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object, super

from collections import namedtuple
import logging
import math
import random

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import (KFold, GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit)
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.metrics import (f1_score, precision_recall_fscore_support as score)

from .helpers import get_feature_extractor, get_label_encoder
from .tagging import get_tags_from_entities, get_entities_from_tags
logger = logging.getLogger(__name__)

# model scoring types
ACCURACY_SCORING = 'accuracy'
LIKELIHOOD_SCORING = 'log_loss'

_NEG_INF = -1e10


class ModelConfig(object):
    """A value object representing a model configuration

    Attributes:
        model_type (str): The name of the model type. Will be used to find the
            model class to instantiate
        example_type (str): The type of the examples which will be passed into
            `fit()` and `predict()`. Used to select feature extractors
        label_type (str): The type of the labels which will be passed into
            `fit()` and returned by `predict()`. Used to select the label
            encoder
        model_settings (dict): Settings specific to the model type specified
        params (dict): Params to pass to the underlying classifier
        param_selection (dict): Configuration for param selection (using cross
            validation)
            {'type': 'shuffle',
             'n': 3,
             'k': 10,
             'n_jobs': 2,
             'scoring': '',
             'grid': {}
            }
        features (dict): The keys are the names of feature extractors and the
            values are either a kwargs dict which will be passed into the
            feature extractor function, or a callable which will be used as to
            extract features

    """
    __slots__ = ['model_type', 'example_type', 'label_type', 'features', 'model_settings', 'params',
                 'param_selection']

    def __init__(self, model_type=None, example_type=None, label_type=None, features=None,
                 model_settings=None, params=None, param_selection=None):
        for arg, val in {'model_type': model_type, 'example_type': example_type,
                         'label_type': label_type, 'features': features}.items():
            if val is None:
                raise TypeError('__init__() missing required argument {!r}'.format(arg))
        if params is None and (param_selection is None or param_selection.get('grid') is None):
            raise ValueError("__init__() One of 'params' and 'param_selection' is required")
        self.model_type = model_type
        self.example_type = example_type
        self.label_type = label_type
        self.features = features
        self.model_settings = model_settings
        self.params = params
        self.param_selection = param_selection

    def to_dict(self):
        """Converts the model config object into a dict

        Returns:
            dict: A dict version of the config
        """
        result = {}
        for attr in self.__slots__:
            result[attr] = getattr(self, attr)
        return result

    def __repr__(self):
        args_str = ', '.join("{}={!r}".format(key, getattr(self, key)) for key in self.__slots__)
        return "{}({})".format(self.__class__.__name__, args_str)


class EvaluatedExample(namedtuple('EvaluatedExample', ['example', 'gold', 'pred',
                                                       'probas'])):
    """Represents the evaluation of a single example

    Attributes:
        example: The example being evaluated
        gold: The expected label for the example
        pred: The predicted label for the example
        proba (dict): Maps labels to their predicted probabilities
    """

    @property
    def is_correct(self):
        return self.gold == self.pred


class ModelEvaluation(namedtuple('ModelEvaluation', ['config', 'results'])):
    """Reprepresents the evaluation of a model at a specific configuration
    using a collection of examples and labels

    Attributes:
        config (ModelConfig): The model config used during evaluation
        results (list of EvaluatedExample): A list of the evaluated examples

    TODO: update to test_results and train_results to calculate bias and variance
    """
    def __init__(self, config, results):
        self.RawResults = namedtuple('RawResults', ['pred', 'gold', 'label_mappings',
                                                    'numeric_labels', 'text_labels',
                                                    'pred_flat', 'gold_flat'])
        self.label_encoder = get_label_encoder(config, config.model_settings['classifier_type'])

    def _make_raw_result(self, pred, gold, label_mappings, numeric_labels, text_labels,
                         pred_flat=None, gold_flat=None):
        return self.RawResults(pred=pred, gold=gold, label_mappings=label_mappings,
                               numeric_labels=numeric_labels, text_labels=text_labels,
                               pred_flat=pred_flat, gold_flat=gold_flat)

    def get_accuracy(self):
        """The accuracy represents the share of examples whose predicted labels
        exactly matched their expected labels.

        Returns:
            float: The accuracy of the model
        """
        num_examples = len(self.results)
        num_correct = len([e for e in self.results if e.is_correct])
        return float(num_correct) / float(num_examples)

    def __repr__(self):
        num_examples = len(self.results)
        num_correct = len(list(self.correct_results()))
        accuracy = self.get_accuracy()
        msg = "<{} score: {:.2%}, {} of {} example{} correct>"
        return msg.format(self.__class__.__name__, accuracy, num_correct, num_examples,
                          '' if num_examples == 1 else 's')

    def correct_results(self):
        """
        Returns:
            iterable: Collection of the examples which were correct
        """
        for result in self.results:
            if result.is_correct:
                yield result

    def incorrect_results(self):
        """
        Returns:
            iterable: Collection of the examples which were incorrect
        """
        for result in self.results:
            if not result.is_correct:
                yield result

    def stats(self):
        """
        Returns statistics for evaluation.
        """
        raise NotImplementedError

    def graphs(self):
        """
        Returns graphs for evaluation.
        """
        raise NotImplementedError

    def raw_results(self):
        """
        Returns raw vectors of the predictions for data scientists to use for any additional
        evaluation metrics or generate graphs of their choice
        """
        raise NotImplementedError

    def _update_raw_result(self, label, label_mappings, val, vec):
        """
        Helper method for updating the text to numeric label mappings and numeric label vectors
        """
        if label_mappings.get(label) is None:
            label_mappings[label] = val
            val += 1
        vec.append(label_mappings[label])
        return label_mappings, val, vec

    def _print_label_stats_table(self, stats, labels, label_mappings):
        title_format = "{:>30}" + "{:>15}" * (len(stats))
        stat_row_format = "{:>30}" + "{:>15.3f}" * (len(stats))
        table_titles = stats.keys()
        print("Statistics by Class: \n")
        print(title_format.format("label", *table_titles))
        for label in labels:
            row = []
            for stat in table_titles:
                row.append(stats[stat][label-1])
            print(stat_row_format.format(label_mappings[label], *row))
        print("\n\n")

    def _print_overall_stats_table(self, stats_overall):
        title_format = "{:>15}" * (len(stats_overall))
        stat_row_format = "{:>15.3f}" * (len(stats_overall))
        table_titles = stats_overall.keys()
        print("Overall Statistics: \n")
        print(title_format.format(*table_titles))
        row = []
        for stat in table_titles:
            row.append(stats_overall[stat])
        print(stat_row_format.format(*row))
        print("\n\n")


class StandardModelEvaluation(ModelEvaluation):
    def raw_results(self):
        """
        Returns raw vectors of the predictions for data scientists to use for any additional
        evaluation metrics or generate graphs of their choice

        Returns:
            namedtuple of:
                pred: vector of predicted classes (numeric value)
                gold: vector of gold classes (numeric value)
                label_mappings: two way dict of the text label to numeric value
                numeric_labels: a list of all the numeric label values
                text_labels: a list of all the text label values

        TODO: check if numeric mappings can be passed in from the model prediction code/parser
              instead of re-generated here
        """
        label_mappings, val = {}, 1
        pred, gold = [], []

        for result in self.results:
            label_mappings, val, pred = self._update_raw_result(result.pred, label_mappings, val,
                                                                pred)
            label_mappings, val, gold = self._update_raw_result(result.gold, label_mappings, val,
                                                                gold)

        text_labels = label_mappings.keys()
        label_mappings.update(dict(reversed(item) for item in label_mappings.items()))
        return self._make_raw_result(pred=pred, gold=gold, label_mappings=label_mappings,
                                     numeric_labels=range(1, val), text_labels=text_labels)

    def stats(self):
        """
        Prints a useful stats table and returns a structured stats object for evaluation

        Returns:
            A tuple of dictionaries. One of overall stats and one of class specific stats.
            Contains precision, recall, f scores, support, bias, and variance

        TODO: calculate bias and variance
        """
        raw_results = self.raw_results()
        labels = raw_results.numeric_labels
        precision, recall, f_beta, support = score(y_true=raw_results.gold, y_pred=raw_results.pred,
                                                   labels=labels)

        stats = {
                    'precision': precision,
                    'recall': recall,
                    'f_beta': f_beta,
                    'support': support,
                }
        self._print_label_stats_table(stats, labels, raw_results.label_mappings)

        f1_weighted = f1_score(y_true=raw_results.gold, y_pred=raw_results.pred, labels=labels,
                               average='weighted')
        f1_macro = f1_score(y_true=raw_results.gold, y_pred=raw_results.pred, labels=labels,
                            average='macro')
        f1_micro = f1_score(y_true=raw_results.gold, y_pred=raw_results.pred, labels=labels,
                            average='micro')

        stats_overall = {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
        self._print_overall_stats_table(stats_overall)

        return stats_overall, stats

    def graphs(self):
        """
        Generates some useful graphs based on the evaluation data
        TODO generate graphs from matplotlib/scikit learn
        """
        return None


class SequenceModelEvaluation(ModelEvaluation):
    def raw_results(self):
        """
        TODO: role evaluation?
        """
        label_mappings, val = {}, 1
        pred, gold = [], []
        pred_flat, gold_flat = [], []

        for result in self.results:
            raw_pred = self.label_encoder.encode([result.pred], examples=[result.example])
            raw_gold = self.label_encoder.encode([result.gold], examples=[result.example])

            vec = []
            for entity in raw_pred:
                label_mappings, val, vec = self._update_raw_result(entity, label_mappings, val, vec)
            pred.append(vec)
            pred_flat.extend(vec)
            vec = []
            for entity in raw_gold:
                label_mappings, val, vec = self._update_raw_result(entity, label_mappings, val, vec)
            gold.append(vec)
            gold_flat.extend(vec)

        text_labels = label_mappings.keys()
        label_mappings.update(dict(reversed(item) for item in label_mappings.items()))
        return self._make_raw_result(pred=pred, gold=gold, label_mappings=label_mappings,
                                     numeric_labels=range(1, val), text_labels=text_labels,
                                     pred_flat=pred_flat, gold_flat=gold_flat)

    def stats(self):
        """
        Prints a useful stats table and returns a structured stats object for evaluation

        Returns:
            A tuple of dictionaries. One of overall stats and one of class specific stats.
            Contains precision, recall, f scores, support, bias, and variance

        TODO: calculate bias and variance
        """
        raw_results = self.raw_results()

        labels = raw_results.numeric_labels
        precision, recall, f_beta, support = score(y_true=raw_results.gold_flat,
                                                   y_pred=raw_results.pred_flat,
                                                   labels=labels)

        stats = {
                    'precision': precision,
                    'recall': recall,
                    'f_beta': f_beta,
                    'support': support,
                }
        self._print_label_stats_table(stats, labels, raw_results.label_mappings)

        f1_weighted = f1_score(y_true=raw_results.gold_flat, y_pred=raw_results.pred_flat,
                               labels=labels, average='weighted')
        f1_macro = f1_score(y_true=raw_results.gold_flat, y_pred=raw_results.pred_flat,
                            labels=labels, average='macro')
        f1_micro = f1_score(y_true=raw_results.gold_flat, y_pred=raw_results.pred_flat,
                            labels=labels, average='micro')

        stats_overall = {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
        self._print_overall_stats_table(stats_overall)

        return stats_overall, stats

    def graphs(self):
        """
        Generates some useful graphs based on the evaluation data
        TODO generate graphs from matplotlib/scikitlearn
        """
        return None


class Model(object):
    """An abstract class upon which all models are based.

    Attributes:
        config (ModelConfig): The configuration for the model
    """
    DEFAULT_CV_SCORING = ACCURACY_SCORING

    def __init__(self, config):
        self.config = config
        self._label_encoder = get_label_encoder(self.config, self.__class__.__name__)
        self._current_params = None
        self._resources = {}
        self._clf = None
        self.cv_loss_ = None

    def fit(self, examples, labels, params=None):
        raise NotImplementedError

    def select_params(self, examples, labels, selection_settings=None):
        raise NotImplementedError

    def _convert_params(self, param_grid, y):
        """Convert the params from the style given by the config to the style
        passed in to the actual classifier.

        Args:
            params_grid (dict): lists of classifier parameter values, keyed by
                parameter name

        Returns:
            (dict): revised params_grid
        """
        raise NotImplementedError

    def predict(self, examples):
        raise NotImplementedError

    def predict_proba(self, examples):
        raise NotImplementedError

    def predict_log_proba(self, examples):
        raise NotImplementedError

    def evaluate(self, examples, labels):
        raise NotImplementedError

    def _get_effective_config(self):
        """Create a model config object for the current effective config (after
        param selection)

        Returns:
            ModelConfig
        """
        config_dict = self.config.to_dict()
        config_dict.pop('param_selection')
        config_dict['params'] = self._current_params
        return ModelConfig(**config_dict)

    def register_resources(self, **kwargs):
        """Registers resources which are accessible to feature extractors

        Args:
            **kwargs: dictionary of resources to register

        """
        self._resources.update(kwargs)

    def get_feature_matrix(self, examples, y=None, fit=False):
        raise NotImplementedError

    def _extract_features(self, example):
        """Gets all features from an example.

        Args:
            example: An example object.

        Returns:
            (dict of str: number): A dict of feature names to their values.
        """
        example_type = self.config.example_type
        feat_set = {}
        for name, kwargs in self.config.features.items():
            if callable(kwargs):
                # a feature extractor function was passed in directly
                feat_extractor = kwargs
            else:
                feat_extractor = get_feature_extractor(example_type, name)(**kwargs)
            feat_set.update(feat_extractor(example, self._resources))
        return feat_set

    def _get_cv_iterator(self, settings):
        if not settings:
            return None
        cv_type = settings['type']
        try:
            cv_iterator = {"k-fold": self._k_fold_iterator,
                           "shuffle": self._shuffle_iterator,
                           "group-k-fold": self._groups_k_fold_iterator,
                           "group-shuffle": self._groups_shuffle_iterator,
                           "stratified-k-fold": self._stratified_k_fold_iterator,
                           "stratified-shuffle": self._stratified_shuffle_iterator,
                           }.get(cv_type)(settings)
        except KeyError:
            raise ValueError('Unknown param selection type: {!r}'.format(cv_type))

        return cv_iterator

    @staticmethod
    def _k_fold_iterator(settings):
        k = settings['k']
        return KFold(n_splits=k)

    @staticmethod
    def _shuffle_iterator(settings):
        k = settings['k']
        n = settings.get('n', k)
        test_size = 1.0 / k
        return ShuffleSplit(n_splits=n, test_size=test_size)

    @staticmethod
    def _groups_k_fold_iterator(settings):
        k = settings['k']
        return GroupKFold(n_splits=k)

    @staticmethod
    def _groups_shuffle_iterator(settings):
        k = settings['k']
        n = settings.get('n', k)
        test_size = 1.0 / k
        return GroupShuffleSplit(n_splits=n, test_size=test_size)

    @staticmethod
    def _stratified_k_fold_iterator(settings):
        k = settings['k']
        return StratifiedKFold(n_splits=k)

    @staticmethod
    def _stratified_shuffle_iterator(settings):
        k = settings['k']
        n = settings.get('n', k)
        test_size = 1.0 / k
        return StratifiedShuffleSplit(n_splits=n, test_size=test_size)

    def requires_resource(self, resource):
        example_type = self.config.example_type
        for name, kwargs in self.config.features.items():
            if callable(kwargs):
                # a feature extractor function was passed in directly
                feature_extractor = kwargs
            else:
                feature_extractor = get_feature_extractor(example_type, name)
            if ('requirements' in feature_extractor.__dict__ and
                    resource in feature_extractor.requirements):
                return True
        return False


class SkLearnModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self._class_encoder = SKLabelEncoder()
        self._feat_vectorizer = DictVectorizer()
        self._feat_selector = self._get_feature_selector()
        self._feat_scaler = self._get_feature_scaler()

    def fit(self, examples, labels, params=None):
        """Trains this model.

        This method inspects instance attributes to determine the classifier
        object and cross-validation strategy, and then fits the model to the
        training examples passed in.

        Args:
            examples (list): A list of examples.
            labels (list): A parallel list to examples. The gold labels
                for each example.
            params (dict, optional): Parameters to use when training. Parameter
                selection will be bypassed if this is provided

        Returns:
            (SkLearnModel): Returns self to match classifier scikit-learn
                interfaces.
        """
        skip_param_selection = params is not None or self.config.param_selection is None
        params = params or self.config.params

        # Prepare resources

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
        y = self._label_encoder.encode(labels)
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

    def select_params(self, examples, labels, selection_settings=None):
        y = self._label_encoder.encode(labels)
        X, y, groups = self.get_feature_matrix(examples, y, fit=True)
        clf, params = self._fit_cv(X, y, groups, selection_settings)
        self._clf = clf
        return params

    def _fit(self, X, y, params):
        """Trains a classifier without cross-validation.

        Args:
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.
            params (dict): Parameters of the classifier

        """
        params = self._convert_params(params, y)
        model_class = self._get_model_constructor()
        return model_class(**params).fit(X, y)

    def _fit_cv(self, X, y, groups=None, selection_settings=None):
        """Summary

        Args:
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.
            selection_settings (None, optional): Description

        """
        selection_settings = selection_settings or self.config.param_selection
        cv_iterator = self._get_cv_iterator(selection_settings)

        if selection_settings is None:
            return self._fit(X, y, self.config.params), self.config.params

        cv_type = selection_settings['type']
        num_splits = cv_iterator.get_n_splits(X, y, groups)
        logger.info('Selecting hyperparameters using %s cross validation with %s split%s', cv_type,
                    num_splits, '' if num_splits == 1 else 's')

        scoring = selection_settings.get('scoring', self.DEFAULT_CV_SCORING)
        n_jobs = selection_settings.get('n_jobs', -1)

        param_grid = self._convert_params(selection_settings['grid'], y)
        model_class = self._get_model_constructor()

        grid_cv = GridSearchCV(estimator=model_class(), scoring=scoring, param_grid=param_grid,
                               cv=cv_iterator, n_jobs=n_jobs)
        model = grid_cv.fit(X, y, groups)

        for idx, params in enumerate(model.cv_results_['params']):
            logger.debug('Candidate parameters: {}'.format(params))
            std_err = 2.0 * model.cv_results_['std_test_score'][idx] / math.sqrt(model.n_splits_)
            if scoring == ACCURACY_SCORING:
                msg = 'Candidate average accuracy: {:.2%} ± {:.2%}'
            elif scoring == LIKELIHOOD_SCORING:
                msg = 'Candidate average log likelihood: {:.4} ± {:.4}'
            logger.debug(msg.format(model.cv_results_['mean_test_score'][idx], std_err))

        if scoring == ACCURACY_SCORING:
            msg = 'Best accuracy: {:.2%}, params: {}'
            self.cv_loss_ = 1 - model.best_score_
        elif scoring == LIKELIHOOD_SCORING:
            msg = 'Best log likelihood: {:.4}, params: {}'
            self.cv_loss_ = - model.best_score_
        logger.info(msg.format(model.best_score_, model.best_params_))

        return model.best_estimator_, model.best_params_

    def predict(self, examples):
        X, _, _ = self.get_feature_matrix(examples)
        y = self._clf.predict(X)
        predictions = self._class_encoder.inverse_transform(y)
        return self._label_encoder.decode(predictions)

    def predict_proba(self, examples):
        X, _, _ = self.get_feature_matrix(examples)
        return self._predict_proba(X, self._clf.predict_proba)

    def predict_log_proba(self, examples):
        X, _, _ = self.get_feature_matrix(examples)
        predictions = self._predict_proba(X, self._clf.predict_log_proba)

        # JSON can't reliably encode infinity, so replace it with large number
        for row in predictions:
            _, probas = row
            for label, proba in probas.items():
                if proba == -np.Infinity:
                    probas[label] = _NEG_INF
        return predictions

    def _predict_proba(self, X, predictor):
        predictions = []
        for row in predictor(X):
            class_index = row.argmax()
            probabilities = {}
            top_class = None
            for class_index, proba in enumerate(row):
                raw_class = self._class_encoder.inverse_transform([class_index])[0]
                decoded_class = self._label_encoder.decode([raw_class])[0]
                probabilities[decoded_class] = proba
                if proba > probabilities.get(top_class, -1.0):
                    top_class = decoded_class
            predictions.append((top_class, probabilities))

        return predictions

    def _get_model_constructor(self):
        """Returns the Python class of the actual underlying model"""
        raise NotImplementedError

    def _get_feature_selector(self):
        """Get a feature selector instance based on the feature_selector model
        parameter.

        Returns:
            (Object): a feature selector which returns a reduced feature matrix,
                given the full feature matrix (X) and the class labels (y)
        """
        raise NotImplementedError

    def _get_feature_scaler(self):
        """Get a feature value scaler based on the model settings"""
        if self.config.model_settings is None:
            scale_type = None
        else:
            scale_type = self.config.model_settings.get('feature_scaler')
        scaler = {'std-dev': StandardScaler(with_mean=False),
                  'max-abs': MaxAbsScaler()}.get(scale_type)
        return scaler

    def get_feature_matrix(self, examples, y=None, fit=False):
        """Transforms a list of examples into a feature matrix.

        Args:
            examples (list): The examples.

        Returns:
            (numpy.matrix): The feature matrix.
            (numpy.array): The group labels for examples.
        """
        groups = []
        feats = []
        for idx, example in enumerate(examples):
            feats.append(self._extract_features(example))
            groups.append(idx)

        X, y = self._preprocess_data(feats, y, fit=fit)
        return X, y, groups

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


class LabelEncoder(object):
    """The label encoder is responsible for converting between rich label
    objects such as a ProcessedQuery and basic formats a model can interpret.

    A workbench model use its label encoder at fit time to encode labels into a
    form it can deal with, and at predict time to decode predictions into
    objects
    """
    def __init__(self, config):
        """Initializes an encoder

        Args:
            config (ModelConfig): The model
        """
        self.config = config

    def encode(self, labels, **kwargs):
        """Transforms a list of label objects into a vector of classes.


        Args:
            labels (list): A list of labels to encode
        """
        return labels

    def decode(self, classes, **kwargs):
        """Decodes a vector of classes into a list of labels

        Args:
            classes (list): A list of classes

        Returns:
            list: The decoded labels
        """
        return classes


class EntityLabelEncoder(LabelEncoder):

    def _get_tag_scheme(self):
        return self.config.model_settings.get('tag_scheme', 'IOB').upper()

    def encode(self, labels, **kwargs):
        examples = kwargs['examples']
        scheme = self._get_tag_scheme()
        # Here each label is a list of entities for the corresponding example
        all_tags = []
        for idx, label in enumerate(labels):
            all_tags.extend(get_tags_from_entities(examples[idx], label, scheme))
        return all_tags

    def decode(self, tags_by_example, **kwargs):
        # TODO: support decoding multiple queries at once
        scheme = self._get_tag_scheme()
        examples = kwargs['examples']
        labels = [get_entities_from_tags(examples[idx], tags, scheme)
                  for idx, tags in enumerate(tags_by_example)]
        return labels
