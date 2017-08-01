# coding=utf-8
"""
This module contains all code required to perform multinomial classification
of text.
"""
from __future__ import absolute_import, division, unicode_literals

import logging
import math
import random
from builtins import range, super, zip
from collections import Counter

import numpy as np
from numpy import bincount
from past.utils import old_div
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn_crfsuite.metrics import sequence_accuracy_score

from .helpers import (GAZETTEER_RSC, QUERY_FREQ_RSC, WORD_FREQ_RSC,
                      mask_numerics, register_model)
from .model import EvaluatedExample, Model

_NEG_INF = -1e10

# classifier types
LOG_REG_TYPE = "logreg"
DECISION_TREE_TYPE = "dtree"
RANDOM_FOREST_TYPE = "rforest"
SVM_TYPE = "svm"
SUPER_LEARNER_TYPE = "super-learner"
BASE_MODEL_TYPES = [LOG_REG_TYPE, DECISION_TREE_TYPE, RANDOM_FOREST_TYPE, SVM_TYPE]

# model scoring types
ACCURACY_SCORING = "accuracy"
LIKELIHOOD_SCORING = "log_loss"


logger = logging.getLogger(__name__)


def evaluate_seq2seq(y, y_pred, **kwargs):
    error = 0
    for idx in range(len(y)):
        if y[idx] != y_pred[idx]:
            error += 1
    accuracy = 1 - error / float(len(y))
    print(accuracy)
    return 1 - error / float(len(y))


class TextModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self._class_encoder = SKLabelEncoder()
        self._feat_vectorizer = DictVectorizer()
        self._feat_selector = self._get_feature_selector()
        self._feat_scaler = self._get_feature_scaler()
        self._meta_type = None
        self._meta_feat_vectorizer = DictVectorizer(sparse=False)
        self._base_clfs = {}
        self.cv_loss_ = None
        self.train_acc_ = None

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior.
        """
        attributes = self.__dict__.copy()
        attributes['_resources'] = {WORD_FREQ_RSC: self._resources.get(WORD_FREQ_RSC, {}),
                                    QUERY_FREQ_RSC: self._resources.get(QUERY_FREQ_RSC, {})}
        return attributes

    def _get_model_constructor(self):
        """Returns the class of the actual underlying model"""
        classifier_type = self.config.model_settings['classifier_type']
        try:
            return {LOG_REG_TYPE: LogisticRegression,
                    DECISION_TREE_TYPE: DecisionTreeClassifier,
                    RANDOM_FOREST_TYPE: RandomForestClassifier,
                    SVM_TYPE: SVC}[classifier_type]
        except KeyError:
            msg = '{}: Classifier type {!r} not recognized'
            raise ValueError(msg.format(self.__class__.__name__, classifier_type))

    def evaluate(self, examples, labels):
        """Evaluates a model against the given examples and labels

        Args:
            examples: A list of examples to predict
            labels: A list of expected labels

        Returns:
            ModelEvaluation: an object containing information about the
                evaluation
        """
        # TODO: also expose feature weights?
        predictions = self.predict_proba(examples)

        # Create a model config object for the current effective config (after param selection)
        config = self._get_effective_config()

        evaluations = [EvaluatedExample(e, labels[i], predictions[i][0], predictions[i][1],
                       config.label_type) for i, e in enumerate(examples)]

        model_eval = TextModel(config, evaluations)
        return model_eval

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
        """Trains this TextModel.

        This method inspects instance attributes to determine the classifier
        object and cross-validation strategy, and then fits the model to the
        training examples passed in.

        Args:
            queries (list of Query): A list of queries.
            classes (list of str): A parallel list to queries. The gold labels
                for each query.
            verbose (bool): Whether to show analysis output

        Returns:
            (TextModel): Returns self to match classifier scikit-learn
                interfaces.
        """
        # Prepare resources
        if self._resources.get(WORD_FREQ_RSC) is None and self.requires_resource(WORD_FREQ_RSC):
            self.compile_word_freq_dict(examples)
        if self._resources.get(QUERY_FREQ_RSC) is None and self.requires_resource(QUERY_FREQ_RSC):
            self.compile_query_freq_dict(examples)

        params = params or self.config.params
        skip_param_selection = params is not None or self.config.param_selection is None

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
        params = self._convert_params(params, y, is_grid=False)
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

        crf_scoring = make_scorer(score_func=sequence_accuracy_score)
        scoring = crf_scoring
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
            else:
                msg = 'Candidate average seq2se2 accuracy: {:.2%} ± {:.2%}'
            logger.info(msg.format(model.cv_results_['mean_test_score'][idx], std_err))

        if scoring == ACCURACY_SCORING:
            msg = 'Best accuracy: {:.2%}, params: {}'
            self.cv_loss_ = 1 - model.best_score_
        elif scoring == LIKELIHOOD_SCORING:
            msg = 'Best log likelihood: {:.4}, params: {}'
            self.cv_loss_ = - model.best_score_
        else:
            msg = 'Best seq2seq accuracy: {:.2%}, params: {}'
            self.cv_loss_ = 1 - model.best_score_
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

    def _convert_params(self, param_grid, y, is_grid=True):
        """
        Convert the params from the style given by the config to the style
        passed in to the actual classifier.

        Args:
            param_grid (dict): lists of classifier parameter values, keyed by parameter name

        Returns:
            (dict): revised param_grid
        """
        if 'class_weight' in param_grid:
            raw_weights = param_grid['class_weight'] if is_grid else [param_grid['class_weight']]
            weights = [{k if isinstance(k, int) else self._class_encoder.transform((k,))[0]: v
                        for k, v in cw_dict.items()} for cw_dict in raw_weights]
            param_grid['class_weight'] = weights if is_grid else weights[0]
        elif 'class_bias' in param_grid:
            # interpolate between class_bias=0 => class_weight=None
            # and class_bias=1 => class_weight='balanced'
            class_count = bincount(y)
            classes = self._class_encoder.classes_
            weights = []
            raw_bias = param_grid['class_bias'] if is_grid else [param_grid['class_bias']]
            for class_bias in raw_bias:
                # these weights are same as sklearn's class_weight='balanced'
                balanced_w = [old_div(len(y), (float(len(classes)) * c)) for c in class_count]
                balanced_tuples = list(zip(list(range(len(classes))), balanced_w))

                weights.append({c: (1 - class_bias) + class_bias * w for c, w in balanced_tuples})
            param_grid['class_weight'] = weights if is_grid else weights[0]
            del param_grid['class_bias']

        return param_grid

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

    def register_resources(self, gazetteers=None, word_freqs=None, query_freqs=None):
        """Loads resources that are built outside the classifier, e.g. gazetteers

        Args:
            gazetteers (dict of Gazetteer): domain gazetteer data
            word_freqs (dict of int): unigram frequencies in queries
            query_freqs (dict of int): whole query index with frequencies
        """
        if gazetteers is not None:
            self._resources[GAZETTEER_RSC] = gazetteers
            if self._meta_type == SUPER_LEARNER_TYPE:
                for model in self._base_clfs.values():
                    model.register_resources(gazetteers=gazetteers)
        if word_freqs is not None:
            self._resources[WORD_FREQ_RSC] = word_freqs
            if self._meta_type == SUPER_LEARNER_TYPE:
                for model in self._base_clfs.values():
                    model.register_resources(word_freqs=word_freqs)
        if query_freqs is not None:
            self._resources[QUERY_FREQ_RSC] = query_freqs
            if self._meta_type == SUPER_LEARNER_TYPE:
                for model in self._base_clfs.values():
                    model.register_resources(query_freqs=query_freqs)

    def compile_word_freq_dict(self, queries):
        """Compiles unigram frequency dictionary of normalized query tokens

        Args:
            queries (list of Query): A list of all queries
        """
        # Unigram frequencies
        tokens = [mask_numerics(tok) for q in queries
                  for tok in q.normalized_tokens]
        freq_dict = Counter(tokens)

        self.register_resources(word_freqs=freq_dict)

    def compile_query_freq_dict(self, queries):
        """Compiles frequency dictionary of normalized query strings

        Args:
            queries (list of Query): A list of all queries
        """
        # Whole query frequencies, with singletons removed
        query_dict = Counter([u'<{}>'.format(q.normalized_text) for q in queries])
        for query in query_dict:
            if query_dict[query] < 2:
                query_dict[query] = 0
        query_dict += Counter()

        self.register_resources(query_freqs=query_dict)


register_model('text', TextModel)
