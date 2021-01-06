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
This module contains all code required to perform multinomial classification
of text.
"""
import logging
import operator
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .helpers import (
    CHAR_NGRAM_FREQ_RSC,
    QUERY_FREQ_RSC,
    WORD_FREQ_RSC,
    WORD_NGRAM_FREQ_RSC,
    register_model,
)
from .model import EvaluatedExample, Model, StandardModelEvaluation

_NEG_INF = -1e10

# classifier types
LOG_REG_TYPE = "logreg"
DECISION_TREE_TYPE = "dtree"
RANDOM_FOREST_TYPE = "rforest"
SVM_TYPE = "svm"
SUPER_LEARNER_TYPE = "super-learner"
BASE_MODEL_TYPES = [LOG_REG_TYPE, DECISION_TREE_TYPE, RANDOM_FOREST_TYPE, SVM_TYPE]

# default model scoring type
ACCURACY_SCORING = "accuracy"


logger = logging.getLogger(__name__)


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
        attributes["_resources"] = {
            rname: self._resources.get(rname, {})
            for rname in [
                WORD_FREQ_RSC,
                QUERY_FREQ_RSC,
                WORD_NGRAM_FREQ_RSC,
                CHAR_NGRAM_FREQ_RSC,
            ]
        }
        return attributes

    def _get_model_constructor(self):
        """Returns the class of the actual underlying model"""
        classifier_type = self.config.model_settings["classifier_type"]
        try:
            return {
                LOG_REG_TYPE: LogisticRegression,
                DECISION_TREE_TYPE: DecisionTreeClassifier,
                RANDOM_FOREST_TYPE: RandomForestClassifier,
                SVM_TYPE: SVC,
            }[classifier_type]
        except KeyError as e:
            msg = "{}: Classifier type {!r} not recognized"
            raise ValueError(msg.format(self.__class__.__name__, classifier_type)) from e

    def _get_cv_scorer(self, selection_settings):
        """
        Returns the scorer to use based on the selection settings and classifier type,
        defaulting to accuracy.
        """
        return selection_settings.get("scoring", ACCURACY_SCORING)

    def evaluate(self, examples, labels):
        """Evaluates a model against the given examples and labels

        Args:
            examples: A list of examples to predict
            labels: A list of expected labels

        Returns:
            ModelEvaluation: an object containing information about the \
                evaluation
        """
        # TODO: also expose feature weights?
        predictions = self.predict_proba(examples)

        # Create a model config object for the current effective config (after param selection)
        config = self._get_effective_config()

        evaluations = [
            EvaluatedExample(
                e, labels[i], predictions[i][0], predictions[i][1], config.label_type
            )
            for i, e in enumerate(examples)
        ]

        model_eval = StandardModelEvaluation(config, evaluations)
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
            (TextModel): Returns self to match classifier scikit-learn \
                interfaces.
        """
        params = params or self.config.params
        skip_param_selection = params is not None or self.config.param_selection is None

        # Shuffle to prevent order effects
        indices = list(range(len(labels)))
        random.shuffle(indices)
        examples = [examples[i] for i in indices]
        labels = [labels[i] for i in indices]

        distinct_labels = set(labels)
        if len(set(distinct_labels)) <= 1:
            return self

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

    def _fit(self, examples, labels, params=None):
        """Trains a classifier without cross-validation.

        Args:
            examples (numpy.matrix): The feature matrix for a dataset.
            labels (numpy.array): The target output values.
            params (dict): Parameters of the classifier

        """
        params = self._convert_params(params, labels, is_grid=False)
        model_class = self._get_model_constructor()
        params = self._clean_params(model_class, params)
        return model_class(**params).fit(examples, labels)

    def predict(self, examples, dynamic_resource=None):
        X, _, _ = self.get_feature_matrix(examples, dynamic_resource=dynamic_resource)
        y = self._clf.predict(X)
        predictions = self._class_encoder.inverse_transform(y)
        return self._label_encoder.decode(predictions)

    def predict_proba(self, examples, dynamic_resource=None):
        X, _, _ = self.get_feature_matrix(examples, dynamic_resource=dynamic_resource)
        return self._predict_proba(X, self._clf.predict_proba)

    def predict_log_proba(self, examples, dynamic_resource=None):
        X, _, _ = self.get_feature_matrix(examples, dynamic_resource=dynamic_resource)
        predictions = self._predict_proba(X, self._clf.predict_log_proba)

        # JSON can't reliably encode infinity, so replace it with large number
        for row in predictions:
            _, probas = row
            for label, proba in probas.items():
                if proba == -np.Infinity:
                    probas[label] = _NEG_INF
        return predictions

    def view_extracted_features(self, example, dynamic_resource=None):
        return self._extract_features(
            example, dynamic_resource=dynamic_resource, tokenizer=self.tokenizer
        )

    def _get_feature_weight(self, feat_name, label_class):
        """Retrieves the feature weight from the coefficient matrix. If there are only two
         classes, the feature vector is actually collapsed into one so we need some logic to
         handle that case.

        Args:
            feat_name (str) : The feature name
            label_class (int): The index of the label

        Returns:
            (ndarray float): The ndarray with a single float element
        """
        if len(self._class_encoder.classes_) == 2 and label_class >= 1:
            return np.array([0.0])
        else:
            return self._clf.coef_[
                label_class, self._feat_vectorizer.vocabulary_[feat_name]
            ]

    def inspect(self, example, gold_label=None, dynamic_resource=None):
        """This class takes an example and returns a 2D list for every feature with feature
          name, feature value, feature weight and their product for the predicted label. If gold
          label is passed in, we will also include the feature value and weight for the gold
          label and returns the log probability of the difference.

        Args:
            example (Query): The query to be predicted
            gold_label (str): The gold label for this string
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference

        Returns:
            (list of lists): A 2D array that includes every feature, their value, weight and \
             probability
        """
        if not isinstance(self._clf, LogisticRegression):
            logging.warning(
                "Currently inspection is only available for Logistic Regression Model"
            )
            return []

        try:
            gold_class = self._class_encoder.transform([gold_label])
        except ValueError:
            logger.warning("Unable to decode label `%s`", gold_label)
            gold_class = None

        pred_label = self.predict([example], dynamic_resource=dynamic_resource)[0]
        pred_class = self._class_encoder.transform([pred_label])
        features = self._extract_features(
            example, dynamic_resource=dynamic_resource, tokenizer=self.tokenizer
        )

        logging.info("Predicted: %s.", pred_label)

        if gold_class is None:
            columns = ["Feature", "Value", "Pred_W({0})".format(pred_label), "Pred_P"]
        else:
            columns = [
                "Feature",
                "Value",
                "Pred_W({0})".format(pred_label),
                "Pred_P",
                "Gold_W({0})".format(gold_label),
                "Gold_P",
                "Diff",
            ]
            logging.info("Gold: %s.", gold_label)

        inspect_table = [columns]

        # Get all active features sorted alphabetically by name
        features = sorted(features.items(), key=operator.itemgetter(0))
        for feature in features:
            feat_name = feature[0]
            feat_value = feature[1]

            # Features we haven't seen before won't be in our vectorizer
            # e.g., an exact match feature for a query we've never seen before
            if feat_name not in self._feat_vectorizer.vocabulary_:
                continue

            weight = self._get_feature_weight(feat_name, pred_class)
            product = feat_value * weight

            if gold_class is None:
                row = [
                    feat_name,
                    round(feat_value, 4),
                    weight.round(4),
                    product.round(4),
                    "-",
                    "-",
                    "-",
                ]
            else:
                gold_w = self._get_feature_weight(feat_name, gold_class)
                gold_p = feat_value * gold_w
                diff = gold_p - product
                row = [
                    feat_name,
                    round(feat_value, 4),
                    weight.round(4),
                    product.round(4),
                    gold_w.round(4),
                    gold_p.round(4),
                    diff.round(4),
                ]

            inspect_table.append(row)

        return inspect_table

    def _predict_proba(self, X, predictor):
        predictions = []
        for row in predictor(X):
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

    def get_feature_matrix(self, examples, y=None, fit=False, dynamic_resource=None):
        """Transforms a list of examples into a feature matrix.

        Args:
            examples (list): The examples.

        Returns:
            (tuple): tuple containing:

                * (numpy.matrix): The feature matrix.
                * (numpy.array): The group labels for examples.
        """
        groups = []
        feats = []
        for idx, example in enumerate(examples):
            feats.append(
                self._extract_features(example, dynamic_resource, self.tokenizer)
            )
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
        if "class_weight" in param_grid:
            raw_weights = (
                param_grid["class_weight"] if is_grid else [param_grid["class_weight"]]
            )
            weights = [
                {
                    k
                    if isinstance(k, int)
                    else self._class_encoder.transform((k,))[0]: v
                    for k, v in cw_dict.items()
                }
                for cw_dict in raw_weights
            ]
            param_grid["class_weight"] = weights if is_grid else weights[0]
        elif "class_bias" in param_grid:
            # interpolate between class_bias=0 => class_weight=None
            # and class_bias=1 => class_weight='balanced'
            class_count = np.bincount(y)
            classes = self._class_encoder.classes_
            weights = []
            raw_bias = (
                param_grid["class_bias"] if is_grid else [param_grid["class_bias"]]
            )
            for class_bias in raw_bias:
                # these weights are same as sklearn's class_weight='balanced'
                balanced_w = [(len(y) / len(classes) / c) for c in class_count]
                balanced_tuples = list(zip(list(range(len(classes))), balanced_w))

                weights.append(
                    {c: (1 - class_bias) + class_bias * w for c, w in balanced_tuples}
                )
            param_grid["class_weight"] = weights if is_grid else weights[0]
            del param_grid["class_bias"]

        return param_grid

    def _get_feature_selector(self):
        """Get a feature selector instance based on the feature_selector model
        parameter

        Returns:
            (Object): a feature selector which returns a reduced feature matrix, \
                given the full feature matrix, X and the class labels, y
        """
        if self.config.model_settings is None:
            selector_type = None
        else:
            selector_type = self.config.model_settings.get("feature_selector")
        selector = {
            "l1": SelectFromModel(LogisticRegression(penalty="l1", C=1)),
            "f": SelectPercentile(),
        }.get(selector_type)
        return selector

    def _get_feature_scaler(self):
        """Get a feature value scaler based on the model settings"""
        if self.config.model_settings is None:
            scale_type = None
        else:
            scale_type = self.config.model_settings.get("feature_scaler")
        scaler = {
            "std-dev": StandardScaler(with_mean=False),
            "max-abs": MaxAbsScaler(),
        }.get(scale_type)
        return scaler


register_model("text", TextModel)
