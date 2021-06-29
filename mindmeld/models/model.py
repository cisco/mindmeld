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

"""This module contains base classes for models defined in the models subpackage."""
import copy
import json
import logging
import math
import os
from abc import ABC, abstractmethod
from inspect import signature

from sklearn.externals import joblib
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from .helpers import (
    create_model,
    CHAR_NGRAM_FREQ_RSC,
    ENABLE_STEMMING,
    GAZETTEER_RSC,
    WORD_NGRAM_FREQ_RSC,
    QUERY_FREQ_RSC,
    SYS_TYPES_RSC,
    WORD_FREQ_RSC,
    SENTIMENT_ANALYZER,
    get_feature_extractor,
    get_label_encoder,
    ingest_dynamic_gazetteer,
)
from .pytorch_utils import encoders as pyt_encoders
from .._version import get_mm_version
from ..exceptions import ClassifierLoadError
from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class ModelConfig:
    """A value object representing a model configuration.

    Attributes:
        model_type (str): The name of the model type. Will be used to find the
            model class to instantiate
        example_type (str): The type of the examples which will be passed into
            `fit()` and `predict()`. Used to select feature extractors
        label_type (str): The type of the labels which will be passed into
            `fit()` and returned by `predict()`. Used to select the label encoder
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
        train_label_set (regex pattern): The regex pattern for finding training
            file names.
        test_label_set (regex pattern): The regex pattern for finding testing
            file names.
    """

    __slots__ = [
        "model_type",
        "example_type",
        "label_type",
        "features",
        "model_settings",
        "params",
        "param_selection",
        "train_label_set",
        "test_label_set",
    ]

    def __init__(
        self,
        model_type=None,
        example_type=None,
        label_type=None,
        features=None,
        model_settings=None,
        params=None,
        param_selection=None,
        train_label_set=None,
        test_label_set=None,
    ):
        for arg, val in {
            "model_type": model_type,
            "example_type": example_type,
            "label_type": label_type,
            "features": features,
        }.items():
            if val is None:
                raise TypeError("__init__() missing required argument {!r}".format(arg))
        if params is None and (
            param_selection is None or param_selection.get("grid") is None
        ):
            raise ValueError(
                "__init__() One of 'params' and 'param_selection' is required"
            )
        self.model_type = model_type
        self.example_type = example_type
        self.label_type = label_type
        self.features = features
        self.model_settings = model_settings
        self.params = params
        self.param_selection = param_selection
        self.train_label_set = train_label_set
        self.test_label_set = test_label_set

    def __repr__(self):
        args_str = ", ".join(
            "{}={!r}".format(key, getattr(self, key)) for key in self.__slots__
        )
        return "{}({})".format(self.__class__.__name__, args_str)

    def to_dict(self):
        """Converts the model config object into a dict

        Returns:
            dict: A dict version of the config
        """
        result = {}
        for attr in self.__slots__:
            result[attr] = getattr(self, attr)
        return result

    def to_json(self):
        """Converts the model config object to JSON

        Returns:
            str: JSON representation of the classifier
        """
        return json.dumps(self.to_dict(), sort_keys=True)

    def resolve_config(self, new_config):
        """This method resolves any config incompatibility issues by
        loading the latest settings from the app config to the current config

        Args:
            new_config (ModelConfig): The ModelConfig representing the app's latest config
        """
        new_settings = ["train_label_set", "test_label_set"]
        logger.warning(
            "Loading missing properties %s from app " "configuration file", new_settings
        )
        for setting in new_settings:
            setattr(self, setting, getattr(new_config, setting))

    def get_ngram_lengths_and_thresholds(self, rname):
        """
        Returns the n-gram lengths and thresholds to extract to optimize resource collection

        Args:
            rname (string): Name of the resource

        Returns:
            (tuple): tuple containing:

                * lengths (list of int): list of n-gram lengths to be extracted
                * thresholds (list of int): thresholds to be applied to corresponding n-gram lengths
        """
        lengths = thresholds = None
        # if it's not the n-gram feature, we don't need length and threshold information
        if rname == CHAR_NGRAM_FREQ_RSC:
            feature_name = "char-ngrams"
        elif rname == WORD_NGRAM_FREQ_RSC:
            feature_name = "bag-of-words"
        else:
            return lengths, thresholds

        # feature name varies based on whether it's for a classifier or tagger
        if self.model_type == "text":
            if feature_name in self.features:
                lengths = self.features[feature_name]["lengths"]
                thresholds = self.features[feature_name].get(
                    "thresholds", [1] * len(lengths)
                )
        elif self.model_type == "tagger":
            feature_name = feature_name + "-seq"
            if feature_name in self.features:
                lengths = self.features[feature_name][
                    "ngram_lengths_to_start_positions"
                ].keys()
                thresholds = self.features[feature_name].get(
                    "thresholds", [1] * len(lengths)
                )

        return lengths, thresholds

    def required_resources(self):
        """Returns the resources this model requires

        Returns:
            set: set of required resources for this model
        """
        # get list of resources required by feature extractors
        required_resources = set()
        for name in self.features:
            feature = get_feature_extractor(self.example_type, name)
            required_resources.update(feature.__dict__.get("requirements", []))
        return required_resources


class BaseModel(ABC):
    """
    A minimalistic abstract class upon which all models are based.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.mindmeld_version = get_mm_version()
        self._resources = {}

    @abstractmethod
    def initialize_resources(self, resource_loader, examples=None, labels=None):
        raise NotImplementedError

    @abstractmethod
    def fit(self, examples, labels, params=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, examples, dynamic_resource=None):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, examples):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, examples, labels):
        raise NotImplementedError

    def dump(self, path, metadata=None):

        metadata = metadata or {}

        # every XxxModel derived from baseModel has one default .pkl dump file that contains a
        #   dictionary with at least the key 'model_config' whose value is a dictionary of the
        #   model configs
        if 'model_config' not in metadata:
            metadata.update({'model_config': self.config})

        # make directory if necessary
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        joblib.dump(metadata, path)

    @classmethod
    def load(cls, path):

        try:
            metadata = joblib.load(path)
        except (OSError, IOError) as e:
            msg = "Unable to load {}. Pickle at {!r} cannot be read."
            raise ClassifierLoadError(msg.format(cls.__name__, path)) from e

        if not isinstance(metadata, dict):
            # backwards compatability
            #   when a serializable model is saved as-is & now has to be retrieved;
            #   notably, the serialized model also consists of the model config;
            #   in this case, metadata = model
            metadata = {"model": metadata, "model_config": metadata.config}

        if 'model_config' not in metadata:
            msg = f"Unable to obtain model_config from dump location- {path}." \
                  f"Please re-build the models."
            raise KeyError(msg)

        return metadata

    def view_extracted_features(self, example, dynamic_resource=None):
        raise NotImplementedError

    def register_resources(self, **kwargs):
        """Registers resources which are accessible to feature extractors

        Args:
            **kwargs: dictionary of resources to register
        """
        self._resources.update(kwargs)

    def get_resource(self, name):
        return self._resources.get(name)


class Model(BaseModel):
    """An abstract class upon which all models are based.

    Attributes:
        config (ModelConfig): The configuration for the model
    """

    # model scoring type
    LIKELIHOOD_SCORING = "log_loss"

    def __init__(self, config):
        super().__init__(config)
        self._label_encoder = get_label_encoder(self.config)
        self._current_params = None
        self._clf = None
        self.cv_loss_ = None

    def _fit(self, examples, labels, params=None):
        raise NotImplementedError

    def _get_model_constructor(self):
        raise NotImplementedError

    @property
    def tokenizer(self):
        tokenizer = self._resources.get("tokenizer")
        if not tokenizer:
            logger.error(
                "The tokenizer resource has not been registered "
                "to the model. Using default tokenizer."
            )
            tokenizer = Tokenizer()
        return tokenizer

    def _fit_cv(self, examples, labels, groups=None, selection_settings=None):
        """Called by the fit method when cross validation parameters are passed in. Runs cross
        validation and returns the best estimator and parameters.

        Args:
            examples (list): A list of examples. Should be in the format expected by the \
                             underlying estimator.
            labels (list): The target output values.
            groups (None, optional): Same length as examples and labels. Used to group \
                                     examples when splitting the dataset into train/test
            selection_settings (dict, optional): A dictionary containing the cross \
                                                 validation selection settings.

        """
        selection_settings = selection_settings or self.config.param_selection
        cv_iterator = self._get_cv_iterator(selection_settings)

        if selection_settings is None:
            return self._fit(examples, labels, self.config.params), self.config.params

        cv_type = selection_settings["type"]
        num_splits = cv_iterator.get_n_splits(examples, labels, groups)
        logger.info(
            "Selecting hyperparameters using %s cross-validation with %s split%s",
            cv_type,
            num_splits,
            "" if num_splits == 1 else "s",
        )

        scoring = self._get_cv_scorer(selection_settings)
        n_jobs = selection_settings.get("n_jobs", -1)

        param_grid = self._convert_params(selection_settings["grid"], labels)
        model_class = self._get_model_constructor()
        estimator, param_grid = self._get_cv_estimator_and_params(
            model_class, param_grid
        )
        # set GridSearchCV's return_train_score attribute to False improves cross-validation
        # runtime perf as it doesn't have to compute training scores and which we don't consume
        grid_cv = GridSearchCV(
            estimator=estimator,
            scoring=scoring,
            param_grid=param_grid,
            cv=cv_iterator,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        model = grid_cv.fit(examples, labels, groups)

        for idx, params in enumerate(model.cv_results_["params"]):
            logger.debug("Candidate parameters: %s", params)
            std_err = (
                2.0
                * model.cv_results_["std_test_score"][idx]
                / math.sqrt(model.n_splits_)
            )
            if scoring == Model.LIKELIHOOD_SCORING:
                msg = "Candidate average log likelihood: {:.4} ± {:.4}"
            else:
                msg = "Candidate average accuracy: {:.2%} ± {:.2%}"
            # pylint: disable=logging-format-interpolation
            logger.debug(msg.format(model.cv_results_["mean_test_score"][idx], std_err))

        if scoring == Model.LIKELIHOOD_SCORING:
            msg = "Best log likelihood: {:.4}, params: {}"
            self.cv_loss_ = -model.best_score_
        else:
            msg = "Best accuracy: {:.2%}, params: {}"
            self.cv_loss_ = 1 - model.best_score_

        best_params = self._process_cv_best_params(model.best_params_)
        # pylint: disable=logging-format-interpolation
        logger.info(msg.format(model.best_score_, best_params))

        return model.best_estimator_, model.best_params_

    def _get_cv_scorer(self, selection_settings):
        """
        Returns the scorer to use based on the selection settings and classifier type.
        """
        raise NotImplementedError

    @staticmethod
    def _clean_params(model_class, params):
        """
        Make sure that the params to be passed into model construction meet the model's expected
        params
        """
        expected_params = signature(model_class).parameters.keys()
        if len(expected_params) == 1 and "parameters" in expected_params:
            # if there is only one key, this is our custom TaggerModel which is supposed to be a
            # pass-through and pass everything to the actual sklearn model
            return params
        result = copy.deepcopy(params)
        for param in params:
            if param not in expected_params:
                msg = (
                    "Unexpected param `{param}`, dropping it from model config.".format(
                        param=param
                    )
                )
                logger.warning(msg)
                result.pop(param)
        return result

    def _get_cv_estimator_and_params(self, model_class, param_grid):
        param_grid = self._clean_params(model_class, param_grid)
        if "warm_start" in signature(model_class).parameters.keys():
            # Warm start helps speed up cross-validation for some models such as random forest
            return model_class(warm_start=True), param_grid
        else:
            return model_class(), param_grid

    @staticmethod
    def _process_cv_best_params(best_params):
        return best_params

    def select_params(self, examples, labels, selection_settings=None):
        """Selects the best set of hyper-parameters for a given set of examples and true labels
            through cross-validation

        Args:
            examples: A list of example queries
            labels: A list of labels associated with the queries
            selection_settings: A dictionary of parameter lists to select from

        Returns:
            dict: A dictionary of optimized parameters to use
        """
        raise NotImplementedError

    def _convert_params(self, param_grid, y, is_grid=True):
        """Convert the params from the style given by the config to the style
        passed in to the actual classifier.

        Args:
            param_grid (dict): lists of classifier parameter values, keyed by \
                parameter name
            y (list): A list of labels
            is_grid (bool, optional): Indicates whether param_grid is actually a grid \
                or a params dict.
        """
        raise NotImplementedError

    def _get_effective_config(self):
        """Create a model config object for the current effective config (after \
        param selection)

        Returns:
            ModelConfig
        """
        config_dict = self.config.to_dict()
        config_dict.pop("param_selection")
        config_dict["params"] = self._current_params
        return ModelConfig(**config_dict)

    def get_feature_matrix(self, examples, y=None, fit=False):
        raise NotImplementedError

    def _extract_features(self, example, dynamic_resource=None, tokenizer=None):
        """Gets all features from an example.

        Args:
            example: An example object.
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference
            tokenizer (Tokenizer): The component used to normalize entities in dynamic_resource

        Returns:
            (dict of str: number): A dict of feature names to their values.
        """
        example_type = self.config.example_type
        feat_set = {}
        workspace_resource = ingest_dynamic_gazetteer(
            self._resources, dynamic_resource, tokenizer
        )
        workspace_features = copy.deepcopy(self.config.features)
        enable_stemming = workspace_features.pop(ENABLE_STEMMING, False)

        for name, kwargs in workspace_features.items():
            if callable(kwargs):
                # a feature extractor function was passed in directly
                feat_extractor = kwargs
            else:
                kwargs[ENABLE_STEMMING] = enable_stemming
                feat_extractor = get_feature_extractor(example_type, name)(**kwargs)
            feat_set.update(feat_extractor(example, workspace_resource))
        return feat_set

    def _get_cv_iterator(self, settings):
        if not settings:
            return None
        cv_type = settings["type"]
        try:
            cv_iterator = {
                "k-fold": self._k_fold_iterator,
                "shuffle": self._shuffle_iterator,
                "group-k-fold": self._groups_k_fold_iterator,
                "group-shuffle": self._groups_shuffle_iterator,
                "stratified-k-fold": self._stratified_k_fold_iterator,
                "stratified-shuffle": self._stratified_shuffle_iterator,
            }.get(cv_type)(settings)
        except KeyError as e:
            raise ValueError("Unknown param selection type: {!r}".format(cv_type)) from e

        return cv_iterator

    @staticmethod
    def _k_fold_iterator(settings):
        k = settings["k"]
        return KFold(n_splits=k)

    @staticmethod
    def _shuffle_iterator(settings):
        k = settings["k"]
        n = settings.get("n", k)
        test_size = 1.0 / k
        return ShuffleSplit(n_splits=n, test_size=test_size)

    @staticmethod
    def _groups_k_fold_iterator(settings):
        k = settings["k"]
        return GroupKFold(n_splits=k)

    @staticmethod
    def _groups_shuffle_iterator(settings):
        k = settings["k"]
        n = settings.get("n", k)
        test_size = 1.0 / k
        return GroupShuffleSplit(n_splits=n, test_size=test_size)

    @staticmethod
    def _stratified_k_fold_iterator(settings):
        k = settings["k"]
        return StratifiedKFold(n_splits=k)

    @staticmethod
    def _stratified_shuffle_iterator(settings):
        k = settings["k"]
        n = settings.get("n", k)
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
            if (
                "requirements" in feature_extractor.__dict__
                and resource in feature_extractor.requirements
            ):
                return True
        return False

    ##################
    # abstract methods
    ##################

    def initialize_resources(self, resource_loader, examples=None, labels=None):
        """Load the required resources for feature extractors. Each feature extractor uses \
        @requires decorator to declare required resources. Based on feature list in model config \
        a list of required resources are compiled, and the passed in resource loader is then used \
        to load the resources accordingly.

        Args:
            resource_loader (ResourceLoader): application resource loader object
            examples (list): Optional. A list of examples.
            labels (list): Optional. A parallel list to examples. The gold labels \
                           for each example.
        """
        # get list of resources required by feature extractors
        required_resources = self.config.required_resources()
        enable_stemming = ENABLE_STEMMING in required_resources
        resource_builders = {}
        for rname in required_resources:
            if rname in self._resources:
                continue
            if rname == GAZETTEER_RSC:
                self._resources[rname] = resource_loader.get_gazetteers()
            elif rname == SENTIMENT_ANALYZER:
                self._resources[rname] = resource_loader.get_sentiment_analyzer()
            elif rname == SYS_TYPES_RSC:
                self._resources[rname] = resource_loader.get_sys_entity_types(labels)
            elif rname == WORD_FREQ_RSC:
                resource_builders[rname] = resource_loader.WordFreqBuilder()
            elif rname == CHAR_NGRAM_FREQ_RSC:
                l, t = self.config.get_ngram_lengths_and_thresholds(rname)
                resource_builders[rname] = resource_loader.CharNgramFreqBuilder(l, t)
            elif rname == WORD_NGRAM_FREQ_RSC:
                l, t = self.config.get_ngram_lengths_and_thresholds(rname)
                resource_builders[rname] = \
                    resource_loader.WordNgramFreqBuilder(l, t, enable_stemming)
            elif rname == QUERY_FREQ_RSC:
                resource_builders[rname] = resource_loader.QueryFreqBuilder(enable_stemming)

        if resource_builders:
            for query in examples:
                for rname, builder in resource_builders.items():
                    builder.add(query)
            for rname, builder in resource_builders.items():
                self._resources[rname] = builder.get_resource()

        # Always initialize the global resource for tokenization, which is not a
        # feature-specific resource
        self._resources["tokenizer"] = resource_loader.get_tokenizer()


class PytorchModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self._label_encoder = get_label_encoder(self.config)
        self._class_encoder = SKLabelEncoder()
        self._clf = None  # need to edit this

    def _get_model_constructor(self):
        raise NotImplementedError

    def initialize_resources(self, resource_loader, examples=None, labels=None):

        # Always initialize the global resource for tokenization, which is not a
        # feature-specific resource
        self._resources["tokenizer"] = resource_loader.get_tokenizer()

    def fit(self, examples, labels, params=None):
        params = params or self.config.params

        if len(set(labels)) <= 1:
            return self

        # Encode classes
        y = self._label_encoder.encode(labels)
        try:
            # runs without Error for tagger models
            flat_y = sum(y, [])
            is_flattened = True
        except TypeError:
            # meaning it is a text model
            flat_y = y
            is_flattened = False
        encoded_flat_y = self._class_encoder.fit_transform(flat_y)
        if is_flattened:
            seq_lengths = [len(_y) for _y in y]
            y = []
            start_idx = 0
            for seq_length in seq_lengths:
                y.append(encoded_flat_y[start_idx: start_idx + seq_length])
                start_idx += seq_length
        else:
            y = list(encoded_flat_y)

        self._clf = self._get_model_constructor()()  # gets the class name only
        self._clf.fit([ex.text for ex in examples], y, **params)

        return self

    def dump(self, path, metadata=None):

        metadata = metadata or {}
        metadata.update({
            "model": self,
            "model_config": self.config,
            "serializable": False}
        )

        # dump clf
        self._clf.dump(path)

        # dump metadata
        super().dump(path, metadata)

    @classmethod
    def load(cls, path):

        # load metadata
        metadata = super().load(path)
        model_config = metadata.get("model_config")
        model = create_model(model_config)

        # gets the class name and then loads
        model._clf = self._get_model_constructor().load(path)  # .load() is a classmethod
        metadata["model"] = model

        return metadata

    def predict(self, examples, dynamic_resource=None):
        y = self._clf.predict(examples)
        predictions = self._class_encoder.inverse_transform(y)
        return self._label_encoder.decode(predictions)

    def predict_proba(self, examples):

        # snippet mostly re-used from text_model.py/TextModel/_predict_proba()
        predictions = []
        for row in self._clf.predict_proba(examples):
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

    def evaluate(self, examples, labels):
        raise NotImplementedError

    @staticmethod
    def get_model_folder_from_model_path(model_path: str):
        if not model_path.endswith(".pkl"):
            msg = "Unsupported model_path provided to create a folder name in featurizers. " \
                  "The supplied model path must end with '.pkl'. "
            raise ValueError(msg)
        model_folder = model_path.split(".pkl")[0]
        os.makedirs(model_folder, exist_ok=True)
        return model_folder

    def _get_encoder_constructor(self):
        """Returns the class of the actual underlying model"""
        model_type = self.config.model_type
        try:
            return {
                "text": pyt_encoders.xxx,
                "tagger": pyt_encoders.xxx,
            }[model_type]
        except KeyError as e:
            msg = "{}: Model type {!r} not recognized"
            raise ValueError(msg.format(self.__class__.__name__, model_type)) from e
