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
import pickle
from abc import ABC, abstractmethod
from inspect import signature
from typing import Union, Type, Dict, Any, Tuple, List, Pattern, Set

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

from ._util import _is_module_available
from .evaluation import EntityModelEvaluation, StandardModelEvaluation
from .helpers import (
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
from .nn_utils.helpers import EmbedderType
from .._version import get_mm_version
from ..core import ProcessedQuery, QueryEntity
from ..resource_loader import ResourceLoader, ProcessedQueryList as PQL
from ..text_preparation.text_preparation_pipeline import (
    TextPreparationPipelineFactory,
    TextPreparationPipeline
)

# for backwards compatability for sklearn models serialized and dumped in previous version
from .labels import LabelEncoder, EntityLabelEncoder  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

Examples = Union[PQL.QueryIterator, PQL.ListIterator]
Labels = Union[PQL.DomainIterator, PQL.IntentIterator, PQL.EntitiesIterator]


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
        model_type: str = None,
        example_type: str = None,
        label_type: str = None,
        features: Dict = None,
        model_settings: Dict = None,
        params: Dict = None,
        param_selection: Dict = None,
        train_label_set: Pattern[str] = None,
        test_label_set: Pattern[str] = None,
    ):
        for arg, val in {
            "model_type": model_type,
            "label_type": label_type,
        }.items():
            if val is None:
                raise TypeError("__init__() missing required argument {!r}".format(arg))
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

    def to_dict(self) -> Dict:
        """Converts the model config object into a dict

        Returns:
            dict: A dict version of the config
        """
        result = {}
        for attr in self.__slots__:
            result[attr] = getattr(self, attr)
        return result

    def to_json(self) -> str:
        """Converts the model config object to JSON

        Returns:
            str: JSON representation of the classifier
        """
        return json.dumps(self.to_dict(), sort_keys=True)

    def resolve_config(self, new_config: "ModelConfig"):
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

    def get_ngram_lengths_and_thresholds(self, rname: str) -> Tuple:
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

    def required_resources(self) -> Set:
        """Returns the resources this model requires

        Returns:
            set: set of required resources for this model
        """
        # get list of resources required by feature extractors
        required_resources = set()
        if self.features:
            for name in self.features:
                feature = get_feature_extractor(self.example_type, name)
                required_resources.update(feature.__dict__.get("requirements", []))
        return required_resources


class AbstractModel(ABC):
    """
    A minimalistic abstract class upon which all models are based.

    In order to maintain backwards compatability, the skeleton of this class is designed based on
    all the access points of Classifier class and its sub-classes. In addition, it also introduces
    the decoupled way of dumping/loading across different model types (meaning not all models are
    dumped/loaded the same way). Furthermore, methods for validation are also introduced so as to
    cater to the model specific config validations. Lastly, this skeleton also includes some
    common properties that could be used across all model types.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.mindmeld_version = get_mm_version()
        self._resources = {}

        self._validate_model_configs()

    @abstractmethod
    def initialize_resources(
        self, resource_loader: ResourceLoader, examples: Examples = None, labels: Labels = None
    ):
        raise NotImplementedError

    @abstractmethod
    def fit(self, examples: Examples, labels: Labels, params: Dict = None):
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, examples: Examples, dynamic_resource: Dict = None
    ) -> Union[List[Any], List[List[Any]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self, examples: Examples, dynamic_resource: Dict = None
    ) -> Union[List[Tuple[str, Dict[str, float]]], Tuple[Tuple[QueryEntity, float]]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, examples: Examples, labels: Labels
    ) -> Union[StandardModelEvaluation, EntityModelEvaluation]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Type["AbstractModel"]:
        raise NotImplementedError

    @classmethod
    def load_model_config(cls, path: str) -> ModelConfig:
        """
        Dumps the model's configs. Raises a FileNotFoundError if no configs file is found.
        For backwards compatability wherein TextModel was serialized and dumped, the textModel file
        is loaded using joblib and then the config is obtained from its public variables.

        Args:
            path (str): The path where the model is dumped
        """

        try:
            model_configs_save_path = cls._get_model_config_save_path(path)
            model_config = pickle.load(open(model_configs_save_path, "rb"))
        except FileNotFoundError as e:  # backwards compatability for sklearn-based model classes
            metadata = joblib.load(path)
            # metadata here can be a serialized model (eg. TextModel) or a dict (eg. TaggerModel)
            if isinstance(metadata, dict):
                # compatability with previously dumped EntityRecognizers and RoleClassifiers
                try:
                    # sklearn TaggerModel used by EntityRecognizer
                    model_config = metadata["model_config"]
                except KeyError:
                    try:
                        # sklearn TextModel used by RoleClassifier (w/ a non-NoneType "model")
                        model_config = metadata["model"].config
                    except AttributeError:
                        # sklearn TextModel used by RoleClassifier (w/ a NoneType "model")
                        #   in latest version, nothing gets dumped at the dump
                        #       path: 'path/to/dump/<entity_name>-role.pkl' if the self._model in
                        #       RoleClassifier is None because of a check in Classifier.dump()
                        #   in previous version, a dictionary '{'model': None, 'roles': set()}'
                        #       is dumped at the path: 'path/to/dump/<entity_name>-role.pkl'
                        #       although the self._model in RoleClassifier is None
                        msg = f"Model config data cold not be identified from existing dump at " \
                              f"path: {path}. Assuming that the dumped model is NoneType and " \
                              f"belongs to a role classifier"
                        raise FileNotFoundError(msg) from e
            else:
                # compatability with previously dumped DomainClassifiers and IntentClassifiers
                #   in this case, metadata = model which was serialized and dumped
                model_config = metadata.config

        return model_config

    def dump(self, path: str):
        """
        Dumps the model's configs and calls the child model's dump method

        Args:
            path (str): The path to dump the model to
        """

        # every subclass of ABCModel has one .pkl dump file that contains a
        #   dictionary with at least the key 'model_config' whose value is a model configs dict
        #   this pickle file is sought in `load_model_config()` method
        self._dump_model_config(path)

        # call this subclass-implemeneted method to allow models to dump in their own style
        # (eg. serialized dump for sklearn-based models, .bin files for pytorch-based models, etc.)
        self._dump(path)

    def _dump_model_config(self, path: str):
        """
        Dumps the model's configs
        """

        model_configs_save_path = self._get_model_config_save_path(path)
        pickle.dump(self.config, open(model_configs_save_path, "wb"))

    @abstractmethod
    def _dump(self, path: str):
        """
        Dumps the model and calls the underlying algo to dump its state.

        Args:
            path (str): The path to dump the model to
        """
        pass

    @staticmethod
    def _get_model_config_save_path(path: str) -> str:
        head, ext = os.path.splitext(path)
        model_config_save_path = head + ".config" + ext
        os.makedirs(os.path.dirname(model_config_save_path), exist_ok=True)
        return model_config_save_path

    def view_extracted_features(
        self, example: ProcessedQuery, dynamic_resource: Dict = None
    ) -> List[Dict]:
        # Not implemeneted unless overwritten by child class
        raise NotImplementedError

    def register_resources(self, **kwargs):  # pylint: disable=no-self-use
        # Resources for feature extractors are not required for deep neural models
        del kwargs
        pass

    def get_resource(self, name) -> Any:
        return self._resources.get(name)

    @property
    def text_preparation_pipeline(self) -> TextPreparationPipeline:

        text_preparation_pipeline = self._resources.get("text_preparation_pipeline")

        if not text_preparation_pipeline:
            logger.error(
                "The text_preparation_pipeline resource has not been registered "
                "to the model. Using default text_preparation_pipeline."
            )
            return TextPreparationPipelineFactory.create_default_text_preparation_pipeline()

        return text_preparation_pipeline

    def _validate_model_configs(self):
        pass


class Model(AbstractModel):
    """An abstract class upon which all models are based.

    Attributes:
        config (ModelConfig): The configuration for the model
    """

    # model scoring type
    LIKELIHOOD_SCORING = "log_loss"
    ALLOWED_CLASSIFIER_TYPES: List[str] = NotImplemented

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

    def _fit_cv(self, examples, labels, groups=None, selection_settings=None, fixed_params=None):
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
        if fixed_params:
            for key, val in fixed_params.items():
                if key not in param_grid:
                    param_grid[key] = [val]
                else:
                    logger.info(
                        "Found parameter %s both in params and param_selection. Proceeding with param_selection.. \
                        (If you did not set this, it could be a Mindmeld default.)",
                        key
                    )
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

    def register_resources(self, **kwargs):
        """Registers resources which are accessible to feature extractors

        Args:
            **kwargs: dictionary of resources to register
        """
        self._resources.update(kwargs)

    def get_feature_matrix(self, examples, y=None, fit=False):
        raise NotImplementedError

    def _extract_features(self, example, dynamic_resource=None, text_preparation_pipeline=None):
        """Gets all features from an example.

        Args:
            example: An example object.
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference
            text_preparation_pipeline (TextPreparationPipeline): MindMeld text processing object

        Returns:
            (dict of str: number): A dict of feature names to their values.
        """
        example_type = self.config.example_type
        feat_set = {}
        workspace_resource = ingest_dynamic_gazetteer(
            self._resources, dynamic_resource, text_preparation_pipeline
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
        self._resources[
            "text_preparation_pipeline"
        ] = resource_loader.get_text_preparation_pipeline()

    def _validate_model_configs(self) -> Union[TypeError, ValueError]:

        for arg, val in {
            "features": self.config.features,
            "example_type": self.config.example_type
        }.items():
            if val is None:
                raise TypeError("__init__() missing required argument {!r}".format(arg))

        if self.config.params is None and (
            self.config.param_selection is None or self.config.param_selection.get("grid") is None
        ):
            raise ValueError(
                "__init__() One of 'params' and 'param_selection' is required"
            )


class PytorchModel(AbstractModel):
    ALLOWED_CLASSIFIER_TYPES: List[str] = NotImplemented  # to be implemented in child classes

    def __init__(self, config):
        if not _is_module_available('torch'):
            raise ImportError("Install the extra 'torch' library by runnning "
                              "'pip install mindmeld[torch]' to use pytorch based neural models")

        super().__init__(config)
        self._label_encoder = get_label_encoder(self.config)
        self._class_encoder = SKLabelEncoder()
        self._query_text_type = None
        self._clf = None

    def initialize_resources(self, resource_loader, examples=None, labels=None):
        del resource_loader, examples, labels

    @staticmethod
    def _validate_training_data(examples: List[Any], labels: Union[List[int], List[List[int]]]):
        if len(examples) != len(labels):
            msg = f"Number of 'labels' ({len(labels)}) must be same as number of 'examples' " \
                  f"({len(examples)})"
            raise AssertionError(msg)

    def _set_query_text_type(self, params: Dict = None, default: str = None):
        """
        Returns the query text type to use for obtaining training examples from Query objects.

        Args:
            params (Dict, optional): The config params passed in to train the model
            default (str, optional): The default text type to use in case no related configs found
        """

        if params is None and self._query_text_type:
            # this condition is satisfied during loading of models
            return

        # While the key 'query_text_type' in config params allows for end-users to configure the
        # choice of text_type to be used, it also needs the user to have knowledge about different
        # text types in a Query object. Three values are available as of now-
        # ["text", "processed_text", "normalized_text"].
        query_text_type = params.get("query_text_type", default) if params else default

        # consider raw text for pretrained transformer models
        if not query_text_type:
            if params and EmbedderType(params.get("embedder_type")) == EmbedderType.BERT:
                query_text_type = "text"

        # if a NoneType value is found, use the processed_text by default
        query_text_type = query_text_type or "processed_text"

        # validation
        allowed_text_types = ["text", "processed_text", "normalized_text"]
        if query_text_type not in allowed_text_types:
            msg = f"The params 'query_text_type' can only be among " \
                  f"{allowed_text_types} but found value {query_text_type}."
            logger.error(msg)
            raise ValueError(msg)

        self._query_text_type = query_text_type  # this var is dumped and loaded when loading models

    def _get_texts_from_examples(self, examples: PQL.QueryIterator) -> List[str]:
        """
        Method that decides which text type- processed_text or raw_text -that needs to be used for
        neural model training/inference.

        Args:
            examples (QueryIterator): A list of ProcessedQuery objects.

        Returns:
            texts (List[str]): A list of strings obtained from the query objects based on the
                provided input configs
        """
        if not self._query_text_type:
            msg = "The instance attribute '_query_text_type' must be set by calling " \
                  "_set_query_text_type() method before calling the " \
                  "_get_texts_from_examples() method."
            logger.debug(msg)
            raise ValueError(msg)
        return [getattr(example, self._query_text_type) for example in examples]


class AbstractModelFactory(ABC):
    """
    Abstract class for individual model factories like TextModelFactory and TaggerModelFactory
    """

    @abstractmethod
    def get_model_cls(self, config: ModelConfig) -> Type[AbstractModel]:
        raise NotImplementedError
