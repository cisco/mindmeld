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
This module contains the base class for all the machine-learned classifiers in MindMeld.
"""
import json
import logging
import os
from abc import ABC, abstractmethod

from sklearn.externals import joblib

from ..constants import DEFAULT_TEST_SET_REGEX, DEFAULT_TRAIN_SET_REGEX
from ..core import Query
from ..exceptions import ClassifierLoadError
from ..models import ModelConfig, create_model
from ..resource_loader import ProcessedQueryList

logger = logging.getLogger(__name__)


class ClassifierConfig:
    """A value object representing a classifier configuration

        Attributes:
            model_type (str): The name of the model type. Will be used to find the \
                model class to instantiate.
            model_settings (dict): Settings specific to the model type specified.
            params (dict): Params to pass to the underlying classifier.
            param_selection (dict): Configuration for param selection (using cross \
                validation). For example:
                {'type': 'shuffle',
                'n': 3,
                'k': 10,
                'n_jobs': 2,
                'scoring': '',
                'grid': {}
                }
            features (dict): The keys are the names of feature extractors and the \
                values are either a kwargs dict which will be passed into the \
                feature extractor function, or a callable which will be used as to \
                extract features.
    """

    __slots__ = [
        "model_type",
        "features",
        "model_settings",
        "params",
        "param_selection",
    ]

    def __init__(
        self,
        model_type=None,
        features=None,
        model_settings=None,
        params=None,
        param_selection=None,
    ):
        """Initializes a classifier configuration"""
        for arg, val in {"model_type": model_type, "features": features}.items():
            if val is None:
                raise TypeError("__init__() missing required argument {!r}".format(arg))
        if params is None and (
            param_selection is None or param_selection.get("grid") is None
        ):
            raise ValueError(
                "__init__() One of 'params' and 'param_selection' is required"
            )
        self.model_type = model_type
        self.features = features
        self.model_settings = model_settings
        self.params = params
        self.param_selection = param_selection

    def to_dict(self):
        """Converts the model config object into a dict.

        Returns:
            (dict): A dict version of the config.
        """
        result = {}
        for attr in self.__slots__:
            result[attr] = getattr(self, attr)
        return result

    def __repr__(self):
        args_str = ", ".join(
            "{}={!r}".format(key, getattr(self, key)) for key in self.__slots__
        )
        return "{}({})".format(self.__class__.__name__, args_str)

    @classmethod
    def from_model_config(cls, model_config):
        config = model_config.to_dict()
        config.pop("example_type")
        config.pop("label_type")
        config.pop("train_label_set")
        config.pop("test_label_set")
        return cls(**config)

    def to_json(self):
        """Converts the model config object to JSON.

        Returns:
            (str): JSON representation of the classifier.
        """
        return json.dumps(self.to_dict(), sort_keys=True)


class Classifier(ABC):
    """The base class for all the machine-learned classifiers in MindMeld. A classifier is a \
    machine-learned model that categorizes input examples into one of the pre-determined class \
    labels. Among other functionality, each classifier provides means by which to fit a \
    statistical model on a given training dataset and then use the trained model to make \
    predictions on new unseen data.

        Attributes:
            ready (bool): Whether the classifier is ready.
            dirty (bool): Whether the classifier has unsaved changes to its model.
            config (ClassifierConfig): The classifier configuration.
            hash (str): A hash representing the inputs into the model.
    """

    CLF_TYPE = None
    """Classifier type (`str`)."""

    def __init__(self, resource_loader):
        """Initializes a classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
        """
        self._resource_loader = resource_loader
        self._model = None  # will be set when model is fit or loaded
        self.ready = False
        self.dirty = False
        self.config = None
        self.hash = ""

    def fit(self,
            queries=None,
            label_set=None,
            incremental_timestamp=None,
            load_cached=True,
            **kwargs):
        """Trains a statistical model for classification using the provided training examples and
        model configuration.

        Args:
            queries (list(ProcessedQuery) or ProcessedQueryList, optional): A list of queries
                 to train on. If not specified the queries will be loaded from the label_set.
            label_set (str): A label set to load. If not specified, the default
                 training set will be loaded.
            incremental_timestamp (str, optional): The timestamp folder to cache models in
            model_type (str, optional): The type of machine learning model to use. If omitted, the
                 default model type will be used.
            model_settings (dict): Settings specific to the model type specified
            features (dict): Features to extract from each example instance to form the feature
                 vector used for model training. If omitted, the default feature set for the model
                 type will be used.
            params (dict): Params to pass to the underlying classifier
            params_selection (dict): The grid of hyper-parameters to search, for finding the optimal
                 hyper-parameter settings for the model. If omitted, the default hyper-parameter
                 search grid will be used.
            param_selection (dict): Configuration for param selection (using cross-validation)
                {'type': 'shuffle',
                'n': 3,
                'k': 10,
                'n_jobs': 2,
                'scoring': '',
                'grid': { 'C': [100, 10000, 1000000]}}
            features (dict): The keys are the names of feature extractors and the
                values are either a kwargs dict which will be passed into the
                feature extractor function, or a callable which will be used as to
                extract features.
            load_cached (bool): If the model is cached on disk, load it into memory.

        Returns:
            True if model was loaded and fit, False if a valid cached model exists but was not
            loaded (controlled by the load_cached arg).

        Examples:
            Fit using default the configuration.

                >>> clf.fit()

            Fit using a 'special' label set.

                >>> clf.fit(label_set='special')

            Fit using given params, bypassing cross-validation. This is useful for speeding up
            train times if you are confident the params are optimized.

                >>> clf.fit(params={'C': 10000000})

            Fit using given parameter selection settings (also known as cross-validation settings).

                >>> clf.fit(param_selection={})

            Fit using a custom set of features, including a custom feature extractor.
            This is only for advanced users.

                >>> clf.fit(features={
                        'in-gaz': {}, // gazetteer features
                        'contrived': lambda exa, res: {'contrived': len(exa.text) == 26}
                    })
        """

        # create model with given params
        model_config = self._get_model_config(**kwargs)
        model = create_model(model_config)

        # resolve query set
        label_set = label_set or model_config.train_label_set or DEFAULT_TRAIN_SET_REGEX
        queries = self._resolve_queries(queries, label_set)

        new_hash = self._get_model_hash(model_config, queries)
        cached_model = self._resource_loader.hash_to_model_path.get(new_hash)

        if incremental_timestamp and cached_model:
            logger.info("No need to fit.  Previous model is cached.")
            if load_cached:
                self.load(cached_model)
                return True
            return False

        examples, labels = self._get_examples_and_labels(queries)

        if not examples:
            logger.warning(
                "Could not fit model since no relevant examples were found. "
                'Make sure the labeled queries for training are placed in "%s" '
                "files in your MindMeld project.",
                label_set,
            )
            return True
        num_labels = len(set(labels))
        if num_labels <= 1:
            phrase = ["are no classes", "is only one class"][num_labels]
            logger.info("Not doing anything for fit since there %s.", phrase)
            return True

        model.initialize_resources(self._resource_loader, examples, labels)
        model.fit(examples, labels)
        self._model = model
        self.config = ClassifierConfig.from_model_config(self._model.config)
        self.hash = new_hash

        self.ready = True
        self.dirty = True
        return True

    def _resolve_queries(self, queries=None, label_set=None):
        """
        Resolve queries and/or label_set into a ProcessedQueryList.
        queries is preferred over label_set.

        Args:
            queries (ProcessedQueryList or list(ProcessedQuery): A set of queries to use for
                the operation.
            label_set (str): The label set to load queries from

        Returns:
            ProcessedQueryList: The set of queries
        """
        if not queries:
            queries = self._get_queries_from_label_set(label_set)
        elif not isinstance(queries, ProcessedQueryList):
            queries = ProcessedQueryList.from_in_memory_list(queries)
        return queries

    def predict(self, query, time_zone=None, timestamp=None, dynamic_resource=None):
        """Predicts a class label for the given query using the trained classification model

        Args:
            query (Query or str): The input query
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference

        Returns:
            str: The predicted class label
        """
        if not self._model:
            logger.error("You must fit or load the model before running predict")
            return None
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(
                query, time_zone=time_zone, timestamp=timestamp
            )
        return self._model.predict([query], dynamic_resource=dynamic_resource)[0]

    def predict_proba(
        self, query, time_zone=None, timestamp=None, dynamic_resource=None
    ):
        """Runs prediction on a given query and generates multiple hypotheses with their
        associated probabilities using the trained classification model

        Args:
            query (Query): The input query
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).
            dynamic_resource (dict, optional):  A dynamic resource to aid NLP inference

        Returns:
            list: a list of tuples of the form (str, float) grouping predicted class labels and \
                their probabilities
        """
        if not self._model:
            logger.error("You must fit or load the model before running predict_proba")
            return []
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(
                query, time_zone=time_zone, timestamp=timestamp
            )

        predict_proba_result = self._model.predict_proba(
            [query], dynamic_resource=dynamic_resource
        )
        class_proba_tuples = list(predict_proba_result[0][1].items())
        return sorted(class_proba_tuples, key=lambda x: x[1], reverse=True)

    def evaluate(self, queries=None, label_set=None):
        """Evaluates the trained classification model on the given test data

        Args:
            queries (Optional(list(ProcessedQuery))): optional list of queries to evaluate
            label_set (str): The label set to use for evaluation.

        Returns:
            ModelEvaluation: A ModelEvaluation object that contains evaluation results
        """
        if not self._model:
            logger.error("You must fit or load the model before running evaluate.")
            return None

        model_config = self._get_model_config()
        label_set = label_set or model_config.test_label_set or DEFAULT_TEST_SET_REGEX
        queries = self._resolve_queries(queries, label_set)

        examples, labels = self._get_examples_and_labels(queries)

        if not examples:
            logger.info(
                "Could not evaluate model since no relevant examples were found. Make sure "
                'the labeled queries for evaluation are placed in "%s" files '
                "in your MindMeld project.",
                label_set,
            )
            return None

        evaluation = self._model.evaluate(examples, labels)
        return evaluation

    def inspect(self, query, gold_label=None, dynamic_resource=None):
        raise NotImplementedError

    def view_extracted_features(
        self, query, time_zone=None, timestamp=None, dynamic_resource=None
    ):
        """Extracts features for the given input based on the model config.

        Args:
            query (Query or str): The input query
            time_zone (str, optional): The name of an IANA time zone, such as \
                'America/Los_Angeles', or 'Asia/Kolkata' \
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).
            dynamic_resource (dict): Dynamic gazetteer to be included for feature extraction.

        Returns:
            dict: The extracted features from the given input
        """
        if not self._model:
            logger.error("You must fit or load the model to initialize resources")
            return None
        if not isinstance(query, Query):
            query = self._resource_loader.query_factory.create_query(
                query, time_zone=time_zone, timestamp=timestamp
            )
        return self._model.view_extracted_features(query, dynamic_resource)

    @staticmethod
    def _get_model_config(loaded_config=None, **kwargs):
        """Updates the loaded configuration with runtime specified options, and creates a model
        configuration object with the final configuration dictionary. If an application config
        exists it should be passed in, if not the default config should be passed in.

        Returns:
            ModelConfig: The model configuration corresponding to the provided config name
        """
        try:
            # If all params required for model config were passed in, use kwargs
            return ModelConfig(**kwargs)
        except (TypeError, ValueError):
            # Use application specified or default config, customizing with provided kwargs
            if not loaded_config:
                logger.warning("loaded_config is not passed in")
            model_config = loaded_config or {}
            model_config.update(kwargs)

            # If a parameter selection grid was passed in at runtime, override params set in the
            # application specified or default config
            if kwargs.get("param_selection") and not kwargs.get("params"):
                model_config.pop("params", None)
        return ModelConfig(**model_config)

    def _data_dump_payload(self):
        return self._model

    def _create_and_dump_payload(self, path):
        joblib.dump(self._data_dump_payload(), path)

    def dump(self, model_path, incremental_model_path=None):
        """Persists the trained classification model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored.
            incremental_model_path (str, optional): The timestamp folder where the cached
                models are stored.
        """
        for path in [model_path, incremental_model_path]:
            if not path:
                continue

            # make directory if necessary
            folder = os.path.dirname(path)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            self._create_and_dump_payload(path)

            hash_path = path + ".hash"
            with open(hash_path, "w") as hash_file:
                hash_file.write(self.hash)

            if path == model_path:
                self.dirty = False

    def unload(self):
        """
        Unloads the model from memory. This helps reduce memory requirements while
        training other models.
        """
        self._model = None
        self.config = None
        self.ready = False

    def load(self, model_path):
        """Loads the trained classification model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        try:
            self._model = joblib.load(model_path)
        except (OSError, IOError) as e:
            msg = "Unable to load {}. Pickle at {!r} cannot be read."
            raise ClassifierLoadError(msg.format(self.__class__.__name__, model_path)) from e
        if self._model is not None:
            if not hasattr(self._model, "mindmeld_version"):
                msg = (
                    "Your trained models are incompatible with this version of MindMeld. "
                    "Please run a clean build to retrain models"
                )
                raise ClassifierLoadError(msg)

            try:
                self._model.config.to_dict()
            except AttributeError:
                # Loaded model config is incompatible with app config.
                self._model.config.resolve_config(self._get_model_config())

            self._model.initialize_resources(self._resource_loader)
            self.config = ClassifierConfig.from_model_config(self._model.config)

        self.hash = self._load_hash(model_path)

        self.ready = True
        self.dirty = False

    @staticmethod
    def _load_hash(model_path):
        hash_path = model_path + ".hash"
        if not os.path.isfile(hash_path):
            return ""
        with open(hash_path, "r") as hash_file:
            model_hash = hash_file.read()
        return model_hash

    @abstractmethod
    def _get_queries_from_label_set(self, label_set=DEFAULT_TRAIN_SET_REGEX):
        """Returns the set of queries loaded from the label_set

        Args:
            label_set (list, optional): A label set to load. If not specified,
                the default training set will be loaded.
        Returns:
            ProcessedQueryList
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_examples_and_labels_hash(self, queries):
        """Returns a hashed string representing the labeled queries

        Args:
            queries (ProcessedQueryList): The queries used to fit this model
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_examples_and_labels(self, queries):
        """Extracts examples and lables extracted from the queries

        Args:
            queries (ProcessedQueryList): The queries to extract examples and lables from

        Returns:
            tuple(ProcessedQueryList.Iterator(Any),
                  ProcessedQueryList.Iterator(Any)): A tuple of iterators
                [0]: the examples, [1]: the labels
        """

    def _get_model_hash(self, model_config, queries):
        """Returns a hash representing the inputs into the model

        Args:
            model_config (ModelConfig): The model configuration
            queries (ProcessedQueryList): The queries used to fit this model

        Returns:
            str: The hash
        """

        # Hash queries
        queries_hash = self._get_examples_and_labels_hash(queries)

        # Hash config
        config_hash = self._resource_loader.hash_string(model_config.to_json())

        # Hash resources
        rsc_strings = []
        for resource in sorted(model_config.required_resources()):
            rsc_strings.append(self._resource_loader.hash_feature_resource(resource))
        rsc_hash = self._resource_loader.hash_list(rsc_strings)

        return self._resource_loader.hash_list([queries_hash, config_hash, rsc_hash])

    def __repr__(self):
        msg = "<{} ready: {!r}, dirty: {!r}>"
        return msg.format(self.__class__.__name__, self.ready, self.dirty)
