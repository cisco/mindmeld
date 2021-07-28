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

"""This module contains the Memm entity recognizer."""
import logging
import os
import random

from sklearn.externals import joblib

from .evaluation import EntityModelEvaluation, EvaluatedExample
from .helpers import (
    get_label_encoder,
    get_seq_accuracy_scorer,
    get_seq_tag_accuracy_scorer,
    ingest_dynamic_gazetteer,
)
from .model import ModelConfig, Model, PytorchModel
from .taggers.crf import ConditionalRandomFields
from .taggers.memm import MemmModel
from ..exceptions import MindMeldError

try:
    from .taggers.lstm import LstmModel
except ImportError:
    LstmModel = None

logger = logging.getLogger(__name__)


class TaggerModel(Model):
    """A machine learning classifier for tags.

    This class manages feature extraction, training, cross-validation, and
    prediction. The design goal is that after providing initial settings like
    hyperparameters, grid-searchable hyperparameters, feature extractors, and
    cross-validation settings, TaggerModel manages all of the details
    involved in training and prediction such that the input to training or
    prediction is Query objects, and the output is class names, and no data
    manipulation is needed from the client.

    Attributes:
        classifier_type (str): The name of the classifier type. Currently
            recognized values are "memm","crf", and "lstm"
        hyperparams (dict): A kwargs dict of parameters that will be used to
            initialize the classifier object.
        grid_search_hyperparams (dict): Like 'hyperparams', but the values are
            lists of parameters. The training process will grid search over the
            Cartesian product of these parameter lists and select the best via
            cross-validation.
        feat_specs (dict): A mapping from feature extractor names, as given in
            FEATURE_NAME_MAP, to a kwargs dict, which will be passed into the
            associated feature extractor function.
        cross_validation_settings (dict): A dict that contains "type", which
            specifies the name of the cross-validation strategy, such as
            "k-folds" or "shuffle". The remaining keys are parameters
            specific to the cross-validation type, such as "k" when the type is
            "k-folds".
    """

    # classifier types
    CRF_TYPE = "crf"
    MEMM_TYPE = "memm"
    LSTM_TYPE = "lstm"
    ALLOWED_CLASSIFIER_TYPES = [CRF_TYPE, MEMM_TYPE, LSTM_TYPE]

    # for default model scoring types
    ACCURACY_SCORING = "accuracy"
    SEQ_ACCURACY_SCORING = "seq_accuracy"
    SEQUENCE_MODELS = ["crf"]

    DEFAULT_FEATURES = {
        "bag-of-words-seq": {
            "ngram_lengths_to_start_positions": {1: [-2, -1, 0, 1, 2], 2: [-2, -1, 0, 1]}
        },
        "in-gaz-span-seq": {},
        "sys-candidates-seq": {"start_positions": [-1, 0, 1]},
    }

    def __init__(self, config):
        if not config.features:
            config_dict = config.to_dict()
            config_dict["features"] = TaggerModel.DEFAULT_FEATURES
            config = ModelConfig(**config_dict)

        super().__init__(config)

        # Get model classifier and initialize
        self._clf = self._get_model_constructor()()
        self._clf.setup_model(self.config)

        self._no_entities = False
        self.types = None

    def __getstate__(self):
        """Returns the information needed to pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior. For the _resources field,
        we save the resources that are memory intensive
        """
        attributes = self.__dict__.copy()
        attributes["_resources"] = {}
        resources_to_persist = set(["sys_types"])
        for key in resources_to_persist:
            attributes["_resources"][key] = self.__dict__["_resources"][key]

        return attributes

    def _get_model_constructor(self):
        """Returns the python class of the actual underlying model"""
        classifier_type = self.config.model_settings["classifier_type"]
        try:
            if classifier_type == TaggerModel.LSTM_TYPE and LstmModel is None:
                msg = (
                    "{}: Classifier type {!r} dependencies not found. Install the "
                    "mindmeld[tensorflow] extra to use this classifier type."
                )
                raise ValueError(msg.format(self.__class__.__name__, classifier_type))
            return {
                TaggerModel.MEMM_TYPE: MemmModel,
                TaggerModel.CRF_TYPE: ConditionalRandomFields,
                TaggerModel.LSTM_TYPE: LstmModel,
            }[classifier_type]
        except KeyError as e:
            msg = "{}: Classifier type {!r} not recognized"
            raise ValueError(msg.format(self.__class__.__name__, classifier_type)) from e

    def _fit(self, examples, labels, params=None):
        """Trains a classifier without cross-validation.

        Args:
            examples (list of mindmeld.core.Query): a list of queries to train on
            labels (list of tuples of mindmeld.core.QueryEntity): a list of expected labels
            params (dict): Parameters of the classifier
        """
        self._clf.set_params(**params)
        return self._clf.fit(examples, labels)

    def _convert_params(self, param_grid, y, is_grid=True):
        """
        Convert the params from the style given by the config to the style
        passed in to the actual classifier.

        Args:
            param_grid (dict): lists of classifier parameter values, keyed by parameter name

        Returns:
            (dict): revised param_grid
        """
        return param_grid

    def _get_cv_scorer(self, selection_settings):
        """
        Returns the scorer to use based on the selection settings and classifier type,
        defaulting to tag accuracy.
        """
        classifier_type = self.config.model_settings["classifier_type"]

        # Sets the default scorer based on the classifier type
        if classifier_type in TaggerModel.SEQUENCE_MODELS:
            default_scorer = get_seq_tag_accuracy_scorer()
        else:
            default_scorer = TaggerModel.ACCURACY_SCORING

        # Gets the scorer based on what is passed in to the selection settings (reverts to
        # default if nothing is passed in)
        scorer = selection_settings.get("scoring", default_scorer)
        if scorer == TaggerModel.SEQ_ACCURACY_SCORING:
            if classifier_type not in TaggerModel.SEQUENCE_MODELS:
                logger.error(
                    "Sequence accuracy is only available for the following models: "
                    "%s. Using tag level accuracy instead...",
                    str(TaggerModel.SEQUENCE_MODELS),
                )
                return TaggerModel.ACCURACY_SCORING
            return get_seq_accuracy_scorer()
        elif (
            scorer == TaggerModel.ACCURACY_SCORING and
            classifier_type in TaggerModel.SEQUENCE_MODELS
        ):
            return get_seq_tag_accuracy_scorer()
        else:
            return scorer

    def unload(self):
        self._clf = None
        self._current_params = None
        self._label_encoder = None
        self._no_entities = None

    def get_feature_matrix(self, examples, y=None, fit=False):
        raise NotImplementedError

    def select_params(self, examples, labels, selection_settings=None):
        raise NotImplementedError

    def fit(self, examples, labels, params=None):
        """Trains the model.

        Args:
            examples (ProcessedQueryList.QueryIterator): A list of queries to train on.
            labels (ProcessedQueryList.EntitiesIterator): A list of expected labels.
            params (dict): Parameters of the classifier.
        """
        skip_param_selection = params is not None or self.config.param_selection is None
        params = params or self.config.params

        # Shuffle to prevent order effects
        indices = list(range(len(labels)))
        random.shuffle(indices)
        examples.reorder(indices)
        labels.reorder(indices)

        types = [entity.entity.type for label in labels for entity in label]
        self.types = types
        if len(set(types)) == 0:
            self._no_entities = True
            logger.info(
                "There are no labels in this label set, so we don't " "fit the model."
            )
            return self
        # Extract labels - label encoders are the same accross all entity recognition models
        self._label_encoder = get_label_encoder(self.config)
        y = self._label_encoder.encode(labels, examples=examples)

        # Extract features
        X, y, groups = self._clf.extract_features(
            examples, self.config, self._resources, y, fit=True
        )

        # Fit the model
        if skip_param_selection:
            self._clf = self._fit(X, y, params)
            self._current_params = params
        else:
            # run cross validation to select params
            if self._clf.__class__ == LstmModel:
                raise MindMeldError("The LSTM model does not support cross-validation")

            _, best_params = self._fit_cv(X, y, groups)
            self._clf = self._fit(X, y, best_params)
            self._current_params = best_params

        return self

    def view_extracted_features(self, query, dynamic_resource=None):
        """Returns a dictionary of extracted features and their weights for a given query

        Args:
            query (mindmeld.core.Query): The query to extract features from
            dynamic_resource (dict): The dynamic resource used along with the query

        Returns:
            list: A list of dictionaries of extracted features and their weights
        """
        workspace_resource = ingest_dynamic_gazetteer(
            self._resources, dynamic_resource=dynamic_resource,
            text_preparation_pipeline=self.text_preparation_pipeline
        )
        return self._clf.extract_example_features(
            query, self.config, workspace_resource
        )

    def predict(self, examples, dynamic_resource=None):
        """
        Args:
            examples (list of mindmeld.core.Query): a list of queries to train on
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference

        Returns:
            (list of tuples of mindmeld.core.QueryEntity): a list of predicted labels
        """
        if self._no_entities:
            return [()]

        workspace_resource = ingest_dynamic_gazetteer(
            self._resources, dynamic_resource=dynamic_resource,
            text_preparation_pipeline=self.text_preparation_pipeline
        )
        predicted_tags = self._clf.extract_and_predict(
            examples, self.config, workspace_resource
        )
        # Decode the tags to labels
        labels = [
            self._label_encoder.decode([example_predicted_tags], examples=[example])[0]
            for example_predicted_tags, example in zip(predicted_tags, examples)
        ]
        return labels

    def predict_proba(self, examples, dynamic_resource=None):
        """
        Args:
            examples (list of mindmeld.core.Query): a list of queries to train on
            dynamic_resource (dict, optional): A dynamic resource to aid NLP inference

        Returns:
            list of tuples of (mindmeld.core.QueryEntity): a list of predicted labels \
            with confidence scores
        """
        if self._no_entities:
            return []

        workspace_resource = ingest_dynamic_gazetteer(
            self._resources, dynamic_resource=dynamic_resource,
            text_preparation_pipeline=self.text_preparation_pipeline
        )
        predicted_tags_probas = self._clf.predict_proba(
            examples, self.config, workspace_resource
        )
        tags, probas = zip(*predicted_tags_probas[0])
        entity_confidence = []
        entities = self._label_encoder.decode([tags], examples=[examples[0]])[0]
        for entity in entities:
            entity_proba = \
                probas[entity.normalized_token_span.start: entity.normalized_token_span.end + 1]
            # We assume that the score of the least likely tag in the sequence as the confidence
            # score of the entire entity sequence
            entity_confidence.append(min(entity_proba))
        predicted_labels_scores = tuple(zip(entities, entity_confidence))
        return predicted_labels_scores

    def evaluate(self, examples, labels):
        """Evaluates a model against the given examples and labels

        Args:
            examples: A list of examples to predict
            labels: A list of expected labels

        Returns:
            ModelEvaluation: an object containing information about the \
                evaluation
        """
        if self._no_entities:
            logger.info(
                "There are no labels in this label set, so we don't "
                "run model evaluation."
            )
            return

        predictions = self.predict(examples)

        evaluations = [
            EvaluatedExample(e, labels[i], predictions[i], None, self.config.label_type)
            for i, e in enumerate(examples)
        ]

        config = self._get_effective_config()
        model_eval = EntityModelEvaluation(config, evaluations)
        return model_eval

    def _dump(self, path):

        # In TaggerModel, unlike TextModel, two dumps happen,
        # one, the underneath classifier and two, the tagger model's metadata

        metadata = {"serializable": self._clf.is_serializable}

        if self._clf.is_serializable:
            metadata.update({
                "model": self
            })
        else:
            # underneath tagger dump for LSTM model, returned `model_dir` is None for MEMM & CRF
            self._clf.dump(path)
            metadata.update({
                "current_params": self._current_params,
                "label_encoder": self._label_encoder,
                "no_entities": self._no_entities,
                "model_config": self.config
            })

        # dump model metadata
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(metadata, path)

    @classmethod
    def load(cls, path):
        """
        Load the model state to memory.

        Args:
            path (str): The path to dump the model to
        """

        # load model metadata
        metadata = joblib.load(path)

        # The default is True since < MM 3.2.0 models are serializable by default
        is_serializable = metadata.get("serializable", True)

        # If model is serializable, it can be loaded and used as-is. But if not serializable,
        #   it means we need to create an instance and load necessary details for it to be used.
        if not is_serializable:
            model = cls(metadata["model_config"])

            # misc resources load
            try:
                model._current_params = metadata["current_params"]
                model._label_encoder = metadata["label_encoder"]
                model._no_entities = metadata["no_entities"]
            except KeyError:  # backwards compatability
                model_dir = metadata["model"]
                tagger_vars = joblib.load(model_dir, ".tagger_vars")
                model._current_params = tagger_vars["current_params"]
                model._label_encoder = tagger_vars["label_encoder"]
                model._no_entities = tagger_vars["no_entities"]

            # underneath tagger load
            model._clf.load(model_dir)

            # replace model dump directory with actual model
            metadata["model"] = model

        return metadata["model"]


class PytorchTaggerModel(PytorchModel):
    ALLOWED_CLASSIFIER_TYPES = ["embedder", "cnn", "lstm"]
    pass


class AutoTaggerModel:

    @staticmethod
    def get_model_class(config: ModelConfig):

        CLASSES = [TaggerModel, PytorchTaggerModel]
        classifier_type = config.model_settings["classifier_type"]

        for _class in CLASSES:
            if classifier_type in _class.ALLOWED_CLASSIFIER_TYPES:
                return _class

        msg = f"Invalid 'classifier_type': {classifier_type}. " \
              f"Allowed types are: {[_class.ALLOWED_CLASSIFIER_TYPES for _class in CLASSES]}"
        raise ValueError(msg)
