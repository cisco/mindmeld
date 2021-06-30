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

"""This module contains some helper functions for the models package"""
import json
import logging
import os
import re
from tempfile import mkstemp

import nltk
from sklearn.metrics import make_scorer

from ..gazetteer import Gazetteer
from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)

FEATURE_MAP = {}
MODEL_MAP = {}
LABEL_MAP = {}
EMBEDDER_MAP = {}
ANNOTATOR_MAP = {}
AUGMENTATION_MAP = {}

# Example types
QUERY_EXAMPLE_TYPE = "query"
ENTITY_EXAMPLE_TYPE = "entity"

# Label types
CLASS_LABEL_TYPE = "class"
ENTITIES_LABEL_TYPE = "entities"

# resource/requirements names
GAZETTEER_RSC = "gazetteers"
QUERY_FREQ_RSC = "q_freq"
SYS_TYPES_RSC = "sys_types"
ENABLE_STEMMING = "enable-stemming"
WORD_FREQ_RSC = "w_freq"
WORD_NGRAM_FREQ_RSC = "w_ngram_freq"
CHAR_NGRAM_FREQ_RSC = "c_ngram_freq"
SENTIMENT_ANALYZER = "vader_classifier"
OUT_OF_BOUNDS_TOKEN = "<$>"
OUT_OF_VOCABULARY = "OOV"
IN_VOCABULARY = "IV"
DEFAULT_SYS_ENTITIES = [
    "sys_time",
    "sys_temperature",
    "sys_volume",
    "sys_amount-of-money",
    "sys_email",
    "sys_url",
    "sys_number",
    "sys_ordinal",
    "sys_duration",
    "sys_phone-number",
]


def create_model(config):
    """Creates a model instance using the provided configuration

    Args:
        config (ModelConfig): A model configuration

    Returns:
        Model: a configured model

    Raises:
        ValueError: When model configuration is invalid
    """
    try:
        return MODEL_MAP[config.model_type](config)
    except KeyError as e:
        msg = "Invalid model configuration: Unknown model type {!r}"
        raise ValueError(msg.format(config.model_type)) from e


def create_annotator(config):
    """Creates an annotator instance using the provided configuration

    Args:
        config (dict): A model configuration

    Returns:
        Annotator: An Annotator class

    Raises:
        ValueError: When model configuration is invalid or required key is missing
    """
    if "annotator_class" not in config:
        raise KeyError(
            "Missing required argument in AUTO_ANNOTATOR_CONFIG: 'annotator_class'"
        )
    if config["annotator_class"] in ANNOTATOR_MAP:
        return ANNOTATOR_MAP[config.pop("annotator_class")](**config)
    else:
        msg = "Invalid model configuration: Unknown model type {!r}"
        raise KeyError(msg.format(config["annotator_class"]))


def get_feature_extractor(example_type, name):
    """Gets a feature extractor given the example type and name

    Args:
        example_type (str): The type of example
        name (str): The name of the feature extractor

    Returns:
        function: A feature extractor wrapper
    """
    return FEATURE_MAP[example_type][name]


def get_label_encoder(config):
    """Gets a label encoder given the label type from the config

    Args:
        config (ModelConfig): A model configuration

    Returns:
        LabelEncoder: The appropriate LabelEncoder object for the given config
    """
    return LABEL_MAP[config.label_type](config)


def create_embedder_model(app_path, config):
    """Creates and loads an embedder model

    Args:
        config (dict): Model settings passed in as a dictionary with
            'embedder_type' being a required key

    Returns:
        Embedder: An instance of appropriate embedder class

    Raises:
        ValueError: When model configuration is invalid or required key is missing
    """

    if "model_settings" in config and config["model_settings"]:
        # when config = {"model_settings": {"embedder_type": ..., "..": ...}}
        embedder_config = config["model_settings"]
    else:
        # when config = {"embedder_type": ..., "..": ...}}
        embedder_config = config

    embedder_type = embedder_config.get("embedder_type")
    if not embedder_type:
        raise KeyError(
            "Missing required argument in config supplied to create embedder model: 'embedder_type'"
        )

    try:
        return EMBEDDER_MAP[embedder_type](app_path, **embedder_config)
    except KeyError as e:
        msg = "Invalid model configuration: Unknown embedder type {!r}"
        raise ValueError(msg.format(embedder_type)) from e


def register_model(model_type, model_class):
    """Registers a model for use with `create_model()`

    Args:
        model_type (str): The model type as specified in model configs
        model_class (class): The model to register
    """
    MODEL_MAP[model_type] = model_class


def register_query_feature(feature_name):
    """Registers query feature

    Args:
        feature_name (str): The name of the query feature

    Returns:
        (func): the feature extractor
    """
    return register_feature(QUERY_EXAMPLE_TYPE, feature_name=feature_name)


def register_entity_feature(feature_name):
    """Registers entity feature

    Args:
        feature_name (str): The name of the entity feature

    Returns:
        (func): the feature extractor
    """
    return register_feature(ENTITY_EXAMPLE_TYPE, feature_name=feature_name)


def register_annotator(annotator_class_name, annotator_class):
    """Registers an Annotator class for use with `create_annotator()`

    Args:
        annotator_class_name (str): The annotator class name as specified in the config
        model_class (class): The annotator class to register
    """
    ANNOTATOR_MAP[annotator_class_name] = annotator_class


def register_augmentor(augmentor_name, augmentor_class):
    """Registers an Annotator class for use with `create_annotator()`

    Args:
        annotator_class_name (str): The annotator class name as specified in the config
        model_class (class): The annotator class to register
    """
    AUGMENTATION_MAP[augmentor_name] = augmentor_class


def register_feature(feature_type, feature_name):
    """
    Decorator for adding feature extractor mappings to FEATURE_MAP

    Args:
        feature_type: 'query' or 'entity'
        feature_name: The name of the feature, used in config.py

    Returns:
        (func): the feature extractor
    """

    def add_feature(func):
        if feature_type not in {QUERY_EXAMPLE_TYPE, ENTITY_EXAMPLE_TYPE}:
            raise TypeError("Feature type can only be 'query' or 'entity'")

        # Add func to feature map with given type and name
        if feature_type in FEATURE_MAP:
            FEATURE_MAP[feature_type][feature_name] = func
        else:
            FEATURE_MAP[feature_type] = {feature_name: func}
        return func

    return add_feature


def register_label(label_type, label_encoder):
    """Register a label encoder for use with
    `get_label_encoder()`

    Args:
        label_type (str): The label type of the label encoder
        label_encoder (LabelEncoder): The label encoder class to register

    Raises:
        ValueError: If the label type is already registered
    """
    if label_type in LABEL_MAP:
        msg = "Label encoder for label type {!r} is already registered.".format(
            label_type
        )
        raise ValueError(msg)

    LABEL_MAP[label_type] = label_encoder


def register_embedder(embedder_type, embedder):
    if embedder_type in EMBEDDER_MAP:
        msg = "Embedder of type {!r} is already registered.".format(embedder_type)
        raise ValueError(msg)

    EMBEDDER_MAP[embedder_type] = embedder


def mask_numerics(token):
    """Masks digit characters in a token

    Args:
        token (str): A string

    Returns:
        str: A masked string for digit characters
    """
    if token.isdigit():
        return "#NUM"
    else:
        return re.sub(r"\d", "8", token)


def get_ngram(tokens, start, length):
    """Gets a ngram from a list of tokens.

    Handles out-of-bounds token positions with a special character.

    Args:
        tokens (list of str): Word tokens.
        start (int): The index of the desired ngram's start position.
        length (int): The length of the n-gram, e.g. 1 for unigram, etc.

    Returns:
        (str) An n-gram in the input token list.
    """

    ngram_tokens = []
    for index in range(start, start + length):
        token = (
            OUT_OF_BOUNDS_TOKEN if index < 0 or index >= len(tokens) else tokens[index]
        )
        ngram_tokens.append(token)
    return " ".join(ngram_tokens)


def get_ngrams_upto_n(tokens, n):
    """This function returns a generator that returns ngram tuples with length upto n

    Args:
        tokens (list of str): Word tokens.
        n (int): The length of n-gram upto which the ngram tokens are generated

    Returns:
        tuple: ngram, (token index start, token index end)
    """
    if n == 0:
        return []
    for length, i in enumerate(range(1, n + 1)):
        for idx, j in enumerate(nltk.ngrams(tokens, i)):
            yield j, (idx, idx + length)


def get_seq_accuracy_scorer():
    """
    Returns a scorer that can be used by sklearn's GridSearchCV based on the
    sequence_accuracy_scoring method below.
    """
    return make_scorer(score_func=sequence_accuracy_scoring)


def get_seq_tag_accuracy_scorer():
    """
    Returns a scorer that can be used by sklearn's GridSearchCV based on the
    sequence_tag_accuracy_scoring method below.
    """
    return make_scorer(score_func=sequence_tag_accuracy_scoring)


def sequence_accuracy_scoring(y_true, y_pred):
    """Accuracy score which calculates two sequences to be equal only if all of
        their predicted tags are equal.

    Args:
        y_true (list): A sequence of true expected labels
        y_pred (list): A sequence of predicted labels

    Returns:
        float: The sequence-level accuracy when comparing the predicted labels \
            against the true expected labels
    """
    total = len(y_true)
    if not total:
        return 0

    matches = sum(
        1 for yseq_true, yseq_pred in zip(y_true, y_pred) if yseq_true == yseq_pred
    )

    return float(matches) / float(total)


def sequence_tag_accuracy_scoring(y_true, y_pred):
    """Accuracy score which calculates the number of tags that were predicted
        correctly.

    Args:
        y_true (list): A sequence of true expected labels
        y_pred (list): A sequence of predicted labels

    Returns:
        float: The tag-level accuracy when comparing the predicted labels \
            against the true expected labels
    """
    y_true_flat = [tag for seq in y_true for tag in seq]
    y_pred_flat = [tag for seq in y_pred for tag in seq]

    total = len(y_true_flat)
    if not total:
        return 0

    matches = sum(
        1
        for (y_true_tag, y_pred_tag) in zip(y_true_flat, y_pred_flat)
        if y_true_tag == y_pred_tag
    )

    return float(matches) / float(total)


def entity_seqs_equal(expected, predicted):
    """
    Returns true if the expected entities and predicted entities all match, returns
    false otherwise. Note that for entity comparison, we compare that the span, text,
    and type of all the entities match.

    Args:
        expected (list of core.Entity): A list of the expected entities for some query
        predicted (list of core.Entity): A list of the predicted entities for some query
    """
    if len(expected) != len(predicted):
        return False
    for expected_entity, predicted_entity in zip(expected, predicted):
        if expected_entity.entity.type != predicted_entity.entity.type:
            return False
        if expected_entity.span != predicted_entity.span:
            return False
        if expected_entity.text != predicted_entity.text:
            return False
    return True


def merge_gazetteer_resource(resource, dynamic_resource, tokenizer):
    """
    Returns a new resource that is a merge between the original resource and the dynamic
    resource passed in for only the gazetteer values

    Args:
        resource (dict): The original resource built from the app
        dynamic_resource (dict): The dynamic resource passed in
        tokenizer (Tokenizer): This component is used to normalize entities in dyn gaz

    Returns:
        dict: The merged resource
    """
    return_obj = {}
    for key in resource:
        # Pass by reference if not a gazetteer key
        if key != GAZETTEER_RSC:
            return_obj[key] = resource[key]
            continue

        # Create a dict from scratch if we match the gazetteer key
        return_obj[key] = {}
        for entity_type in resource[key]:
            # If the entity type is in the dyn gaz, we merge the data. Else,
            # just pass by reference the original resource data
            if entity_type in dynamic_resource[key]:
                new_gaz = Gazetteer(entity_type, tokenizer)
                # We deep copy here since shallow copying will also change the
                # original resource's data during the '_update_entity' op.
                new_gaz.from_dict(resource[key][entity_type])

                for entity in dynamic_resource[key][entity_type]:
                    new_gaz._update_entity(
                        tokenizer.normalize(entity),
                        dynamic_resource[key][entity_type][entity],
                    )

                # The new gaz created is a deep copied version of the merged gaz data
                return_obj[key][entity_type] = new_gaz.to_dict()
            else:
                return_obj[key][entity_type] = resource[key][entity_type]
    return return_obj


def ingest_dynamic_gazetteer(resource, dynamic_resource=None, tokenizer=None):
    """Ingests dynamic gazetteers from the app and adds them to the resource

    Args:
        resource (dict): The original resource
        dynamic_resource (dict, optional): The dynamic resource that needs to be ingested
        tokenizer (Tokenizer): This used to normalize the entities in the dynamic resource

    Returns:
        (dict): A new resource with the ingested dynamic resource
    """
    if not dynamic_resource or GAZETTEER_RSC not in dynamic_resource:
        return resource
    tokenizer = tokenizer or Tokenizer()
    workspace_resource = merge_gazetteer_resource(resource, dynamic_resource, tokenizer)
    return workspace_resource


def requires(resource):
    """
    Decorator to enforce the resource dependencies of the active feature extractors

    Args:
        resource (str): the key of a classifier resource which must be initialized before
            the given feature extractor is used

    Returns:
        (func): the feature extractor
    """

    def add_resource(func):
        req = func.__dict__.get("requirements", set())
        req.add(resource)
        func.requirements = req
        return func

    return add_resource


class FileBackedList:
    """
    FileBackedList implements an interface for simple list use cases
    that is backed by a temporary file on disk.  This is useful for
    simple list processing in a memory efficient way.
    """

    def __init__(self):
        self.num_lines = 0
        self.file_handle = None
        fd, self.filename = mkstemp()
        os.close(fd)

    def __len__(self):
        return self.num_lines

    def append(self, line):
        if self.file_handle is None:
            self.file_handle = open(self.filename, "w")
        self.file_handle.write(json.dumps(line))
        self.file_handle.write("\n")
        self.num_lines += 1

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()
        os.unlink(self.filename)

    def __iter__(self):
        # Flush out any remaining data to be written
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        return FileBackedList.Iterator(self)

    class Iterator:
        def __init__(self, source):
            self.source = source
            self.file_handle = open(source.filename, "r")

        def __len__(self):
            return len(self.source)

        def __next__(self):
            try:
                line = next(self.file_handle)
                return json.loads(line)
            except Exception as e:
                self.file_handle.close()
                self.file_handle = None
                if not isinstance(e, StopIteration):
                    logger.error("Error reading from FileBackedList")
                raise

        def __del__(self):
            if self.file_handle:
                self.file_handle.close()
