# -*- coding: utf-8 -*-
"""This module contains some helper functions for the models package"""
from sklearn.metrics import make_scorer

import re

from ..gazetteer import Gazetteer
from ..tokenizer import Tokenizer

FEATURE_MAP = {}
MODEL_MAP = {}
LABEL_MAP = {}

# Example types
QUERY_EXAMPLE_TYPE = 'query'
ENTITY_EXAMPLE_TYPE = 'entity'

# Label types
CLASS_LABEL_TYPE = 'class'
ENTITIES_LABEL_TYPE = 'entities'


# resource/requirements names
GAZETTEER_RSC = 'gazetteers'
QUERY_FREQ_RSC = 'q_freq'
SYS_TYPES_RSC = 'sys_types'
ENABLE_STEMMING = 'enable-stemming'
WORD_FREQ_RSC = 'w_freq'
WORD_NGRAM_FREQ_RSC = 'w_ngram_freq'
CHAR_NGRAM_FREQ_RSC = 'c_ngram_freq'
OUT_OF_BOUNDS_TOKEN = '<$>'
DEFAULT_SYS_ENTITIES = ['sys_time', 'sys_temperature', 'sys_volume', 'sys_amount-of-money',
                        'sys_email', 'sys_url', 'sys_number', 'sys_ordinal', 'sys_duration',
                        'sys_phone-number']


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
    except KeyError:
        msg = 'Invalid model configuration: Unknown model type {!r}'
        raise ValueError(msg.format(config.model_type))


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


def register_model(model_type, model_class):
    """Registers a model for use with `create_model()`

    Args:
        model_type (str): The model type as specified in model configs
        model_class (class): The model to register
    """
    if model_type in MODEL_MAP:
        raise ValueError('Model {!r} is already registered.'.format(model_type))

    MODEL_MAP[model_type] = model_class


def register_query_feature(feature_name):
    return register_feature(QUERY_EXAMPLE_TYPE, feature_name=feature_name)


def register_entity_feature(feature_name):
    return register_feature(ENTITY_EXAMPLE_TYPE, feature_name=feature_name)


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
        msg = 'Label encoder for label type {!r} is already registered.'.format(label_type)
        raise ValueError(msg)

    LABEL_MAP[label_type] = label_encoder


def mask_numerics(token):
    """Masks digit characters in a token"""
    if token.isdigit():
        return '#NUM'
    else:
        return re.sub(r'\d', '8', token)


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
    for index in range(start, start+length):
        token = (OUT_OF_BOUNDS_TOKEN if index < 0 or index >= len(tokens)
                 else tokens[index])
        ngram_tokens.append(token)
    return ' '.join(ngram_tokens)


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
    """
    Accuracy score which calculates two sequences to be equal only if all of
    their predicted tags are equal.
    """
    total = len(y_true)
    if not total:
        return 0

    matches = sum(1 for yseq_true, yseq_pred in zip(y_true, y_pred)
                  if yseq_true == yseq_pred)

    return float(matches) / float(total)


def sequence_tag_accuracy_scoring(y_true, y_pred):
    """
    Accuracy score which calculates the number of tags that were predicted
    correctly.
    """
    y_true_flat = [tag for seq in y_true for tag in seq]
    y_pred_flat = [tag for seq in y_pred for tag in seq]

    total = len(y_true_flat)
    if not total:
        return 0

    matches = sum(1 for (y_true_tag, y_pred_tag) in zip(y_true_flat, y_pred_flat)
                  if y_true_tag == y_pred_tag)

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
                new_gaz = Gazetteer(entity_type)
                # We deep copy here since shallow copying will also change the
                # original resource's data during the '_update_entity' op.
                new_gaz.from_dict(resource[key][entity_type])

                for entity in dynamic_resource[key][entity_type]:
                    new_gaz._update_entity(
                        tokenizer.normalize(entity),
                        dynamic_resource[key][entity_type][entity])

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
        req = func.__dict__.get('requirements', set())
        req.add(resource)
        func.requirements = req
        return func

    return add_resource
