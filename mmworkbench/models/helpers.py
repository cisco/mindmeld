# -*- coding: utf-8 -*-
"""This module contains some helper functions for the models package"""
from __future__ import unicode_literals
from sklearn.metrics import make_scorer

import re

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
WORD_FREQ_RSC = 'w_freq'

OUT_OF_BOUNDS_TOKEN = '<$>'


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


def extract_sequence_features(example, example_type, feature_config, resources):
    """Extracts feature dicts for each token in an example.

    Args:
        example (mmworkbench.core.Query): an query
        example_type (str): The type of example
        feature_config (dict): The config for features
        resources (dict): Resources of this model
    Returns:
        (list dict): features
    """
    feat_seq = []
    for name, kwargs in feature_config.items():
        if callable(kwargs):
            # a feature extractor function was passed in directly
            feat_extractor = kwargs
        else:
            feat_extractor = get_feature_extractor(example_type, name)(**kwargs)

        update_feat_seq = feat_extractor(example, resources)
        if not feat_seq:
            feat_seq = update_feat_seq
        else:
            for idx, features in enumerate(update_feat_seq):
                feat_seq[idx].update(features)

    return feat_seq


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


def register_features(example_type, features):
    """Register a set of feature extractors for use with
    `get_feature_extractor()`

    Args:
        example_type (str): The example type of the feature extractors
        features (dict): Features extractor templates keyed by name

    Raises:
        ValueError: If the example type is already registered
    """
    if example_type in FEATURE_MAP:
        msg = 'Features for example type {!r} are already registered.'.format(example_type)
        raise ValueError(msg)

    FEATURE_MAP[example_type] = features


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


def get_entity_scorer():
    return make_scorer(score_func=entity_accuracy_scoring)


def entity_accuracy_scoring(expected, predicted):
    num_examples = len(expected)
    num_correct = sum(1 for expected_seq, predicted_seq in zip(expected, predicted)
                      if entity_seqs_equal(expected_seq, predicted_seq))
    return float(num_correct) / float(num_examples)


def entity_seqs_equal(expected, predicted):
    if len(expected) != len(predicted):
        return False
    for i in range(len(expected)):
        if expected[i].entity.type != predicted[i].entity.type:
            return False
        if expected[i].span != predicted[i].span:
            return False
        if expected[i].text != predicted[i].text:
            return False
    return True
