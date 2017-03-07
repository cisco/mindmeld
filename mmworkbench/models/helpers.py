# -*- coding: utf-8 -*-
"""This module contains some helper functions for the models package"""

import re

FEATURE_MAP = {}
MODEL_MAP = {}

# resource/requirements names
GAZETTEER_RSC = "gaz"
WORD_FREQ_RSC = "w_freq"
QUERY_FREQ_RSC = "q_freq"


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


def model(model_type):
    """Decorator for registering a model class


    """

    def _decorator(cls):
        register_model(model_type, cls)
    return _decorator


def register_model(model_type, model_class):
    """Helper for registering models

    Args:
        model_type (str): The model type as specified in model configs
        model_class (type): The model to register

    """
    if model_type in MODEL_MAP:
        raise ValueError('Model {!r} is already registered.'.format(model_type))

    MODEL_MAP[model_type] = model_class


def register_features(example_type, features):
    """Helper for registering feature extractors for a particular example type

    Args:
        example_type (str): The example type of the feature extractors
        features (dict): Features listed by name

    Raises:
        ValueError: If the example type is already registered

    """
    if example_type in FEATURE_MAP:
        msg = 'Features for example type {!r} are already registered.'.format(example_type)
        raise ValueError(msg)

    FEATURE_MAP[example_type] = features


def mask_numerics(token):
    """Masks digit characters in a token"""
    if token.isdigit():
        return '#NUM'
    else:
        return re.sub(r'\d', '8', token)
