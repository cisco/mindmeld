#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_config
----------------------------------

Tests for config module.
"""
# pylint: disable=locally-disabled,redefined-outer-name

import os

from mindmeld.components._config import (
    _expand_parser_config,
    get_classifier_config,
    get_custom_action_config,
    get_max_history_len,
)

APP_PATH = os.path.dirname(os.path.abspath(__file__))

BASIC_PARSER_CONFIG = {
    "product": ["quantity", "size", "option"],
    "store": ["location"],
    "option": ["size"],
}


def test_list_parser_config():
    """Tests that a parser config with list specs is correctly expanded"""
    config = {"option": ["size"]}

    actual = _expand_parser_config(config)
    expected = {
        "option": {
            "size": {
                "left": True,
                "right": True,
                "min_instances": 0,
                "max_instances": None,
                "precedence": "left",
                "linking_words": frozenset(),
            }
        }
    }

    assert actual == expected


def test_dict_parser_config():
    """Tests that a parser config with dict specs is correctly expanded"""
    config = {"option": {"size": {}}}

    actual = _expand_parser_config(config)
    expected = {
        "option": {
            "size": {
                "left": True,
                "right": True,
                "min_instances": 0,
                "max_instances": None,
                "precedence": "left",
                "linking_words": frozenset(),
            }
        }
    }

    assert actual == expected


def test_dict_parser_config_2():
    """Tests that a parser config with dict specs is correctly expanded"""
    config = {"option": {"size": {"left": False}}}

    actual = _expand_parser_config(config)
    expected = {
        "option": {
            "size": {
                "left": False,
                "right": True,
                "min_instances": 0,
                "max_instances": None,
                "precedence": "left",
                "linking_words": frozenset(),
            }
        }
    }

    assert actual == expected


def test_get_classifier_config():
    """Tests that the default config is returned when an app specified config doesn't exist."""
    actual = get_classifier_config("domain", APP_PATH)["param_selection"]

    expected = {
        "type": "k-fold",
        "k": 10,
        "grid": {"fit_intercept": [True, False], "C": [10, 100, 1000, 10000, 100000]},
    }

    assert actual == expected


def test_get_classifier_config2():
    """Tests that the app specified config is returned over the default config."""
    actual = get_classifier_config("intent", APP_PATH, domain="domain")[
        "param_selection"
    ]

    expected = {
        "type": "k-fold",
        "k": 5,
        "grid": {
            "fit_intercept": [True, False],
            "C": [1, 20, 300],
            "class_bias": [1, 0],
        },
    }

    assert actual == expected


def test_get_classifier_config_func():
    """Tests that the app config provider is called."""
    actual = get_classifier_config(
        "entity", APP_PATH, domain="domain", intent="intent"
    )["params"]

    expected = {"penalty": "l2", "C": 100, "solver": "liblinear"}

    assert actual == expected


def test_get_classifier_config_func_error():
    """Tests robustness to exceptions raised by a config provider."""
    actual = get_classifier_config("entity", APP_PATH, domain="domain", intent="error")[
        "params"
    ]

    expected = {"error": "intent", "penalty": "l2", "C": 100, "solver": "liblinear"}

    assert actual == expected


def test_custom_action_config(kwik_e_mart_app_path):
    actual = get_custom_action_config(kwik_e_mart_app_path)

    assert "url" in actual
    assert actual["url"] == "http://0.0.0.0:8080/"


def test_custom_action_config_no_config(home_assistant_app_path):
    actual = get_custom_action_config(home_assistant_app_path)

    assert actual is None


def test_max_history_len_config(food_ordering_app_path):
    actual = get_max_history_len(food_ordering_app_path)

    assert actual == 5
