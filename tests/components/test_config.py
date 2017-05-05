#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_config
----------------------------------

Tests for config module.
"""
# pylint: disable=locally-disabled,redefined-outer-name

import os

import pytest

from mmworkbench.components._config import _expand_parser_config

BASIC_PARSER_CONFIG = {
    'product': ['quantity', 'size', 'option'],
    'store': ['location'],
    'option': ['size']
}


def test_list_parser_config():
    """Tests that a parser config with list specs is correctly expanded"""
    config = {'option': ['size']}

    actual = _expand_parser_config(config)
    expected = {
        'option': {
            'size': {
                'left': True,
                'right': True,
                'min_instances': 0,
                'max_instances': None,
                'precedence': 'left',
                'linking_words': frozenset()
            }
        }
    }

    assert actual == expected


def test_dict_parser_config():
    """Tests that a parser config with dict specs is correctly expanded"""
    config = {'option': {'size': {}}}

    actual = _expand_parser_config(config)
    expected = {
        'option': {
            'size': {
                'left': True,
                'right': True,
                'min_instances': 0,
                'max_instances': None,
                'precedence': 'left',
                'linking_words': frozenset()
            }
        }
    }

    assert actual == expected


def test_dict_parser_config_2():
    """Tests that a parser config with dict specs is correctly expanded"""
    config = {'option': {'size': {'left': False}}}

    actual = _expand_parser_config(config)
    expected = {
        'option': {
            'size': {
                'left': False,
                'right': True,
                'min_instances': 0,
                'max_instances': None,
                'precedence': 'left',
                'linking_words': frozenset()
            }
        }
    }

    assert actual == expected
