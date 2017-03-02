#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_path
----------------------------------

Tests for `path` module.
"""

import os

from mmworkbench import path

APP_NAME = 'kwik-e-mart'
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)

DOMAIN_NAME = 'store_info'
DOMAINS = set([DOMAIN_NAME])

INTENTS = set(['exit', 'find_nearest_store', 'get_store_hours', 'greet', 'help'])


def test_get_domains():
    domains = set(path.get_domains(APP_PATH))
    assert len(domains) == 1
    assert DOMAIN_NAME in domains


def test_get_intents():
    intents = path.get_intents(APP_PATH, DOMAIN_NAME)
    assert intents == INTENTS


def test_get_labeled_query_tree():
    tree = path.get_labeled_query_tree(APP_PATH)
    assert set(tree.keys()) == DOMAINS
    assert set(tree[DOMAIN_NAME].keys()) == INTENTS


def test_get_entity_types():
    entity_types = path.get_entity_types(APP_PATH)
    assert len(entity_types) == 1
    assert 'location' in entity_types


def test_get_indexes():
    indexes = path.get_indexes(APP_PATH)
    assert len(indexes) == 1
    assert 'stores' in indexes
