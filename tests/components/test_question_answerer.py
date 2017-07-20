#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_question_answerer
----------------------------------

Tests for `question_answerer` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import pytest
import os

from mmworkbench.components.question_answerer import QuestionAnswerer
from mmworkbench.components._elasticsearch_helpers import create_es_client

ENTITY_TYPE = 'store_name'
APP_PATH = '../kwik_e_mart'
DATA_FILE_PATH = os.path.dirname(__file__) + "/../kwik_e_mart/data/stores.json"

@pytest.fixture
def es_client():
    """An Elasticsearch client"""
    return create_es_client()


@pytest.fixture
def answerer(resource_loader, es_client):
    QuestionAnswerer.load_kb(app_name='kwik_e_mart', index_name='store_name',
                             data_file=DATA_FILE_PATH)
    qa = QuestionAnswerer(APP_PATH)
    es_client.indices.flush(index='_all')
    return qa


def test_basic_search(answerer):
    """Test basic search."""

    # retrieve object using ID
    res = answerer.get(index='store_name', id='20')
    assert len(res) > 0

    # simple text query
    res = answerer.get(index='store_name', store_name='peanut')
    assert len(res) > 0

    # simple text query
    res = answerer.get(index='store_name', store_name='Springfield Heights')
    assert len(res) > 0

    # multiple text queries
    res = answerer.get(index='store_name', store_name='peanut', address='peanut st')
    assert len(res) > 0


def test_advanced_search(answerer):
    """Test advanced search."""

    s = answerer.build_search(index='store_name')
    res = s.query(store_name='peanut').execute()
    assert len(res) > 0


def test_partial_match(answerer):

    # test partial match
    res = answerer.get(index='store_name', store_name='Garden')
    assert len(res) > 0


def test_sort_by_distance(answerer):
    # retrieve object using ID
    res = answerer.get(index='store_name', _sort='location', _sort_type='distance',
                       _sort_location='44.24,-123.12')
    assert len(res) > 0
    assert res[0].get('id') == '19'
