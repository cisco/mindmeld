#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_markup
----------------------------------

Tests for `markup` module.
"""
from __future__ import unicode_literals

import pytest

from mmworkbench.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


def test_tokenize_raw(tokenizer):
    tokens = tokenizer.tokenize_raw('Test: Query for $500,000.')

    assert len(tokens)
    assert tokens[0]['text'] == 'Test:'
    assert tokens[0]['start'] == 0
    assert tokens[3]['start'] == 16
    assert tokens[3]['start'] == 16


def test_tokenize(tokenizer):
    tokens = tokenizer.tokenize('Test: Query for $500,000. Chyea!')

    assert len(tokens)
    assert tokens[0]['entity'] == 'test'
    assert tokens[0]['raw-entity'] == 'Test:'
    assert tokens[1]['raw-start'] == 6
    assert tokens[3]['raw-entity'] == '$500,000.'
    assert tokens[3]['raw-start'] == 16
    assert tokens[3]['entity'] == '$500,000'
    assert tokens[4]['entity'] == 'chyea'
    assert tokens[4]['raw-entity'] == 'Chyea!'
    assert tokens[4]['raw-start'] == 26


def test_normalize(tokenizer):
    normalized_text = tokenizer.normalize('Test: Query for $500,000.', False)

    assert normalized_text == 'test query for $500,000'


def test_normalize2(tokenizer):
    normalized_text = tokenizer.normalize('Test: Query for test.12.345..test,test', False)

    assert normalized_text == 'test query for test 12.345 test test'


def test_normalize3(tokenizer):
    normalized_text = tokenizer.normalize('Test: awesome band sigur rós.', False)

    assert normalized_text == 'test awesome band sigur ros'


def test_normalize4(tokenizer):
    normalized_text = tokenizer.normalize("D'Angelo's new album", False)

    assert normalized_text == "d angelo s new album"


def test_mapping(tokenizer):
    raw = 'Test: 1. 2. 3.'
    normalized = tokenizer.normalize(raw)

    assert normalized == 'test 1 2 3'

    forward, backward = tokenizer.generate_character_index_mappings(raw, normalized)

    assert forward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 4,
        6: 5,
        7: 5,
        8: 6,
        9: 7,
        10: 7,
        11: 8,
        12: 9,
        13: 9
    }

    assert backward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 5,
        5: 6,
        6: 8,
        7: 9,
        8: 11,
        9: 12,
    }
