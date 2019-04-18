#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_markup
----------------------------------

Tests for `markup` module.
"""
# pylint: disable=I0011,W0621
import pytest

from mindmeld.tokenizer import Tokenizer


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
    assert tokens[0]['raw_entity'] == 'Test:'
    assert tokens[1]['raw_start'] == 6
    assert tokens[3]['raw_entity'] == '$500,000.'
    assert tokens[3]['raw_start'] == 16
    assert tokens[3]['entity'] == '$500,000'
    assert tokens[4]['entity'] == 'chyea'
    assert tokens[4]['raw_entity'] == 'Chyea!'
    assert tokens[4]['raw_start'] == 26


def test_normalize(tokenizer):
    normalized_text = tokenizer.normalize('Test: Query for $500,000.', False)

    assert normalized_text == 'test query for $500,000'


def test_normalize_2(tokenizer):
    normalized_text = tokenizer.normalize('Test: Query for test.12.345..test,test', False)

    assert normalized_text == 'test query for test 12.345 test test'


def test_normalize_3(tokenizer):
    normalized_text = tokenizer.normalize('Test: awesome band sigur rós.', False)

    assert normalized_text == 'test awesome band sigur ros'


def test_normalize_4(tokenizer):
    normalized_text = tokenizer.normalize("D'Angelo's new album", False)

    assert normalized_text == "d angelo s new album"


def test_normalize_5(tokenizer):
    raw = 'is s.o.b. ,, gonna be on at 8 p.m.?'
    normalized = tokenizer.normalize(raw, False)

    assert normalized == 'is s o b gonna be on at 8 p m'


def test_normalize_apos(tokenizer):

    # verify that apostrophe at the end of a possessive form is removed
    raw = "join Dennis' pmr"
    normalized = tokenizer.normalize(raw, True)

    assert normalized == "join dennis pmr"

    # verify that apostrophe in the middle of an entity is not removed
    raw = "join O'reilly's pmr"
    normalized = tokenizer.normalize(raw, True)

    assert normalized == "join o'reilly 's pmr"


def test_mapping(tokenizer):
    raw = 'Test: 1. 2. 3.'
    normalized = tokenizer.normalize(raw)

    assert normalized == 'test 1 2 3'

    forward, backward = tokenizer.get_char_index_map(raw, normalized)

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


def test_mapping_2(tokenizer):
    raw = 'is s.o.b. ,, gonna be on at 8 p.m.?'
    normalized = tokenizer.normalize(raw, False)

    assert normalized == 'is s o b gonna be on at 8 p m'

    forward, backward = tokenizer.get_char_index_map(raw, normalized)

    assert forward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 7,
        9: 8,
        10: 8,
        11: 8,
        12: 8,
        13: 9,
        14: 10,
        15: 11,
        16: 12,
        17: 13,
        18: 14,
        19: 15,
        20: 16,
        21: 17,
        22: 18,
        23: 19,
        24: 20,
        25: 21,
        26: 22,
        27: 23,
        28: 24,
        29: 25,
        30: 26,
        31: 27,
        32: 28,
        33: 28,
        34: 28
    }

    assert backward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 9,
        9: 13,
        10: 14,
        11: 15,
        12: 16,
        13: 17,
        14: 18,
        15: 19,
        16: 20,
        17: 21,
        18: 22,
        19: 23,
        20: 24,
        21: 25,
        22: 26,
        23: 27,
        24: 28,
        25: 29,
        26: 30,
        27: 31,
        28: 32
    }
