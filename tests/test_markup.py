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
from mmworkbench import markup


MARKED_UP_STRS = [
    'show me houses under {[600,000|num:number] dollars|price}',
    'show me houses under {[$600,000|num:number]|price}',
    'show me houses under {[1.5|num:number] million dollars|price}',
    'play {s.o.b.|track}',
    "what's on at {[8 p.m.|num:time]|range}?",
    'is {s.o.b.|show} gonna be on at {[8 p.m.|num:time]|range}?',
    'this is a {role model|type|role}',
    'this query has no entities'
]

MARKED_DOWN_STRS = [
    'show me houses under 600,000 dollars',
    'show me houses under $600,000',
    'show me houses under 1.5 million dollars',
    'play s.o.b.',
    "what's on at 8 p.m.?",
    'is s.o.b. gonna be on at 8 p.m.?',
    'this is a role model',
    'this query has no entities'
]


@pytest.fixture
def tokenizer():
    return Tokenizer()


def test_mark_down():
    text = 'is {s.o.b.|show} gonna be on at {[8 p.m.|num:time]|range}?'
    marked_down = markup.mark_down(text)
    assert marked_down == 'is s.o.b. gonna be on at 8 p.m.?'


def test_parse_basic_query(tokenizer):
    markup_text = 'This is a test query string'
    processed_query = markup.create_processed_query(markup_text, tokenizer)
    assert processed_query
    assert processed_query.query


def test_parse_entity(tokenizer):
    markup_text = 'When does the {Elm Street|store_name} store close?'

    processed_query = markup.create_processed_query(markup_text, tokenizer)

    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]
    assert entity.start == 14
    assert entity.end == 24
    assert entity.normalized_text == 'elm street'
    assert entity.entity.type == 'store_name'


def test_parse_numerics(tokenizer):
    text = 'show me houses under {[600,000|num:number] dollars|price}'
    processed_query = markup.create_processed_query(text, tokenizer)

    assert processed_query
    assert len(processed_query.entities) == 0

    entity = processed_query.entities[0]
    assert entity.raw_text == '600,000 dollars'
    assert entity.type == 'price'


def test_parse_numerics_2(tokenizer):
    text = 'show me houses under {[$600,000|num:number]|price}'
    processed_query = markup.create_processed_query(text, tokenizer)
    assert processed_query


def test_parse_numerics_3(tokenizer):
    text = 'show me houses under {[1.5|num:number] million dollars|price}'
    processed_query = markup.create_processed_query(text, tokenizer)
    assert processed_query


def test_parse_special_chars(tokenizer):
    text = 'play {s.o.b.|track}'
    # 'play {s o b|track}'
    processed_query = markup.create_processed_query(text, tokenizer)
    entities = processed_query.entities

    assert len(entities)
    entity = entities[0]
    # assert entity.raw_text == 's.o.b.'
    assert entity.normalized_text == 's o b'
    assert entity.start == 5
    assert entity.end == 10


def test_parse_special_chars_2(tokenizer):
    text = "what's on {[8 p.m.|num:time]|range}?"
    processed_query = markup.create_processed_query(text, tokenizer)
    entities = processed_query.entities

    assert len(entities)
    assert entities[0].normalized_text == '8 p m'
    assert entities[0].start == 10
    assert entities[0].end == 15


def test_parse_special_chars_3(tokenizer):
    text = 'is {s.o.b.|show} gonna be on at {[8 p.m.|num:time]|range}?'
    processed_query = markup.create_processed_query(text, tokenizer)
    entities = processed_query.entities

    assert len(entities)
    assert entities[0].start == 3
    assert entities[0].end == 8
    assert entities[0].normalized_text == 's o b'

    assert entities[1].start == 25
    assert entities[1].end == 30
    assert entities[1].normalized_text == '8 p m'


def test_parse_special_chars_4(tokenizer):
    text = 'is {s.o.b.|show} ,, gonna be on at {[8 p.m.|num:time]|range}?'
    processed_query = markup.create_processed_query(text, tokenizer)
    entities = processed_query.entities

    assert entities[0].start == 3
    assert entities[0].end == 8
    assert entities[0].normalized_text == 's o b'

    assert entities[1].start == 28
    assert entities[1].end == 33
    assert entities[1].normalized_text == '8 p m'


@pytest.mark.focus
def test_parse_special_chars_5(tokenizer):
    text = 'what christmas movies   are  , showing {[at 8pm|num:time]|range}'

    processed_query = markup.create_processed_query(text, tokenizer)
    entities = processed_query.entities

    assert entities[0].start == 39
    assert entities[0].end == 45
    assert entities[0].normalized_text == 'at 8pm'
