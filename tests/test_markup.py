#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_markup
----------------------------------

Tests for `markup` module.
"""
from __future__ import unicode_literals

import pytest

from mmworkbench import markup

from mmworkbench.core import Entity, NestedEntity, ProcessedQuery, QueryEntity, Span

MARKED_UP_STRS = [
    'show me houses under {[600,000|sys:number] dollars|price}',
    'show me houses under {[$600,000|sys:number]|price}',
    'show me houses under {[1.5|sys:number] million dollars|price}',
    'play {s.o.b.|track}',
    "what's on at {[8 p.m.|sys:time]|range}?",
    'is {s.o.b.|show} gonna be on at {[8 p.m.|sys:time]|range}?',
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


@pytest.mark.mark_down
def test_mark_down():
    """Tests the mark down function"""
    text = 'is {s.o.b.|show} gonna be {[on at 8 p.m.|sys:time]|range}?'
    marked_down = markup.mark_down(text)
    assert marked_down == 'is s.o.b. gonna be on at 8 p.m.?'


@pytest.mark.load
def test_load_basic_query(query_factory):
    """Tests loading a basic query with no entities"""
    markup_text = 'This is a test query string'
    processed_query = markup.load_query(markup_text, query_factory)
    assert processed_query
    assert processed_query.query


@pytest.mark.load
def test_load_entity(query_factory):
    """Tests loading a basic query with an entity"""
    markup_text = 'When does the {Elm Street|store_name} store close?'

    processed_query = markup.load_query(markup_text, query_factory)

    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]
    assert entity.span.start == 14
    assert entity.span.end == 23
    assert entity.normalized_text == 'elm street'
    assert entity.entity.type == 'store_name'
    assert entity.entity.text == 'Elm Street'


@pytest.mark.load
@pytest.mark.system
def test_load_system(query_factory):
    """Tests loading a query with a system entity"""
    text = 'show me houses under {600,000 dollars|sys:currency}'
    processed_query = markup.load_query(text, query_factory)

    assert processed_query
    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]
    assert entity.text == '600,000 dollars'
    assert entity.entity.type == 'sys:currency'
    assert entity.span.start == 21
    assert not isinstance(entity.entity.value, str)

    assert entity.entity.value == {'unit': '$', 'value': 600000}


@pytest.mark.load
@pytest.mark.system
@pytest.mark.nested
def test_load_nested_system(query_factory):
    """Tests loading a query with a nested system entity"""
    text = 'show me houses under {[600,000|sys:number] dollars|price}'
    processed_query = markup.load_query(text, query_factory)

    assert processed_query
    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]
    assert entity.text == '600,000 dollars'
    assert entity.entity.type == 'price'

    assert entity.span == Span(21, 35)

    assert not isinstance(entity.entity.value, str)
    assert 'children' in entity.entity.value
    assert len(entity.entity.value['children']) == 1
    nested = entity.entity.value['children'][0]
    assert nested.text == '600,000'
    assert nested.span == Span(0, 6)
    assert nested.entity.type == 'sys:number'
    assert nested.entity.value == {'value': 600000}


@pytest.mark.load
@pytest.mark.system
@pytest.mark.nested
def test_load_nested_system_2(query_factory):
    """Tests loading a query with a nested system entity"""
    text = 'show me houses under {$[600,000|sys:number]|price}'
    processed_query = markup.load_query(text, query_factory)
    assert processed_query
    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]
    assert entity.text == '$600,000'
    assert entity.entity.type == 'price'
    assert entity.span == Span(21, 28)

    assert not isinstance(entity.entity.value, str)
    assert 'children' in entity.entity.value
    assert len(entity.entity.value['children']) == 1
    nested = entity.entity.value['children'][0]
    assert nested.text == '600,000'
    assert nested.entity.value == {'value': 600000}
    assert nested.span == Span(1, 7)


@pytest.mark.load
@pytest.mark.system
@pytest.mark.nested
def test_load_nested_system_3(query_factory):
    """Tests loading a query with a nested system entity"""
    text = 'show me houses under {[1.5 million|sys:number] dollars|price}'
    processed_query = markup.load_query(text, query_factory)

    assert processed_query


@pytest.mark.load
@pytest.mark.system
@pytest.mark.nested
def test_load_nested_system_4(query_factory):
    """Tests dumping a query with multiple nested system entities"""
    text = 'show me houses {between [600,000|sys:number] and [1,000,000|sys:number] dollars|price}'
    processed_query = markup.load_query(text, query_factory)

    assert processed_query
    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]
    assert entity.text == 'between 600,000 and 1,000,000 dollars'
    assert entity.entity.type == 'price'
    assert entity.span == Span(15, 51)

    assert not isinstance(entity.entity.value, str)
    assert 'children' in entity.entity.value
    assert len(entity.entity.value['children']) == 2
    lower, upper = entity.entity.value['children']

    assert lower.text == '600,000'
    assert lower.entity.value == {'value': 600000}
    assert lower.span == Span(8, 14)

    assert upper.text == '1,000,000'
    assert upper.entity.value == {'value': 1000000}
    assert upper.span == Span(20, 28)


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars(query_factory):
    """Tests loading a query with special characters"""
    text = 'play {s.o.b.|track}'
    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    assert len(entities)
    entity = entities[0]
    assert entity.text == 's.o.b.'
    assert entity.normalized_text == 's o b'
    assert entity.span.start == 5
    assert entity.span.end == 10


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_2(query_factory):
    """Tests loading a query with special characters"""
    text = "what's on at {[8 p.m.|sys:time]|range}?"
    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    assert len(entities) == 1

    entity = entities[0]
    assert entity.text == '8 p.m.'
    assert entity.normalized_text == '8 p m'
    assert entity.span == Span(13, 18)
    assert entity.entity.type == 'range'

    nested = entity.entity.value['children'][0]
    assert nested.text == '8 p.m.'
    assert nested.span == Span(0, 5)
    assert nested.entity.type == 'sys:time'
    assert nested.entity.value['value']


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_3(query_factory):
    """Tests loading a query with special characters"""
    text = 'is {s.o.b.|show} gonna be {[on at 8 p.m.|sys:time]|range}?'
    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    expected_entity = QueryEntity.from_query(processed_query.query, Span(3, 8), entity_type='show')
    assert entities[0] == expected_entity

    assert entities[1].entity.type == 'range'
    assert entities[1].span == Span(19, 30)
    assert 'children' in entities[1].entity.value
    assert entities[1].entity.value['children'][0].entity.type == 'sys:time'


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_4(query_factory):
    """Tests loading a query with special characters"""
    text = 'is {s.o.b.|show} ,, gonna be on at {[8 p.m.|sys:time]|range}?'

    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    expected_entity = QueryEntity.from_query(processed_query.query, Span(3, 8), entity_type='show')
    assert entities[0] == expected_entity

    assert entities[1].entity.type == 'range'
    assert entities[1].span == Span(28, 33)
    assert 'children' in entities[1].entity.value
    assert entities[1].entity.value['children'][0].entity.type == 'sys:time'


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_5(query_factory):
    """Tests loading a query with special characters"""
    text = 'what christmas movies   are  , showing at {[8pm|sys:time]|range}'

    processed_query = markup.load_query(text, query_factory)

    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]

    assert entity.span == Span(42, 44)
    assert entity.normalized_text == '8pm'


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_6(query_factory):
    """Tests loading a query with special characters"""
    text = "what's on {after [8 p.m.|sys:time]|range}?"
    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    assert len(entities) == 1

    assert entities[0].text == 'after 8 p.m.'
    assert entities[0].normalized_text == 'after 8 p m'
    assert entities[0].span == Span(10, 21)


@pytest.mark.dump
def test_dump_basic(query_factory):
    """Tests dumping a basic query"""
    query_text = 'A basic query'
    query = query_factory.create_query(query_text)
    processed_query = ProcessedQuery(query)

    assert markup.dump_query(processed_query) == query_text


@pytest.mark.dump
def test_dump_entity(query_factory):
    """Tests dumping a basic query with an entity"""
    query_text = 'When does the Elm Street store close?'
    query = query_factory.create_query(query_text)
    entities = [QueryEntity.from_query(query, Span(14, 23), entity_type='store_name')]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = 'When does the {Elm Street|store_name} store close?'
    assert markup.dump_query(processed_query) == markup_text


@pytest.mark.dump
def test_dump_entities(query_factory):
    """Tests dumping a basic query with two entities"""
    query_text = 'When does the Elm Street store close on Monday?'
    query = query_factory.create_query(query_text)
    entities = [QueryEntity.from_query(query, Span(14, 23), entity_type='store_name'),
                QueryEntity.from_query(query, Span(40, 45), entity_type='sys:time')]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = 'When does the {Elm Street|store_name} store close on {Monday|sys:time}?'
    assert markup.dump_query(processed_query) == markup_text


@pytest.mark.dump
@pytest.mark.nested
def test_dump_nested(query_factory):
    """Tests dumping a query with a nested system entity"""
    query_text = 'show me houses under 600,000 dollars'
    query = query_factory.create_query(query_text)

    nested = NestedEntity.from_query(query, Span(0, 6), parent_offset=21, entity_type='sys:number')
    raw_entity = Entity('600,000 dollars', 'price', value={'children': [nested]})
    entities = [QueryEntity.from_query(query, Span(21, 35), entity=raw_entity)]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = 'show me houses under {[600,000|sys:number] dollars|price}'
    assert markup.dump_query(processed_query) == markup_text


@pytest.mark.dump
@pytest.mark.nested
def test_dump_multi_nested(query_factory):
    """Tests dumping a query with multiple nested system entities"""
    query_text = 'show me houses between 600,000 and 1,000,000 dollars'
    query = query_factory.create_query(query_text)

    lower = NestedEntity.from_query(query, Span(8, 14), parent_offset=15, entity_type='sys:number')
    upper = NestedEntity.from_query(query, Span(20, 28), parent_offset=15, entity_type='sys:number')
    raw_entity = Entity('between 600,000 dollars and 1,000,000', 'price',
                        value={'children': [lower, upper]})
    entities = [QueryEntity.from_query(query, Span(15, 51), entity=raw_entity)]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = ('show me houses {between [600,000|sys:number] and '
                   '[1,000,000|sys:number] dollars|price}')

    assert markup.dump_query(processed_query) == markup_text
