#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_core
----------------------------------

Tests for `core` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mmworkbench.core import (Entity, QueryEntity, Span, NestedEntity,
                              TEXT_FORM_RAW, TEXT_FORM_PROCESSED, TEXT_FORM_NORMALIZED,
                              sort_by_lowest_time_grain)


@pytest.fixture
def query(query_factory):
    """Returns a basic query"""
    return query_factory.create_query('Test: One. 2. 3.')


def test_query(query_factory):
    """Tests creation of a query"""
    text = 'Test: 1. 2. 3.'
    query = query_factory.create_query(text)

    assert query.text == text
    assert query.processed_text == text
    assert query.normalized_text == 'test 1 2 3'


def test_query_with_leading_special_chars(query_factory):
    """Tests creation of a query"""
    text = ' Test: 1. 2. 3.'
    query = query_factory.create_query(text)

    assert query.text == text
    assert query.processed_text == text
    assert query.normalized_text == 'test 1 2 3'

    text = '"Test": 1. 2. 3.'
    query = query_factory.create_query(text)

    assert query.text == text
    assert query.processed_text == text
    assert query.normalized_text == 'test 1 2 3'

    text = '(Test": 1. 2. 3.)'
    query = query_factory.create_query(text)

    assert query.text == text
    assert query.processed_text == text
    assert query.normalized_text == 'test 1 2 3'

    text = '(())'
    query = query_factory.create_query(text)

    assert query.text == text
    assert query.processed_text == text
    assert query.normalized_text == ''


def test_transform_index_forward(query):
    """Tests transforming a char index from raw to processed or normalized"""
    raw_index = 6
    raw_char = query.text[raw_index]

    proc_index = query.transform_index(raw_index, TEXT_FORM_RAW, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    norm_index = query.transform_index(raw_index, TEXT_FORM_RAW, TEXT_FORM_NORMALIZED)
    norm_char = query.normalized_text[norm_index]

    assert raw_char == 'O'

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert norm_index == 5
    assert norm_char == 'o'


def test_transform_index_backward(query):
    """Tests transforming a char index from normalized to processed or raw"""
    norm_index = 5
    norm_char = query.normalized_text[norm_index]

    proc_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    raw_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_char = query.text[raw_index]

    assert norm_char == 'o'

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert raw_index == 6
    assert raw_char == 'O'


def test_transform_index_backward_2(query):
    """Tests transforming a char index from normalized to processed or raw"""
    norm_index = 7
    norm_char = query.normalized_text[norm_index]

    proc_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    raw_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_char = query.text[raw_index]

    assert norm_char == 'e'

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert raw_index == 8
    assert raw_char == 'e'


def test_transform_index_backward_3(query):
    """Tests transforming a char index from normalized to processed or raw"""
    norm_index = 8
    norm_char = query.normalized_text[norm_index]

    proc_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_char = query.processed_text[proc_index]

    raw_index = query.transform_index(norm_index, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_char = query.text[raw_index]

    assert norm_char == ' '

    assert proc_index == raw_index
    assert proc_char == raw_char

    assert raw_index == 10
    assert raw_char == ' '


def test_transform_span_forward(query):
    """Tests transforming a char span from raw to processed or normalized"""
    raw_span = Span(0, 9)
    raw_text = query.text[raw_span.start:raw_span.end + 1]

    proc_span = query.transform_span(raw_span, TEXT_FORM_RAW, TEXT_FORM_PROCESSED)
    proc_text = query.processed_text[proc_span.start:proc_span.end + 1]

    norm_span = query.transform_span(raw_span, TEXT_FORM_RAW, TEXT_FORM_NORMALIZED)
    norm_text = query.normalized_text[norm_span.start:norm_span.end + 1]

    assert raw_text == 'Test: One.'

    assert proc_span == raw_span
    assert proc_text == raw_text

    assert norm_span == Span(0, 7)
    assert norm_text == 'test one'


def test_transform_span_backward(query):
    """Tests transforming a char span from normalized to processed or raw"""
    norm_span = Span(0, 7)
    norm_text = query.normalized_text[norm_span.start:norm_span.end + 1]

    proc_span = query.transform_span(norm_span, TEXT_FORM_NORMALIZED, TEXT_FORM_PROCESSED)
    proc_text = query.processed_text[proc_span.start:proc_span.end + 1]

    raw_span = query.transform_span(norm_span, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)
    raw_text = query.text[raw_span.start:raw_span.end + 1]

    assert norm_text == 'test one'

    assert proc_span == raw_span
    assert proc_text == raw_text

    assert raw_span == Span(0, 8)
    assert raw_text == 'Test: One'


def test_span_iter():
    """Tests the __iter__ implementation for Span objects"""
    span = Span(3, 5)
    indexes = list(span)
    assert indexes == [3, 4, 5]


def test_query_equality(query_factory):
    """Tests query equality"""
    query_a = query_factory.create_query('Hello. There.')
    query_b = query_factory.create_query('Hello. There.')

    assert query_a == query_b


def test_query_equality_2(query_factory):
    """Tests query equality"""
    query_a = query_factory.create_query('Hello. There.', time_zone='America/Los_Angeles')
    query_b = query_factory.create_query('Hello. There.', time_zone='America/Bahia')

    assert query_a != query_b


def test_query_entity_equality():
    """Tests query entity equality"""
    entity_a = QueryEntity(('Entity', 'Entity', 'entity'), (Span(0, 5),)*3, (Span(0, 0),)*3,
                           Entity('text', 'type', 'role', 'value', 'display'))
    entity_b = QueryEntity(('Entity', 'Entity', 'entity'), (Span(0, 5),)*3, (Span(0, 0),)*3,
                           Entity('text', 'type', 'role', 'value', 'display'))

    assert entity_a == entity_b


def test_entity_equality():
    """Tests entity equality"""
    entity_a = Entity('text', 'type', 'role', 'value', 'display')
    entity_b = Entity('text', 'type', 'role', 'value', 'display')

    assert entity_a == entity_b


def test_entity_string():
    """Tests string representation of entities"""
    entity = NestedEntity(('Entity', 'Entity', 'entity'), (Span(0, 5),)*3, (Span(0, 0),)*3,
                          Entity('text', 'type', 'role', 'value', 'display'))
    assert str(entity) == "type:role 'Entity' 0-5"


def test_nested_and_query_entity_equality():
    """Tests NestedEntity and QueryEntity equality operations"""
    entity_a = NestedEntity(('Entity', 'Entity', 'entity'), (Span(0, 5),)*3, (Span(0, 0),)*3,
                            Entity('text', 'type', 'role', 'value', 'display'))
    entity_b = QueryEntity(('Entity', 'Entity', 'entity'), (Span(0, 5),)*3, (Span(0, 0),)*3,
                           Entity('text', 'type', 'role', 'value', 'display'))

    assert entity_a == entity_b
    assert not entity_a != entity_b

    entity_a = NestedEntity(('Entity123', 'Entity', 'entity'), (Span(0, 5),)*3, (Span(0, 0),)*3,
                            Entity('text', 'type', 'role', 'value', 'display'))
    entity_b = QueryEntity(('Entity', 'Entity', 'entity'), (Span(0, 5),)*3, (Span(0, 0),)*3,
                           Entity('text', 'type', 'role', 'value', 'display'))

    assert not entity_a == entity_b
    assert entity_a != entity_b


def test_create_entity_from_query(query_factory):
    """Tests the QueryEntity generated has the correct character indices based on raw query"""
    query = query_factory.create_query("!!Connect me with Paul's meeting room please.")
    norm_span = Span(16, 19)
    entity = QueryEntity.from_query(query, normalized_span=norm_span, entity_type='test_type')

    assert entity.span.start == 18
    assert entity.span.end == 21


def test_query_time_zone(query_factory):
    """Tests that a query created with a time uses the correct time zone for system entities"""
    query = query_factory.create_query('Today at noon', time_zone='America/Bahia')

    assert len(query.system_entity_candidates) == 4

    entity_times = [qe.entity.value['value'][10:] for qe in query.system_entity_candidates]

    # time in 'America/Bahia' is always -3 (no daylight savings time)
    assert 'T12:00:00.000-03:00' in entity_times


def test_query_time_zone_2(query_factory):
    """Tests that a query created with a time uses the correct time zone for system entities"""
    query = query_factory.create_query('Today at noon', time_zone='UTC')

    assert len(query.system_entity_candidates) == 4

    entity_times = [qe.entity.value['value'][10:] for qe in query.system_entity_candidates]
    assert 'T12:00:00.000+00:00' in entity_times


def test_query_timestamp(query_factory):
    """Tests that a query created with a time uses the correct time zone for system entities"""
    query = query_factory.create_query('Today at noon', time_zone='America/Bahia',
                                       timestamp=1516748906000)

    assert len(query.system_entity_candidates) == 4

    entity_predictions = [qe.entity.value['value'] for qe in query.system_entity_candidates]

    # time in 'America/Bahia' is always -3 (no daylight savings time)
    assert '2018-01-23T12:00:00.000-03:00' in entity_predictions


def test_sort_system_entities(query_factory):
    """Tests that sorting sys:time QueryEntities works correctly"""
    query = query_factory.create_query(
        'Can we do today or next week or 3pm or 2017 or in 20 minutes '
        'or next quarter or at 6:30pm or in February',
        timestamp=1516748906000
    )

    time_entities = [e for e in query.system_entity_candidates if e.entity.type == 'sys_time']

    assert len(time_entities) == 19

    time_entities = sort_by_lowest_time_grain(time_entities)

    assert time_entities[0].entity.value['grain'] == 'year'
    assert time_entities[1].entity.value['grain'] == 'year'
    assert time_entities[2].entity.value['grain'] == 'quarter'
    assert time_entities[3].entity.value['grain'] == 'month'
    assert time_entities[4].entity.value['grain'] == 'month'
    assert time_entities[5].entity.value['grain'] == 'week'
    assert time_entities[6].entity.value['grain'] == 'day'
    assert time_entities[7].entity.value['grain'] == 'hour'
    assert time_entities[8].entity.value['grain'] == 'hour'
    assert time_entities[9].entity.value['grain'] == 'hour'
    assert time_entities[10].entity.value['grain'] == 'hour'
    assert time_entities[11].entity.value['grain'] == 'hour'
    assert time_entities[12].entity.value['grain'] == 'minute'
    assert time_entities[13].entity.value['grain'] == 'minute'
    assert time_entities[14].entity.value['grain'] == 'minute'
    assert time_entities[15].entity.value['grain'] == 'minute'
    assert time_entities[16].entity.value['grain'] == 'minute'
    assert time_entities[17].entity.value['grain'] == 'second'
    assert time_entities[18].entity.value['grain'] == 'second'
    # Note: could not find a query that would yield a millisecond entity
