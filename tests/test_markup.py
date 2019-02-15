#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_markup
----------------------------------

Tests for `markup` module.
"""
import pytest

from mmworkbench import markup

from mmworkbench.core import Entity, NestedEntity, ProcessedQuery, QueryEntity, Span

MARKED_UP_STRS = [
    'show me houses under {[600,000|sys_number] dollars|price}',
    'show me houses under {[$600,000|sys_number]|price}',
    'show me houses under {[1.5|sys_number] million dollars|price}',
    'play {s.o.b.|track}',
    "what's on at {[8 p.m.|sys_time]|range}?",
    'is {s.o.b.|show} gonna be on at {[8 p.m.|sys_time]|range}?',
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
    text = 'is {s.o.b.|show} gonna be {{on at 8 p.m.|sys_time}|range}?'
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
    text = 'show me houses under {600,000 dollars|sys_amount-of-money}'
    processed_query = markup.load_query(text, query_factory)

    assert processed_query
    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]
    assert entity.text == '600,000 dollars'
    assert entity.entity.type == 'sys_amount-of-money'
    assert entity.span.start == 21
    assert not isinstance(entity.entity.value, str)

    assert entity.entity.value == {'unit': '$', 'value': 600000}


@pytest.mark.dump
@pytest.mark.system
@pytest.mark.role
def test_load_system_role(query_factory):
    """Tests loading a basic query with an entity with a role"""
    text = ('What stores are open between {3|sys_time|open_hours} and '
            '{5|sys_time|close_hours}')

    processed_query = markup.load_query(text, query_factory)

    assert len(processed_query.entities) == 2

    entity = processed_query.entities[0]
    assert entity.span.start == 29
    assert entity.span.end == 29
    assert entity.normalized_text == '3'
    assert entity.entity.type == 'sys_time'
    assert entity.entity.text == '3'
    assert entity.entity.role == 'open_hours'

    entity = processed_query.entities[1]
    assert entity.span.start == 35
    assert entity.span.end == 35
    assert entity.normalized_text == '5'
    assert entity.entity.type == 'sys_time'
    assert entity.entity.text == '5'
    assert entity.entity.role == 'close_hours'


@pytest.mark.load
@pytest.mark.system
@pytest.mark.nested
def test_load_nested(query_factory):
    """Tests loading a query with a nested system entity"""
    text = 'show me houses under {{600,000|sys_number} dollars|price}'

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
    assert nested.entity.type == 'sys_number'
    assert nested.entity.value == {'value': 600000}


@pytest.mark.load
@pytest.mark.system
@pytest.mark.nested
def test_load_nested_2(query_factory):
    """Tests loading a query with a nested system entity"""
    text = 'show me houses under {${600,000|sys_number}|price}'
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
def test_load_nested_3(query_factory):
    """Tests loading a query with a nested system entity"""
    text = 'show me houses under {{1.5 million|sys_number} dollars|price}'
    processed_query = markup.load_query(text, query_factory)

    assert processed_query


@pytest.mark.load
@pytest.mark.system
@pytest.mark.nested
def test_load_nested_4(query_factory):
    """Tests dumping a query with multiple nested system entities"""
    text = 'show me houses {between {600,000|sys_number} and {1,000,000|sys_number} dollars|price}'
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
    text = "what's on at {{8 p.m.|sys_time}|range}?"
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
    assert nested.entity.type == 'sys_time'
    assert nested.entity.value['value']


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_3(query_factory):
    """Tests loading a query with special characters"""
    text = 'is {s.o.b.|show} gonna be {{on at 8 p.m.|sys_time}|range}?'
    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    expected_entity = QueryEntity.from_query(processed_query.query, Span(3, 8), entity_type='show')
    assert entities[0] == expected_entity

    assert entities[1].entity.type == 'range'
    assert entities[1].span == Span(19, 30)
    assert 'children' in entities[1].entity.value
    assert entities[1].entity.value['children'][0].entity.type == 'sys_time'


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_4(query_factory):
    """Tests loading a query with special characters"""
    text = 'is {s.o.b.|show} ,, gonna be on at {{8 p.m.|sys_time}|range}?'

    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    expected_entity = QueryEntity.from_query(processed_query.query, Span(3, 8), entity_type='show')
    assert entities[0] == expected_entity

    assert entities[1].entity.type == 'range'
    assert entities[1].span == Span(28, 33)
    assert 'children' in entities[1].entity.value
    assert entities[1].entity.value['children'][0].entity.type == 'sys_time'


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_5(query_factory):
    """Tests loading a query with special characters"""
    text = 'what christmas movies   are  , showing at {{8pm|sys_time}|range}'

    processed_query = markup.load_query(text, query_factory)

    assert len(processed_query.entities) == 1

    entity = processed_query.entities[0]

    assert entity.span == Span(42, 44)
    assert entity.normalized_text == '8pm'


@pytest.mark.load
@pytest.mark.special
def test_load_special_chars_6(query_factory):
    """Tests loading a query with special characters"""
    text = "what's on {after {8 p.m.|sys_time}|range}?"
    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    assert len(entities) == 1

    assert entities[0].text == 'after 8 p.m.'
    assert entities[0].normalized_text == 'after 8 p m'
    assert entities[0].span == Span(10, 21)


@pytest.mark.load
@pytest.mark.group
def test_load_group(query_factory):
    """Tests loading a query with an entity group"""
    text = "a [{large|size} {latte|product} with {nonfat milk|option}|product] please"

    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    assert len(entities) == 3

    assert entities[0].text == 'large'
    assert entities[0].entity.type == 'size'
    assert entities[0].span == Span(2, 6)
    assert entities[0].parent == entities[1]

    assert entities[1].text == 'latte'
    assert entities[1].entity.type == 'product'
    assert entities[1].span == Span(8, 12)
    assert entities[1].children == (entities[0], entities[2])

    assert entities[2].text == 'nonfat milk'
    assert entities[2].entity.type == 'option'
    assert entities[2].span == Span(19, 29)
    assert entities[2].parent == entities[1]


@pytest.mark.load
@pytest.mark.group
def test_load_group_nested(query_factory):
    """Tests loading a query with a nested entity group"""
    text = ('Order [{one|quantity} {large|size} {Tesora|product} with [{medium|size} '
            '{cream|option}|option] and [{medium|size} {sugar|option}|option]|product]')

    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    assert len(entities) == 7

    assert entities[0].text == 'one'
    assert entities[0].entity.type == 'quantity'
    assert entities[0].span == Span(6, 8)
    assert entities[0].parent == entities[2]

    assert entities[1].text == 'large'
    assert entities[1].entity.type == 'size'
    assert entities[1].span == Span(10, 14)
    assert entities[1].parent == entities[2]

    assert entities[2].text == 'Tesora'
    assert entities[2].entity.type == 'product'
    assert entities[2].span == Span(16, 21)
    assert entities[2].children == (entities[0], entities[1], entities[4], entities[6])

    assert entities[3].text == 'medium'
    assert entities[3].entity.type == 'size'
    assert entities[3].span == Span(28, 33)
    assert entities[3].parent == entities[4]

    assert entities[4].text == 'cream'
    assert entities[4].entity.type == 'option'
    assert entities[4].span == Span(35, 39)
    assert entities[4].parent == entities[2]
    assert entities[4].children == (entities[3],)

    assert entities[5].text == 'medium'
    assert entities[5].entity.type == 'size'
    assert entities[5].span == Span(45, 50)
    assert entities[5].parent == entities[6]

    assert entities[6].text == 'sugar'
    assert entities[6].entity.type == 'option'
    assert entities[6].span == Span(52, 56)
    assert entities[6].parent == entities[2]
    assert entities[6].children == (entities[5],)


@pytest.mark.load
@pytest.mark.group
def test_load_groups(query_factory):
    """Tests loading a query with multiple top level entity groups"""
    text = ('Order [{one|quantity} {large|size} {Tesora|product} with '
            '[{medium|size} {cream|option}|option]|product] from '
            '[{Philz|store} in {Downtown Sunnyvale|location}|store]')

    processed_query = markup.load_query(text, query_factory)
    entities = processed_query.entities

    assert len(entities) == 7

    assert entities[0].text == 'one'
    assert entities[0].entity.type == 'quantity'
    assert entities[0].span == Span(6, 8)
    assert entities[0].parent == entities[2]

    assert entities[1].text == 'large'
    assert entities[1].entity.type == 'size'
    assert entities[1].span == Span(10, 14)
    assert entities[1].parent == entities[2]

    assert entities[2].text == 'Tesora'
    assert entities[2].entity.type == 'product'
    assert entities[2].span == Span(16, 21)
    assert entities[2].children == (entities[0], entities[1], entities[4])

    assert entities[3].text == 'medium'
    assert entities[3].entity.type == 'size'
    assert entities[3].span == Span(28, 33)
    assert entities[3].parent == entities[4]

    assert entities[4].text == 'cream'
    assert entities[4].entity.type == 'option'
    assert entities[4].span == Span(35, 39)
    assert entities[4].parent == entities[2]
    assert entities[4].children == (entities[3],)

    assert entities[5].text == 'Philz'
    assert entities[5].entity.type == 'store'
    assert entities[5].span == Span(46, 50)
    assert entities[5].children == (entities[6],)

    assert entities[6].text == 'Downtown Sunnyvale'
    assert entities[6].entity.type == 'location'
    assert entities[6].span == Span(55, 72)
    assert entities[6].parent == entities[5]


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
    assert markup.dump_query(processed_query, no_entity=True) == query_text


@pytest.mark.dump
def test_dump_role(query_factory):
    """Tests dumping a basic query with an entity with a role"""
    query_text = 'What stores are open between 3 and 5'
    query = query_factory.create_query(query_text)
    entities = [
        QueryEntity.from_query(query, Span(29, 29), entity_type='sys_time', role='open_hours'),
        QueryEntity.from_query(query, Span(35, 35), entity_type='sys_time', role='close_hours')
    ]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = ('What stores are open between {3|sys_time|open_hours} and '
                   '{5|sys_time|close_hours}')
    entity_text = 'What stores are open between {3|sys_time} and {5|sys_time}'
    assert markup.dump_query(processed_query) == markup_text
    assert markup.dump_query(processed_query, no_role=True) == entity_text
    assert markup.dump_query(processed_query, no_role=True, no_entity=True) == query_text


@pytest.mark.dump
def test_dump_entities(query_factory):
    """Tests dumping a basic query with two entities"""
    query_text = 'When does the Elm Street store close on Monday?'
    query = query_factory.create_query(query_text)
    entities = [QueryEntity.from_query(query, Span(14, 23), entity_type='store_name'),
                QueryEntity.from_query(query, Span(40, 45), entity_type='sys_time')]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = 'When does the {Elm Street|store_name} store close on {Monday|sys_time}?'
    assert markup.dump_query(processed_query) == markup_text
    assert markup.dump_query(processed_query, no_entity=True) == query_text


@pytest.mark.dump
@pytest.mark.nested
def test_dump_nested(query_factory):
    """Tests dumping a query with a nested system entity"""
    query_text = 'show me houses under 600,000 dollars'
    query = query_factory.create_query(query_text)

    nested = NestedEntity.from_query(query, Span(0, 6), parent_offset=21, entity_type='sys_number')
    raw_entity = Entity('600,000 dollars', 'price', value={'children': [nested]})
    entities = [QueryEntity.from_query(query, Span(21, 35), entity=raw_entity)]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = 'show me houses under {{600,000|sys_number} dollars|price}'
    assert markup.dump_query(processed_query) == markup_text
    assert markup.dump_query(processed_query, no_group=True) == markup_text
    assert markup.dump_query(processed_query, no_entity=True) == query_text


@pytest.mark.dump
@pytest.mark.nested
def test_dump_multi_nested(query_factory):
    """Tests dumping a query with multiple nested system entities"""
    query_text = 'show me houses between 600,000 and 1,000,000 dollars'
    query = query_factory.create_query(query_text)

    lower = NestedEntity.from_query(query, Span(8, 14), parent_offset=15, entity_type='sys_number')
    upper = NestedEntity.from_query(query, Span(20, 28), parent_offset=15, entity_type='sys_number')
    raw_entity = Entity('between 600,000 dollars and 1,000,000', 'price',
                        value={'children': [lower, upper]})
    entities = [QueryEntity.from_query(query, Span(15, 51), entity=raw_entity)]
    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = ('show me houses {between {600,000|sys_number} and '
                   '{1,000,000|sys_number} dollars|price}')

    assert markup.dump_query(processed_query) == markup_text
    assert markup.dump_query(processed_query, no_group=True) == markup_text
    assert markup.dump_query(processed_query, no_entity=True) == query_text


@pytest.mark.dump
@pytest.mark.group
def test_dump_group(query_factory):
    """Tests dumping a query with an entity group"""
    query_text = 'a large latte with nonfat milk please'
    query = query_factory.create_query(query_text)

    size = QueryEntity.from_query(query, Span(2, 6), entity_type='size')
    option = QueryEntity.from_query(query, Span(19, 29), entity_type='option')
    product = QueryEntity.from_query(query, Span(8, 12), entity_type='product',
                                     children=(size, option))

    processed_query = ProcessedQuery(query, entities=[size, product, option])
    markup_text = "a [{large|size} {latte|product} with {nonfat milk|option}|product] please"
    entity_text = "a {large|size} {latte|product} with {nonfat milk|option} please"
    group_text = "a [large latte with nonfat milk|product] please"

    assert markup.dump_query(processed_query) == markup_text
    assert markup.dump_query(processed_query, no_group=True) == entity_text
    assert markup.dump_query(processed_query, no_entity=True) == group_text
    assert markup.dump_query(processed_query, no_group=True, no_entity=True) == query_text


@pytest.mark.dump
@pytest.mark.group
def test_dump_group_nested(query_factory):
    """Tests dumping a query with nested entity groups"""
    query_text = 'Order one large Tesora with medium cream and medium sugar'

    query = query_factory.create_query(query_text)
    entities = [
        QueryEntity.from_query(query, Span(6, 8), entity_type='quantity'),
        QueryEntity.from_query(query, Span(10, 14), entity_type='size'),
        QueryEntity.from_query(query, Span(16, 21), entity_type='product'),
        QueryEntity.from_query(query, Span(28, 33), entity_type='size'),
        QueryEntity.from_query(query, Span(35, 39), entity_type='option'),
        QueryEntity.from_query(query, Span(45, 50), entity_type='size'),
        QueryEntity.from_query(query, Span(52, 56), entity_type='option')
    ]
    entities[4] = entities[4].with_children((entities[3],))
    entities[6] = entities[6].with_children((entities[5],))
    entities[2] = entities[2].with_children((entities[0], entities[1], entities[4], entities[6]))

    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = ('Order [{one|quantity} {large|size} {Tesora|product} with [{medium|size} '
                   '{cream|option}|option] and [{medium|size} {sugar|option}|option]|product]')
    entity_text = ('Order {one|quantity} {large|size} {Tesora|product} with {medium|size} '
                   '{cream|option} and {medium|size} {sugar|option}')
    group_text = ('Order [one large Tesora with [medium '
                  'cream|option] and [medium sugar|option]|product]')

    assert markup.dump_query(processed_query) == markup_text
    assert markup.dump_query(processed_query, no_group=True) == entity_text
    assert markup.dump_query(processed_query, no_entity=True) == group_text
    assert markup.dump_query(processed_query, no_group=True, no_entity=True) == query_text


@pytest.mark.dump
@pytest.mark.group
def test_dump_group_nested_2(query_factory):
    """Tests dumping a query with nested entity groups"""
    query_text = 'Can I get one curry sauce with my rice ball with house salad'

    query = query_factory.create_query(query_text)
    entities = [
        QueryEntity.from_query(query, Span(10, 12), entity_type='sys_number', role='quantity'),
        QueryEntity.from_query(query, Span(14, 24), entity_type='option'),
        QueryEntity.from_query(query, Span(34, 59), entity_type='dish')
    ]
    entities[1] = entities[1].with_children((entities[0],))
    entities[2] = entities[2].with_children((entities[1],))

    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = ('Can I get [[{one|sys_number|quantity} {curry sauce|option}|option] '
                   'with my {rice ball with house salad|dish}|dish]')
    entity_text = ('Can I get {one|sys_number|quantity} {curry sauce|option} '
                   'with my {rice ball with house salad|dish}')
    role_text = ('Can I get {one|quantity} curry sauce '
                 'with my rice ball with house salad')
    group_text = ('Can I get [[one curry sauce|option] '
                  'with my rice ball with house salad|dish]')

    assert markup.dump_query(processed_query) == markup_text
    assert markup.dump_query(processed_query, no_group=True) == entity_text
    assert markup.dump_query(processed_query, no_group=True, no_entity=True) == role_text
    assert markup.dump_query(processed_query, no_entity=True, no_role=True) == group_text
    assert markup.dump_query(processed_query,
                             no_group=True, no_entity=True, no_role=True) == query_text


@pytest.mark.dump
@pytest.mark.group
def test_dump_groups(query_factory):
    """Tests dumping a query with multiple top level entity groups"""
    query_text = 'Order one large Tesora with medium cream from Philz in Downtown Sunnyvale'

    query = query_factory.create_query(query_text)
    entities = [
        QueryEntity.from_query(query, Span(6, 8), entity_type='quantity'),
        QueryEntity.from_query(query, Span(10, 14), entity_type='size'),
        QueryEntity.from_query(query, Span(16, 21), entity_type='product'),
        QueryEntity.from_query(query, Span(28, 33), entity_type='size'),
        QueryEntity.from_query(query, Span(35, 39), entity_type='option'),
        QueryEntity.from_query(query, Span(46, 50), entity_type='store'),
        QueryEntity.from_query(query, Span(55, 72), entity_type='location')
    ]
    entities[4] = entities[4].with_children((entities[3],))
    entities[2] = entities[2].with_children((entities[0], entities[1], entities[4]))
    entities[5] = entities[5].with_children((entities[6],))

    processed_query = ProcessedQuery(query, entities=entities)

    markup_text = ('Order [{one|quantity} {large|size} {Tesora|product} with '
                   '[{medium|size} {cream|option}|option]|product] from '
                   '[{Philz|store} in {Downtown Sunnyvale|location}|store]')

    assert markup.dump_query(processed_query) == markup_text


@pytest.mark.load
@pytest.mark.dump
@pytest.mark.group
def test_load_dump_groups(query_factory):
    """Tests that load_query and dump_query are reversible"""
    text = ('Order [{one|quantity} {large|size} {Tesora|product} with '
            '[{medium|size} {cream|option}|option]|product] from '
            '[{Philz|store} in {Downtown Sunnyvale|location}|store]')

    processed_query = markup.load_query(text, query_factory)

    markup_text = markup.dump_query(processed_query)

    assert text == markup_text


@pytest.mark.load
@pytest.mark.dump
@pytest.mark.group
def test_load_dump_groups_roles(query_factory):
    """Tests that load_query and dump_query are reversible"""
    text = ('Order [{one|sys_number|quantity} {large|size} {Tesora|product|dish} with '
            '[{medium|size} {cream|option|addin}|option]|product]')

    processed_query = markup.load_query(text, query_factory)

    markup_text = markup.dump_query(processed_query)

    assert text == markup_text


@pytest.mark.load
@pytest.mark.dump
def test_load_dump_2(query_factory):
    """Tests that load_query and dump_query are reversible"""
    text = ("i'm extra hungry get me a {chicken leg|dish}, [{1|quantity} "
            "{kheema nan|dish}|dish] [{2|quantity} regular {nans|dish}|dish] "
            "[{one|quantity} {chicken karahi|dish}|dish], [{1|quantity} "
            "{saag paneer|dish}|dish] and [{1|quantity} {chicken biryani|dish}|dish]")

    processed_query = markup.load_query(text, query_factory)

    markup_text = markup.dump_query(processed_query)

    assert text == markup_text


def test_bootstrap_query_with_entities(query_factory):
    query_text = 'Can I get one curry sauce with my rice ball with house salad'

    query = query_factory.create_query(query_text)
    entities = [
        QueryEntity.from_query(query, Span(10, 12), entity_type='sys_number', role='quantity'),
        QueryEntity.from_query(query, Span(14, 24), entity_type='option'),
        QueryEntity.from_query(query, Span(34, 59), entity_type='dish')
    ]
    entities[1] = entities[1].with_children((entities[0],))
    entities[2] = entities[2].with_children((entities[1],))
    confidence = {
        'domains': {
            'food': 0.95,
            'music': 0.05
        },
        'intents': {
            'get_comestibles': 0.99,
            'reorder': 0.01
        },
        'entities': [
            {'sys_number': 0.9},
            {'option': 0.99},
            {'dish': 0.65}
        ],
        'roles': [
            {'quantity': 0.8, 'quality': 0.2},
            None,
            None
        ]
    }

    processed_query = ProcessedQuery(
        query, domain='food', intent='get_comestibles', entities=entities, confidence=confidence
    )
    bootstrap_data = markup.bootstrap_query_row(processed_query, show_confidence=True)

    expected_data = {
        'query': ('Can I get [[{one|sys_number|quantity} {curry sauce|option}|option] '
                  'with my {rice ball with house salad|dish}|dish]'),
        'domain': 'food',
        'domain_conf': 0.95,
        'intent': 'get_comestibles',
        'intent_conf': 0.99,
        'entity_conf': 0.65,
        'role_conf': 0.8
    }
    assert bootstrap_data == expected_data


def test_bootstrap_query_no_entity(query_factory):
    """"Tests bootstrap output for a query without entities"""
    query_text = 'cancel the timer'
    query = query_factory.create_query(query_text)
    confidence = {
        'domains': {
            'times_and_dates': 0.95,
            'espionage': 0.05
        },
        'intents': {
            'stop_timer': 0.9,
            'start_timer': 0.07,
            'cut_blue_wire': 0.03
        },
        'entities': [],
        'roles': []
    }

    processed_query = ProcessedQuery(
        query, domain='times_and_dates', intent='stop_timer', entities=[], confidence=confidence
    )
    bootstrap_data = markup.bootstrap_query_row(processed_query, show_confidence=True)

    expected_data = {
        'query': 'cancel the timer',
        'domain': 'times_and_dates',
        'domain_conf': 0.95,
        'intent': 'stop_timer',
        'intent_conf': 0.9,
        'entity_conf': 1.0,
        'role_conf': 1.0
    }
    assert bootstrap_data == expected_data
