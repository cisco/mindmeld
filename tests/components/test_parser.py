#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_parser
----------------------------------

Tests for parser module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mindmeld import markup
from mindmeld.components.parser import Parser
from mindmeld.exceptions import ParserTimeout


class TestBasicParser:
    """A set of tests for a basic parser with no constraints"""
    CONFIG = {'head': ['dependent']}

    @classmethod
    def setup_class(cls):
        """Creates the parser for this group of tests"""
        cls.parser = Parser(config=cls.CONFIG)

    def test_no_entities(self):
        """Tests the parser returns no groups when there are no entities"""
        query = markup.load_query('Hello there')
        entities = self.parser.parse_entities(query.query, query.entities)

        assert entities == ()

    def test_singleton(self):
        """Tests the parser returns no groups when a head has no dependents"""
        query = markup.load_query('Hello {there|head}')

        entities = self.parser.parse_entities(query.query, query.entities, timeout=None)

        assert entities == query.entities

    def test_left(self):
        """Tests the parser attaches dependents from the left"""
        query = markup.load_query('{Hello|dependent} {there|head}')
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 2
        assert entities[0].parent == entities[1]
        assert entities[1].children == (entities[0],)

    def test_right(self):
        """Tests the parser attaches dependents from the right"""
        query = markup.load_query('{Hello|head} {there|dependent}')
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 2
        assert entities[0].children == (entities[1],)
        assert entities[1].parent == entities[0]

    def test_distance(self):
        """Tests the parser attaches dependents to their nearest head"""
        query = markup.load_query('{Hello|head} {there|dependent} my {friend|head}')
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 3
        assert entities[0].children == (entities[1],)
        assert entities[1].parent == entities[0]
        assert entities[2].children is None

    def test_unconfigured(self):
        """Tests the parser functions when unconfigured entities are present"""
        query = markup.load_query('{Hello|head} {there|other}')
        entities = self.parser.parse_entities(query.query, query.entities)

        assert entities


class TestRoleParser:
    """A set of tests for a parser which has nested groups"""
    CONFIG = {
        'dish|beverage': ['option|beverage', 'size'],
        'dish': ['option', 'size']
    }

    @classmethod
    def setup_class(cls):
        """Creates the parser for this group of tests"""
        cls.parser = Parser(config=cls.CONFIG)

    def test_generic(self):
        """Tests groups where no roles are specified in the config"""
        query = markup.load_query('{noodles|dish|main_course} with {tofu|option}')
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 2
        assert entities[0].children == (entities[1],)
        assert entities[1].parent == entities[0]

    def test_with_role(self):
        """Tests groups when roles are explicitly specified in the config"""
        text = '{large|size} {latte|dish|beverage} {ice|option|beverage}'
        query = markup.load_query(text)
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 3
        assert entities[0].parent == entities[1]
        assert entities[1].children == (entities[0], entities[2])

        text = 'I’d like a {muffin|dish|baked_good} with {no sugar|option|beverage}'
        query = markup.load_query(text)
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 2
        assert entities[0].children is None
        assert entities[1].parent is None

        text = 'I’d like a {latte|dish|beverage} with {maple syrup|option|general}'
        query = markup.load_query(text)
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 2
        assert entities[0].children is None
        assert entities[1].parent is None


class TestNestedParser:
    """A set of tests for a parser which has nested groups"""
    CONFIG = {
        'dish': ['option', 'size'],
        'option': ['size']
    }

    @classmethod
    def setup_class(cls):
        """Creates the parser for this group of tests"""
        cls.parser = Parser(config=cls.CONFIG)

    def test_standalone_option(self):
        """Tests that an option can exist as a standalone group"""
        query = markup.load_query('{light|size} {ice|option}')
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 2
        assert entities[0].parent == entities[1]
        assert entities[1].children == (entities[0],)

    def test_nested(self):
        """Tests that an option can exist as a standalone group"""
        text = '{large|size} {latte|dish} {light|size} {ice|option}'
        query = markup.load_query(text)
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 4
        assert entities[0].parent == entities[1]
        assert entities[1].children == (entities[0], entities[3])
        assert entities[2].parent == entities[3]
        assert entities[3].children == (entities[2],)


class TestMaxInstancesParser:
    """A set of tests for a parser which has max instance constraints on groups"""
    CONFIG = {
        'dish': {
            'option': {},
            'size': {'max_instances': 1}  # only one size per dish
        },
        'option': {
            'size': {'max_instances': 1}  # only one size per option
        }
    }

    @classmethod
    def setup_class(cls):
        """Creates the parser for this group of tests"""
        cls.parser = Parser(config=cls.CONFIG)

    def test_max_instances(self):
        """Tests that parser respects the max instances constraint"""
        text = '{light|size} {medium|size} {latte|dish}'
        query = markup.load_query(text)

        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 3
        assert entities[0].parent is None
        assert entities[1].parent == entities[2]
        assert entities[2].children == (entities[1],)

    def test_distance_override(self):
        """Tests that parser correctly allocates one size per dish,
        overriding distance in the process.
        """
        text = '{latte|dish} size {medium|size}, {mocha|dish} size {large|size}'
        query = markup.load_query(text)
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 4
        assert entities[0].children == (entities[1],)
        assert entities[1].parent == entities[0]
        assert entities[2].children == (entities[3],)
        assert entities[3].parent == entities[2]


class TestParserLinkWords:
    """A set of tests for a parser with link words"""
    CONFIG = {
        'dish': {
            'option': {'linking_words': {'with'}},
            'size': {},
        }
    }

    @classmethod
    def setup_class(cls):
        """Creates the parser for this group of tests"""
        cls.parser = Parser(config=cls.CONFIG)

    def test_link_word(self):
        """Tests that parser considers link words, overriding default distance calculation."""
        text = 'A {pizza|dish} with {olives|option}, {breadsticks|dish} and a {coke|dish}'
        query = markup.load_query(text)
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len(entities) == 4
        assert entities[0].children == (entities[1],)
        assert entities[1].parent == entities[0]
        assert entities[2].children is None
        assert entities[3].children is None

    def test_link_word_negative(self):
        """Tests that parser does not apply link words for other dependent types."""
        text = 'A {pepperoni pizza|dish} with {large|size} {coke|dish}'
        query = markup.load_query(text)
        entities = self.parser.parse_entities(query.query, query.entities)

        assert len([e for e in entities if e.parent is None and e.children is not None]) == 1
        assert entities[0].children is None
        assert entities[1].parent == entities[2]
        assert entities[2].children == (entities[1],)


def test_parser_timeout():
    """Tests that the parser throws a ParserTimeout exception on very ambiguous queries
    which take long to entities.
    """
    config = {
        'name': {
            'form': {'max_instances': 1},
            'size': {'max_instances': 1},
            'number': {'max_instances': 1, 'right': False},
            'option': {'linking_words': ['with']}
        }
    }
    parser = Parser(config=config)

    text = ('{venti|size} {jade citrus|name} with {one|number} bag of '
            '{peach tranquility|name} and {one|number} bag {jade citrus|name} '
            '{2 pumps peppermint|option} {no hot water|option} sub {steamed|option} '
            '{lemonade|option} {4|number} {honeys|option}')

    query = markup.load_query(text)

    with pytest.raises(ParserTimeout):
        parser.parse_entities(query.query, query.entities, handle_timeout=False)
