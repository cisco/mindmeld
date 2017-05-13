#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_parser
----------------------------------

Tests for parser module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import pytest

from mmworkbench import markup
from mmworkbench.core import EntityGroup
from mmworkbench.components.parser import Parser
from mmworkbench.exceptions import ParserTimeout


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
        parse = self.parser.parse_entities(query.query, query.entities)

        assert parse == []

    def test_singleton(self):
        """Tests the parser returns no groups when a head has no dependents"""
        query = markup.load_query('Hello {there|head}')
        parse = self.parser.parse_entities(query.query, query.entities)

        assert parse == []

    def test_left(self):
        """Tests the parser attaches dependents from the left"""
        query = markup.load_query('{Hello|dependent} {there|head}')
        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 1
        assert parse[0].head == query.entities[1]
        assert parse[0].dependents == (query.entities[0],)

    def test_right(self):
        """Tests the parser attaches dependents from the right"""
        query = markup.load_query('{Hello|head} {there|dependent}')
        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 1
        assert parse[0].head == query.entities[0]
        assert parse[0].dependents == (query.entities[1],)

    def test_distance(self):
        """Tests the parser attaches dependents to their nearest head"""
        query = markup.load_query('{Hello|head} {there|dependent} my {friend|head}')
        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 1
        assert parse[0].head == query.entities[0]
        assert parse[0].dependents == (query.entities[1],)


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
        query = markup.load_query('[{light|size} {ice|option}|option]')
        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 1
        assert parse[0].head == query.entities[1]
        assert parse[0].dependents == (query.entities[0],)

    def test_nested(self):
        """Tests that an option can exist as a standalone group"""
        text = '[{large|size} {latte|dish} [{light|size} {ice|option}|option]|dish]'
        query = markup.load_query(text)
        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 1
        assert parse[0].head == query.entities[1]
        assert len(parse[0].dependents) == 2
        assert parse[0].dependents[0] == query.entities[0]
        assert isinstance(parse[0].dependents[1], EntityGroup)
        assert parse[0].dependents[1].head == query.entities[3]
        assert parse[0].dependents[1].dependents == (query.entities[2],)


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
        text = '{light|size} [{medium|size} {latte|dish}|dish]'
        query = markup.load_query(text)

        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 1
        assert parse[0].head == query.entities[2]
        assert parse[0].dependents == (query.entities[1],)

    def test_distance_override(self):
        """Tests that parser correctly allocates one size per dish,
        overriding distance in the process.
        """
        text = '[{latte|dish} size {medium|size}|dish], [{mocha|dish} size {large|size}|dish]'
        query = markup.load_query(text)
        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 2
        assert parse[0].head == query.entities[0]
        assert parse[0].dependents == (query.entities[1],)
        assert parse[1].head == query.entities[2]
        assert parse[1].dependents == (query.entities[3],)


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
        text = 'A [{pizza|dish} with {olives|option}|dish], {breadsticks|dish} and a {coke|dish}'
        query = markup.load_query(text)
        parse = self.parser.parse_entities(query.query, query.entities)

        assert len(parse) == 1
        assert parse[0].head == query.entities[0]
        assert parse[0].dependents == (query.entities[1],)

    def test_link_word_negative(self):
        """Tests that parser does not apply link words for other dependent types."""
        text = 'A {pepperoni pizza|dish} with [{large|size} {coke|dish}|dish]'
        query = markup.load_query(text)
        parse = self.parser.parse_entities(query.query, query.entities)
        assert len(parse) == 1
        assert parse[0].head == query.entities[2]
        assert parse[0].dependents == (query.entities[1],)


def test_parser_timeout():
    """Tests that the parser throws a ParserTimeout exception on very ambiguous queries
    which take long to parse.
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

    text = ('[{venti|size} {jade citrus|name}|name] with [{one|number} bag of '
            '{peach tranquility|name}|name] and [{one|number} bag {jade citrus|name} '
            '{2 pumps peppermint|option} {no hot water|option} sub {steamed|option} '
            '{lemonade|option} {4|number} {honeys|option}|name]')

    query = markup.load_query(text)

    with pytest.raises(ParserTimeout):
        parser.parse_entities(query.query, query.entities, handle_timeout=False)
