# -*- coding: utf-8 -*-
"""The markup module contains functions for interacting with the MindMeld Markup language for
representing annotations of query text inline.
"""
from __future__ import unicode_literals

import re

from .core import Entity, ProcessedQuery, QueryEntity, Span

ENTITY_PATTERN = re.compile(r'\{(.*?)\}')
SYSTEM_ENTITY_PATTERN = re.compile(r'\[(.*?)\|sys:(.*?)\]')


def load_query(markup, query_factory, domain=None, intent=None, is_gold=False):
    """Creates a processed query object from marked up query text.

    Args:
        markup (str): The marked up query text
        query_factory (QueryFactory): An object which can create
        domain (str): The name of the domain annotated for the query
        intent (str): The name of the intent annotated for the query
        is_gold (bool): True if the markup passed in is a reference, human-labeled example

    Returns:
        ProcessedQuery: a processed query
    """

    raw_text = mark_down(markup)
    query = query_factory.create_query(raw_text)
    entities = _parse_entities(markup, query=query)

    return ProcessedQuery(query, domain=domain, intent=intent, entities=entities, is_gold=is_gold)


def dump_query(processed_query):
    """Converts a processed query into marked up query text.

    Args:
        processed_query (ProcessedQuery): The query to convert

    Returns:
        str: A marked up representation of the query
    """
    raw_text = processed_query.query.raw_text
    markup = _mark_up(raw_text, processed_query.entities)
    return markup


def validate_markup(markup):
    """Checks whether the markup text is well-formed.

    Args:
        markup (str): Description

    Returns:
        bool: True if the markup is valid
    """
    pass


def _parse(markup):
    """Given marked up text returns marked down text and a list of entities.

    Args:
        markup (str): marked up text

    """

    pass


def mark_down(markup):
    return mark_down_entities(mark_down_numerics(markup))


def _mark_up(raw_text, entities=None, numerics=None):
    entities = entities or []
    numerics = numerics or []
    # TODO: also mark up numerics
    return _mark_up_entities(raw_text, entities)


def _parse_entities(markup, query=None):
    entities = []
    for match in ENTITY_PATTERN.finditer(markup):
        prefix = mark_down(markup[:match.start()])
        start = len(prefix)
        clean_match_str = mark_down_numerics(match.group(1))
        components = clean_match_str.split('|')
        if len(components) == 2:
            entity_text, entity_type = components
            role_name = None
        else:
            entity_text, entity_type, role_name = components

        end = start + len(entity_text) - 1

        span = Span(start, end)
        raw_entity = Entity(entity_type, role=role_name)
        entities.append(QueryEntity.from_query(query, raw_entity, span))

    return entities


def mark_down_entities(markup):
    def markeddown(match):
        entity = match.group(1)
        r2 = '\|[^]]*$'
        return re.sub(r2, '', entity)
    return re.sub(ENTITY_PATTERN, markeddown, markup)


def _mark_up_entities(query_str, entities):
    # remove existing markup just in case
    query_str = mark_down_entities(query_str)

    # make sure entities are sorted
    sorted_entities = sorted(entities, key='start')
    new_query = ''
    cursor = 0

    # add each entity
    for entity in sorted_entities:
        start = entity.start
        end = entity.end
        new_query += query_str[cursor:start]
        if entity.role is None:
            new_query += "{{{}|{}}}".format(query_str[start:end], entity.type)
        else:
            new_query += "{{{}|{}|{}}}".format(query_str[start:end], entity.type, entity.role)
        cursor = end
    new_query += query_str[cursor:]
    return new_query


# TODO: figure out whether we need numerics

def _parse_numerics(markup):
    numerics = []
    query_str = mark_down(markup)
    for match in SYSTEM_ENTITY_PATTERN.finditer(query_str):
        entity_text = match.group(1)
        prefix = mark_down(query_str[:match.start()])

        start = len(prefix)
        end = start - 1 + len(entity_text)
        numerics.append({'entity': entity_text, 'type': match.group(2),
                         'start': start, 'end': end})
    return numerics


def mark_down_numerics(markup):
    # TODO: figure out whether we need this
    return re.sub(SYSTEM_ENTITY_PATTERN, r'\1', markup)


def _mark_up_numerics(markup, numerics):
    # TODO: implement this if we need it
    return markup
