# -*- coding: utf-8 -*-
"""The markup module contains functions for interacting with the MindMeld Markup language for
representing annotations of query text inline.
"""
from __future__ import unicode_literals

import re

from .core import Entity, NestedEntity, ProcessedQuery, QueryEntity, Span

ENTITY_PATTERN = re.compile(r'\{(.*?)\}')
NESTED_ENTITY_PATTERN = re.compile(r'\[(.*?)\]')


class MarkupError(Exception):
    pass


class SystemEntityMarkupError(MarkupError):
    pass


def load_query(markup, query_factory, domain=None, intent=None, is_gold=False):
    """Creates a processed query object from marked up query text.

    Args:
        markup (str): The marked up query text
        query_factory (QueryFactory): An object which can create queries
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


def validate_markup(markup, query_factory):
    """Checks whether the markup text is well-formed.

    Args:
        markup (str): The marked up query text
        query_factory (QueryFactory): An object which can create queries

    Returns:
        bool: True if the markup is valid
    """
    return NotImplemented


def _mark_up(raw_text, entities=None, numerics=None):
    entities = entities or []
    numerics = numerics or []
    # TODO: also mark up nested entities
    return _mark_up_entities(raw_text, entities)


def _parse_entities(markup, query):
    entities = []
    for match in ENTITY_PATTERN.finditer(markup):
        start = len(mark_down(markup[:match.start()]))
        match_text = match.group(1)
        clean_match_str = mark_down_nested(match_text)
        components = clean_match_str.split('|')
        if len(components) == 2:
            entity_text, entity_type = components
            role = None
        elif len(components) == 3:
            entity_text, entity_type, role = components
        else:
            raise MarkupException('Invalid entity mark up: too many pipes')
        end = start + len(entity_text) - 1

        # get entity text excluding type and role
        marked_entity_text = match_text[:match_text.find('|', match_text.rfind(']'))]
        nested = _parse_nested(marked_entity_text, query, start)
        if len(nested):
            value = {'children': nested}
        else:
            value = entity_text
        span = Span(start, end)
        raw_entity = None
        if Entity.is_system_entity(entity_type):
            raw_entity = _resolve_system_entity(query, span, entity_type)
        if raw_entity is None:
            raw_entity = Entity(entity_type, role=role, value=value)
        entities.append(QueryEntity.from_query(query, raw_entity, span))
    return entities


def _parse_nested(markup, query, offset):
    """Parses the markup within an entity for nested entities

    Args:
        markup (str): The text inside an entity to be parsed for nested entities
        query (Query): A query object for the cleaned up markup
        offset (int): The offset from the start of the raw query text to the
            start of the markup

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    entities = []
    for match in NESTED_ENTITY_PATTERN.finditer(markup):
        prefix = mark_down(markup[:match.start()])
        start = len(prefix)
        components = match.group(1).split('|')
        if len(components) == 2:
            entity_text, entity_type = components
            role = None
        elif len(components) == 3:
            entity_text, entity_type, role = components
        else:
            raise MarkupError('Invalid entity mark up: too many pipes')
        end = start + len(entity_text) - 1
        span = Span(start, end)
        raw_entity = None
        if Entity.is_system_entity(entity_type):
            raw_entity = _resolve_system_entity(query, span.shift(offset), entity_type)
        else:
            raw_entity = Entity(entity_type, role=role, value=entity_text)

        entities.append(NestedEntity.from_query(query, raw_entity, offset, span))

    return entities


def _resolve_system_entity(query, span, entity_type):
    for candidate in query.get_system_entity_candidates(set((entity_type,))):
        if candidate.span == span and candidate.entity.type == entity_type:
            return candidate.entity

    msg = 'Unable to resolve system entity of type {!r} for {!r}'
    raise SystemEntityMarkupError(msg.format(entity_type, span.slice(query.text)))


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


def _mark_up_nested(query_str, entities):
    # remove existing markup just in case
    query_str = mark_down(query_str)

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
            new_query += "[{}|{}]".format(query_str[start:end], entity.type)
        else:
            new_query += "[{}|{}|{}]".format(query_str[start:end], entity.type, entity.role)
        cursor = end
    new_query += query_str[cursor:]
    return new_query


def mark_down(markup):
    """Removes all entity mark up from a string

    Args:
        markup (str): A marked up string

    Returns:
        str: A clean string with no mark up
    """
    return mark_down_entities(mark_down_nested(markup))


def mark_down_entities(markup):
    """Removes top level entity mark up from a string

    Args:
        markup (str): A marked up string

    Returns:
        str: A clean string with no top level entity mark up
    """
    return _mark_down(markup)


def mark_down_nested(markup):
    """Removes nested entity mark up from a string

    Args:
        markup (str): A marked up string

    Returns:
        str: A clean string with no nested entities marked up
    """
    return _mark_down(markup, nested=True)


def _mark_down(markup, nested=False):
    def _replace(match):
        entity = match.group(1)
        pattern = r'\|[^]]*$'
        return re.sub(pattern, '', entity)
    pattern = NESTED_ENTITY_PATTERN if nested else ENTITY_PATTERN
    return re.sub(pattern, _replace, markup)
