# -*- coding: utf-8 -*-
"""The markup module contains functions for interacting with the MindMeld Markup language for
representing annotations of query text inline.
"""
from __future__ import unicode_literals
from future.utils import raise_from

import re

from .core import Entity, NestedEntity, ProcessedQuery, QueryEntity, Span
from .ser import resolve_system_entity, SystemEntityResolutionError

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
    try:
        entities = _parse_entities(markup, query=query)
    except SystemEntityResolutionError as exc:
        msg = "Unable to load query {!r}: {}"
        raise_from(SystemEntityMarkupError(msg.format(markup, exc)), exc)

    return ProcessedQuery(query, domain=domain, intent=intent, entities=entities, is_gold=is_gold)


def load_query_file(file_path, query_factory, domain, intent, is_gold=False):
    """Loads the queries from the specified file

    Args:
        domain (str): The domain of the query file
        intent (str): The intent of the query file
        filename (str): The name of the query file

    """
    queries = []
    import codecs
    with codecs.open(file_path, encoding='utf-8') as queries_file:
        for line in queries_file:
            line = line.strip()
            # only create query if line is not empty string
            query_text = line.split('\t')[0].strip()
            if query_text:
                if query_text[0] == '-':
                    continue

            query = load_query(query_text, query_factory, domain=domain, intent=intent, is_gold=is_gold)
            queries.append(query)
    return queries


def dump_query(processed_query):
    """Converts a processed query into marked up query text.

    Args:
        processed_query (ProcessedQuery): The query to convert

    Returns:
        str: A marked up representation of the query
    """
    raw_text = processed_query.query.text
    markup = _mark_up_entities(raw_text, processed_query.entities)
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
            raise MarkupError('Invalid entity mark up: too many pipes')
        end = start + len(entity_text) - 1

        # get entity text excluding type and role
        marked_entity_text = match_text[:match_text.find('|', match_text.rfind(']'))]
        nested = _parse_nested(marked_entity_text, query, start)
        value = {'children': nested} if len(nested) else None
        span = Span(start, end)
        raw_entity = None
        if Entity.is_system_entity(entity_type):
            raw_entity = resolve_system_entity(query, entity_type, span).entity
        if raw_entity is None:
            raw_entity = Entity(entity_text, entity_type, role=role, value=value)
        entities.append(QueryEntity.from_query(query, span, entity=raw_entity))
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
            raw_entity = resolve_system_entity(query, entity_type, span.shift(offset)).entity
        else:
            raw_entity = Entity(entity_text, entity_type, role=role)

        entities.append(NestedEntity.from_query(query, span, entity=raw_entity,
                                                parent_offset=offset))

    return entities


def _mark_up_entities(query_str, entities):
    entities = entities or []

    # remove existing markup just in case
    query_str = mark_down_entities(query_str)

    # make sure entities are sorted
    sorted_entities = sorted(entities, key=lambda x: x.span.start)
    marked_text = ''
    cursor = 0

    # add each entity
    for entity in sorted_entities:
        marked_text += query_str[cursor:entity.span.start]
        entity_text = _mark_up_nested(entity)
        if entity.entity.role is None:
            marked_text += "{{{}|{}}}".format(entity_text, entity.entity.type)
        else:
            marked_text += "{{{}|{}|{}}}".format(entity_text, entity.entity.type,
                                                 entity.entity.role)
        cursor = entity.span.end + 1
    marked_text += query_str[cursor:]
    return marked_text


def _mark_up_nested(entity):
    outer_text = entity.text
    nested_entities = []
    if isinstance(entity.entity.value, dict):
        nested_entities = entity.entity.value.get('children', [])

    # remove existing markup just in case
    outer_text = mark_down(outer_text)

    # make sure entities are sorted
    sorted_entities = sorted(nested_entities, key=lambda x: x.span.start)
    marked_text = ''
    cursor = 0

    # add each entity
    for entity in sorted_entities:
        marked_text += outer_text[cursor:entity.span.start]
        if entity.entity.role is None:
            marked_text += "[{}|{}]".format(entity.text, entity.entity.type)
        else:
            marked_text += "[{}|{}|{}]".format(entity.text, entity.entity.type, entity.entity.role)
        cursor = entity.span.end + 1
    marked_text += outer_text[cursor:]
    return marked_text


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
