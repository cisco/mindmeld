# -*- coding: utf-8 -*-
"""This module contains constants and functions for sequence tagging."""
from __future__ import unicode_literals
from builtins import zip

import logging

from ...core import QueryEntity, Span, TEXT_FORM_RAW, TEXT_FORM_NORMALIZED
from ...ser import resolve_system_entity, SystemEntityResolutionError

logger = logging.getLogger(__name__)

START_TAG = 'START'
B_TAG = 'B'
I_TAG = 'I'
O_TAG = 'O'
E_TAG = 'E'
S_TAG = 'S'
OUT_OF_BOUNDS_TOKEN = '<$>'


def get_tags_from_entities(query, scheme='IOB'):
    """Get joint app and system IOB tags from a query's entities.

    Args:
        query (ProcessedQuery): An annotated query instance.

    Returns:
        (list of str): The tags for each token in the query. A tag has four
            parts separated by '|'. The first two are the IOB status for
            app entities followed by the type of app entity or
            '' if the IOB status is 'O'. The last two are like the first two,
            but for system facets.
    """

    # Normal entities
    app_entities = [e for e in query.entities if not e.entity.is_system_entity]
    iobs, app_types = _get_tags_from_entities(query, app_entities, scheme)

    # System entities
    # This algorithm assumes that the query system entities are well-formed and
    # only occur as standalone or fully inside an app entity.
    sys_entities = [e for e in query.entities if e.entity.is_system_entity]
    sys_iobs, sys_types = _get_tags_from_entities(query, sys_entities, scheme)

    tags = ['|'.join(args) for args in
            zip(iobs, app_types, sys_iobs, sys_types)]

    return tags


def _get_tags_from_entities(query, entities, scheme='IOB'):
    normalized_tokens = query.query.normalized_tokens
    iobs = [O_TAG for _ in normalized_tokens]
    types = ['' for _ in normalized_tokens]

    # tag I and type for all tag schemes
    for entity in entities:

        for i in entity.normalized_token_span:
            iobs[i] = I_TAG
            types[i] = entity.entity.type

    # Replace I with B/E/S when appropriate
    if scheme in ('IOB', 'IOBES'):
        for entity in entities:
            iobs[entity.normalized_token_span.start] = B_TAG
    if scheme == 'IOBES':
        for entity in entities:
            if len(entity.normalized_token_span) == 1:
                iobs[entity.normalized_token_span.end] = S_TAG
            else:
                iobs[entity.normalized_token_span.end] = E_TAG

    return iobs, types


def get_entities_from_tags(query, tags):
    """From a set of joint IOB tags, parse the app and system entities.

    This performs the reverse operation of get_tags_from_entities.

    Args:
        query (Query): Any query instance.
        tags (list of str): Joint app and system tags, like those
            created by get_tags_from_entities.

    Returns:
        (list of QueryEntity) The tuple containing the list of entities.
    """

    normalized_tokens = query.normalized_tokens

    entities = []

    def _append_entity(token_start, entity_type, tokens):
        prefix = ' '.join(normalized_tokens[:token_start])
        # If there is a prefix, we have to add one for the whitespace
        start = len(prefix) + 1 if len(prefix) else 0
        end = start - 1 + len(' '.join(tokens))
        norm_span = Span(start, end)
        entity = QueryEntity.from_query(query, normalized_span=norm_span, entity_type=entity_type)
        entities.append(entity)
        logger.debug("Appended {}".format(entity))

    def _append_system_entity(token_start, token_end, entity_type):
        msg = "Looking for '{}' between {} and {}"
        logger.debug(msg.format(entity_type, token_start, token_end))
        prefix = ' '.join(normalized_tokens[:token_start])
        # If there is a prefix, we have to add one for the whitespace
        start = len(prefix) + 1 if len(prefix) else 0
        end = start - 1 + len(' '.join(normalized_tokens[token_start:token_end]))
        norm_span = Span(start, end)

        span = query.transform_span(norm_span, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)

        try:
            entity = resolve_system_entity(query, entity_type, span)
            entities.append(entity)
            logger.debug("Appended system entity {}".format(entity))
        except SystemEntityResolutionError:
            msg = "Found no matching system entity {}-{}, {!r}"
            logger.debug(msg.format(token_start, token_end, entity_type))

    entity_tokens = []
    entity_start = None
    prev_ent_type = ''
    sys_entity_start = None
    prev_sys_type = ''

    for tag_idx, tag in enumerate(tags):
        iob, ent_type, sys_iob, sys_type = tag.split('|')

        # Close sysem entity and reset if the tag indicates a new entity
        if (sys_entity_start is not None and
                (sys_iob in (O_TAG, B_TAG, S_TAG) or sys_type != prev_sys_type)):
            logger.debug("System entity closed at prev")
            _append_system_entity(sys_entity_start, tag_idx, prev_sys_type)
            sys_entity_start = None
            prev_sys_type = ''

        # Close regular facet and reset if the tag indicates a new facet
        if (entity_start is not None and
                (iob in (O_TAG, B_TAG, S_TAG) or ent_type != prev_ent_type)):
            logger.debug("Entity closed at prev")
            _append_entity(entity_start, prev_ent_type, entity_tokens)
            entity_start = None
            prev_ent_type = ''
            entity_tokens = []

            # Cut short any numeric facet that might continue beyond the facet
            if sys_entity_start is not None:
                _append_system_entity(sys_entity_start, tag_idx, prev_sys_type)
            sys_entity_start = None
            prev_sys_type = ''

        # Check if a regular facet has started
        if iob in (B_TAG, S_TAG) or ent_type not in ('', prev_ent_type):
            entity_start = tag_idx
        # Check if a numeric facet has started
        if sys_iob in (B_TAG, S_TAG) or sys_type not in ('', prev_sys_type):
            sys_entity_start = tag_idx

        # Append the current token to the current entity, if applicable.
        if iob != O_TAG and entity_start is not None:
            entity_tokens.append(normalized_tokens[tag_idx])

        # Close the numeric facet if the tag indicates it closed
        if (sys_entity_start is not None and
                sys_iob in (E_TAG, S_TAG)):
            logger.debug("System entity closed here")
            _append_system_entity(sys_entity_start, tag_idx+1, sys_type)
            sys_entity_start = None
            sys_type = ''

        # Close the regular facet if the tag indicates it closed
        if (entity_start is not None and
                iob in (E_TAG, S_TAG)):
            logger.debug("Entity closed here")
            _append_entity(entity_start, ent_type, entity_tokens)
            entity_start = None
            ent_type = ''
            entity_tokens = []

            # Cut short any numeric facet that might continue beyond the facet
            if sys_entity_start is not None:
                _append_system_entity(sys_entity_start, tag_idx+1, sys_type)
            sys_entity_start = None
            sys_type = ''

        prev_ent_type = ent_type
        prev_sys_type = sys_type

    # Handle facets that end with the end of the query
    if entity_start is not None:
        logger.debug("Entity closed at end: {}".format(entity_tokens))
        _append_entity(entity_start, prev_ent_type, entity_tokens)
    else:
        logger.debug("Entity did not end: {}".format(entity_start))
    if sys_entity_start is not None:
        _append_system_entity(sys_entity_start, len(tags), prev_sys_type)

    return entities
