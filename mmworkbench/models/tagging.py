# -*- coding: utf-8 -*-
"""This module contains constants and functions for sequence tagging."""
from __future__ import absolute_import, unicode_literals
from builtins import zip

import logging

from ..core import QueryEntity, Span, TEXT_FORM_RAW, TEXT_FORM_NORMALIZED
from ..ser import resolve_system_entity, SystemEntityResolutionError

logger = logging.getLogger(__name__)

START_TAG = 'START'
B_TAG = 'B'
I_TAG = 'I'
O_TAG = 'O'
E_TAG = 'E'
S_TAG = 'S'


def get_tags_from_entities(query, entities, scheme='IOB'):
    """Get joint app and system IOB tags from a query's entities.

    Args:
        query (Query): A query instance.
        entities (List of QueryEntity): A list of queries found in the query

    Returns:
        (list of str): The tags for each token in the query. A tag has four
            parts separated by '|'. The first two are the IOB status for
            app entities followed by the type of app entity or
            '' if the IOB status is 'O'. The last two are like the first two,
            but for system entities.
    """
    entities = [e for e in entities]
    iobs, types = _get_tags_from_entities(query, entities, scheme)
    tags = ['|'.join(args) for args in zip(iobs, types)]
    return tags


def _get_tags_from_entities(query, entities, scheme='IOB'):
    normalized_tokens = query.normalized_tokens
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


def get_entities_from_tags(query, tags, scheme='IOB'):
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

    def _is_system_entity(entity_type):
        if entity_type.split('_')[0] == 'sys':
            return True
        return False

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

    for tag_idx, tag in enumerate(tags):
        iob, ent_type = tag.split('|')

        # Close entity and reset if the tag indicates a new entity
        if (entity_start is not None and
                (iob in (O_TAG, B_TAG, S_TAG) or ent_type != prev_ent_type)):
            logger.debug("Entity closed at prev")
            if _is_system_entity(prev_ent_type):
                _append_system_entity(entity_start, tag_idx, prev_ent_type)
            else:
                _append_entity(entity_start, prev_ent_type, entity_tokens)
            entity_start = None
            prev_ent_type = ''
            entity_tokens = []

        # Check if an entity has started
        if iob in (B_TAG, S_TAG) or ent_type not in ('', prev_ent_type):
            entity_start = tag_idx

            if _is_system_entity(ent_type):
                # During predict time, we construct sys_candidates for the input query.
                # These candidates are "global" sys_candidates, in that the entire query
                # is sent to Mallard to extract sys_candidates and not just a span range
                # within the query. However, the tagging model could more restrictive in
                # its classifier, so a sub-span of the original sys_candidate could be tagged
                # as a sys_entity. For example, the query "set alarm for 1130", mallard
                # provides the following sys_time entity candidate: "for 1130". However,
                # our entity recognizer only tags the token "1130" as a sys-time entity,
                # and not "at". Therefore, when we append system entities for this query,
                # we pick the start of the sys_entity to be the sys_candidate's start span
                # if the tagger identified a sys_entity within that sys_candidate's span
                # range of the same sys_entity type. Else, we just use the tag_idx tracked
                # in the control logic.

                picked_by_existing_system_entity_candidates = False

                for sys_candidate in query.get_system_entity_candidates(ent_type):

                    start_span = sys_candidate.normalized_token_span.start
                    end_span = sys_candidate.normalized_token_span.end

                    if start_span <= tag_idx <= end_span:
                        # We currently don't prioritize any sys_candidate if there are
                        # multiple candidates that meet this conditional.
                        # TODO: Assess if a priority is needed
                        entity_start = sys_candidate.normalized_token_span.start
                    picked_by_existing_system_entity_candidates = True

                if not picked_by_existing_system_entity_candidates:
                    entity_start = tag_idx

        # Append the current token to the current entity, if applicable.
        if iob != O_TAG and entity_start is not None and not _is_system_entity(ent_type):
            entity_tokens.append(normalized_tokens[tag_idx])

        # Close the entity if the tag indicates it closed
        if (entity_start is not None and iob in (E_TAG, S_TAG)):
            logger.debug("Entity closed here")
            if _is_system_entity(ent_type):
                _append_system_entity(entity_start, tag_idx+1, ent_type)
            else:
                _append_entity(entity_start, ent_type, entity_tokens)
            entity_start = None
            ent_type = ''
            entity_tokens = []

        prev_ent_type = ent_type

    # Handle entities that end with the end of the query
    if entity_start is not None:
        logger.debug("Entity closed at end")
        if _is_system_entity(prev_ent_type):
            _append_system_entity(entity_start, len(tags), prev_ent_type)
        else:
            _append_entity(entity_start, prev_ent_type, entity_tokens)
    else:
        logger.debug("Entity did not end: {}".format(entity_start))

    return tuple(entities)
