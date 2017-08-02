# -*- coding: utf-8 -*-
"""
This module contains all code required to perform sequence tagging.
"""
from __future__ import print_function, absolute_import, unicode_literals, division
from builtins import zip

from ...core import QueryEntity, Span, TEXT_FORM_RAW, TEXT_FORM_NORMALIZED
from ...ser import resolve_system_entity, SystemEntityResolutionError

import logging

logger = logging.getLogger(__name__)


START_TAG = 'START'
B_TAG = 'B'
I_TAG = 'I'
O_TAG = 'O'
E_TAG = 'E'
S_TAG = 'S'


class Tagger(object):
    def __init__(self, **parameters):
        """
        Args:
            config (ModelConfig): model configuration
        """
        self.config = parameters['config']
        # Default tag scheme to IOB
        self._tag_scheme = self.config.model_settings.get('tag_scheme', 'IOB').upper()
        # Placeholders
        self._resources = {}
        self._clf = None
        self._current_params = {}

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior. For the _resources field,
        we save the resources that are memory intensive
        """
        attributes = self.__dict__.copy()
        resources_to_persist = set(['sys_types'])
        for key in list(attributes['_resources'].keys()):
            if key not in resources_to_persist:
                del attributes['_resources'][key]
        return attributes

    def fit(self, examples, labels, resources=None):
        """Trains the model

        Args:
            labeled_queries (list of mmworkbench.core.Query): a list of queries to train on
            labels (list of tuples of mmworkbench.core.QueryEntity): a list of predicted labels
        """
        raise NotImplementedError

    def predict(self, examples):
        """Predicts for a list of examples
        Args:
            examples (list of mmworkbench.core.Query): a list of queries to be predicted
        Returns:
            (list of tuples of mmworkbench.core.QueryEntity): a list of predicted labels
        """
        raise NotImplementedError

    def get_params(self, deep=True):
        """Returns a dict of the __init__ parameters of the model
        """
        # return self._current_params
        raise NotImplementedError

    def set_params(self, **parameters):
        """Sets the parameters
        """
        # for parameter, value in parameters.items():
        #     self.setattr(parameter, value)
        # return self

        # for parameter, value in parameters.items():
        #     self._current_params[parameter] = value
        # return self
        raise NotImplementedError

    def _get_model_constructor(self):
        """Returns the python class of the actual underlying model"""
        raise NotImplementedError


"""
Helpers for taggers
"""


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

    # Normal entities
    app_entities = [e for e in entities if not e.entity.is_system_entity]
    iobs, app_types = _get_tags_from_entities(query, app_entities, scheme)

    # System entities
    # This algorithm assumes that the query system entities are well-formed and
    # only occur as standalone or fully inside an app entity.
    sys_entities = [e for e in entities if e.entity.is_system_entity]
    sys_iobs, sys_types = _get_tags_from_entities(query, sys_entities, scheme)

    tags = ['|'.join(args) for args in
            zip(iobs, app_types, sys_iobs, sys_types)]

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

        # Close regular entity and reset if the tag indicates a new entity
        if (entity_start is not None and
                (iob in (O_TAG, B_TAG, S_TAG) or ent_type != prev_ent_type)):
            logger.debug("Entity closed at prev")
            _append_entity(entity_start, prev_ent_type, entity_tokens)
            entity_start = None
            prev_ent_type = ''
            entity_tokens = []

            # Cut short any numeric entity that might continue beyond the entity
            if sys_entity_start is not None:
                _append_system_entity(sys_entity_start, tag_idx, prev_sys_type)
            sys_entity_start = None
            prev_sys_type = ''

        # Check if a regular entity has started
        if iob in (B_TAG, S_TAG) or ent_type not in ('', prev_ent_type):
            entity_start = tag_idx
        # Check if a numeric entity has started
        if sys_iob in (B_TAG, S_TAG) or sys_type not in ('', prev_sys_type):
            sys_entity_start = tag_idx

        # Append the current token to the current entity, if applicable.
        if iob != O_TAG and entity_start is not None:
            entity_tokens.append(normalized_tokens[tag_idx])

        # Close the numeric entity if the tag indicates it closed
        if (sys_entity_start is not None and
                sys_iob in (E_TAG, S_TAG)):
            logger.debug("System entity closed here")
            _append_system_entity(sys_entity_start, tag_idx+1, sys_type)
            sys_entity_start = None
            sys_type = ''

        # Close the regular entity if the tag indicates it closed
        if (entity_start is not None and
                iob in (E_TAG, S_TAG)):
            logger.debug("Entity closed here")
            _append_entity(entity_start, ent_type, entity_tokens)
            entity_start = None
            ent_type = ''
            entity_tokens = []

            # Cut short any numeric entity that might continue beyond the entity
            if sys_entity_start is not None:
                _append_system_entity(sys_entity_start, tag_idx+1, sys_type)
            sys_entity_start = None
            sys_type = ''

        prev_ent_type = ent_type
        prev_sys_type = sys_type

    # Handle entities that end with the end of the query
    if entity_start is not None:
        logger.debug("Entity closed at end: {}".format(entity_tokens))
        _append_entity(entity_start, prev_ent_type, entity_tokens)
    else:
        logger.debug("Entity did not end: {}".format(entity_start))
    if sys_entity_start is not None:
        _append_system_entity(sys_entity_start, len(tags), prev_sys_type)

    return tuple(entities)
