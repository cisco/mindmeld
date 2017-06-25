# -*- coding: utf-8 -*-
"""This module contains the system entity recognizer."""
from __future__ import absolute_import, unicode_literals
from builtins import str

import logging
import json
import sys

import requests

from .core import Entity, QueryEntity, Span
from .exceptions import SystemEntityResolutionError

logger = logging.getLogger(__name__)

MALLARD_URL = "http://localhost:2626"
MALLARD_ENDPOINT = "parse"


def get_candidates(query, entity_types=None, span=None):
    """Identifies candidate system entities in the given query

    Args:
        query (Query): The query to examine
        entity_types (list of str): The entity types to consider

    Returns:
        list of QueryEntity: The system entities found in this
    """

    dims = _dimensions_from_entity_types(entity_types)
    response = parse_numerics(query.text, dimensions=dims)
    return [_mallard_item_to_entity(query, item) for item in response['data']]


def parse_numerics(sentence, dimensions=None, language='eng', reference_time=''):
    """Calls Mallard API to extract numerical entities from a sentence.

    Args:
        sentence (str): A raw sentence.
        dimensions (None or list of str): The list of types (e.g. volume,
            temperature) to restrict the output to. If None, include all types
    Returns:
        (dict) The JSON result from Mallard for the given query.
    """
    if sentence == '':
        return {'data': []}
    url = '/'.join([MALLARD_URL, MALLARD_ENDPOINT])
    data = {
        'text': sentence,
        'language': language
    }
    if dimensions is not None:
        data["dimensions"] = dimensions
    if reference_time:
        data['reference_time'] = reference_time

    try:
        response = requests.request('POST', url, json=data)
        return response.json()
    except requests.ConnectionError:
        logger.error('Unable to connect to Mallard.')
        raise RuntimeError('Unable to connect to Mallard. Is it running?')
    except Exception as ex:
        logger.error('Numerical Entity Recognizer Error %s\nURL: %r\nData: %s', ex, url,
                     json.dumps(data))
        sys.exit('\nThe numerical parser service encountered the following ' +
                 'error:\n' + str(ex) + '\nURL: ' + url + '\nRaw data: ' + str(data) +
                 '\nPlease check your data and ensure Mallard is running. You may ' +
                 "run Mallard with 'python start-nparse.py'.")


def resolve_system_entity(query, entity_type, span):
    """Resolves a system entity in the provided query at the specified span

    Args:
        query (Query): The query containing the entity
        entity_type (str): The type of the entity
        span (Span): The character span of the entity in the query

    Returns:
        Entity: The resolved entity

    Raises:
        SystemEntityResolutionError:
    """
    for candidate in query.get_system_entity_candidates(set((entity_type,))):
        if candidate.span == span and candidate.entity.type == entity_type:
            return candidate

    # If no matching candidate was found, try parsing only this entity
    for raw_candidate in parse_numerics(span.slice(query.text))['data']:
        candidate = _mallard_item_to_entity(query, raw_candidate, offset=span.start)
        # If the candidate matches the entire entity, return it
        if candidate.span == span and candidate.entity.type == entity_type:
            return candidate

    msg = 'Unable to resolve system entity of type {!r} for {!r}'
    raise SystemEntityResolutionError(msg.format(entity_type, span.slice(query.text)))


def _mallard_item_to_entity(query, item, offset=0):
    """Converts an item from mallard into a QueryEntity

    Args:
        query (Query): The query
        item (dict): The mallard item
        offset (int, optional): The offset into the query that the item's
            indexing begins

    Returns:
        QueryEntity: The query entity described by the mallard item
    """
    value = {}

    start = int(item['entity']['start'])
    end = int(item['entity']['end']) - 1

    confidence = -float(item['likelihood']) * int(item['rule_count'])

    if 'unit' in item:
        value['unit'] = item['unit']
    if 'grain' in item:
        value['grain'] = str(item['grain'])

    if item['dimension'] == 'time':
        if 'operator' in item and item['operator'] == 'less-than':
            num_type = 'interval'
            value['value'] = ["0001-01-01T00:00:00.000-08:00", item['value'][0]]
        elif 'operator' in item and item['operator'] == 'greater-than':
            num_type = 'interval'
            value['value'] = [item['value'][0], "9999-12-31T00:00:00.000-08:00"]
        elif 'operator' in item and item['operator'] == 'between':
            num_type = 'interval'
            value['value'] = item['value']
        else:
            if len(item['value']) > 1:
                num_type = 'interval'
                value['value'] = item['value']
            else:
                num_type = 'time'
                value['value'] = str(item['value'][0])
    elif item['dimension'] == 'currency':
        num_type = 'currency'
        value['value'] = item['value'][0]
    elif item['dimension'] == 'number':
        num_type = 'number'
        value['value'] = item['value'][0]
    else:
        num_type = str(item['dimension'])
        value['value'] = str(item['value'][0])
    entity_type = "sys_{}".format(num_type)

    entity = Entity(item['entity']['text'], entity_type, value=value, confidence=confidence)
    return QueryEntity.from_query(query, Span(start + offset, end + offset), entity=entity)


def _dimensions_from_entity_types(entity_types):
    entity_types = entity_types or []
    dims = [et.split('_')[1] for et in (entity_types or []) if et.startswith('sys_')]
    if len(dims) == 0:
        dims = None
    return dims
