# -*- coding: utf-8 -*-
'This module contains the system entity recognizer.'
from __future__ import unicode_literals
from builtins import object

import logging
import json
import sys

import requests

from .core import Entity, QueryEntity, Span


logger = logging.getLogger(__name__)

MALLARD_URL = "http://localhost:2626"
MALLARD_ENDPOINT = "parse"


class SystemEntityRecognizer(object):

    def __init__(self, entity_types=None):
        self._entity_types = entity_types

    def get_candidates(self, query, entity_types=None):
        """Identifies candidate system entities in the given query

        Args:
            query (Query): The query to examine
            entity_types (list of str): The entity types to consider

        Returns:
            list of QueryEntity: The system entities found in this
        """

        entity_types = entity_types or self._entity_types
        dims = mallard_dimensions_from_entity_types(entity_types)
        response = parse_numerics(query.text, dimensions=dims)
        return [mallard_item_to_entity(query, item) for item in response['data']]


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
    except Exception as e:
        print('Numerical Entity Recognizer Error: ' + str(e))
        print('URL: ' + url)
        print('Data: ' + json.dumps(data))
        sys.exit('\nThe numerical parser service encountered the following ' +
                 'error:\n' + str(e) + '\nURL: ' + url + '\nRaw data: ' + str(data) +
                 '\nPlease check your data and ensure Mallard is running. You may ' +
                 "run Mallard with 'python start-nparse.py'.")


def mallard_item_to_entity(query, item):
    """Converts an item from mallard into a QueryEntity

    Args:
        query (Query): The query
        item (dict): The mallard item

    Returns:
        QueryEntity: The query entity described by the mallard item
    """
    value = {}

    start = int(item['entity']['start'])
    end = int(item['entity']['end']) - 1

    confidence = -float(item['likelihood']) * int(item['rule_count'])

    if 'unit' in item:
        value['unit'] = item['unit'].encode('utf-8')
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
                num_type = str(item['dimension'])
                value['value'] = str(item['value'][0])
    else:
        num_type = str(item['dimension'])
        value['value'] = str(item['value'][0])
    entity_type = "sys:{}".format(num_type)
    entity = Entity(entity_type, value=value, confidence=confidence)
    return QueryEntity.from_query(query, entity, Span(start, end))


def mallard_dimensions_from_entity_types(entity_types):
    entity_types = entity_types or []
    dims = [et.split(':')[1] for et in (entity_types or []) if et.startswith('sys:')]
    if len(dims) == 0:
        dims = None
    return dims


def resolve_entity_conflicts(query, entities):
    """This method takes a list containing entities for a query, and resolves
    any entity conflicts. The resolved list is returned.

    If two facets in a query conflict with each other, use the following logic:
        - If the target facet is a subset of another facet, then delete the
          target facet.
        - If the target facet shares the identical span as another facet,
          then keep the one with the highest confidence.
        - If the target facet overlaps with another facet, then keep the one
          with the highest confidence.

    Args:
        query (Query): The query in which entities were recognized
        entities (list of QueryEntity): A list of entities to resolve

    Returns:
        list of QueryEntity: A filtered list of entities

    """
    filtered = [e for e in entities]
    i = 0
    while i < len(filtered):
        include_target = True
        target = filtered[i]
        j = i + 1
        while j < len(filtered):
            other = filtered[j]
            if is_superset(target, other) and not is_same_span(target, other):
                logger.debug('Removing {{{1:s}|{2:s}}} facet in query {0:d} since it is a '
                             'subset of another.'.format(i, other.raw_text, other.entity.type))
                del filtered[j]
                continue
            elif is_subset(target, other) and not is_same_span(target, other):
                logger.debug('Removing {{{1:s}|{2:s}}} facet in query {0:d} since it is a '
                             'subset of another.'.format(i, target.raw_text, target.entity.type))
                del filtered[i]
                include_target = False
                break
            elif is_same_span(target, other) or is_overlapping(target, other):
                if target.entity.confidence >= other.entity.confidence:
                    logger.debug('Removing {{{1:s}|{2:s}}} facet in query {0:d} since it overlaps '
                                 'with another.'.format(i, other.raw_text, other.entity.type))
                    del filtered[j]
                    continue
                elif target.entity.confidence < other.entity.confidence:
                    logger.debug('Removing {{{1:s}|{2:s}}} facet in query {0:d} since it overlaps '
                                 'with another.'.format(i, target.raw_text, target.entity.type))
                    del filtered[i]
                    include_target = False
                    break
            j += 1
        if include_target:
            i += 1

    return filtered


def is_subset(target, other):
    return ((target.start >= other.start) and
            (target.end <= other.end))


def is_superset(target, other):
    return ((target.start <= other.start) and
            (target.end >= other.end))


def is_same_span(target, other):
    return (is_superset(target, other) and is_subset(target, other))


def is_overlapping(target, other):
    target_range = range(target.start, target.end + 1)
    predicted_range = range(other.start, other.end + 1)
    overlap = set(target_range).intersection(predicted_range)
    return (overlap and not is_subset(target, other) and
            not is_superset(target, other))
