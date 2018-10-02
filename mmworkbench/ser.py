# -*- coding: utf-8 -*-
"""This module contains the system entity recognizer."""
import logging
import json
import sys

import requests

from .core import Entity, QueryEntity, Span
from .exceptions import SystemEntityResolutionError

logger = logging.getLogger(__name__)

MALLARD_URL = "http://localhost:2626"
MALLARD_ENDPOINT = "parse"


def get_candidates(query, entity_types=None, language=None, time_zone=None, timestamp=None):
    """Identifies candidate system entities in the given query

    Args:
        query (Query): The query to examine
        entity_types (list of str): The entity types to consider
        language (str, optional): Language as specified using a 639-2 code.
            If omitted, English is assumed.
        time_zone (str, optional): An IANA time zone id such as 'America/Los_Angeles'.
            If not specified, the system time zone is used.
        timestamp (long, optional): A unix timestamp used as the reference time.
            If not specified, the current system time is used. If `time_zone`
            is not also specified, this parameter is ignored.

    Returns:
        list of QueryEntity: The system entities found in the query
    """
    dims = _dimensions_from_entity_types(entity_types)
    language = language or query.language
    time_zone = time_zone or query.time_zone
    timestamp = timestamp or query.timestamp
    response = parse_numerics(query.text, dimensions=dims, language=language,
                              time_zone=time_zone, timestamp=timestamp)

    if (int(response.get('status', -1)) == 200) and ('data' in response.keys()):
        return [e for e in [_mallard_item_to_query_entity(query, item) for item in response['data']]
                if entity_types is None or e.entity.type in entity_types]

    logger.debug("Mallard did not process query: {} with dims: {} correctly and "
                 "returned response: {}".format(query.text, str(dims), str(response)))
    return []


def get_candidates_for_text(text, entity_types=None, span=None, language=None,
                            time_zone=None, timestamp=None):
    """Identifies candidate system entities in the given text

    Args:
        text (str): The text to examine
        entity_types (list of str): The entity types to consider
        language (str, optional): Language as specified using a 639-2 code.
            If omitted, English is assumed.
        time_zone (str, optional): An IANA time zone id such as 'America/Los_Angeles'.
            If not specified, the system time zone is used.
        timestamp (long, optional): A unix timestamp used as the reference time.
            If not specified, the current system time is used. If `time_zone`
            is not also specified, this parameter is ignored.

    Returns:
        list of dict: The system entities found in the text
    """

    dims = _dimensions_from_entity_types(entity_types)
    response = parse_numerics(text, dimensions=dims)
    items = []
    for item in response['data']:
        entity = _mallard_item_to_entity(item)
        if entity_types is None or entity.type in entity_types:
            item['entity_type'] = entity.type
            items.append(item)
    return items


def parse_numerics(sentence, dimensions=None, language='eng', time_zone=None, timestamp=None):
    """Calls Mallard API to extract numerical entities from a sentence.

    Args:
        sentence (str): A raw sentence.
        dimensions (None or list of str): The list of types (e.g. volume,
            temperature) to restrict the output to. If None, include all types
        language (str, optional): Language of the sentence specified using a 639-2 code.
            If omitted, English is assumed.
        time_zone (str, optional): An IANA time zone id such as 'America/Los_Angeles'.
            If not specified, the system time zone is used.
        timestamp (long, optional): A unix timestamp used as the reference time.
            If not specified, the current system time is used. If `time_zone`
            is not also specified, this parameter is ignored.

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
        data['dimensions'] = dimensions
    if time_zone:
        data['timeZone'] = time_zone
    if timestamp:
        data['timestamp'] = timestamp

    try:
        response = requests.request('POST', url, json=data)
        return response.json()
    except requests.ConnectionError:
        logger.debug('Unable to connect to Mallard.')
        raise RuntimeError("Unable to connect to Mallard. Make sure it's running by typing "
                           "'mmworkbench num-parse' at the command line.")
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
    alternates = []
    for candidate in query.system_entity_candidates:
        if candidate.span == span:
            if candidate.entity.type == entity_type:
                return candidate
            else:
                alternates.append(candidate)

    language = query.language
    time_zone = query.time_zone
    timestamp = query.timestamp

    mallard_candidates = parse_numerics(span.slice(query.text), language=language,
                                        time_zone=time_zone, timestamp=timestamp)['data']
    mallard_text_val_to_candidate = {}

    # If no matching candidate was found, try parsing only this entity
    # Refer to this ticket for how we prioritize mallard candidates:
    # https://mindmeldinc.atlassian.net/browse/WB3-54
    #
    # For secondary candidate picking, we prioritize candidates as follows:
    # a) candidate matches both span range and entity type
    # b) candidate with the most number of matching characters to the user
    # annotation
    # c) candidate whose span matches either the start or end user annotation
    # span

    for raw_candidate in mallard_candidates:
        candidate = _mallard_item_to_query_entity(query, raw_candidate, offset=span.start)

        if candidate.entity.type == entity_type:
            # If the candidate matches the entire entity, return it
            if candidate.span == span:
                return candidate
            else:
                mallard_text_val_to_candidate.setdefault(candidate.text, []).append(candidate)

    # Sort mallard matching candidates by the length of the value
    best_mallard_candidate_names = list(mallard_text_val_to_candidate.keys())
    best_mallard_candidate_names.sort(key=len, reverse=True)

    if best_mallard_candidate_names:
        default_mallard_candidate = None
        longest_matched_mallard_candidate = best_mallard_candidate_names[0]

        for candidate in mallard_text_val_to_candidate[longest_matched_mallard_candidate]:
            if candidate.span.start == span.start or candidate.span.end == span.end:
                return candidate
            else:
                default_mallard_candidate = candidate

        return default_mallard_candidate

    msg = 'Unable to resolve system entity of type {!r} for {!r}.'
    msg = msg.format(entity_type, span.slice(query.text))
    if alternates:
        msg += ' Entities found for the following types {!r}'.format([a.entity.type
                                                                      for a in alternates])

    raise SystemEntityResolutionError(msg)


def _mallard_item_to_query_entity(query, item, offset=0):
    """Converts an item from mallard into a QueryEntity

    Args:
        query (Query): The query
        item (dict): The mallard item
        offset (int, optional): The offset into the query that the item's
            indexing begins

    Returns:
        QueryEntity: The query entity described by the mallard item
    """
    start = int(item['entity']['start']) + offset
    end = int(item['entity']['end']) - 1 + offset
    entity = _mallard_item_to_entity(item)
    return QueryEntity.from_query(query, Span(start, end), entity=entity)


def _mallard_item_to_entity(item):
    """Converts an item from mallard into an Entity

    Args:
        query (Query): The query
        item (dict): The mallard item
        offset (int, optional): The offset into the query that the item's
            indexing begins

    Returns:
        Entity: The entity described by the mallard item
    """
    value = {}

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

    return Entity(item['entity']['text'], entity_type, value=value, confidence=confidence)


def _dimensions_from_entity_types(entity_types):
    entity_types = entity_types or []
    dims = set()
    for entity_type in entity_types:
        if entity_type == 'sys_interval':
            dims.add('time')
        if entity_type.startswith('sys_'):
            dims.add(entity_type.split('_')[1])
    dims = list(dims)
    if not dims:
        dims = None
    return dims
