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

DUCKLING_URL = "http://localhost:8000"
DUCKLING_ENDPOINT = "parse"

SUCCESSFUL_HTTP_CODE = 200


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
    response, response_code = parse_numerics(query.text, dimensions=dims, language=language,
                                             time_zone=time_zone, timestamp=timestamp)
    if response_code == 200:
        return [e for e in [_duckling_item_to_query_entity(query, item) for item in response]
                if entity_types is None or e.entity.type in entity_types]

    logger.debug("Duckling did not process query: {} with dims: {} correctly and "
                 "returned response: {}".format(query.text, str(dims), str(response)))
    return []


# TODO - Remove? Never called
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
    response, response_code = parse_numerics(text, dimensions=dims)
    if response_code == 200:
        items = []
        for item in response:
            entity = _duckling_item_to_entity(item)
            if entity_types is None or entity.type in entity_types:
                item['entity_type'] = entity.type
                items.append(item)
        return items
    else:
        logger.debug("Ducking did not process text: {} with dims: {} correctly and "
                     "return response: {}".format(text, str(dims), str(response)))
        return []


def parse_numerics(sentence, dimensions=None, language='EN', time_zone=None, timestamp=None):
    """Calls Duckling API to extract numerical entities from a sentence.

    Args:
        sentence (str): A raw sentence.
        dimensions (None or list of str): The list of types (e.g. volume,
            temperature) to restrict the output to. If None, include all types
        language (str, optional): Language of the sentence specified using a 639-1 code.
            If omitted, English is assumed.
        time_zone (str, optional): An IANA time zone id such as 'America/Los_Angeles'.
            If not specified, the system time zone is used.
        timestamp (long, optional): A unix millisecond timestamp used as the reference time.
            If not specified, the current system time is used. If `time_zone`
            is not also specified, this parameter is ignored.

    Returns:
        response (list, dict): Duckling response that consist of a list of dicts, each
            corresponding to a single prediction
        response_code (int): http status code

    """
    if sentence == '':
        return {}, SUCCESSFUL_HTTP_CODE
    url = '/'.join([DUCKLING_URL, DUCKLING_ENDPOINT])
    data = {
        'text': sentence,
        'lang': language,
        'latent': True,
    }
    if dimensions is not None:
        # TODO - Passing in these dimensions doesn't affect Duckling output, DOES work on Postman
        data['dims'] = dimensions
    if time_zone:
        data['tz'] = time_zone
    if timestamp:
        if len(str(timestamp)) != 13:
            logger.debug("Warning: Possible non-millisecond unix timestamp passed in.")
        data['reftime'] = timestamp
    try:
        response = requests.request('POST', url, data=data)
        return response.json(), response.status_code
    except requests.ConnectionError:
        logger.debug('Unable to connect to Duckling.')
        raise RuntimeError("Unable to connect to Duckling. Make sure it's running by ...")  # TODO
    except Exception as ex:
        logger.error('Numerical Entity Recognizer Error %s\nURL: %r\nData: %s', ex, url,
                     json.dumps(data))
        sys.exit('\nThe numerical parser service encountered the following ' +
                 'error:\n' + str(ex) + '\nURL: ' + url + '\nRaw data: ' + str(data) +
                 '\nPlease check your data and ensure Duckling is running. You may ' +
                 "run Duckling by ... ")  # TODO


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
            # Returns the first entity type that matches even if there are multiple
            if candidate.entity.type == entity_type:
                return candidate
            else:
                alternates.append(candidate)

    language = query.language
    time_zone = query.time_zone
    timestamp = query.timestamp

    duckling_candidates, response_codes = parse_numerics(span.slice(query.text), language=language,
                                                        time_zone=time_zone, timestamp=timestamp)
    duckling_text_val_to_candidate = {}

    # If no matching candidate was found, try parsing only this entity
    # Refer to this ticket for how we prioritize duckling candidates:
    # https://mindmeldinc.atlassian.net/browse/WB3-54
    #
    # For secondary candidate picking, we prioritize candidates as follows:
    # a) candidate matches both span range and entity type
    # b) candidate with the most number of matching characters to the user
    # annotation
    # c) candidate whose span matches either the start or end user annotation
    # span

    for raw_candidate in duckling_candidates:
        candidate = _duckling_item_to_query_entity(query, raw_candidate, offset=span.start)

        if candidate.entity.type == entity_type:
            # If the candidate matches the entire entity, return it
            if candidate.span == span:
                return candidate
            else:
                duckling_text_val_to_candidate.setdefault(candidate.text, []).append(candidate)

    # Sort duckling matching candidates by the length of the value
    best_duckling_candidate_names = list(duckling_text_val_to_candidate.keys())
    best_duckling_candidate_names.sort(key=len, reverse=True)

    if best_duckling_candidate_names:
        default_duckling_candidate = None
        longest_matched_duckling_candidate = best_duckling_candidate_names[0]

        for candidate in duckling_text_val_to_candidate[longest_matched_duckling_candidate]:
            if candidate.span.start == span.start or candidate.span.end == span.end:
                return candidate
            else:
                default_duckling_candidate = candidate

        return default_duckling_candidate

    msg = 'Unable to resolve system entity of type {!r} for {!r}.'
    msg = msg.format(entity_type, span.slice(query.text))
    if alternates:
        msg += ' Entities found for the following types {!r}'.format([a.entity.type
                                                                      for a in alternates])

    raise SystemEntityResolutionError(msg)


def _duckling_item_to_query_entity(query, item, offset=0):
    """Converts an item from duckling into a QueryEntity

    Args:
        query (Query): The query
        item (dict): The duckling item
        offset (int, optional): The offset into the query that the item's
            indexing begins

    Returns:
        QueryEntity: The query entity described by the duckling item or nothing if blank query
    """
    if item:
        start = int(item['start']) + offset
        end = int(item['end']) - 1 + offset
        entity = _duckling_item_to_entity(item)
        return QueryEntity.from_query(query, Span(start, end), entity=entity)
    else:
        return


def _duckling_item_to_entity(item):
    """Converts an item from duckling into an Entity

    Args:
        query (Query): The query
        item (dict): The duckling item
        offset (int, optional): The offset into the query that the item's
            indexing begins

    Returns:
        Entity: The entity described by the duckling item
    """
    value = {}
    dimension = item['dim']

    # These dimensions have no 'type' key in the 'value' dict
    if dimension == 'email' or dimension == 'phone-number' or dimension == 'url':
        num_type = dimension
        value['value'] = item['value']['value']

    # Remaining dimensions have a type key
    # amount-of-money, distance, duration, numeral, ordinal, quantity, temperature, time, volume
    else:
        type_ = item['value']['type']
        # num_type = f'{dimension}-{type_}'  # e.g. time-interval, temperature-value, etc
        num_type = dimension

        if type_ == 'value':
            value['value'] = item['value']['value']
        elif type_ == 'interval':
            from_ = None
            to_ = None

            if 'from' in item['value']:
                from_ = item['value']['from']['value']
            if 'to' in item['value']:
                to_ = item['value']['to']['value']

            # Some intervals will only contain one value. The other value will be None in that case
            value['value'] = (from_, to_)

        # Get the unit if it exists
        if 'unit' in item['value']:
            value['unit'] = item['value']['unit']

        # Special handling of time dimension grain
        if dimension == 'time':
            if type_ == 'value':
                value['grain'] = item['value'].get('grain')
            elif type_ == 'interval':
                if 'from' in item['value']:
                    value['grain'] = item['value']['from'].get('grain')
                elif 'to' in item['value']:
                    value['grain'] = item['value']['to'].get('grain')

    entity_type = "sys_{}".format(num_type)
    return Entity(item['body'], entity_type, value=value)


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
