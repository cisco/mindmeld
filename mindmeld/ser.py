# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the system entity recognizer."""
import logging
import json
from enum import Enum
import pycountry

from .core import Entity, QueryEntity, Span, _sort_by_lowest_time_grain
from .exceptions import SystemEntityResolutionError
from .system_entity_recognizer import SystemEntityRecognizer
from .components.request import validate_language_code, validate_locale_code

logger = logging.getLogger(__name__)

SUCCESSFUL_HTTP_CODE = 200


class DucklingDimension(Enum):
    AMOUNT_OF_MONEY = 'amount-of-money'
    DISTANCE = 'distance'
    DURATION = 'duration'
    NUMERAL = 'numeral'
    ORDINAL = 'ordinal'
    QUANTITY = 'quantity'
    TEMPERATURE = 'temperature'
    VOLUME = 'volume'
    EMAIL = 'email'
    PHONE_NUMBER = 'phone-number'
    URL = 'url'
    TIME = 'time'


def get_candidates(query, entity_types=None, locale=None,
                   language=None, time_zone=None, timestamp=None):
    """Identifies candidate system entities in the given query.

    Args:
        query (Query): The query to examine
        entity_types (list of str): The entity types to consider
        locale (str, optional): The locale representing the ISO 639-1 language code and \
            ISO3166 alpha 2 country code separated by an underscore character.
        language (str, optional): Language as specified using a 639-1/2 code.
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
    response, response_code = parse_numerics(
        query.text, dimensions=dims, locale=locale, language=language,
        time_zone=time_zone, timestamp=timestamp)
    if response_code == SUCCESSFUL_HTTP_CODE:
        return [e for e in [_duckling_item_to_query_entity(query, item) for item in response]
                if entity_types is None or e.entity.type in entity_types]

    logger.debug("System Entity Recognizer service did not process query: %s with dims: %s "
                 "correctly and returned response: %s", query.text, str(dims), str(response))
    return []


def get_candidates_for_text(text, entity_types=None, language='en'):
    """Identifies candidate system entities in the given text.

    Args:
        text (str): The text to examine
        entity_types (list of str): The entity types to consider
        language (str): Language code
    Returns:
        list of dict: The system entities found in the text
    """
    dims = _dimensions_from_entity_types(entity_types)
    response, response_code = parse_numerics(text, dimensions=dims, language=language)
    if response_code == SUCCESSFUL_HTTP_CODE:
        items = []
        for item in response:
            entity = _duckling_item_to_entity(item)
            if entity_types is None or entity.type in entity_types:
                item['entity_type'] = entity.type
                items.append(item)
        return items
    else:
        logger.debug("System Entity Recognizer service did not process query: %s with dims: %s "
                     "correctly and returned response: %s", text, str(dims), str(response))
        return []


def parse_numerics(sentence, dimensions=None, language=None, locale=None,
                   time_zone=None, timestamp=None):
    """Calls System Entity Recognizer service API to extract numerical entities from a sentence.
    Args:
        sentence (str): A raw sentence.
        dimensions (None or list of str): The list of types (e.g. volume, \
            temperature) to restrict the output to. If None, include all types
        language (str, optional): Language of the sentence specified using a 639-1/2 code.
            If both locale and language are provided, the locale is used. If neither are
            provided, the EN language code is used.
        locale (str, optional): The locale representing the ISO 639-1 language code and \
            ISO3166 alpha 2 country code separated by an underscore character.
        time_zone (str, optional): An IANA time zone id such as 'America/Los_Angeles'. \
            If not specified, the system time zone is used.
        timestamp (long, optional): A unix millisecond timestamp used as the reference time. \
            If not specified, the current system time is used. If `time_zone` \
            is not also specified, this parameter is ignored.
    Returns:
        (tuple): A tuple containing:
            * response (list, dict): Response from the System Entity Recognizer service that
            consists of a list of dicts, each corresponding to a single prediction or just a
            dict, corresponding to a single prediction.
            * response_code (int): http status code.
    """
    if sentence == '':
        logger.error('Empty query passed to the system entity resolver')
        return {}, SUCCESSFUL_HTTP_CODE

    data = {
        'text': sentence,
        'latent': True,
    }

    language = validate_language_code(language)
    locale = validate_locale_code(locale)

    # If a ISO 639-2 code is provided, we attempt to convert it to
    # ISO 639-1 since the dependent system entity resolver requires this
    if language and len(language) == 3:
        iso639_2_code = pycountry.languages.get(alpha_3=language.lower())
        try:
            language = getattr(iso639_2_code, 'alpha_2').upper()
        except AttributeError:
            language = None

    if locale and language:
        language_code_of_locale = locale.split('_')[0]
        if language_code_of_locale.lower() != language.lower():
            logger.error('Language code %s and Locale code do not match %s, '
                         'using only the locale code for processing', language, locale)
            # The system entity recognizer prefers the locale code over the language code,
            # so we bias towards sending just the locale code when the codes dont match.
            language = None

    # If the locale is invalid, we use the default
    if not language and not locale:
        language = 'EN'
        locale = 'en_US'

    if locale:
        data['locale'] = locale

    if language:
        data['lang'] = language.upper()

    if dimensions is not None:
        data['dims'] = json.dumps(dimensions)

    if time_zone:
        data['tz'] = time_zone

    if timestamp:
        if len(str(timestamp)) != 13:
            logger.debug("Warning: Possible non-millisecond unix timestamp passed in.")
        if len(str(timestamp)) == 10:
            # Convert a second grain unix timestamp to millisecond
            timestamp *= 1000
        data['reftime'] = timestamp

    return SystemEntityRecognizer.get_instance().get_response(data)


def resolve_system_entity(query, entity_type, span):
    """Resolves a system entity in the provided query at the specified span.

    Args:
        query (Query): The query containing the entity
        entity_type (str): The type of the entity
        span (Span): The character span of the entity in the query

    Returns:
        Entity: The resolved entity

    Raises:
        SystemEntityResolutionError:
    """
    span_filtered_candidates = list(
        filter(lambda candidate: candidate.span == span, query.system_entity_candidates))

    entity_type_filtered_candidates = list(
        filter(lambda candidate: candidate.entity.type == entity_type, span_filtered_candidates))

    if entity_type == 'sys_time':
        entity_type_filtered_candidates = \
            _sort_by_lowest_time_grain(entity_type_filtered_candidates)

    if len(entity_type_filtered_candidates) > 0:
        return entity_type_filtered_candidates[-1]

    language = query.language
    time_zone = query.time_zone
    timestamp = query.timestamp

    duckling_candidates, _ = parse_numerics(
        span.slice(query.text), language=language,
        time_zone=time_zone, timestamp=timestamp)
    duckling_text_val_to_candidate = {}

    # If no matching candidate was found, try parsing only this entity
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
    if span_filtered_candidates:
        msg += ' Entities found for the following types {!r}'.format(
            [a.entity.type for a in span_filtered_candidates])

    raise SystemEntityResolutionError(msg)


def _duckling_item_to_query_entity(query, item, offset=0):
    """Converts an item from the output of duckling into a QueryEntity

    Args:
        query (Query): The query to construct the QueryEntity from
        item (dict): The duckling item
        offset (int, optional): The offset into the query that the item's
            indexing begins

    Returns:
        QueryEntity: The query entity described by the duckling item or \
            None if no item is present
    """
    if item:
        start = int(item['start']) + offset
        end = int(item['end']) - 1 + offset
        entity = _duckling_item_to_entity(item)
        return QueryEntity.from_query(query, Span(start, end), entity=entity)
    else:
        return


def _duckling_item_to_entity(item):
    """Converts an item from the output of duckling into an Entity

    Args:
        query (Query): The query to construct the QueryEntity from
        item (dict): The duckling item
        offset (int, optional): The offset into the query that the item's
            indexing begins

    Returns:
        Entity: The entity described by the duckling item
    """
    value = {}
    dimension = item['dim']

    # These dimensions have no 'type' key in the 'value' dict
    if dimension in map(lambda x: x.value, [DucklingDimension.EMAIL,
                                            DucklingDimension.PHONE_NUMBER,
                                            DucklingDimension.URL]):
        num_type = dimension
        value['value'] = item['value']['value']
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
        if dimension == DucklingDimension.TIME.value:
            if type_ == 'value':
                value['grain'] = item['value'].get('grain')
            elif type_ == 'interval':

                # Want to predict time intervals as sys_interval
                num_type = 'interval'
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
    if not dims:
        return None
    return list(dims)
