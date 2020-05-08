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

from .system_entity_recognizer import DucklingRecognizer

logger = logging.getLogger(__name__)


def get_candidates(
    query,
    entity_types=None,
    locale=None,
    language=None,
    time_zone=None,
    timestamp=None,
    sys_recognizer=None,
):
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
        sys_recognizer (SystemEntityRecognizer): System Entity Recognizer, default to Duckling

    Returns:
        list of QueryEntity: The system entities found in the query
    """
    dims = _dimensions_from_entity_types(entity_types)
    language = language or query.language
    time_zone = time_zone or query.time_zone
    timestamp = timestamp or query.timestamp
    recognizer = sys_recognizer or DucklingRecognizer()

    response, response_code = recognizer.parse(
        query.text,
        dimensions=dims,
        locale=locale,
        language=language,
        time_zone=time_zone,
        timestamp=timestamp,
    )

    if response_code == SUCCESSFUL_HTTP_CODE:
        return [
            e
            for e in [_duckling_item_to_query_entity(query, item) for item in response]
            if entity_types is None or e.entity.type in entity_types
        ]

    logger.debug(
        "System Entity Recognizer service did not process query: %s with dims: %s "
        "correctly and returned response: %s",
        query.text,
        str(dims),
        str(response),
    )
    return []


def get_candidates_for_text(
    text, entity_types=None, language=None, locale=None, url=None
):
    """Identifies candidate system entities in the given text.

    Args:
        text (str): The text to examine
        entity_types (list of str): The entity types to consider
        language (str): Language code
        locale (str): Locale code
        sys_recognizer (SystemEntityRecognizer): System Entity Recognizer, default to Duckling
    Returns:
        list of dict: The system entities found in the text
    """
    dims = _dimensions_from_entity_types(entity_types)
    response, response_code = parse_numerics(
        text, dimensions=dims, language=language, locale=locale, url=url
    )
    if response_code == SUCCESSFUL_HTTP_CODE:
        items = []
        for item in response:
            entity = _duckling_item_to_entity(item)
            if entity_types is None or entity.type in entity_types:
                item["entity_type"] = entity.type
                items.append(item)
        return items
    else:
        logger.debug(
            "System Entity Recognizer service did not process query: %s with dims: %s "
            "correctly and returned response: %s",
            text,
            str(dims),
            str(response),
        )
        return []


def _dimensions_from_entity_types(entity_types):
    entity_types = entity_types or []
    dims = set()
    for entity_type in entity_types:
        if entity_type == "sys_interval":
            dims.add("time")
        if entity_type.startswith("sys_"):
            dims.add(entity_type.split("_")[1])
    if not dims:
        return None
    return list(dims)
