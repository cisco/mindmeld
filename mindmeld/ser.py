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
import warnings

from .system_entity_recognizer import DucklingRecognizer

logger = logging.getLogger(__name__)


def get_candidates(
    query, entity_types=None, locale=None, language=None, time_zone=None, timestamp=None
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

    Returns:
        list of QueryEntity: The system entities found in the query
    """
    msg = "get_candidates is deprecated in favor of DucklingRecognizer.get_candidates."
    warnings.warn(msg)
    return DucklingRecognizer.get_instance().get_candidates(
        query,
        entity_types=entity_types,
        locale=locale,
        language=language,
        time_zone=time_zone,
        timestamp=timestamp,
    )


def get_candidates_for_text(text, entity_types=None, language=None, locale=None):
    """Identifies candidate system entities in the given text.

    Args:
        text (str): The text to examine
        entity_types (list of str): The entity types to consider
        language (str): Language code
        locale (str): Locale code
    Returns:
        list of dict: The system entities found in the text
    """
    msg = "get_candiates_for_text is deprecated in favor of" \
          " DucklingRecognizer.get_candidates_for_text."
    warnings.warn(msg)
    return DucklingRecognizer.get_instance().get_candidates_for_text(
        text, entity_types=entity_types, language=language, locale=locale
    )


def parse_numerics(
    sentence,
    dimensions=None,
    language=None,
    locale=None,
    time_zone=None,
    timestamp=None,
):
    """Calls System Entity Recognizer service API to extract numerical entities from a sentence.

    Args:
        sentence (str): A raw sentence.
        dimensions (None or list of str): The list of types (e.g. volume, \
            temperature) to restrict the output to. If None, include all types.
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
            - response (list, dict): Response from the System Entity Recognizer service that \
            consists of a list of dicts, each corresponding to a single prediction or just a \
            dict, corresponding to a single prediction.
            - response_code (int): http status code.
    """
    msg = "parse_numerics is deprecated in favor of DucklingRecognizer.parse."
    warnings.warn(msg)
    return DucklingRecognizer.get_instance().parse(
        sentence,
        dimensions=dimensions,
        language=language,
        locale=locale,
        time_zone=time_zone,
        timestamp=timestamp,
    )


def resolve_system_entity(query, entity_type, span):
    """Resolves a system entity in the provided query at the specified span.

    Args:
        query (Query): The query containing the entity
        entity_type (str): The type of the entity
        span (Span): The character span of the entity in the query

    Returns:
        Entity: The resolved entity

    Raises:
        SystemEntityResolutionError
    """
    msg = "resolve_system_entity is deprecated in favor " \
          "of DucklingRecognizer.resolve_system_entity."
    warnings.warn(msg)
    return DucklingRecognizer.get_instance().resolve_system_entity(query, entity_type, span)
