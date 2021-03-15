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
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum

import pycountry
import requests

from .components._config import (
    DEFAULT_DUCKLING_URL,
    get_system_entity_url_config,
    is_duckling_configured,
)
from .components.schemas import (
    validate_language_code,
    validate_locale_code,
    validate_timestamp,
)
from .core import Entity, QueryEntity, Span, _sort_by_lowest_time_grain
from .exceptions import MindMeldError, SystemEntityResolutionError

SUCCESSFUL_HTTP_CODE = 200
SYS_ENTITY_REQUEST_TIMEOUT = os.environ.get("MM_SYS_ENTITY_REQUEST_TIMEOUT", 3.0)
try:
    if float(SYS_ENTITY_REQUEST_TIMEOUT) <= 0.0:
        raise MindMeldError(
            "MM_SYS_ENTITY_REQUEST_TIMEOUT env var has to be > 0.0 seconds."
        )
except ValueError as e:
    raise MindMeldError(
        "MM_SYS_ENTITY_REQUEST_TIMEOUT env var has to be a float value."
    ) from e

logger = logging.getLogger(__name__)


class SystemEntityError(Exception):
    pass


class DucklingDimension(Enum):
    AMOUNT_OF_MONEY = "amount-of-money"
    DISTANCE = "distance"
    DURATION = "duration"
    NUMERAL = "numeral"
    ORDINAL = "ordinal"
    QUANTITY = "quantity"
    TEMPERATURE = "temperature"
    VOLUME = "volume"
    EMAIL = "email"
    PHONE_NUMBER = "phone-number"
    URL = "url"
    TIME = "time"


class SystemEntityRecognizer(ABC):
    """SystemEntityRecognizer is the external parsing service used to extract
    system entities. It is intended to be used as a singleton, so it's
    initialized only once during NLP object construction.
    """

    _instance = None

    @staticmethod
    def get_instance():
        """Static access method. If there is no instance instantiated, we instantiate
        NoOpSystemEntityRecognizer.

        Returns:
            (SystemEntityRecognizer): A SystemEntityRecognizer instance
        """
        if not SystemEntityRecognizer._instance:
            SystemEntityRecognizer._instance = NoOpSystemEntityRecognizer.get_instance()
        return SystemEntityRecognizer._instance

    @staticmethod
    def set_system_entity_recognizer(system_entity_recognizer=None, app_path=None):
        """We set the global System Entity Recognizer to be the one configured from the
        application's path.

        Args:
              system_entity_recognizer: A system entity recognizer
              app_path (str): The application path

        Returns:
            (SystemEntityRecognizer)
        """
        if system_entity_recognizer and isinstance(
            system_entity_recognizer, SystemEntityRecognizer
        ):
            SystemEntityRecognizer._instance = system_entity_recognizer
        elif app_path:
            SystemEntityRecognizer._instance = SystemEntityRecognizer.load_from_app_path(
                app_path
            )
        else:
            raise SystemEntityError(
                "Either `system_entity_recognizer` or `app_path` must be valid."
            )

    @staticmethod
    def load_from_app_path(app_path):
        """If the application configuration is empty, we do not use Duckling.

        Otherwise, we return the Duckling recognizer with the URL defined in the application's
          config, default to the DEFAULT_DUCKLING_URL.

        Args:
              app_path (str): Application path

        Returns:
            (SystemEntityRecognizer)
        """
        if not app_path:
            raise SystemEntityError(
                "App path must be valid to load entity recognizer config."
            )

        if is_duckling_configured(app_path):
            url = get_system_entity_url_config(app_path=app_path)
            return DucklingRecognizer.get_instance(url)
        else:
            return NoOpSystemEntityRecognizer.get_instance()

    @abstractmethod
    def parse(self, sentence, **kwargs):
        """Calls System Entity Recognizer service API to extract numerical entities from a sentence.

        Args:
            sentence (str): A raw sentence.

        Returns:
            (tuple): A tuple containing:
                - response (list, dict): Response from the System Entity Recognizer service that \
                consists of a list of dicts, each corresponding to a single prediction or just a \
                dict, corresponding to a single prediction.
                - response_code (int): http status code.
        """
        pass

    @abstractmethod
    def resolve_system_entity(self, query, entity_type, span):
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
        pass

    @abstractmethod
    def get_candidates(self, query, entity_types=None, **kwargs):
        """Identifies candidate system entities in the given query.

        Args:
            query (Query): The query to examine
            entity_types (list of str): The entity types to consider

        Returns:
            list of QueryEntity: The system entities found in the query
        """
        pass

    @abstractmethod
    def get_candidates_for_text(self, text, entity_types=None, **kwargs):
        """Identifies candidate system entities in the given text.

        Args:
            text (str): The text to examine
            entity_types (list of str): The entity types to consider

        Returns:
            list of dict: The system entities found in the text
        """
        pass


class NoOpSystemEntityRecognizer(SystemEntityRecognizer):
    """
    This is a no-ops recognizer which returns empty list and 200.
    """

    _instance = None

    def __init__(self):
        if self._instance:
            raise SystemEntityError("NoOpSystemEntityRecognizer is a singleton.")

        NoOpSystemEntityRecognizer._instance = self

    @staticmethod
    def get_instance():
        if not NoOpSystemEntityRecognizer._instance:
            NoOpSystemEntityRecognizer()

        return NoOpSystemEntityRecognizer._instance

    def parse(self, sentence, **kwargs):
        return [], SUCCESSFUL_HTTP_CODE

    def resolve_system_entity(self, query, entity_type, span):
        return

    def get_candidates(self, query, entity_types=None, **kwargs):
        return []

    def get_candidates_for_text(self, text, entity_types=None, **kwargs):
        return []


class DucklingRecognizer(SystemEntityRecognizer):
    _instance = None

    def __init__(self, url=DEFAULT_DUCKLING_URL):
        """Private constructor for SystemEntityRecognizer. Do not directly
        construct the DucklingRecognizer object. Instead, use the
        static get_instance method.

        Args:
            url (str): Duckling URL
        """
        if DucklingRecognizer._instance:
            raise SystemEntityError("DucklingRecognizer is a singleton")

        self.url = url
        DucklingRecognizer._instance = self

    @staticmethod
    def get_instance(url=None):
        """Static access method.
        We get an instance for the Duckling URL. If there is no URL being passed,
          default to DEFAULT_DUCKLING_URL.

        Args:
            url: Duckling URL.

        Returns:
            (DucklingRecognizer): A DucklingRecognizer instance
        """
        url = url or DEFAULT_DUCKLING_URL
        if not DucklingRecognizer._instance:
            DucklingRecognizer(url=url)
        return DucklingRecognizer._instance

    def get_response(self, data):
        """
        Send a post request to Duckling, data is a dictionary with field `text`.
        Return a tuple consisting the JSON response and a response code.

        Args:
            data (dict)

        Returns:
            (dict, int)
        """
        try:
            response = requests.request(
                "POST", self.url, data=data, timeout=float(SYS_ENTITY_REQUEST_TIMEOUT)
            )

            if response.status_code == requests.codes["ok"]:
                response_json = response.json()
                return response_json, response.status_code
            else:
                raise SystemEntityError("System entity status code is not 200.")
        except requests.ConnectionError:
            sys.exit(
                "Unable to connect to the system entity recognizer. Make sure it's "
                "running by typing 'mindmeld num-parse' at the command line."
            )
        except Exception as ex:  # pylint: disable=broad-except
            logger.error(
                "Numerical Entity Recognizer Error: %s\nURL: %r\nData: %s",
                ex,
                self.url,
                json.dumps(data),
            )
            sys.exit(
                "\nThe system entity recognizer encountered the following "
                + "error:\n"
                + str(ex)
                + "\nURL: "
                + self.url
                + "\nRaw data: "
                + str(data)
                + "\nPlease check your data and ensure Numerical parsing service is running. "
                "Make sure it's running by typing "
                "'mindmeld num-parse' at the command line."
            )

    def parse(
        self,
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

        Returns:
            (tuple): A tuple containing:
                - response (list, dict): Response from the System Entity Recognizer service that \
                consists of a list of dicts, each corresponding to a single prediction or just a \
                dict, corresponding to a single prediction.
                - response_code (int): http status code.
        """
        if sentence == "":
            logger.error("Empty query passed to the system entity resolver")
            return [], SUCCESSFUL_HTTP_CODE

        data = {
            "text": sentence,
            "latent": True,
        }

        language = validate_language_code(language)
        locale = validate_locale_code(locale)

        # If a ISO 639-2 code is provided, we attempt to convert it to
        # ISO 639-1 since the dependent system entity resolver requires this
        if language and len(language) == 3:
            iso639_2_code = pycountry.languages.get(alpha_3=language.lower())
            try:
                language = getattr(iso639_2_code, "alpha_2").upper()
            except AttributeError:
                language = None

        if locale and language:
            language_code_of_locale = locale.split("_")[0]
            if language_code_of_locale.lower() != language.lower():
                logger.error(
                    "Language code %s and Locale code do not match %s, "
                    "using only the locale code for processing",
                    language,
                    locale,
                )
                # The system entity recognizer prefers the locale code over the language code,
                # so we bias towards sending just the locale code when the codes dont match.
                language = None

        # If the locale is invalid, we use the default
        if not language and not locale:
            language = "EN"
            locale = "en_US"

        if locale:
            data["locale"] = locale

        if language:
            data["lang"] = language.upper()

        if dimensions is not None:
            data["dims"] = json.dumps(dimensions)

        if time_zone:
            data["tz"] = time_zone

        if timestamp:
            data["reftime"] = validate_timestamp(str(timestamp))

        # Currently we rely on Duckling for parsing numerical data but in the future we can use
        # other system entity recognizer too
        return self.get_response(data)

    def resolve_system_entity(self, query, entity_type, span):
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
        span_filtered_candidates = list(
            filter(
                lambda candidate: candidate.span == span, query.system_entity_candidates
            )
        )

        entity_type_filtered_candidates = list(
            filter(
                lambda candidate: candidate.entity.type == entity_type,
                span_filtered_candidates,
            )
        )

        if entity_type == "sys_time":
            entity_type_filtered_candidates = _sort_by_lowest_time_grain(
                entity_type_filtered_candidates
            )

        if len(entity_type_filtered_candidates) > 0:
            # Duckling ranks sys_interval candidates with incomplete
            # "to" duration time interval higher than candidates with complete
            # "to" duration time interval. Therefore, we recommend the complete
            # candidate over the incomplete one when all the candidates have the
            # same "from" duration time.
            if entity_type == "sys_interval":
                from_vals = set()
                candidates_with_from_and_to_vals = []
                for candidate in entity_type_filtered_candidates:
                    from_val, to_val = candidate.entity.value["value"]
                    from_vals.add(from_val)
                    if from_val and to_val:
                        candidates_with_from_and_to_vals.append(candidate)

                if len(candidates_with_from_and_to_vals) > 0 and len(from_vals) == 1:
                    # All of the candidates have the same "from" time
                    return candidates_with_from_and_to_vals[0]

            # Duckling sorts most probable entity candidates higher than
            # the lower probable candidates. So we return the best possible
            # candidate in this case when multiple duckling candidates are
            # returned.
            return entity_type_filtered_candidates[0]

        language = query.language
        time_zone = query.time_zone
        timestamp = query.timestamp

        duckling_candidates, _ = self.parse(
            span.slice(query.text),
            language=language,
            time_zone=time_zone,
            timestamp=timestamp,
        )
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
            candidate = duckling_item_to_query_entity(
                query, raw_candidate, offset=span.start
            )

            if candidate.entity.type == entity_type:
                # If the candidate matches the entire entity, return it
                if candidate.span == span:
                    return candidate
                else:
                    duckling_text_val_to_candidate.setdefault(
                        candidate.text, []
                    ).append(candidate)

        # Sort duckling matching candidates by the length of the value
        best_duckling_candidate_names = list(duckling_text_val_to_candidate.keys())
        best_duckling_candidate_names.sort(key=len, reverse=True)

        if best_duckling_candidate_names:
            default_duckling_candidate = None
            longest_matched_duckling_candidate = best_duckling_candidate_names[0]

            for candidate in duckling_text_val_to_candidate[
                longest_matched_duckling_candidate
            ]:
                if candidate.span.start == span.start or candidate.span.end == span.end:
                    return candidate
                else:
                    default_duckling_candidate = candidate

            return default_duckling_candidate

        msg = "Unable to resolve system entity of type {!r} for {!r}."
        msg = msg.format(entity_type, span.slice(query.text))
        if span_filtered_candidates:
            msg += " Entities found for the following types {!r}".format(
                [a.entity.type for a in span_filtered_candidates]
            )

        raise SystemEntityResolutionError(msg)

    def get_candidates(
        self,
        query,
        entity_types=None,
        locale=None,
        language=None,
        time_zone=None,
        timestamp=None,
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
        dims = dimensions_from_entity_types(entity_types)
        language = language or query.language
        time_zone = time_zone or query.time_zone
        timestamp = timestamp or query.timestamp

        response, response_code = self.parse(
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
                for e in [
                    duckling_item_to_query_entity(query, item) for item in response
                ]
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
        self, text, entity_types=None, language=None, locale=None
    ):
        """Identifies candidate system entities in the given text.

        Args:
            text (str): The text to examine
            entity_types (list of str): The entity types to consider
            language (str): Language code
            locale (str): Locale code

        Returns:
            list of dict: The system entities found in the text
        """
        dims = dimensions_from_entity_types(entity_types)
        response, response_code = self.parse(
            text, dimensions=dims, language=language, locale=locale
        )
        if response_code == SUCCESSFUL_HTTP_CODE:
            items = []
            for item in response:
                entity = duckling_item_to_entity(item)
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


def _construct_interval_helper(interval_item):
    from_ = interval_item.get("from", {}).get("value", None)
    to_ = interval_item.get("to", {}).get("value", None)
    return from_, to_


def duckling_item_to_entity(item):
    """Converts an item from the output of duckling into an Entity

    Args:
        item (dict): The duckling item

    Returns:
        Entity: The entity described by the duckling item
    """
    value = {}
    dimension = item["dim"]

    # These dimensions have no 'type' key in the 'value' dict
    if dimension in map(
        lambda x: x.value,
        [
            DucklingDimension.EMAIL,
            DucklingDimension.PHONE_NUMBER,
            DucklingDimension.URL,
        ],
    ):
        num_type = dimension
        value["value"] = item["value"]["value"]
        if "values" in item["value"]:
            value["alternate_values"] = item["value"]["values"]
    else:
        type_ = item["value"]["type"]
        # num_type = f'{dimension}-{type_}'  # e.g. time-interval, temperature-value, etc
        num_type = dimension

        if type_ == "value":
            value["value"] = item["value"]["value"]
            if "values" in item["value"]:
                value["alternate_values"] = item["value"]["values"]
        elif type_ == "interval":
            # Some intervals will only contain one value. The other value will be None in that case
            value["value"] = _construct_interval_helper(item["value"])
            if "values" in item["value"]:
                value["alternate_values"] = [
                    _construct_interval_helper(interval_item)
                    for interval_item in item["value"]["values"]
                ]

        # Get the unit if it exists
        if "unit" in item["value"]:
            value["unit"] = item["value"]["unit"]

        # Special handling of time dimension grain
        if dimension == DucklingDimension.TIME.value:
            if type_ == "value":
                value["grain"] = item["value"].get("grain")
            elif type_ == "interval":

                # Want to predict time intervals as sys_interval
                num_type = "interval"
                if "from" in item["value"]:
                    value["grain"] = item["value"]["from"].get("grain")
                elif "to" in item["value"]:
                    value["grain"] = item["value"]["to"].get("grain")

    entity_type = "sys_{}".format(num_type)
    return Entity(item["body"], entity_type, value=value)


def duckling_item_to_query_entity(query, item, offset=0):
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
        start = int(item["start"]) + offset
        end = int(item["end"]) - 1 + offset
        entity = duckling_item_to_entity(item)
        return QueryEntity.from_query(query, Span(start, end), entity=entity)
    else:
        return


def dimensions_from_entity_types(entity_types):
    """
    Args:
        entity_types (list)

    Returns:
        (list)
    """
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
