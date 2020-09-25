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
import logging

import attr
import immutables
import pycountry
from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError

logger = logging.getLogger(__name__)


def validate_language_code(param=None):
    """Validates language code parameters
    Args:
        param (str, optional): The language code parameter

    Returns:
        str: A validated language code or None if unvalidated
    """
    if not param:
        return None
    if not isinstance(param, str):
        logger.error("Invalid %r param: %s is not of type %s.", "language", param, str)
        return None

    # The pycountry APIs need the param to be in lowercase for processing
    param = param.lower()

    if len(param) != 2 and len(param) != 3:
        logger.error(
            "Invalid %r param: %s is not a valid ISO 639-1 or ISO 639-2 language code.",
            "locale",
            param,
        )
        return None

    if len(param) == 2 and not pycountry.languages.get(alpha_2=param):
        logger.error(
            "Invalid %r param: %s is not a valid ISO 639-1 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for valid codes.",
            "locale",
            param,
        )
        return None

    if len(param) == 3 and not pycountry.languages.get(alpha_3=param):
        logger.error(
            "Invalid %r param: %s is not a valid ISO 639-2 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes for valid codes.",
            "locale",
            param,
        )
        return None

    return param


def validate_locale_code(param=None):
    """Validates the locale code parameters
    Args:
        param (str, optional): The locale code parameter

    Returns:
        str: A validated locale code or None if unvalidated
    """
    if not param:
        return None
    if not isinstance(param, str):
        logger.error("Invalid %r param: %s is not of type %s.", "locale", param, str)
        return None

    if len(param.split("_")) != 2:
        logger.error("Invalid %r param: Not a valid locale.", param)
        return None

    language_code = param.split("_")[0].lower()
    if not validate_language_code(language_code):
        logger.error(
            "Invalid %r param: %s is not a valid ISO 639-1 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for valid codes.",
            "locale",
            language_code,
        )
        return None

    # pycountry requires the country code to be upper-cased
    country_code = param.split("_")[1].upper()
    if not pycountry.countries.get(alpha_2=country_code):
        logger.error(
            "Invalid %r param: %s is not a valid ISO3166 alpha 2 country code. "
            "See https://www.iso.org/obp/ui/#search for valid codes.",
            "locale",
            country_code,
        )
        return None

    # return the validated locale
    return language_code + "_" + country_code


def _validate_time_zone(param=None):
    """Validates time zone parameters

    Args:
        param (str, optional): The time zone parameter

    Returns:
        str: The passed in time zone
    """
    if not param:
        return None
    if not isinstance(param, str):
        logger.warning(
            "Invalid %r param: %s is not of type %s.", "time_zone", param, str
        )
        return None
    try:
        timezone(param)
    except UnknownTimeZoneError:
        logger.warning(
            "Invalid %r param: %s is not a valid time zone.", "time_zone", param
        )
        return None
    return param


def _validate_generic(name, ptype):
    def validator(param):
        if not isinstance(param, ptype):
            logger.warning(
                "Invalid %r param: %s is not of type %s.", name, param, ptype
            )
            param = None
        return param

    return validator


PARAM_VALIDATORS = {
    "allowed_intents": _validate_generic("allowed_intents", (tuple, list)),
    "target_dialogue_state": _validate_generic("target_dialogue_state", str),
    "time_zone": _validate_time_zone,
    "language": validate_language_code,
    "locale": validate_locale_code,
    "timestamp": _validate_generic("timestamp", int),
    "dynamic_resource": _validate_generic("dynamic_resource", immutables.Map),
}


@attr.s(frozen=False, kw_only=True)
class Params:
    """
    A class that contains parameters that modify how the user query is parsed.

    Attributes:
        allowed_intents (list, str): A list of intents that you can set to force the language
            processor to choose from.
        target_dialogue_state (str): The name of the dialogue handler that you want to reach in
            the next turn.
        time_zone (str):  The name of an IANA time zone, such as 'America/Los_Angeles', or
            'Asia/Kolkata'.
        timestamp (long): A unix time stamp for the request accurate to the nearest second.
        locale (str, optional): The locale representing the ISO 639-1/2 language code and ISO3166
            alpha 2 country code separated by an underscore character.
        language (str, optional): The language code representing ISO 639-1/2
            language codes.
        dynamic_resource (dict): A dictionary containing data used to influence the language
            classifiers by adding resource data for the given turn.
    """

    allowed_intents = attr.ib(default=attr.Factory(tuple))
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
    language = attr.ib(default=None)
    locale = attr.ib(default=None)
    dynamic_resource = attr.ib(default=attr.Factory(dict))

    def validate_param(self, name):
        """
        Perform validation on the value of a specific parameter in the Params object.

        Args:
            name (str): Name of the parameter to be validated.

        Returns:
            bool: True/False depending on success of the validation, None if the param name does
                not exist.
        """
        validator = PARAM_VALIDATORS.get(name)
        param = vars(self).get(name)
        if param:
            return validator(param)
        return param

    def validate_dm_params(self, handler_map):
        """
        Validate that the value of the 'target_dialogue_state' parameter is a valid dialogue state
            for the application and returns that value in a dictionary.

        Args:
            handler_map (dict): Mapping from dialogue state to the function handler that gets
                called when in the state.

        Returns:
            dict: single item dictionary with the parameter value if valid and None if not.
        """
        target_dialogue_state = self.validate_param("target_dialogue_state")
        if target_dialogue_state and target_dialogue_state not in handler_map:
            logger.error(
                "Target dialogue state %s does not match any dialogue state names "
                "in for the application. Not applying the target dialogue state "
                "this turn.",
                target_dialogue_state,
            )
            return {"target_dialogue_state": None}
        return {"target_dialogue_state": target_dialogue_state}

    def validate_nlp_params(self):
        """
        Validate language, locale, time zone, timestamp, and dynamic resource parameters
            and return the params as a dictionary.

        Returns:
            dict: Mapping from parameter name to bool depending on validation.
        """
        return {
            param: self.validate_param(param)
            for param in (
                "time_zone",
                "timestamp",
                "dynamic_resource",
                "language",
                "locale",
            )
        }

    def to_dict(self):
        fields = [
            "allowed_intents",
            "target_dialogue_state",
            "time_zone",
            "timestamp",
            "language",
            "locale",
            "dynamic_resource",
        ]
        _dic = {field: vars(self).get(field) for field in fields}
        # converting from immutable map to just dictionary
        _dic["dynamic_resource"] = {
            key: _dic["dynamic_resource"][key] for key in _dic["dynamic_resource"]
        }

        return _dic


@attr.s(frozen=True, kw_only=True)
class FrozenParams(Params):
    """
    An immutable version of the Params object.

    Attributes:
        allowed_intents (list, str): A list of intents that you can set to force the language
            processor to choose from.
        target_dialogue_state (str): The name of the dialogue handler that you want to reach in
            the next turn.
        time_zone (str):  The name of an IANA time zone, such as 'America/Los_Angeles', or
            'Asia/Kolkata'.
        language (str): The language code representing ISO 639-1/2 language codes
        locale (str, optional): The locale representing the ISO 639-1/2 language code and
            ISO3166 alpha 2 country code separated by an underscore character.
        timestamp (long): A unix time stamp for the request accurate to the nearest second.
        dynamic_resource (dict): A dictionary containing data used to influence the language
            classifiers by adding resource data for the given turn.
    """

    allowed_intents = attr.ib(default=attr.Factory(tuple), converter=tuple)
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
    language = attr.ib(default=None)
    locale = attr.ib(default=None)
    dynamic_resource = attr.ib(default=immutables.Map(), converter=immutables.Map)


def tuple_elems_to_immutable_map(value):
    """Custom attrs converter. Converts a list of elements into a list of immutables.Map
    objects.
    """
    return tuple([immutables.Map(i) for i in value])


@attr.s(frozen=True, kw_only=True)  # pylint: disable=too-many-instance-attributes
class Request:
    """
    The Request is an object passed in through the Dialogue Manager and contains all the
    information provided by the application client for the dialogue handler to act on. Note: the
    Request object is read-only since it represents the client state, which should not be mutated.

    Attributes:
        domains (str): Domain of the current query.
        intent (str): Intent of the current query.
        entities (list): A list of entities in the current query.
        history (list): List of previous and current responder objects (de-serialized) up to the
            current conversation.
        text (str): The query text.
        frame (): Immutables Map of stored data across multiple dialogue turns.
        params (Params): An object that modifies how MindMeld process the current turn.
        context (): Immutables Map containing front-end client state that is passed to the
            application from the client in the request.
        confidences (): Immutables Map of keys ``domains``, ``intents``, ``entities`` and ``roles``
            containing confidence probabilities across all labels for each classifier.
        nbest_transcripts_text (tuple): List of alternate n-best transcripts from an ASR system
        nbest_transcripts_entities (tuple): List of lists of extracted entities for each of the
            n-best transcripts.
        nbest_aligned_entities (tuple): List of lists of aligned entities for each of the n-best
            transcripts.
    """

    domain = attr.ib(default=None)
    intent = attr.ib(default=None)
    entities = attr.ib(
        default=attr.Factory(tuple), converter=tuple_elems_to_immutable_map
    )
    history = attr.ib(
        default=attr.Factory(tuple), converter=tuple_elems_to_immutable_map
    )
    text = attr.ib(default=None)
    frame = attr.ib(default=immutables.Map(), converter=immutables.Map)
    params = attr.ib(default=FrozenParams())
    context = attr.ib(default=immutables.Map(), converter=immutables.Map)
    confidences = attr.ib(default=immutables.Map(), converter=immutables.Map)
    nbest_transcripts_text = attr.ib(
        default=attr.Factory(tuple), converter=tuple_elems_to_immutable_map
    )
    nbest_transcripts_entities = attr.ib(
        default=attr.Factory(tuple), converter=tuple_elems_to_immutable_map
    )
    nbest_aligned_entities = attr.ib(
        default=attr.Factory(tuple), converter=tuple_elems_to_immutable_map
    )

    def to_dict(self):
        return {
            "text": self.text,
            "domain": self.domain,
            "intent": self.intent,
            "context": dict(self.context),
            "params": self.params.to_dict(),
            "frame": dict(self.frame),
        }
