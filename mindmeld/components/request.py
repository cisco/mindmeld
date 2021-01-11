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
from typing import Optional, Dict
import attr
import immutables
import pycountry
from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError
from marshmallow import EXCLUDE, Schema
from marshmallow import fields

logger = logging.getLogger(__name__)


def validate_language_code(value: Optional[str]) -> Optional[str]:
    """Validates language code parameters
    Args:
        value (str): The language code parameter

    Returns:
        str: A validated language code or None if unvalidated
    """
    if not value:
        return None

    if not isinstance(value, str):
        logger.error("Invalid language param: %s is not of type %s.", value, str)
        return None

    # The pycountry APIs need the param to be in lowercase for processing
    value = value.lower()

    if len(value) != 2 and len(value) != 3:
        logger.error(
            "Invalid language param: %s is not a valid ISO 639-1 or ISO 639-2 language code.",
            value,
        )
        return None

    if len(value) == 2 and not pycountry.languages.get(alpha_2=value):
        logger.error(
            "Invalid language param: %s is not a valid ISO 639-1 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for valid codes.",
            value,
        )
        return None

    if len(value) == 3 and not pycountry.languages.get(alpha_3=value):
        logger.error(
            "Invalid language param: %s is not a valid ISO 639-2 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes for valid codes.",
            value,
        )
        return None

    return value


def validate_locale_code(value: Optional[str]) -> Optional[str]:
    """Validates the locale code parameters
    Args:
        value (str): The locale code parameter

    Returns:
        str: A validated locale code or None if unvalidated
    """
    if not value:
        return None

    if not isinstance(value, str):
        logger.error("Invalid locale_code param: %s is not of type %s.", value, str)

    if len(value.split("_")) != 2:
        logger.error("Invalid locale_code param: %s is not a valid locale.", value)
        return None

    language_code = value.split("_")[0].lower()
    if not validate_language_code(language_code):
        logger.error(
            "Invalid locale_code param: %s is not a valid ISO 639-1 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for valid codes.",
            language_code,
        )
        return None

    # pycountry requires the country code to be upper-cased
    country_code = value.split("_")[1].upper()
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


def validate_locale_code_with_ref_language_code(locale: Optional[str],
                                                reference_language_code: str) -> Optional[str]:
    """This function makes sure the locale is consistent with the app's language code"""
    locale = validate_locale_code(locale)
    # if the developer or app doesnt specify the locale, we just use the default locale
    if not locale:
        return

    if locale.split("_")[0].lower() != reference_language_code.lower():
        logger.error(
            "Locale %s is inconsistent with app language code %s. "
            "Set the language code in the config.py file."
            "Using the default locale code instead.", locale, reference_language_code
        )
        return

    return locale


class LanguageCodeField(fields.String):

    def _serialize(self,
                   value,
                   attribute,  # pylint: disable=unused-argument
                   obj,  # pylint: disable=unused-argument
                   **kwargs):
        if value is None:
            return ""
        return str(value)

    def _deserialize(self,
                     value,
                     attribute,  # pylint: disable=unused-argument
                     data,  # pylint: disable=unused-argument
                     **kwargs):
        try:
            return validate_language_code(value)
        except ValueError as error:
            logger.warning(
                "Invalid language param: %s has a wrong value that caused %s.", value, error
            )
            return None


class LocaleCodeField(fields.String):

    def _serialize(self,
                   value,
                   attribute,  # pylint: disable=unused-argument
                   obj,  # pylint: disable=unused-argument
                   **kwargs):
        if value is None:
            return ""
        return str(value)

    def _deserialize(self,
                     value,
                     attribute,  # pylint: disable=unused-argument
                     data,  # pylint: disable=unused-argument
                     **kwargs):
        try:
            return validate_locale_code(value)
        except ValueError as error:
            logger.warning(
                "Invalid locale_code param: %s has a wrong value that caused %s.", value, error
            )
            return None


class TimeZoneField(fields.String):

    def _serialize(self,
                   value,
                   attribute,  # pylint: disable=unused-argument
                   obj,  # pylint: disable=unused-argument
                   **kwargs):
        if value is None:
            return ""
        return str(value)

    def _deserialize(self,
                     value,
                     attribute,  # pylint: disable=unused-argument
                     data,  # pylint: disable=unused-argument
                     **kwargs):
        try:
            return timezone(value)
        except ValueError as error:
            logger.warning(
                "Invalid time_zone param: %s has a wrong value that caused %s.", value, error
            )
            return None
        except UnknownTimeZoneError:
            logger.warning(
                "Invalid time_zone param: %s is not a valid time zone.", value
            )
            return None


class ParamsSchema(Schema):
    allowed_intents = fields.List(fields.String, data_key='allowed_intents')
    time_zone = TimeZoneField(data_key='time_zone', allow_none=True)
    dynamic_resource = fields.Dict(data_key='dynamic_resource')
    language = LanguageCodeField(allow_none=True)
    locale = LocaleCodeField(allow_none=True)
    timestamp = fields.Integer()
    target_dialogue_state = fields.String(allow_none=True)

    class Meta:
        unknown = EXCLUDE


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

    def validate_dm_params(self, handler_map: Dict) -> Dict:
        """
        Validate that the value of the 'target_dialogue_state' parameter is a valid dialogue state
            for the application and returns that value in a dictionary.

        Args:
            handler_map (dict): Mapping from dialogue state to the function handler that gets
                called when in the state.

        Returns:
            dict: single item dictionary with the parameter value if valid and None if not.
        """
        if self.target_dialogue_state and self.target_dialogue_state not in handler_map:
            logger.error(
                "Target dialogue state %s does not match any dialogue state names "
                "in for the application. Not applying the target dialogue state "
                "this turn.",
                self.target_dialogue_state,
            )
            return {"target_dialogue_state": None}
        return {"target_dialogue_state": self.target_dialogue_state}

    def to_dict(self) -> Dict:
        """This method is primarily implemented to return a mutable dictionary for the dynamic_resource
            param and a mutable list for the allowed_intents param.
        """
        _dic = params_schema.dump(self)

        # converting from immutable map to just dictionary
        _dic["dynamic_resource"] = {
            key: _dic["dynamic_resource"][key] for key in _dic["dynamic_resource"]
        }

        _dic["allowed_intents"] = list(_dic["allowed_intents"])
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


class RequestSchema(Schema):
    text = fields.String(required=True)
    domain = fields.String()
    intent = fields.String()
    entities = fields.List(fields.String)
    history = fields.List(fields.Dict)
    params = fields.Nested(ParamsSchema)
    context = fields.Dict()
    confidences = fields.Dict()
    nbest_transcripts_text = fields.List(fields.String)
    nbest_transcripts_entities = fields.List(fields.Dict)
    nbest_aligned_entities = fields.List(fields.Dict)
    request_id = fields.String()


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

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "domain": self.domain,
            "intent": self.intent,
            "context": dict(self.context),
            "params": params_schema.dump(self.params),
            "frame": dict(self.frame),
        }


params_schema = ParamsSchema()
request_schema = RequestSchema()
