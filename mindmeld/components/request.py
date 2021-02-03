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
from typing import Optional, Dict, Any, List
import attr
import immutables
import pycountry
from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError
from marshmallow import EXCLUDE, Schema, fields, ValidationError

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
        raise ValidationError("Invalid language param: %s is not of type str." % value)

    # The pycountry APIs need the param to be in lowercase for processing
    value = value.lower()

    if len(value) != 2 and len(value) != 3:
        raise ValidationError(
            "Invalid language param: %s is not a "
            "valid ISO 639-1 or ISO 639-2 language code." % value
        )

    if len(value) == 2 and not pycountry.languages.get(alpha_2=value):
        raise ValidationError(
            "Invalid language param: %s is not a valid ISO 639-1 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for valid codes." % value
        )

    if len(value) == 3 and not pycountry.languages.get(alpha_3=value):
        raise ValidationError(
            "Invalid language param: %s is not a valid ISO 639-2 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes for valid codes." % value
        )

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
        raise ValidationError("Invalid locale_code param: %s is not of type str" % value)

    if len(value.split("_")) != 2:
        raise ValidationError("Invalid locale_code param: %s is not a valid locale." % value)

    language_code = value.split("_")[0].lower()
    if not validate_language_code(language_code):
        raise ValidationError(
            "Invalid locale_code param: %s is not a valid ISO 639-1 language code. "
            "See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for valid codes." %
            language_code
        )

    # pycountry requires the country code to be upper-cased
    country_code = value.split("_")[1].upper()
    if not pycountry.countries.get(alpha_2=country_code):
        raise ValidationError(
            "Invalid %r param: %s is not a valid ISO3166 alpha 2 country code. "
            "See https://www.iso.org/obp/ui/#search for valid codes." %
            ("locale", country_code)
        )

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


def validate_timestamp(value: str) -> int:
    result = int(value)
    if len(value) > 13:
        raise ValidationError(f"Invalid timestamp {value}, it should be a 13 digit UTC "
                              f"timestamp representation precise to the nearest millisecond. "
                              f"Using the process timestamp instead.")

    if len(value) <= 10:
        # Convert a second grain unix timestamp to millisecond
        logger.debug(
            "Warning: Possible non-millisecond unix timestamp passed in %s. "
            "Multiplying it by 1000 to represent the timestamp in milliseconds.", value
        )
        result *= 1000

    return result


def _validate_allowed_intents(list_of_allowed_intents: List[str],
                              nlp: Any) -> List[str]:
    if not nlp or not list_of_allowed_intents:
        return list_of_allowed_intents

    for allowed_nlp_component in list_of_allowed_intents:
        if not isinstance(allowed_nlp_component, str):
            raise ValidationError(
                f"Invalid allowed_intents param: {allowed_nlp_component} is not of type str"
            )

        nlp_entries = [None, None, None, None]
        entries = allowed_nlp_component.split(".")[:len(nlp_entries)]
        for idx, entry in enumerate(entries):
            nlp_entries[idx] = entry

        domain, intent, _, _ = nlp_entries

        if not domain or domain not in nlp.domains:
            raise ValidationError(
                f"Domain: {domain} is not in the NLP component hierarchy"
            )

        if not intent or (intent != "*" and intent not in nlp.domains[domain].intents):
            raise ValidationError(
                f"Intent: {intent} is not in the NLP component hierarchy"
            )
    return list_of_allowed_intents


def _validate_target_dialogue_state(target_dialogue_state: Optional[str],
                                    dialogue_handler_map: Optional[Dict]) -> Optional[str]:
    if not target_dialogue_state:
        return None

    if not dialogue_handler_map:
        return target_dialogue_state

    if target_dialogue_state not in dialogue_handler_map:
        raise ValidationError(
            f"Target dialogue state {target_dialogue_state} does not match any "
            f"dialogue state names in for the application"
        )
    return target_dialogue_state


class LanguageCodeField(fields.String):

    def _serialize(self,
                   value,
                   attribute,  # pylint: disable=unused-argument
                   obj,  # pylint: disable=unused-argument
                   **kwargs):
        if value is None:
            return
        return str(value)

    def _deserialize(self,
                     value,
                     attribute,  # pylint: disable=unused-argument
                     data,  # pylint: disable=unused-argument
                     **kwargs):
        try:
            return validate_language_code(value)
        except ValueError as error:
            raise ValidationError(
                f"Invalid language param: {value} has a wrong value that caused {str(error)}."
            ) from error


class LocaleCodeField(fields.String):

    def _serialize(self,
                   value,
                   attribute,  # pylint: disable=unused-argument
                   obj,  # pylint: disable=unused-argument
                   **kwargs):
        if value is None:
            return None
        return str(value)

    def _deserialize(self,
                     value,
                     attribute,  # pylint: disable=unused-argument
                     data,  # pylint: disable=unused-argument
                     **kwargs):
        try:
            return validate_locale_code(value)
        except ValueError as error:
            raise ValidationError(
                f"Invalid locale_code param: {value} has a "
                f"wrong value that caused {str(error)}.") from error


class TimeZoneField(fields.String):

    def _serialize(self,
                   value,
                   attribute,  # pylint: disable=unused-argument
                   obj,  # pylint: disable=unused-argument
                   **kwargs):
        if value is None:
            return
        return str(value)

    def _deserialize(self,
                     value,
                     attribute,  # pylint: disable=unused-argument
                     data,  # pylint: disable=unused-argument
                     **kwargs):
        try:
            return timezone(value)
        except ValueError as error:
            raise ValidationError(f"Invalid time_zone param: {value} "
                                  f"has a wrong value that caused {str(error)}.") from error
        except UnknownTimeZoneError as error:
            raise ValidationError(f"Invalid time_zone param: {value} "
                                  f"is not a valid time zone.") from error


class TimestampField(fields.Integer):
    def _serialize(self,
                   value,
                   attribute,  # pylint: disable=unused-argument
                   obj,  # pylint: disable=unused-argument
                   **kwargs):
        if value is None:
            return
        return str(value)

    def _deserialize(self,
                     value,
                     attribute,  # pylint: disable=unused-argument
                     data,  # pylint: disable=unused-argument
                     **kwargs):
        try:
            return validate_timestamp(value)
        except ValueError as error:
            raise ValidationError(f"Invalid timestamp param: {value} has "
                                  f"a wrong value that caused {str(error)}.") from error


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
    timestamp = attr.ib(default=None)
    language = attr.ib(default=None)
    locale = attr.ib(default=None)
    dynamic_resource = attr.ib(default=attr.Factory(dict))

    def to_dict(self) -> Dict[str, Any]:
        """This method is primarily implemented to return a mutable dictionary for the dynamic_resource
            param and a mutable list for the allowed_intents param.
        """
        _dic = params_schema.dump(self)
        # Pop out fields that are set to None
        for field in ['allowed_intents', 'target_dialogue_state', 'timestamp', 'time_zone']:
            if field in _dic and not _dic[field]:
                _dic.pop(field, None)
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
    timestamp = attr.ib(default=None)
    language = attr.ib(default=None)
    locale = attr.ib(default=None)
    dynamic_resource = attr.ib(default=immutables.Map(), converter=immutables.Map)


class ParamsSchema(Schema):
    allowed_intents = fields.Method("serialize_allowed_intents",
                                    deserialize="deserialize_allowed_intents")
    time_zone = TimeZoneField(allow_none=True)
    dynamic_resource = fields.Method("serialize_dynamic_resource",
                                     deserialize="deserialize_dynamic_resource")
    language = LanguageCodeField(allow_none=True)
    locale = LocaleCodeField(allow_none=True)
    timestamp = TimestampField(allow_none=True)
    target_dialogue_state = fields.Method("serialize_target_dialogue_state",
                                          deserialize="deserialize_target_dialogue_state",
                                          allow_none=True)

    def serialize_allowed_intents(self, params: Params) -> List[str]:
        return list(_validate_allowed_intents(params.allowed_intents, self.context.get('nlp')))

    def deserialize_allowed_intents(self, allowed_intents: List[str]) -> List[str]:
        return _validate_allowed_intents(allowed_intents, self.context.get('nlp'))

    def serialize_target_dialogue_state(self, params: Params) -> Optional[str]:
        return _validate_target_dialogue_state(
            params.target_dialogue_state,
            self.context.get('dialogue_handler_map'))

    def deserialize_target_dialogue_state(self, target_dialogue_state: str) -> Optional[str]:
        return _validate_target_dialogue_state(
            target_dialogue_state,
            self.context.get('dialogue_handler_map'))

    def serialize_dynamic_resource(self, params: Params):  # pylint: disable=no-self-use
        return dict(params.dynamic_resource)

    def deserialize_dynamic_resource(self, value):  # pylint: disable=no-self-use
        return immutables.Map(value)

    class Meta:
        unknown = EXCLUDE


def deserialize_to_list_immutable_maps(value):
    """Custom attrs converter. Converts a list of elements into a list of immutables.Map
    objects.
    """
    return tuple([immutables.Map(i) for i in value])


def deserialize_to_lists_of_list_of_immutable_maps(values):
    """Custom attrs converter. Converts a list of elements into a list of immutables.Map
    objects.
    """
    return tuple([deserialize_to_list_immutable_maps(value) for value in values])


def serialize_to_list_of_dicts(values):
    """Custom attrs converter. Converts a list of elements into a list of immutables.Map
    objects.
    """
    return [dict(value) for value in values]


def serialize_to_lists_of_list_of_dicts(values):
    """Custom attrs converter. Converts a list of elements into a list of immutables.Map
    objects.
    """
    return [serialize_to_list_of_dicts(value) for value in values]


@attr.s(frozen=True, kw_only=True)  # pylint: disable=too-many-instance-attributes
class Request:
    """
    The Request is an object passed in through the Dialogue Manager and contains all the
    information provided by the application client for the dialogue handler to act on. Note: the
    Request object is read-only since it represents the client state, which should not be mutated.

    Attributes:
        domains (str): Domain of the current query.
        intent (str): Intent of the current query.
        entities (list of dicts): A list of entities in the current query.
        history (list of dicts): List of previous and current responder objects
            (de-serialized) up to the current conversation.
        text (str): The query text.
        frame (): Immutables Map of stored data across multiple dialogue turns.
        params (Params): An object that modifies how MindMeld process the current turn.
        context (dict): Immutables Map containing front-end client state that is passed to the
            application from the client in the request.
        confidences (dict): Immutables Map of keys ``domains``, ``intents``, ``entities``
            and ``roles`` containing confidence probabilities across all labels for
            each classifier.
        nbest_transcripts_text (tuple): List of alternate n-best transcripts from an ASR system
        nbest_transcripts_entities (tuple): List of lists of extracted entities for each of the
            n-best transcripts.
        nbest_aligned_entities (tuple): List of lists of aligned entities for each of the n-best
            transcripts.
    """
    domain = attr.ib(default=None)
    intent = attr.ib(default=None)
    entities = attr.ib(
        default=attr.Factory(tuple), converter=deserialize_to_list_immutable_maps
    )
    history = attr.ib(
        default=attr.Factory(tuple), converter=deserialize_to_list_immutable_maps
    )
    text = attr.ib(default=None)
    frame = attr.ib(default=immutables.Map(), converter=immutables.Map)
    params = attr.ib(default=FrozenParams())
    context = attr.ib(default=immutables.Map(), converter=immutables.Map)
    confidences = attr.ib(default=immutables.Map(), converter=immutables.Map)
    nbest_transcripts_text = attr.ib(
        default=attr.Factory(tuple), converter=tuple
    )
    nbest_transcripts_entities = attr.ib(
        default=attr.Factory(tuple), converter=deserialize_to_lists_of_list_of_immutable_maps
    )
    nbest_aligned_entities = attr.ib(
        default=attr.Factory(tuple), converter=deserialize_to_lists_of_list_of_immutable_maps
    )
    form = attr.ib(default=attr.Factory(dict))

    def to_dict(self) -> Dict[str, Any]:
        return request_schema.dump(self)


class RequestSchema(Schema):
    text = fields.String(required=True)
    domain = fields.String()
    intent = fields.String()
    entities = fields.Method("serialize_entities",
                             deserialize="deserialize_list_of_maps")
    history = fields.Method("serialize_history",
                            deserialize="deserialize_list_of_maps")
    params = fields.Nested(ParamsSchema)
    frame = fields.Method("serialize_frame",
                          deserialize="deserialize_map")
    context = fields.Method("serialize_context",
                            deserialize="deserialize_map")
    confidences = fields.Method("serialize_confidences",
                                deserialize="deserialize_map")
    nbest_transcripts_text = fields.List(fields.String)
    nbest_transcripts_entities = fields.Method(
        "serialize_nbest_transcripts_entities",
        deserialize="deserialize_list_of_list_of_immutable_maps")
    nbest_aligned_entities = fields.Method(
        "serialize_nbest_aligned_entities",
        deserialize="deserialize_list_of_list_of_immutable_maps")
    request_id = fields.String()

    def deserialize_list_of_maps(self, value):  # pylint: disable=no-self-use
        return deserialize_to_list_immutable_maps(value)

    def deserialize_list_of_list_of_immutable_maps(self, values):  # pylint: disable=no-self-use
        return deserialize_to_lists_of_list_of_immutable_maps(values)

    def serialize_history(self, request: Request):  # pylint: disable=no-self-use
        return serialize_to_list_of_dicts(request.history)

    def serialize_entities(self, request: Request):  # pylint: disable=no-self-use
        return serialize_to_list_of_dicts(request.entities)

    def serialize_nbest_transcripts_entities(self, request: Request):  # pylint: disable=no-self-use
        return serialize_to_lists_of_list_of_dicts(request.nbest_transcripts_entities)

    def serialize_nbest_aligned_entities(self, request: Request):  # pylint: disable=no-self-use
        return serialize_to_lists_of_list_of_dicts(request.nbest_aligned_entities)

    def serialize_confidences(self, request: Request):  # pylint: disable=no-self-use
        return dict(request.confidences)

    def serialize_context(self, request: Request):  # pylint: disable=no-self-use
        return dict(request.context)

    def serialize_frame(self, request: Request):  # pylint: disable=no-self-use
        return dict(request.frame)

    def deserialize_map(self, value):  # pylint: disable=no-self-use
        return immutables.Map(value)


class FormEntitySchema(Schema):
    entity = fields.String()
    role = fields.String()
    responses = fields.List(fields.String())
    retry_responses = fields.List(fields.String())
    value = fields.String()
    default_eval = fields.Boolean(default=True)
    hints = fields.List(fields.String())
    custom_eval = fields.String()


class FormSchema(Schema):
    entities = fields.List(fields.Nested(FormEntitySchema))
    max_retries = fields.Integer()
    exit_msg = fields.String()
    exit_keys = fields.List(fields.String)


class DialogueResponseSchema(Schema):
    frame = fields.Dict()
    params = fields.Nested(ParamsSchema)
    history = fields.List(fields.Dict())
    slots = fields.Dict()
    request = fields.Nested(RequestSchema)
    dialogue_state = fields.String()
    directives = fields.List(fields.Dict())
    form = fields.Nested(FormSchema)


form_schema = FormSchema()
dialogue_response_schema = DialogueResponseSchema()
params_schema = ParamsSchema()
request_schema = RequestSchema()
