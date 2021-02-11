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
    if isinstance(value, int):
        value = str(value)

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


class ParamsSchema(Schema):
    allowed_intents = fields.Method("serialize_allowed_intents",
                                    deserialize="deserialize_allowed_intents",
                                    allow_none=True)
    time_zone = TimeZoneField(allow_none=True)
    dynamic_resource = fields.Method("serialize_dynamic_resource",
                                     deserialize="deserialize_dynamic_resource",
                                     allow_none=True)
    language = LanguageCodeField(allow_none=True)
    locale = LocaleCodeField(allow_none=True)
    timestamp = TimestampField(allow_none=True)
    target_dialogue_state = fields.Method("serialize_target_dialogue_state",
                                          deserialize="deserialize_target_dialogue_state",
                                          allow_none=True)

    def serialize_allowed_intents(self, params) -> List[str]:
        return list(_validate_allowed_intents(params.allowed_intents, self.context.get('nlp')))

    def deserialize_allowed_intents(self, allowed_intents: List[str]) -> List[str]:
        return _validate_allowed_intents(allowed_intents, self.context.get('nlp'))

    def serialize_target_dialogue_state(self, params) -> Optional[str]:
        return _validate_target_dialogue_state(
            params.target_dialogue_state,
            self.context.get('dialogue_handler_map'))

    def deserialize_target_dialogue_state(self, target_dialogue_state: str) -> Optional[str]:
        return _validate_target_dialogue_state(
            target_dialogue_state,
            self.context.get('dialogue_handler_map'))

    def serialize_dynamic_resource(self, params):  # pylint: disable=no-self-use
        return dict(params.dynamic_resource)

    def deserialize_dynamic_resource(self, value):  # pylint: disable=no-self-use
        return immutables.Map(value)

    class Meta:
        unknown = EXCLUDE


class FormEntitySchema(Schema):
    entity = fields.String(required=True)
    role = fields.String(allow_none=True)
    responses = fields.List(fields.String(), allow_none=True)
    retry_response = fields.List(fields.String(), allow_none=True)
    value = fields.Dict(allow_none=True)
    default_eval = fields.Boolean(default=True)
    hints = fields.List(fields.String(), allow_none=True)
    custom_eval = fields.String(allow_none=True)

    def serialize_value(self, form):  # pylint: disable=no-self-use
        if form:
            return form.value or dict(form.value)

    def deserialize_value(self, value):  # pylint: disable=no-self-use
        return value or immutables.Map(value)


class FormSchema(Schema):
    entities = fields.List(fields.Nested(FormEntitySchema), required=True)
    max_retries = fields.Integer(allow_none=True)
    exit_msg = fields.String(allow_none=True)
    exit_keys = fields.List(fields.String, allow_none=True)


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
    form = fields.Method("serialize_form",
                         deserialize="deserialize_map")
    request_id = fields.String()

    def deserialize_list_of_maps(self, value):  # pylint: disable=no-self-use
        return deserialize_to_list_immutable_maps(value)

    def deserialize_list_of_list_of_immutable_maps(self, values):  # pylint: disable=no-self-use
        return deserialize_to_lists_of_list_of_immutable_maps(values)

    def serialize_history(self, request):  # pylint: disable=no-self-use
        return serialize_to_list_of_dicts(request.history)

    def serialize_entities(self, request):  # pylint: disable=no-self-use
        return serialize_to_list_of_dicts(request.entities)

    def serialize_nbest_transcripts_entities(self, request):  # pylint: disable=no-self-use
        return serialize_to_lists_of_list_of_dicts(request.nbest_transcripts_entities)

    def serialize_nbest_aligned_entities(self, request):  # pylint: disable=no-self-use
        return serialize_to_lists_of_list_of_dicts(request.nbest_aligned_entities)

    def serialize_confidences(self, request):  # pylint: disable=no-self-use
        return dict(request.confidences)

    def serialize_context(self, request):  # pylint: disable=no-self-use
        return dict(request.context)

    def serialize_frame(self, request):  # pylint: disable=no-self-use
        return dict(request.frame)

    def serialize_form(self, request):  # pylint: disable=no-self-use
        return dict(request.form)

    def deserialize_map(self, value):  # pylint: disable=no-self-use
        return immutables.Map(value)


class DialogueResponseSchema(Schema):
    frame = fields.Dict(allow_none=True)
    params = fields.Nested(ParamsSchema)
    history = fields.List(fields.Dict())
    slots = fields.Dict(allow_none=True)
    request = fields.Nested(RequestSchema)
    dialogue_state = fields.String(allow_none=True)
    directives = fields.List(fields.Dict())
    form = fields.Dict(allow_none=True)


# default schema validators
DEFAULT_FORM_SCHEMA = FormSchema()
DEFAULT_RESPONSE_SCHEMA = DialogueResponseSchema()
DEFAULT_PARAMS_SCHEMA = ParamsSchema()
DEFAULT_REQUEST_SCHEMA = RequestSchema()
