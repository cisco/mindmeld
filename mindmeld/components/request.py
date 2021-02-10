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
from .schemas import DEFAULT_REQUEST_SCHEMA, DEFAULT_PARAMS_SCHEMA

logger = logging.getLogger(__name__)


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

    def __iter__(self):
        for key, value in DEFAULT_PARAMS_SCHEMA.dump(self).items():
            if value:
                yield key, value


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
    form = attr.ib(default=attr.Factory(tuple), converter=immutables.Map)

    def __iter__(self):
        for key, value in DEFAULT_REQUEST_SCHEMA.dump(self).items():
            if value:
                yield key, value
