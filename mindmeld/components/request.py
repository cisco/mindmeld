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
from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError

logger = logging.getLogger(__name__)


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
        logger.warning("Invalid %r param: %s is not of type %s.", 'time_zone', param, str)
        return None
    try:
        timezone(param)
    except UnknownTimeZoneError:
        logger.warning("Invalid %r param: %s is not a valid time zone.", 'time_zone', param)
        return None
    return param


def _validate_generic(name, ptype):
    def validator(param):
        if not isinstance(param, ptype):
            logger.warning("Invalid %r param: %s is not of type %s.", name, param, ptype)
            param = None
        return param
    return validator


PARAM_VALIDATORS = {
    'allowed_intents': _validate_generic('allowed_intents', tuple),
    'target_dialogue_state': _validate_generic('target_dialogue_state', str),
    'time_zone': _validate_time_zone,
    'timestamp': _validate_generic('timestamp', int),
    'dynamic_resource': _validate_generic('dynamic_resource', immutables.Map)
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
        dynamic_resource (dict): A dictionary containing data used to influence the language
            classifiers by adding resource data for the given turn.
    """
    allowed_intents = attr.ib(default=attr.Factory(tuple))
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
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

    def dm_params(self, handler_map):
        """
        Check that the value of the 'target_dialogue_state' parameter is a valid dialogue state
            for the application.

        Args:
            handler_map (dict): Mapping from dialogue state to the function handler that gets
                called when in the state.

        Returns:
            dict: single item dictionary with the parameter value if valid and None if not.
        """
        target_dialogue_state = self.validate_param('target_dialogue_state')
        if target_dialogue_state and target_dialogue_state not in handler_map:
            logger.error("Target dialogue state %s does not match any dialogue state names "
                         "in for the application. Not applying the target dialogue state "
                         "this turn.", target_dialogue_state)
            return {'target_dialogue_state': None}
        return {'target_dialogue_state': target_dialogue_state}

    def nlp_params(self):
        """
        Validate time zone, timestamp, and dynamic resource parameters.

        Returns:
            dict: Mapping from parameter name to bool depending on validation.
        """
        return {param: self.validate_param(param)
                for param in ('time_zone', 'timestamp', 'dynamic_resource')}


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
        timestamp (long): A unix time stamp for the request accurate to the nearest second.
        dynamic_resource (dict): A dictionary containing data used to influence the language
            classifiers by adding resource data for the given turn.
    """
    allowed_intents = attr.ib(default=attr.Factory(tuple),
                              converter=tuple)
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
    dynamic_resource = attr.ib(default=immutables.Map(),
                               converter=immutables.Map)


@attr.s(frozen=True, kw_only=True)
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
    entities = attr.ib(default=attr.Factory(tuple),
                       converter=tuple)
    history = attr.ib(default=attr.Factory(tuple),
                      converter=tuple)
    text = attr.ib(default=None)
    frame = attr.ib(default=immutables.Map(),
                    converter=immutables.Map)
    params = attr.ib(default=FrozenParams())
    context = attr.ib(default=immutables.Map(),
                      converter=immutables.Map)
    confidences = attr.ib(default=immutables.Map(),
                          converter=immutables.Map)
    nbest_transcripts_text = attr.ib(default=attr.Factory(tuple),
                                     converter=tuple)
    nbest_transcripts_entities = attr.ib(default=attr.Factory(tuple),
                                         converter=tuple)
    nbest_aligned_entities = attr.ib(default=attr.Factory(tuple),
                                     converter=tuple)
