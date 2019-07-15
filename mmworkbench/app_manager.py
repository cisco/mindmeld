# -*- coding: utf-8 -*-
"""
This module contains the application manager
"""
import copy
import logging

from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError

from .components import (
    NaturalLanguageProcessor, DialogueManager, QuestionAnswerer, DialogueContext
)
from .components.dialogue import DialogueResponder
from .resource_loader import ResourceLoader
from .exceptions import AllowedNlpClassesKeyError


logger = logging.getLogger(__name__)


def _validate_generic(name, ptype):
    def validator(param):
        if not isinstance(param, ptype):
            logger.warning("Invalid %r param: %s is not of type %s.", name, param, ptype)
            param = None
        return param
    return validator


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


PARAM_VALIDATORS = {
    'allowed_intents': _validate_generic('allowed_intents', list),

    # TODO: use a better validator for this
    'target_dialogue_state': _validate_generic('target_dialogue_state', str),

    'time_zone': _validate_time_zone,
    'timestamp': _validate_generic('timestamp', int),
    'dynamic_resource': _validate_generic('dynamic_resource', dict)
}


class ApplicationManager:
    """The Application Manager is the core orchestrator of the MindMeld platform. It receives
    a client request from the gateway, and processes that request by passing it through all the
    necessary components of Workbench. Once processing is complete, the application manager
    returns the final response back to the gateway.
    """

    MAX_HISTORY_LEN = 100

    def __init__(self, app_path, nlp=None, question_answerer=None, es_host=None,
                 context_class=None, responder_class=None, preprocessor=None, async_mode=False):
        self.async_mode = async_mode

        self._app_path = app_path
        # If NLP or QA were passed in, use the resource loader from there
        if nlp:
            resource_loader = nlp.resource_loader
            if question_answerer:
                question_answerer.resource_loader = resource_loader
        elif question_answerer:
            resource_loader = question_answerer.resource_loader
        else:
            resource_loader = ResourceLoader.create_resource_loader(
                app_path, preprocessor=preprocessor)

        self._query_factory = resource_loader.query_factory

        self.nlp = nlp or NaturalLanguageProcessor(app_path, resource_loader)
        self.question_answerer = question_answerer or QuestionAnswerer(app_path, resource_loader,
                                                                       es_host)
        self.context_class = context_class or DialogueContext
        self.responder_class = responder_class or DialogueResponder
        self.dialogue_manager = DialogueManager(self.responder_class, async_mode=self.async_mode)

    @property
    def ready(self):
        return self.nlp.ready

    def load(self):
        """Loads all resources required to run a Workbench application."""
        if self.async_mode:
            return self._load_async()

        if self.nlp.ready:
            # if we are ready, don't load again
            return
        self.nlp.load()

    async def _load_async(self):
        if self.nlp.ready:
            # if we are ready, don't load again
            return
        self.nlp.load()
        # TODO: make an async nlp
        # await self.nlp.load()

    def parse(self, text, params=None, context=None, frame=None, history=None, verbose=False):
        """
        Args:
            text (str): The text of the message sent by the user
            params (dict, optional): Contains parameters which modify how text is parsed
            params['allowed_intents'] (list, optional): A list of allowed intents
                for model consideration
            params['target_dialogue_state'] (str, optional): The target dialogue state
            params['time_zone'] (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            params['timestamp'] (long, optional): A unix time stamp for the request (in seconds).
            context (dict, optional): Description
            history (list, optional): Description
            verbose (bool, optional): Description

        Returns:
            (dict): Context object

        .. _IANA tz database:
           https://www.iana.org/time-zones

        .. _List of tz database time zones:
           https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

        """
        if self.async_mode:
            return self._parse_async(text, params=params, context=context, frame=frame,
                                     history=history, verbose=verbose)

        params = params or {}
        context = context or {}
        history = history or []
        frame = frame or {}
        # TODO: what do we do with verbose???

        dm_context, nlp_hierarchy, process_params, dm_params = \
            self._pre_nlp(text, params, context, history, frame)

        processed_query = self.nlp.process(text, nlp_hierarchy, **process_params)

        dm_context = self._pre_dm(dm_context, processed_query)
        dm_response = self.dialogue_manager.apply_handler(dm_context, **dm_params)

        response = self._post_dm(dm_context, dm_response, history)

        return response

    async def _parse_async(self, text, params=None, context=None, frame=None,
                           history=None, verbose=False):
        """
        Args:
            text (str): The text of the message sent by the user
            params (dict, optional): Contains parameters which modify how text is parsed
            params['allowed_intents'] (list, optional): A list of allowed intents
                for model consideration
            params['target_dialogue_state'] (str, optional): The target dialogue state
            params['time_zone'] (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            params['timestamp'] (long, optional): A unix time stamp for the request (in seconds).
            context (dict, optional): Description
            history (list, optional): Description
            verbose (bool, optional): Description

        Returns:
            (dict): Context object

        .. _IANA tz database:
           https://www.iana.org/time-zones

        .. _List of tz database time zones:
           https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

        """
        params = params or {}
        context = context or {}
        history = history or []
        frame = frame or {}

        dm_context, nlp_hierarchy, process_params, dm_params = \
            self._pre_nlp(text, params, context, history, frame)

        processed_query = self.nlp.process(text, nlp_hierarchy, **process_params)

        # TODO: make an async nlp
        # processed_query = await self.nlp.process(text, nlp_hierarchy, **process_params)

        dm_context = self._pre_dm(dm_context, processed_query)
        dm_response = await self.dialogue_manager.apply_handler(dm_context, **dm_params)

        response = self._post_dm(dm_context, dm_response, history)

        return response

    def _pre_nlp(self, text, params, context, history, frame):
        request = {'text': text, 'params': params, 'context': context}

        # validate params
        allowed_intents = self._validate_param(params, 'allowed_intents')
        target_dialogue_state = self._validate_param(params, 'target_dialogue_state')

        # params for dm.apply_handler()
        dm_params = {'target_dialogue_state': target_dialogue_state}

        # params for nlp.process()
        process_params = {param: self._validate_param(params, param)
                          for param in ('time_zone', 'timestamp', 'dynamic_resource')}

        dm_context = self.context_class({
            'request': request,
            'history': history,
            'params': {},  # params for next turn
            'frame': copy.deepcopy(frame),
            'entities': []
        })

        # Validate target dialogue state
        if target_dialogue_state and target_dialogue_state not in self.dialogue_manager.handler_map:
            logger.error("Target dialogue state {} does not match any dialogue state names "
                         "in for the application. Not applying the target dialogue state "
                         "this turn.".format(target_dialogue_state))
            target_dialogue_state = None

        nlp_hierarchy = None
        if allowed_intents:
            try:
                nlp_hierarchy = self.nlp.extract_allowed_intents(allowed_intents)
            except (AllowedNlpClassesKeyError, ValueError, KeyError) as ex:
                # We have to print the error object since it sometimes contains a message
                # and sometimes it doesn't, like a ValueError.
                logger.error(
                    "Validation error '{}' on input allowed intents {}. "
                    "Not applying domain/intent restrictions this "
                    "turn".format(ex, allowed_intents))

        return dm_context, nlp_hierarchy, process_params, dm_params

    @staticmethod
    def _pre_dm(context, processed_query):
        context.update(processed_query)
        context.pop('text')
        return context

    def _post_dm(self, context, dm_response, history):
        context.update(dm_response)

        # Append this item to the history, but don't recursively store history
        history = context.pop('history')
        history.insert(0, context)

        # validate outgoing params
        self._validate_param(context['params'], 'allowed_intents', mode='outgoing')
        self._validate_param(context['params'], 'target_dialogue_state', mode='outgoing')

        # limit length of history
        history = history[:self.MAX_HISTORY_LEN]
        response = copy.deepcopy(context)
        response['history'] = history

        return response

    def add_middleware(self, middleware):
        """Adds middleware for the dialogue manager.

        Args:
            middleware (callable): A dialogue manager middleware function
        """
        self.dialogue_manager.add_middleware(middleware)

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (function): The dialogue state handler function
            **kwargs (dict): A list of options which specify the dialogue rule
        """
        self.dialogue_manager.add_dialogue_rule(name, handler, **kwargs)

    @staticmethod
    def _validate_param(params, name, mode='incoming'):
        validator = PARAM_VALIDATORS.get(name)
        param = params.get(name)
        if param:
            return validator(param)
        return param
