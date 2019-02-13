# -*- coding: utf-8 -*-
"""
This module contains the application manager
"""
import logging

from .components.request import Request, Params, FrozenParams
from .components import (
    NaturalLanguageProcessor, DialogueManager, QuestionAnswerer
)
from .components.dialogue import DialogueResponder
from .resource_loader import ResourceLoader


logger = logging.getLogger(__name__)


class ApplicationManager:
    """The Application Manager is the core orchestrator of the MindMeld platform. It receives
    a client request from the gateway, and processes that request by passing it through all the
    necessary components of Workbench. Once processing is complete, the application manager
    returns the final response back to the gateway.
    """

    MAX_HISTORY_LEN = 100

    def __init__(self, app_path, nlp=None, question_answerer=None, es_host=None,
                 request_class=None, responder_class=None, preprocessor=None, async_mode=False):
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
        self.question_answerer = question_answerer or QuestionAnswerer(
            app_path, resource_loader, es_host)
        self.request_class = request_class or Request
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

    def _pre_dm(self, processed_query, context, params, frame, history):
        request = self.request_class(context=context, history=history, frame=frame,
                                     params=FrozenParams(previous_params=params), **processed_query)

        response = self.responder_class(frame=frame, params=Params(previous_params=params),
                                        slots={}, history=history, request=request,
                                        directives=[])
        return request, response

    def parse(self, text, params=None, context=None, frame=None, history=None, verbose=False):
        """
        Args:
            text (str): The text of the message sent by the user
            params (Params/dict, optional): Contains parameters which modify how text is parsed
            params.allowed_intents (list, optional): A list of allowed intents
                for model consideration
            params.target_dialogue_state (str, optional): The target dialogue state
            params.time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            params.timestamp (long, optional): A unix time stamp for the request (in seconds).
            context (dict, optional): A dictionary of app-specific data
            history (list, optional): A list of previous and current responder objects
                                      through interactions with workbench
            verbose (bool, optional): Flag to return confidence scores for domains and intents

        Returns:
            (dict): A deserialized Responder object

        .. _IANA tz database:
           https://www.iana.org/time-zones

        .. _List of tz database time zones:
           https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

        """
        if self.async_mode:
            return self._parse_async(text, params=params, context=context, frame=frame,
                                     history=history, verbose=verbose)

        params = params or FrozenParams()
        if type(params) == dict:
            params = FrozenParams(**params)
        elif type(params) == Params:
            params = FrozenParams(**DialogueResponder.to_json(params))
        elif not type(params) == FrozenParams:
            raise TypeError("Invalid type for params argument. "
                            "Should be dict or {}".format(FrozenParams.__name__))

        history = history or []
        frame = frame or {}
        context = context or {}

        allowed_intents, nlp_params, dm_params = self._pre_nlp(params, verbose)
        processed_query = self.nlp.process(query_text=text, allowed_intents=allowed_intents,
                                           **nlp_params)
        request, response = self._pre_dm(processed_query=processed_query,
                                         context=context, history=history,
                                         frame=frame, params=params)
        dm_response = self.dialogue_manager.apply_handler(request, response, **dm_params)
        response = self._post_dm(request, dm_response)
        return response

    async def _parse_async(self, text, params=None, context=None, frame=None,
                           history=None, verbose=False):
        """
        Args:
            text (str): The text of the message sent by the user
            params (Params, optional): Contains parameters which modify how text is parsed
            params.allowed_intents (list, optional): A list of allowed intents
                for model consideration
            params.target_dialogue_state (str, optional): The target dialogue state
            params.time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            params.timestamp (long, optional): A unix time stamp for the request (in seconds).
            context (dict, optional): A dictionary of app-specific data
            history (list, optional): A list of previous and current responder objects
                                      through interactions with workbench
            verbose (bool, optional): Flag to return confidence scores for domains and intents

        Returns:
            (dict): A deserialized Responder object

        .. _IANA tz database:
           https://www.iana.org/time-zones

        .. _List of tz database time zones:
           https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

        """
        params = params or FrozenParams()
        if isinstance(params, dict):
            params = FrozenParams(**params)
        elif isinstance(params, Params):
            params = FrozenParams(**DialogueResponder.to_json(params))
        elif not isinstance(params, FrozenParams):
            raise TypeError("Invalid type for params argument. "
                            "Should be dict or {}".format(FrozenParams.__name__))

        context = context or {}
        history = history or []
        frame = frame or {}

        allowed_intents, nlp_params, dm_params = self._pre_nlp(params, verbose)
        processed_query = self.nlp.process(query_text=text,
                                           allowed_intents=allowed_intents,
                                           **nlp_params)
        request, response = self._pre_dm(processed_query=processed_query,
                                         context=context, history=history,
                                         frame=frame, params=params)
        # TODO: make an async nlp
        # processed_query = await self.nlp.process(text, nlp_hierarchy, **process_params)
        dm_response = await self.dialogue_manager.apply_handler(request, response, **dm_params)
        response = self._post_dm(request, dm_response)

        return response

    def _pre_nlp(self, params, verbose=False):
        # validate params
        allowed_intents = params.validate_param('allowed_intents')
        nlp_params = params.nlp_params()
        nlp_params['verbose'] = verbose
        return allowed_intents, nlp_params, params.dm_params(
            self.dialogue_manager.handler_map)

    def _post_dm(self, request, dm_response):
        # Append this item to the history, but don't recursively store history
        prev_request = DialogueResponder.to_json(dm_response)
        prev_request.pop('history')

        # limit length of history
        new_history = (prev_request,) + request.history
        dm_response.history = new_history[:self.MAX_HISTORY_LEN]

        # validate outgoing params
        dm_response.params.validate_param('allowed_intents')
        dm_response.params.validate_param('target_dialogue_state')
        return dm_response

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
