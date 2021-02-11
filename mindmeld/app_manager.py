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

"""
This module contains the application manager
"""
import logging
import copy

from .components import DialogueManager, NaturalLanguageProcessor, QuestionAnswerer
from .components._config import get_max_history_len
from .components.dialogue import DialogueResponder
from .components.request import FrozenParams, Params, Request
from .resource_loader import ResourceLoader

logger = logging.getLogger(__name__)


def freeze_params(params):
    """
    If params is a dictionary or Params we convert it into FrozenParams.
    Otherwise we raise a TypeError.

    Args:
        params (dict, Params): The input params to convert

    Returns:
        FrozenParams: The converted params object
    """
    params = params or FrozenParams()
    if isinstance(params, dict):
        params = FrozenParams(**params)
    elif params.__class__ == Params:
        params = FrozenParams(**dict(params))
    elif not isinstance(params, FrozenParams):
        raise TypeError(
            "Invalid type for params argument. "
            "Should be dict or {}".format(FrozenParams.__name__)
        )
    return params


class ApplicationManager:
    """The Application Manager is the core orchestrator of the MindMeld platform. It receives \
    a client request, and processes that request by passing it through all the necessary \
    components of MindMeld. Once processing is complete, the application manager returns \
    the final response back to the client.

        Attributes:
            async_mode (bool): Whether the application is asynchronous or synchronous.
            nlp (NaturalLanguageProcessor): The natural language processor.
            question_answerer (QuestionAnswerer): The question answerer.
            request_class (Request): Any class that inherits \
                from Request
            responder_class (DialogueResponder): Any class \
                that inherits from the DialogueResponder
            dialogue_manager (DialogueManager): The application's dialogue manager.
    """

    MAX_HISTORY_LEN = 100
    """The max number of turns in history."""

    def __init__(
        self,
        app_path,
        nlp=None,
        question_answerer=None,
        es_host=None,
        request_class=None,
        responder_class=None,
        preprocessor=None,
        async_mode=False,
    ):
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
                app_path, preprocessor=preprocessor
            )

        self._query_factory = resource_loader.query_factory

        self.nlp = nlp or NaturalLanguageProcessor(app_path, resource_loader)
        self.question_answerer = question_answerer or QuestionAnswerer(
            app_path, resource_loader, es_host
        )
        self.request_class = request_class or Request
        self.responder_class = responder_class or DialogueResponder
        self.dialogue_manager = DialogueManager(
            self.responder_class, async_mode=self.async_mode
        )
        self.max_history_len = (
            get_max_history_len(self._app_path) or self.MAX_HISTORY_LEN
        )

    @property
    def ready(self):
        """Whether the nlp component is ready."""
        return self.nlp.ready

    def load(self):
        """Loads all resources required to run a MindMeld application."""
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
        # TODO: make an async nlp
        self.nlp.load()

    def _pre_dm(self, processed_query, context, params, frame, form, history):
        # We pass in the previous turn's responder's params to the current request

        # TODO: Currently, we serialize the form before passing it to the request and response
        # since its hard to deserialize it.
        request = self.request_class(
            context=context,
            history=history,
            frame=frame,
            form=form,
            params=params,
            **processed_query
        )

        # We reset the current turn's responder's params
        response = self.responder_class(
            frame=frame,
            form={},
            params=Params(),
            slots={},
            history=copy.deepcopy(history),
            request=request,
            directives=[],
        )
        return request, response

    def parse(
        self, text, params=None, context=None, frame=None, form=None, history=None, verbose=False
    ):
        """
        Args:
            text (str): The text of the message sent by the user
            params (Params/dict, optional): Contains parameters which modify how text is parsed
            params.allowed_intents (list, optional): A list of allowed intents \
                for model consideration
            params.target_dialogue_state (str, optional): The target dialogue state
            params.time_zone (str, optional): The name of an IANA time zone, such as \
                'America/Los_Angeles', or 'Asia/Kolkata' \
                See the [tz database](https://www.iana.org/time-zones) for more information.
            params.timestamp (long, optional): A unix time stamp for the request (in seconds).
            frame (dict, optional): A dictionary specifying the frame of the conversation
            context (dict, optional): A dictionary of app-specific data
            history (list, optional): A list of previous and current responder objects \
                                      through interactions with MindMeld
            verbose (bool, optional): Flag to return confidence scores for domains and intents

        Returns:
            TODO: Convert to dict
            (Responder): A Responder object

        .. _IANA tz database:
           https://www.iana.org/time-zones

        .. _List of tz database time zones:
           https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

        """
        if self.async_mode:
            return self._parse_async(
                text,
                params=params,
                context=context,
                frame=frame,
                form=form,
                history=history,
                verbose=verbose,
            )
        params = freeze_params(params)
        history = history or []
        frame = frame or {}
        form = form or {}
        context = context or {}
        processed_query = self.nlp.process(query_text=text,
                                           allowed_intents=params.allowed_intents,
                                           locale=params.locale,
                                           language=params.language,
                                           time_zone=params.time_zone,
                                           timestamp=params.timestamp,
                                           dynamic_resource=params.dynamic_resource,
                                           verbose=verbose)
        request, response = self._pre_dm(
            processed_query=processed_query,
            context=context,
            history=history,
            frame=frame,
            form=form,
            params=params,
        )
        dm_responder = self.dialogue_manager.apply_handler(
            request, response, target_dialogue_state=params.target_dialogue_state
        )
        modified_dm_responder = self._post_dm(dm_responder)
        return modified_dm_responder

    async def _parse_async(
        self, text, params=None, context=None, frame=None, form=None, history=None, verbose=False
    ):
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
                                      through interactions with MindMeld
            verbose (bool, optional): Flag to return confidence scores for domains and intents

        Returns:
            @TODO: Convert to dict
            (Responder): A Responder object

        .. _IANA tz database:
           https://www.iana.org/time-zones

        .. _List of tz database time zones:
           https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

        """
        params = freeze_params(params)
        context = context or {}
        history = history or []
        frame = frame or {}
        form = form or {}
        # TODO: make an async nlp
        processed_query = self.nlp.process(query_text=text,
                                           allowed_intents=params.allowed_intents,
                                           locale=params.locale,
                                           language=params.language,
                                           time_zone=params.time_zone,
                                           timestamp=params.timestamp,
                                           dynamic_resource=params.dynamic_resource,
                                           verbose=verbose)

        request, response = self._pre_dm(
            processed_query=processed_query,
            context=context,
            history=history,
            frame=frame,
            form=form,
            params=params,
        )

        dm_responder = await self.dialogue_manager.apply_handler(
            request, response, target_dialogue_state=params.target_dialogue_state
        )
        modified_dm_responder = self._post_dm(dm_responder)
        return modified_dm_responder

    def _post_dm(self, dm_response):
        # Append this item to the history, but don't recursively store history
        prev_request = dict(dm_response)
        prev_request.pop("history", None)
        prev_request["request"].pop("history", None)

        # limit length of history
        new_history = [prev_request, ] + dm_response.history
        dm_response.history = new_history[: self.max_history_len]
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
            kwargs (dict): A list of options which specify the dialogue rule
        """
        self.dialogue_manager.add_dialogue_rule(name, handler, **kwargs)
