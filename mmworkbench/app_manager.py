# -*- coding: utf-8 -*-
"""
This module contains the application manager
"""
from __future__ import absolute_import, unicode_literals
from builtins import object

import copy
import logging

from .components import NaturalLanguageProcessor, DialogueManager, QuestionAnswerer
from .components.dialogue import DialogueResponder
from .resource_loader import ResourceLoader
from .exceptions import AllowedNlpClassesKeyError

logger = logging.getLogger(__name__)


class ApplicationManager(object):
    """The Application Manager is the core orchestrator of the MindMeld platform. It receives
    a client request from the gateway, and processes that request by passing it through all the
    necessary components of Workbench. Once processing is complete, the application manager
    returns the final response back to the gateway.
    """
    def __init__(self, app_path, nlp=None, question_answerer=None, es_host=None,
                 context_class=None, responder_class=None):
        self._app_path = app_path
        # If NLP or QA were passed in, use the resource loader from there
        if nlp:
            resource_loader = nlp.resource_loader
            if question_answerer:
                question_answerer.resource_loader = resource_loader
        elif question_answerer:
            resource_loader = question_answerer.resource_loader
        else:
            resource_loader = ResourceLoader.create_resource_loader(app_path)

        self._query_factory = resource_loader.query_factory

        self.nlp = nlp or NaturalLanguageProcessor(app_path, resource_loader)
        self.question_answerer = question_answerer or QuestionAnswerer(app_path, resource_loader,
                                                                       es_host)
        self.context_class = context_class or dict
        self.responder_class = responder_class or DialogueResponder
        self.dialogue_manager = DialogueManager(self.responder_class)

    @property
    def ready(self):
        return self.nlp.ready

    def load(self):
        """Loads all resources required to run a Workbench application."""
        if self.nlp.ready:
            # if we are ready, don't load again
            return
        self.nlp.load()

    def parse(self, text, payload=None, session=None, frame=None, history=None,
              allowed_intents=None, target_dialogue_state=None, verbose=False):
        """
        Args:
            text (str): The text of the message sent by the user
            payload (dict, optional): Description
            session (dict, optional): Description
            history (list, optional): Description
            allowed_intents (list, optional): A list of allowed intents
            for model consideration
            target_dialogue_state (str, optional): The target dialogue state
            verbose (bool, optional): Description

        Returns:
            (dict): Context object
        """

        session = session or {}
        history = history or []
        frame = frame or {}
        # TODO: what do we do with verbose???

        request = {'text': text, 'session': session}
        if payload:
            request['payload'] = payload

        context = self.context_class(
            {'request': request, 'history': history, 'frame': copy.deepcopy(frame)})

        # Validate target dialogue state
        if target_dialogue_state and target_dialogue_state not in self.dialogue_manager.handler_map:
            logger.error("Target dialogue state {} does not match any dialogue state names "
                         "in for the application. Not applying the target dialogue state "
                         "this turn.".format(target_dialogue_state))
            target_dialogue_state = None

        # We bypass the NLP processing engine if the target dialogue state is specified. This
        # improves performance by decreasing round trip time between the client and wb.
        if not target_dialogue_state:
            nlp_hierarchy = None
            if allowed_intents:
                try:
                    nlp_hierarchy = self.nlp.extract_allowed_intents(allowed_intents)
                except (AllowedNlpClassesKeyError, ValueError, KeyError) as e:
                    # We have to print the error object since it sometimes contains a message
                    # and sometimes it doesn't, like a ValueError.
                    logger.error(
                        "Validation error '{}' on input allowed intents {}. "
                        "Not applying domain/intent restrictions this "
                        "turn".format(e, allowed_intents))

            # TODO: support passing in reference time from session
            query = self._query_factory.create_query(text)

            # TODO: support specifying target domain, etc in payload
            processed_query = self.nlp.process_query(query, nlp_hierarchy)

            context.update(processed_query.to_dict())
            context.pop('text')

        context.update(self.dialogue_manager.apply_handler(context, target_dialogue_state))

        # Append this item to the history, but don't recursively store history
        history = context.pop('history')
        history.insert(0, context)
        response = copy.deepcopy(context)
        response['history'] = history

        return response

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (function): The dialogue state handler function
            **kwargs (dict): A list of options which specify the dialogue rule
        """
        self.dialogue_manager.add_dialogue_rule(name, handler, **kwargs)
