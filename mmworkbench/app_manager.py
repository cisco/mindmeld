# -*- coding: utf-8 -*-
"""
This module contains the application manager
"""
from __future__ import unicode_literals
from builtins import object

import copy

from .components import NaturalLanguageProcessor, DialogueManager, QuestionAnswerer
from .resource_loader import ResourceLoader


class ApplicationManager(object):
    """The Application Manager is the core orchestrator of the MindMeld platform. It receives
    a client request from the gateway, and processes that request by passing it through all the
    necessary components of Workbench. Once processing is complete, the application manager
    returns the final response back to the gateway.
    """
    def __init__(self, app_path, nlp=None, question_answerer=None, es_host=None):
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
        self.dialogue_manager = DialogueManager()
        self.question_answerer = question_answerer or QuestionAnswerer(app_path, resource_loader,
                                                                       es_host)

    @property
    def ready(self):
        return self.nlp.ready

    def load(self):
        """Loads all resources required to run a Workbench application."""
        if self.nlp.ready:
            # if we are ready, don't load again
            return
        self.nlp.load()

    def parse(self, text, payload=None, session=None, frame=None, history=None, verbose=False):
        """
        Args:
            text (str): The text of the message sent by the user
            payload (dict, optional): Description
            session (dict, optional): Description
            history (list, optional): Description
            verbose (bool, optional): Description

        """
        session = session or {}
        history = history or []
        frame = frame or {}
        # TODO: what do we do with verbose???
        # TODO: where is the frame stored?

        request = {'text': text, 'session': session}
        if payload:
            request['payload'] = payload

        # TODO: support passing in reference time from session
        query = self._query_factory.create_query(text)

        # TODO: support specifying target domain, etc in payload
        processed_query = self.nlp.process_query(query)

        context = {'request': request, 'history': history, 'frame': copy.deepcopy(frame)}
        context.update(processed_query.to_dict())
        context.pop('text')
        context.update(self.dialogue_manager.apply_handler(context))

        return context

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (function): The dialogue state handler function
            **kwargs (dict): A list of options which specify the dialogue rule
        """
        self.dialogue_manager.add_dialogue_rule(name, handler, **kwargs)
