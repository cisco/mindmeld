# -*- coding: utf-8 -*-
"""
This module contains the application manager
"""
from __future__ import unicode_literals
from builtins import object

import copy

from .dialogue import DialogueManager
from .processor import NaturalLanguageProcessor
from .question_answerer import QuestionAnswerer


class ApplicationManager(object):
    """This class provides the functionality to manage a workbench application.

    The application manager is responsible for communicating between handling
    conversation requests components, handling req
    """
    def __init__(self, app_path, nlp=None, es_host=None):
        self._app_path = app_path
        self.nlp = nlp or NaturalLanguageProcessor(app_path)
        self._query_factory = self.nlp.resource_loader.query_factory
        self.dialogue_manager = DialogueManager()
        self.question_answerer = QuestionAnswerer(self.nlp.resource_loader, es_host)

    @property
    def ready(self):
        return self.nlp.ready

    def load(self):
        """Loads all resources required to run the application."""
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
