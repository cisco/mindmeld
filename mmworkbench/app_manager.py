# -*- coding: utf-8 -*-
"""
This module contains the application manager
"""
from __future__ import unicode_literals
from builtins import object

from .dialogue import DialogueManager
from .processor.nlp import NaturalLanguageProcessor, create_query_factory, create_resource_loader


class ApplicationManager(object):
    '''This class provides the functionality to manage a workbench application.

    The application manager is responsible for communicating between handling
    conversation requests components, handling req
    '''
    def __init__(self, app_path):
        self._query_factory = create_query_factory(app_path)
        self._resource_loader = create_resource_loader(app_path, self._query_factory)
        self.nlp = NaturalLanguageProcessor(app_path, self._query_factory, self._resource_loader)
        self.dialogue_manager = DialogueManager()

    def load(self):
        """Loads all resources required to run the application."""
        self.nlp.load()

    def parse(self, text, payload=None, session={}, history=None, verbose=False):
        """
        Args:
            text (str): The text of the message sent by the user
            payload (dict, optional): Description
            session (dict, optional): Description
            history (list, optional): Description
            verbose (bool, optional): Description

        """
        # TODO: what do we do with verbose???
        # TODO: where are the slots stored
        # TODO: where is the frame stored? (Is this the same as slots)

        history = history or []

        request = {'text': text, 'session': session}
        if payload:
            request['payload'] = payload

        # TODO: support passing in reference time from session
        query = self._query_factory.create_query(text)

        # TODO: support specifying target domain, etc in payload
        processed_query = self.nlp.process_query(query)

        context = {'request': request, 'history': history}
        context.update(processed_query.to_dict())
        context.pop('text')
        return context

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (function): The dialogue state handler function
            **kwargs (dict): A list of options which specify the dialogue rule
        """
        self.dialogue_manager.add_dialogue_rule(name, handler, **kwargs)
