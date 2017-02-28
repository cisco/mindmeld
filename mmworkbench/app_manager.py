# -*- coding: utf-8 -*-
"""
This module contains the application manager
"""
from __future__ import unicode_literals
from builtins import object

from .dialogue import DialogueManager
from .processor.nlp import NaturalLanguageProcessor

class ApplicationManager(object):
    '''This class provides the functionality to manage a workbench application.

    language parser models. It is decoupled from any specific server
    implementation (underlying server framework, endpoints, request/response
    signatures).
    '''
    def __init__(self, app_path):
        self.nlp = NaturalLanguageProcessor(app_path)
        self.dialogue_manager = DialogueManager()

    def load(self):
        """Loads all resources required to run the application."""
        self.nlp.load()

    def parse(self, request):
        """
        Args:
            request (dict): A dictionary containing the body of the request
                received
        """

        #apply nlp here
        return self.dialogue_manager.apply_handler(request)

    def add_handler(self, handler, pattern=None, name=None, **kwargs):
        """Adds a dialogue handler to the

        Args:
            handler (function): Description
            pattern (TYPE): Description
            name (None, optional): Description
            **kwargs (TYPE): Description
        """
        name = name or handler.__name__
        self.dialogue_manager.add_handler(handler, pattern, name, **kwargs)
