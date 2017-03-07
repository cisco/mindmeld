# -*- coding: utf-8 -*-
"""
This module contains the app component.
"""

from __future__ import unicode_literals
from builtins import object

import logging
import os
import sys
from .app_manager import ApplicationManager
from .server import WorkbenchServer

logger = logging.getLogger(__name__)


class Application(object):

    def __init__(self, import_name):
        self.import_name = import_name
        filename = getattr(sys.modules[import_name], '__file__', None)
        if filename is None:
            raise ValueError('Invalid import name')
        self.app_path = os.path.dirname(os.path.abspath(filename))

        self.app_manager = None
        self._server = None
        self._dialogue_rules = []

    def lazy_init(self, nlp=None):
        if self.app_manager:
            return
        self.app_manager = ApplicationManager(self.app_path, nlp)
        self._server = WorkbenchServer(self.app_manager)

        # Add any pending dialogue rules
        for rule in self._dialogue_rules:
            name, handler, kwargs = rule
            self.add_dialogue_rule(name, handler, **kwargs)
        self._dialogue_rules = None

    def run(self, **kwargs):
        """Runs the application on a local development server."""
        defaults = {'port': 7150, 'host': '0.0.0.0', 'threaded': True}
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        self.lazy_init()
        self.app_manager.load()
        self._server.run(**kwargs)

    def handle(self, **kwargs):
        """A decorator that is used to register dialogue state rules"""

        def _decorator(func):
            name = kwargs.pop('name', None)
            self.add_dialogue_rule(name, func, **kwargs)
            return func
        return _decorator

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (function): The dialogue state handler function
            **kwargs (dict): A list of options which specify the dialogue rule
        """
        if self.app_manager:
            self.app_manager.add_dialogue_rule(name, handler, **kwargs)
        else:
            self._dialogue_rules.append((name, handler, kwargs))

    def cli(self):
        from .cli import cli
        # pylint:
        cli(obj={'app': self})
