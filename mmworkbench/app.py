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
        self._app_manager = ApplicationManager(self.app_path)
        self._server = WorkbenchServer(self._app_manager)

    def run(self, **kwargs):
        """Runs the application on a local development server."""
        defaults = {'port': 7150, 'host': '0.0.0.0', 'threaded': True}
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        self._app_manager.load()
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
        self._app_manager.add_dialogue_rule(name, handler, **kwargs)

    def cli(self):
        from .cli import cli
        cli(obj={'app': self})
