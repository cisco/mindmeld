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

    def handle(self, pattern=None, **kwargs):
        """A decorator that is used to register dialogue state handlers"""
        def _decorator(func):
            self.add_handler(func, pattern, **kwargs)
            return func
        return _decorator

    def add_handler(self, handler, pattern, **kwargs):
        """Adds a handler for a dialogue state
        Args:
            handler (TYPE): Description
            pattern (TYPE): Description
            **kwargs (TYPE): Description
        """
        self._app_manager.add_handler(handler, pattern, **kwargs)

    def cli(self):
        from .cli import cli
        cli(obj={'app': self})
