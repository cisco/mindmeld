# -*- coding: utf-8 -*-
"""
This module contains the app component.
"""
import logging
import os
import sys

from .app_manager import ApplicationManager
from .cli import app_cli
from .server import WorkbenchServer
from .components.dialogue import DialogueResponder, DialogueFlow
from .components.request import Request

logger = logging.getLogger(__name__)


class Application:

    def __init__(self, import_name, request_class=None, responder_class=None, preprocessor=None,
                 async_mode=False):
        self.import_name = import_name
        filename = getattr(sys.modules[import_name], '__file__', None)
        if filename is None:
            raise ValueError('Invalid import name')
        self.app_path = os.path.dirname(os.path.abspath(filename))

        self.app_manager = None
        self._server = None
        self._dialogue_rules = []
        self._middleware = []
        self.request_class = request_class or Request
        self.responder_class = responder_class or DialogueResponder
        self.preprocessor = preprocessor
        self.async_mode = async_mode

    @property
    def question_answerer(self):
        return None if self.app_manager is None else self.app_manager.question_answerer

    def lazy_init(self, nlp=None):
        if self.app_manager:
            return
        self.app_manager = ApplicationManager(
            self.app_path, nlp, responder_class=self.responder_class,
            request_class=self.request_class, preprocessor=self.preprocessor,
            async_mode=self.async_mode)
        self._server = WorkbenchServer(self.app_manager)

        # Add any pending dialogue rules
        for rule in self._dialogue_rules:
            name, handler, kwargs = rule
            self.add_dialogue_rule(name, handler, **kwargs)
        self._dialogue_rules = None
        for middleware in self._middleware:
            self.add_middleware(middleware)
        self._middleware = None

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

    def middleware(self, *args):
        """A decorator that is used to register dialogue handler middleware"""

        def _decorator(func):
            self.add_middleware(func)
            return func

        try:
            # Support syntax: @middleware
            func = args[0]
            if not callable(func):
                raise TypeError
            _decorator(func)
            return func
        except (IndexError, TypeError):
            # Support syntax: @middleware()
            return _decorator

    def add_middleware(self, middleware):
        """Adds middleware for the dialogue manager

        Args:
            middleware (callable): A dialogue manager middleware function
        """
        if self.app_manager:
            self.app_manager.add_middleware(middleware)
        else:
            self._middleware.append(middleware)

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (callable): The dialogue state handler function
            **kwargs (dict): A list of options which specify the dialogue rule
        """
        if self.app_manager:
            self.app_manager.add_dialogue_rule(name, handler, **kwargs)
        else:
            self._dialogue_rules.append((name, handler, kwargs))

    def dialogue_flow(self, **kwargs):
        """Creates a dialogue flow for the application"""

        def _decorator(func):
            name = kwargs.pop('name', func.__name__)
            flow = DialogueFlow(name, func, self, **kwargs)
            return flow

        return _decorator

    def cli(self):
        # pylint:
        app_cli(obj={'app': self})
