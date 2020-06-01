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
This module contains the app component.
"""
import logging
import os
import sys

from .app_manager import ApplicationManager
from .cli import app_cli
from .components._config import get_custom_action_config
from .components.custom_action import (
    CustomAction,
    CustomActionException,
    CustomActionSequence,
)
from .components.dialogue import DialogueFlow, DialogueResponder, AutoEntityFilling
from .components.request import Request
from .server import MindMeldServer

logger = logging.getLogger(__name__)


class Application:
    """The conversational application.

        Attributes:
            import_name (str): The name of the application package.
            app_path (str): The application path.
            app_manager (ApplicationManager): The application manager.
            request_class (Request): Any class that inherits from \
                Request.
            responder_class (DialogueResponder): Any class that \
                inherits from the DialogueResponder.
            preprocessor (Preprocessor): The application preprocessor, if any.
            async_mode (bool): ``True`` if the application is async, ``False`` otherwise.
        """

    def __init__(
        self,
        import_name,
        request_class=None,
        responder_class=None,
        preprocessor=None,
        async_mode=False,
    ):
        self.import_name = import_name
        filename = getattr(sys.modules[import_name], "__file__", None)
        if filename is None:
            raise ValueError("Invalid import name")
        self.app_path = os.path.dirname(os.path.abspath(filename))

        self.app_manager = None
        self._server = None
        self._dialogue_rules = []
        self._middleware = []
        self.request_class = request_class or Request
        self.responder_class = responder_class or DialogueResponder
        self.preprocessor = preprocessor
        self.async_mode = async_mode
        self.custom_action_config = get_custom_action_config(self.app_path)

    @property
    def question_answerer(self):
        """
        The application's Question Answerer, which is initialized as part of the application \
            manager.
        """
        return None if self.app_manager is None else self.app_manager.question_answerer

    def lazy_init(self, nlp=None):
        """
        Initialize the application manager, spin up the server and compile the dialogue rules.
        """
        if self.app_manager:
            return
        self.app_manager = ApplicationManager(
            self.app_path,
            nlp,
            responder_class=self.responder_class,
            request_class=self.request_class,
            preprocessor=self.preprocessor,
            async_mode=self.async_mode,
        )
        self._server = MindMeldServer(self.app_manager)

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
        defaults = {"port": 7150, "host": "0.0.0.0", "threaded": True}
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        self.lazy_init()
        self.app_manager.load()
        self._server.run(**kwargs)

    def handle(self, **kwargs):
        """A decorator that is used to register dialogue state rules"""

        def _decorator(func):
            name = kwargs.pop("name", None)
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
            kwargs (dict): A list of options which specify the dialogue rule
        """
        if self.app_manager:
            self.app_manager.add_dialogue_rule(name, handler, **kwargs)
        else:
            self._dialogue_rules.append((name, handler, kwargs))

    def custom_action(self, **kwargs):
        """Adds a custom action, similar to `add_custom_action` but allows the ordering
        of nlp entities to be more flexible.

        Examples:
            app.custom_action(intent='greeting', action='say_greeting')
            app.custom_action(entity='person', action='greet_person')
        """
        action = kwargs.pop("action", None)
        actions = kwargs.pop("actions", None)

        if not (action or actions):
            raise CustomActionException(
                "`action` or `actions` must be present in arguments."
            )

        if action:
            self.add_custom_action(
                action,
                asynch=kwargs.pop("asynch", False),
                overwrite=kwargs.pop("overwrite", False),
                config=kwargs.pop("config", None),
                **kwargs
            )
        else:
            self.add_custom_action_sequence(
                actions,
                asynch=kwargs.pop("asynch", False),
                overwrite=kwargs.pop("overwrite", False),
                config=kwargs.pop("config", None),
                **kwargs
            )

    def add_custom_action(
        self, action, asynch=False, overwrite=False, config=None, **kwargs
    ):
        """Adds a custom action handler for the dialogue manager.

        Whenever the user hits this state, we invoke the custom action instead and returns
            the appropriate responder.

        Args:
            action (str): The name of the custom action
            asynch (bool): Whether we should invoke this custom action asynchronously
            overwrite (bool): Whether we should overwrite the Responder with fields from the
                response, otherwise we will extend the fields (frame, directives) accordingly.
            config (dict): The custom action config, if different from the application's.
        """
        if not action:
            raise CustomActionException("Argument `action` should not be empty.")

        config = config or self.custom_action_config
        if not config:
            raise CustomActionException("Argument `config` should not be empty.")

        custom_action = CustomAction(action, config, overwrite=overwrite)
        state_name = kwargs.pop("name", "custom_action_{}".format(action))
        if asynch:
            self.add_dialogue_rule(state_name, custom_action.invoke_async, **kwargs)
        else:
            self.add_dialogue_rule(state_name, custom_action.invoke, **kwargs)

    def add_custom_action_sequence(
        self, actions, asynch=False, overwrite=False, config=None, **kwargs
    ):
        """Adds a custom action sequence handler for the dialogue manager.

        Whenever the user hits this state, we invoke the sequence of custom actions and returns
            the appropriate responder.

        Args:
            actions (list): A list of custom actions
            asynch (bool): Whether we should invoke this custom action asynchronously
            overwrite (bool): Whether we should overwrite the Responder with fields from the
                response, otherwise we will extend the fields (frame, directives) accordingly.
            config (dict): The custom action config, if different from the application's.
        """
        if not actions:
            raise CustomActionException("Argument `actions` should not be empty.")

        config = config or self.custom_action_config
        if not config:
            raise CustomActionException("Argument `config` should not be empty.")

        action_seq = CustomActionSequence(actions, config, overwrite=overwrite)
        state_name = kwargs.pop("name", "custom_actions_{}".format(actions))
        if asynch:
            self.add_dialogue_rule(state_name, action_seq.invoke_async, **kwargs)
        else:
            self.add_dialogue_rule(state_name, action_seq.invoke, **kwargs)

    def dialogue_flow(self, **kwargs):
        """Creates a dialogue flow for the application"""

        def _decorator(func):
            name = kwargs.pop("name", func.__name__)
            flow = DialogueFlow(name, func, self, **kwargs)
            return flow

        return _decorator

    def auto_fill(self, name=None, *, form, **kwargs):
        """Creates a flow to fill missing entities"""

        def _decorator(func):
            func_name = name or func.__name__
            if not form or not isinstance(form, dict):
                raise TypeError("Form cannot be empty.")
            auto_fill = AutoEntityFilling(func, form, self)
            self.add_dialogue_rule(func_name, auto_fill, **kwargs)
            return func

        return _decorator

    def cli(self):
        """Initialize the application's command line interface."""
        # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        app_cli(obj={"app": self})
