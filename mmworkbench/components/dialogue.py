# -*- coding: utf-8 -*-
"""This module contains the dialogue manager component of Workbench"""
from __future__ import absolute_import, unicode_literals
from builtins import object, str

import copy
from functools import cmp_to_key
import logging
import random
import json
import os

from .. import path
from ..exceptions import WorkbenchImportError

mod_logger = logging.getLogger(__name__)


class DirectiveNames(object):
    """A constants object for directive names.

    Attributes:
        LIST (str): A directive to display a list.
        LISTEN (str): A directive to listen (start speech recognition).
        REPLY (str): A directive to display a text view.
        RESET (str): Description
        SPEAK (str): A directive to speak text out loud.
        SUGGESTIONS (str): A view for a list of suggestions.
    """

    LIST = 'list'
    LISTEN = 'listen'
    REPLY = 'reply'
    RESET = 'reset'
    SPEAK = 'speak'
    SUGGESTIONS = 'suggestions'


class DirectiveTypes(object):
    """A constants object for directive types.

    Attributes:
        ACTION (str): An action directive
        VIEW (str): A view directive.
    """

    VIEW = 'view'
    ACTION = 'action'


class DialogueStateRule(object):
    """A rule that determines a dialogue state. Each rule represents a pattern that must match in
    order to invoke a particular dialogue state.

    Attributes:
        dialogue_state (str): The name of the dialogue state
        domain (str): The name of the domain to match against
        entity_types (set): The set of entity types to match against
        intent (str): The name of the intent to match against
    """

    logger = mod_logger.getChild('DialogueStateRule')

    def __init__(self, dialogue_state, **kwargs):
        """Initializes a dialogue state rule.

        Args:
            dialogue_state (str): The name of the dialogue state
            domain (str): The name of the domain to match against
            has_entity (str|list|set): A synonym for the ``has_entities`` param
            has_entities (str|list|set): A single entity type or a list of entity types to match
                against.
            intent (str): The name of the intent to match against
        """

        self.dialogue_state = dialogue_state

        key_kwargs = (('domain',), ('intent',), ('has_entity', 'has_entities'))
        valid_kwargs = set()
        for keys in key_kwargs:
            valid_kwargs.update(keys)
        for kwarg in kwargs:
            if kwarg not in valid_kwargs:
                raise TypeError(('DialogueStateRule() got an unexpected keyword argument'
                                 ' \'{!s}\'').format(kwarg))

        resolved = {}
        for keys in key_kwargs:
            if len(keys) == 2:
                single, plural = keys
                if single in kwargs and plural in kwargs:
                    msg = 'Only one of {!r} and {!r} can be specified for a dialogue state rule'
                    raise ValueError(msg.format(single, plural, self.__class__.__name__))
                if single in kwargs:
                    resolved[plural] = {kwargs[single]}
                if plural in kwargs:
                    resolved[plural] = set(kwargs[plural])
            elif keys[0] in kwargs:
                resolved[keys[0]] = kwargs[keys[0]]

        self.domain = resolved.get('domain', None)
        self.intent = resolved.get('intent', None)
        entities = resolved.get('has_entities', None)
        self.entity_types = None
        if entities is not None:
            if isinstance(entities, str):
                # Single entity type passed in
                self.entity_types = frozenset((entities,))
            elif isinstance(entities, (list, set)):
                # List of entity types passed in
                self.entity_types = frozenset(entities)
            else:
                msg = 'Invalid entity specification for dialogue state rule: {!r}'
                raise ValueError(msg.format(entities))

    def apply(self, context):
        """Applies the dialogue state rule to the given context.

        Args:
            context (dict): A request context

        Returns:
            bool: whether or not the context matches
        """
        # Note: this will probably change as the details of "context" are worked out

        # check domain is correct
        if self.domain is not None and self.domain != context['domain']:
            return False

        # check intent is correct
        if self.intent is not None and self.intent != context['intent']:
            return False

        # check expected entity types are present
        if self.entity_types is not None:
            # TODO cache entity types
            entity_types = set()
            for entity in context['entities']:
                entity_types.add(entity['type'])

            if len(self.entity_types & entity_types) < len(self.entity_types):
                return False

        return True

    @property
    def complexity(self):
        """Returns an integer representing the complexity of this dialogue state rule.

        Components of a rule in order of increasing complexity are as follows:
            domains, intents, entity types, entity mappings

        Returns:
            int: A number representing the rule complexity
        """
        complexity = [0] * 3
        if self.domain:
            complexity[0] = 1

        if self.intent:
            complexity[1] = 1

        if self.entity_types:
            complexity[2] = len(self.entity_types)

        return tuple(complexity)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.dialogue_state)

    @staticmethod
    def compare(this, that):
        """Compares the complexity of two dialogue state rules

        Args:
            this (DialogueStateRule): a dialogue state rule
            that (DialogueStateRule): a dialogue state rule

        Returns:
            int: the comparison result
        """
        if not (isinstance(this, DialogueStateRule) and isinstance(that, DialogueStateRule)):
            return NotImplemented
        this_comp = this.complexity
        that_comp = that.complexity

        for idx in range(len(this_comp)-1, -1, -1):
            this_val = this_comp[idx]
            that_val = that_comp[idx]
            if this_val == that_val:
                continue
            return this_val - that_val
        return 0


class DialogueManager(object):

    logger = mod_logger.getChild('DialogueManager')

    def __init__(self, responder_class=None):
        self.handler_map = {}
        self.rules = []
        self.responder_class = responder_class or DialogueResponder

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue state rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (function): The dialogue state handler function
            **kwargs (dict): A list of options to be passed to the DialogueStateRule initializer
        """
        if name is None:
            name = handler.__name__

        rule = DialogueStateRule(name, **kwargs)

        self.rules.append(rule)
        self.rules.sort(key=cmp_to_key(DialogueStateRule.compare), reverse=True)
        if handler is not None:
            old_handler = self.handler_map.get(name)
            if old_handler is not None and old_handler != handler:
                msg = 'Handler mapping is overwriting an existing dialogue state: %s' % name
                raise AssertionError(msg)
            self.handler_map[name] = handler

    def apply_handler(self, context, target_dialogue_state=None):
        """Applies the dialogue state handler for the most complex matching rule

        Args:
            context (dict): Description
            target_dialogue_state (str, optional): The target dialogue state

        Returns:
            dict: A dict containing the dialogue state and directives
        """
        dialogue_state = None

        for rule in self.rules:
            if target_dialogue_state:
                if target_dialogue_state == rule.dialogue_state:
                    dialogue_state = rule.dialogue_state
                    break
            else:
                if rule.apply(context):
                    dialogue_state = rule.dialogue_state
                    break

        if dialogue_state is None:
            msg = 'Failed to find dialogue state for {domain}.{intent}'.format(
                domain=context.get('domain'), intent=context.get('intent'))
            self.logger.info(msg, context)
            handler = self._default_handler
        else:
            handler = self.handler_map[dialogue_state]
        # TODO: prepopulate slots
        slots = {}
        responder = self.responder_class(slots)
        handler(context, responder)

        return {'dialogue_state': dialogue_state, 'directives': responder.directives}

    @staticmethod
    def _default_handler(context, responder):
        # TODO: implement default handler
        pass


class DialogueResponder(object):
    """The dialogue responder helps generate directives and fill slots in the
    system-generated natural language responses.

    Attributes:
        directives (list): A list of directives that the responder has added
        slots (dict): Values to populate the placeholder slots in the natural language
            response
    """
    logger = mod_logger.getChild('DialogueResponder')
    DirectiveNames = DirectiveNames
    DirectiveTypes = DirectiveTypes

    def __init__(self, slots):
        """Initializes a dialogue responder

        Args:
            slots (dict): Values to populate the placeholder slots in the natural language
                response
        """
        self.slots = slots
        self.directives = []

    def reply(self, text):
        """Adds a 'reply' directive

        Args:
            text (str): The text of the reply
        """
        text = self._process_template(text)
        self.display(DirectiveNames.REPLY, payload={'text': text})

    def speak(self, text):
        """Adds a 'speak' directive

        Args:
            text (str): The text to speak aloud
        """
        text = self._process_template(text)
        self.act(DirectiveNames.SPEAK, payload={'text': text})

    def list(self, items):
        """Adds a 'list' view directive

        Args:
            items (list): The list of dictionary objects
        """
        items = items or []
        self.display(DirectiveNames.LIST, payload=items)

    def suggest(self, suggestions):
        """Adds a 'suggestions' directive

        Args:
            suggestions (list): A list of suggestions
        """
        suggestions = suggestions or []
        self.display(DirectiveNames.SUGGESTIONS, payload=suggestions)

    def listen(self):
        """Adds a 'listen' directive."""
        self.act(DirectiveNames.LISTEN)

    def reset(self):
        """Adds a 'reset' directive."""
        self.act(DirectiveNames.RESET)

    def display(self, name, payload=None):
        """Adds an arbitrary directive of type 'view'.

        Args:
            name (str): The name of the directive
            payload (dict, optional): The payload for the view
        """
        self.direct(name, DirectiveTypes.VIEW, payload=payload)

    def act(self, name, payload=None):
        """Adds an arbitrary directive of type 'action'.

        Args:
            name (str): The name of the directive
            payload (dict, optional): The payload for the action
        """
        self.direct(name, DirectiveTypes.ACTION, payload=payload)

    def direct(self, name, dtype, payload=None):
        """Adds an arbitrary directive

        Args:
            name (str): The name of the directive
            dtype (str): The type of the directive
            payload (dict, optional): The payload for the view
        """

        directive = {'name': name, 'type': dtype}
        if payload:
            directive['payload'] = payload

        self.directives.append(directive)

    def respond(self, directive):
        """Adds an arbitrary directive.

        Args:
            directive (dict): A directive.
        """
        self.logger.warning('respond() is deprecated. Instead use direct().')
        self.directives.append(directive)

    def prompt(self, text):
        """Alias for `reply()`. Deprecated.

        Args:
            text (str): The text of the reply
        """
        self.logger.warning('prompt() is deprecated. '
                            'Please use reply() and listen() instead')
        self.reply(text)

    @staticmethod
    def _choose(items):
        """Chooses a random item from items"""
        if isinstance(items, (tuple, list)):
            return random.choice(items)
        elif isinstance(items, set):
            return random.choice(tuple(items))
        return items

    def _process_template(self, text):
        return self._choose(text).format(**self.slots)


def _get_app_module(app_path):
    # Get the absolute path from the relative path (such as home_assistant/app.py)
    app_path = os.path.abspath(app_path)
    package_name = os.path.basename(app_path)
    module_path = path.get_app_module_path(app_path)

    if not os.path.isfile(module_path):
        raise WorkbenchImportError('Cannot import the app at {path}.'.format(app=module_path))

    try:
        path.load_app_package(app_path)

        import imp
        app_module = imp.load_source(
            '{package_name}.app'.format(package_name=package_name), module_path)
        app = app_module.app
        return app
    except ImportError as ex:
        raise WorkbenchImportError(ex.msg)


class Conversation(object):
    """The conversation object is a very basic workbench client.

    It can be useful for testing out dialogue flows in python.

    Example:
        >>> convo = Conversation(app_path='path/to/my/app')
        >>> convo.say('Hello')
        ['Hello. I can help you find store hours. How can I help?']
        >>> convo.say('Is the store on elm open?')
        ['The 23 Elm Street Kwik-E-Mart is open from 7:00 to 19:00.']

    Attributes:
        history (list): The history of the conversation. Most recent messages
        session (dict): Description
    """

    logger = mod_logger.getChild('Conversation')

    def __init__(self, app=None, app_path=None, nlp=None, session=None):
        """
        Args:
            app (Application, optional): An initialized app object. Either app or app_path must
                be given.
            app_path (None, optional): The path to the app data. Used to create an app object.
                Either app or app_path must be given.
            nlp (NaturalLanguageProcessor, optional): A natural language processor for the app.
                If passed, changes to this processor will affect the response from `say()`
            session (dict, optional): The session to be used in the conversation
        """
        app = app or _get_app_module(app_path)
        app.lazy_init(nlp)
        self._app_manager = app.app_manager
        if not self._app_manager.ready:
            self._app_manager.load()
        self.session = session or {}
        self.history = []
        self.frame = {}
        self.default_params = {}
        self.params = {}

    def say(self, text, params=None):
        """Send a message in the conversation. The message will be
        processed by the app based on the current state of the conversation and
        returns the extracted messages from the directives.

        Args:
            text (str): The text of a message
            params (dict): The params to use with this message

        Returns:
            list of str: A text representation of the dialogue responses
        """
        response = self.process(text, params=params)

        # handle directives
        response_texts = [self._follow_directive(a) for a in response['directives']]
        return response_texts

    def process(self, text, params=None):
        """Send a message in the conversation. The message will be processed by
        the app based on the current state of the conversation and returns
        the response.

        Args:
            text (str): The text of a message
            params (dict): The params to use with this message

        Returns:
            (dictionary): The dictionary Response
        """
        external_params = params or copy.deepcopy(self.default_params)
        params = copy.deepcopy(self.params)
        params.update(external_params)

        response = self._app_manager.parse(text, params=params, session=self.session,
                                           frame=self.frame, history=self.history)

        self.history = response['history']
        self.frame = response['frame']
        self.params = response['params']

        return response

    def _follow_directive(self, directive):
        msg = ''
        try:
            if directive['name'] == DirectiveNames.REPLY:
                msg = directive['payload']['text']
            elif directive['name'] == DirectiveNames.SUGGESTIONS:
                suggestions = directive['payload']
                if not suggestions:
                    raise ValueError
                msg = 'Suggestion{}:'.format('' if len(suggestions) == 1 else 's')
                texts = []
                for idx, suggestion in enumerate(suggestions):
                    if idx > 0:
                        msg += ', {!r}'
                    else:
                        msg += ' {!r}'

                    texts.append(self._generate_suggestion_text(suggestion))
                msg = msg.format(*texts)
            elif directive['name'] == DirectiveNames.COLLECTION:
                msg = '\n'.join(
                    [json.dumps(item, indent=4, sort_keys=True) for item in directive['payload']])
        except (KeyError, ValueError, AttributeError):
            msg = "Unsupported response: {!r}".format(directive)

        return msg

    @staticmethod
    def _generate_suggestion_text(suggestion):
        pieces = []
        if 'text' in suggestion:
            pieces.append(suggestion['text'])
        if suggestion['type'] != 'text':
            pieces.append('({})'.format(suggestion['type']))

        return ' '.join(pieces)

    def reset(self):
        self.history = []
        self.frame = {}
        self.params = {}
