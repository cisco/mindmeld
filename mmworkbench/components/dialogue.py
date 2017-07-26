# -*- coding: utf-8 -*-
"""This module contains the dialogue manager component of Workbench"""
from __future__ import absolute_import, unicode_literals
from builtins import object, str

from functools import cmp_to_key
import logging
import random

from .. import path

logger = logging.getLogger(__name__)

SHOW_REPLY = 'show-reply'
SHOW_PROMPT = 'show-prompt'
SHOW_SUGGESTIONS = 'show-suggestions'


class DialogueStateRule(object):
    """A rule that determines a dialogue state. Each rule represents a pattern that must match in
    order to invoke a particular dialogue state.

    Attributes:
        dialogue_state (str): The name of the dialogue state
        domain (str): The name of the domain to match against
        entity_types (set): The set of entity types to match against
        intent (str): The name of the intent to match against
    """
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

    def __init__(self):
        self.handler_map = {}
        self.rules = []

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

    def apply_handler(self, context):
        """Applies the dialogue state handler for the most complex matching rule

        Args:
            context (dict): Description

        Returns:
            dict: A dict containing the dialogue datae and client actions
        """
        dialogue_state = None
        for rule in self.rules:
            if rule.apply(context):
                dialogue_state = rule.dialogue_state
                break

        if dialogue_state is None:
            logger.info('Failed to find dialogue state', context)
            handler = self._default_handler
        else:
            handler = self.handler_map[dialogue_state]
        slots = {}  # TODO: where should slots come from??
        responder = DialogueResponder(slots)
        handler(context, slots, responder)
        return {'dialogue_state': dialogue_state, 'client_actions': responder.client_actions}

    @staticmethod
    def _default_handler(context, slots, responder):
        # TODO: implement default handler
        pass


class DialogueResponder(object):
    """The dialogue responder helps generate client actions and fill slots in the
    system-generated natural language responses.

    Attributes:
        client_actions (list): A list of client actions that the responder has added
        slots (dict): Values to populate the placeholder slots in the natural language
            response
    """
    def __init__(self, slots):
        """Initializes a dialogue responder

        Args:
            slots (dict): Values to populate the placeholder slots in the natural language
                response
        """
        self.slots = slots
        self.client_actions = []

    def reply(self, text):
        """Sends a 'show-reply' client action

        Args:
            text (str): The text of the reply
        """
        self._reply(text)

    def prompt(self, text):
        """Sends a 'show-prompt' client action

        Args:
            text (str): The text of the prompt
        """
        self._reply(text, action=SHOW_PROMPT)

    def _reply(self, text, action=SHOW_REPLY):
        """Convenience method as reply and prompt are basically the same."""
        text = self._choose(text)
        self.respond({
            'name': action,
            'message': {'text': text.format(**self.slots)}
        })

    def show(self, things):
        raise NotImplementedError

    def suggest(self, suggestions=None):
        suggestions = suggestions or []
        self.respond({
            'name': SHOW_SUGGESTIONS,
            'message': suggestions
        })

    def respond(self, action):
        """Sends an arbitrary client action.

        Args:
            action (dict): A client action

        """
        self.client_actions.append(action)

    @staticmethod
    def _choose(items):
        """Chooses a random item from items"""
        if isinstance(items, tuple) or isinstance(items, list):
            return random.choice(items)
        elif isinstance(items, set):
            items = random.choice(tuple(items))
        return items


def _get_app_module(app_path):
    module_path = path.get_app_module_path(app_path)

    import imp
    app_module = imp.load_source('app_module', module_path)
    app = app_module.app
    return app


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

    def say(self, text):
        """Send a message in the conversation. The message will be processed by the app based on
        the current state of the conversation.

        Args:
            text (str): The text of a message

        Returns:
            list of str: A text representation of the dialogue responses
        """
        response = self._app_manager.parse(text, session=self.session, frame=self.frame,
                                           history=self.history)
        response.pop('history')
        self.history.insert(0, response)
        self.frame = response['frame']

        # handle client actions
        response_texts = [self._handle_client_action(a) for a in response['client_actions']]
        return response_texts

    def _handle_client_action(self, action):
        try:
            if action['name'] in set((SHOW_REPLY, SHOW_PROMPT)):
                msg = action['message']['text']
            elif action['name'] == SHOW_SUGGESTIONS:
                suggestions = action['message']
                if not len(suggestions):
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
        except (KeyError, ValueError, AttributeError):
            msg = "Unsupported response: {!r}".format(action)

        return msg

    @staticmethod
    def _generate_suggestion_text(suggestion):
        pieces = []
        if 'text' in suggestion:
            pieces.append(suggestion['text'])
        if suggestion['type'] != 'text':
            pieces.append('({})'.format(suggestion['type']))

        return ' '.join(pieces)
