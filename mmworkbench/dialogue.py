# -*- coding: utf-8 -*-
"""This module contains the dialogue manager"""
from __future__ import unicode_literals
from builtins import object, str


class DialogueStateRule(object):
    """A rule for resolving dialogue states

    Attributes:
        dialogue_state (str): The name of the dialogue state
        domain (str): Description
        entity_mappings (dict): Description
        entity_types (set): Description
        intent (str): Description
    """
    def __init__(self, dialogue_state, **kwargs):

        self.dialogue_state = dialogue_state

        resolved = {}
        key_kwargs = [('domain',), ('intent',), ('entity', 'entities')]
        for keys in key_kwargs:
            if len(keys) == 2:
                single, plural = keys
                if single in kwargs and plural in kwargs:
                    msg = 'Only one of {!r} and {!r} can be sepcified for a dialogue state rule'
                    raise ValueError(msg.format(single, plural, self.__class__.__name__))
                if single in kwargs:
                    resolved[plural] = frozenset((kwargs[single],))
                if plural in kwargs:
                    resolved[plural] = frozenset(kwargs[plural])
            elif keys[0] in kwargs:
                resolved[keys[0]] = kwargs[keys[0]]


        self.domain = resolved.get('domain', None)
        self.intent = resolved.get('intent', None)
        entities = resolved.get('entities', None)
        self.entity_types = None
        self.entity_mappings = None
        if entities is not None:
            if isinstance(entities, str):
                # Single entity type passed in
                self.entity_types = frozenset((entities,))
            elif isinstance(entities, list) or isinstance(entities, set):
                # List of entity types passed in
                self.entity_types = frozenset(entities)
            elif isinstance(entities, dicts):
                self.entity_mappings = entities
            else:
                msg = 'Invalid entity specification for dialogue state rule: {!r}'
                raise ValueError(msg.format(entities))

    def apply(self, context):
        """Applies the rule to the given context.

        Args:
            context (dict): A request context

        Returns:
            bool: whether or not the context matches
        """
        # Note: this will probably change as the details of "context" are worked out


        # check domain is correct
        if self.domain is not None and self.domain != context.domain:
            return False

        # check intent is correct
        if self.intent is not None and self.intent != context.intent:
            return False

        # check expected entity types are present
        if (self.entity_types is not None and
                len(self.entity_types & context.entity_types) < len(self.entity_types)):
            return False

        # check entity mapping
        if self.entity_mappings is not None:
            matched_entities = set()
            for entity in context.entities:
                if (entity.type in self.entity_mappings and
                        entity.value == self.entity_mappings[entity.type]):
                    matched_entities.add(entity.type)

            # if there is not a matched entity for each mapping, fail
            if len(matched_entities) < len(self.entity_mappings):
                return False

        return True

    @property
    def complexity(self):
        """Returns an integer representing the complexity of this rule.

        Components of a rule in order of increasing complexity are as follows:
            domains, intents, entity types, entity mappings

        Returns:
            int:
        """
        complexity = 0
        if self.domain:
            complexity += 1
        if self.intent:
            complexity += 1 << 1
        # TODO: handle specification of multiple entity types or entity mappings
        if self.entity_types:
            complexity += 1 << 2
        if self.entity_mappings:
            complexity += 1 << 3

        return complexity

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)


class DialogueManager(object):

    def __init__(self):
        self.handler_map = {}
        self.rules = []

    def add_dialogue_rule(self, name, handler, **kwargs):
        """Adds a dialogue rule for the dialogue manager.

        Args:
            name (str): The name of the dialogue state
            handler (function): The dialogue state handler function
            **kwargs (dict): A list of options to be passed to the
                DialogueStateRule initializer
        """
        if name is None:
            name = handler.__name__

        rule = DialogueStateRule(name, **kwargs)

        self.rules.append(rule)
        self.rules.sort(key=lambda x: x.complexity)
        if handler is not None:
            old_handler = self.handler_map.get(name)
            if old_handler is not None and old_handler != handler:
                msg = 'Handler mapping is overwriting an existing dialogue state: %s' % name
                raise AssertionError(msg)
            self.handler_map[name] = handler

    def apply_handler(self, context):
        """Applies the handler for the most complex matching rule

        Args:
            context (TYPE): Description

        Returns:
            TYPE: Description
        """
        name = None
        for rule in self.rules:
            if rule.apple(context):
                name = rule.name
                break

        if name is None:
            # TODO: implement default handler
            handler = lambda x: x
        else:
            handler = self.handler_map[name]
        response = handler(context)
        return response
