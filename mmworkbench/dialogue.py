# -*- coding: utf-8 -*-
"""This module contains the dialogue manager"""
from __future__ import unicode_literals
from builtins import object


class DialogueStateRule(object):

    def __init__(self, dialogue_state, pattern=None, domain=None, intent=None, **kwargs):
        self.dialogue_state = dialogue_state
        self._pattern = pattern
        self._domain = domain
        self._intent = intent
        # TODO: extend this for fancier rules

    def apply(self, context):
        """Applies the rule to the given context

        Args:
            context (TYPE): Description

        Returns:
            bool: whether or not the context matches
        """
        # TODO: implement!
        return False

    @property
    def complexity(self):
        """Returns the

        Returns:
            int:
        """
        complexity = 0
        if self._domain:
            complexity += 1
        if self._intent:
            complexity += 1 << 1
        if self._pattern:
            complexity += 1 << 2

        # TODO: extend this for fancier rules
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

    def add_handler(self, handler, pattern, name, **kwargs):
        """Connects a URL rule.  Works exactly like the :meth:`route`
        decorator.  If a view_func is provided it will be registered with the
        endpoint.

        Args:
            handler (function): Description
            pattern (TYPE): Description
            name (None, optional): Description
            **kwargs (TYPE): Description
        """
        self.handler_map[name] = handler
        self.rules.append(DialogueStateRule(name, pattern, **kwargs))
        self.rules.sort(key=lambda x: x.complexity)

    def apply_handler(self, context):
        """Finds the

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
