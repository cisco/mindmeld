# -*- coding: utf-8 -*-
"""
This module contains the language parser component.
"""

from __future__ import unicode_literals
from builtins import object


class Parser(object):
    """A language parser which is used to group entities in a given query."""

    def __init__(self, resource_loader, domain, intent):
        self._resource_loader = resource_loader
        self.domain = domain
        self.intent = intent
