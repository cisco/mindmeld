# -*- coding: utf-8 -*-
"""
This module contains the language parser component of the Workbench natural language processor
"""

from __future__ import unicode_literals
from builtins import object


class Parser(object):
    """A language parser which is used to extract relations between entities in a given query and
    group related entities together."""

    def __init__(self, resource_loader, domain, intent):
        self._resource_loader = resource_loader
        self.domain = domain
        self.intent = intent
