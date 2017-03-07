# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from .app import Application
from .dialogue import Conversation
from .processor.nlp import NaturalLanguageProcessor

__author__ = 'MindMeld, Inc.'
__email__ = 'contact@mindmeld.com'
__version__ = '3.0.0.dev'

__all__ = ['Application', 'Conversation', 'NaturalLanguageProcessor']
