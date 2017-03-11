# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from .app import Application
from .core import config_logs
from .dialogue import Conversation
from .processor import NaturalLanguageProcessor
from .question_answerer import QuestionAnswerer


__author__ = 'MindMeld, Inc.'
__email__ = 'contact@mindmeld.com'
__version__ = '3.0.0.dev'

__all__ = ['config_logs', 'Application', 'Conversation', 'NaturalLanguageProcessor',
           'QuestionAnswerer']
