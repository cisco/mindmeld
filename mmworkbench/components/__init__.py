# -*- coding: utf-8 -*-
"""This module contains the components of the workbench platform"""
from .dialogue import Conversation, DialogueManager, DialogueResponder
from .nlp import NaturalLanguageProcessor
from .question_answerer import QuestionAnswerer
from .entity_resolver import EntityResolver
from .preprocessor import Preprocessor


__all__ = ['Conversation', 'DialogueResponder', 'DialogueManager',
           'NaturalLanguageProcessor', 'QuestionAnswerer', 'EntityResolver', 'Preprocessor']
