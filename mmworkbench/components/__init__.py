# -*- coding: utf-8 -*-
"""This module contains the components of the workbench platform"""
from .dialogue import Conversation, DialogueManager
from .nlp import NaturalLanguageProcessor
from .question_answerer import QuestionAnswerer
from .entity_resolver import EntityResolver


__all__ = ['Conversation', 'DialogueManager', 'NaturalLanguageProcessor', 'QuestionAnswerer',
           'EntityResolver']
