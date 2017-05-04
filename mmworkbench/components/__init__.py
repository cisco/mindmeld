# -*- coding: utf-8 -*-
"""This module contains the components of the workbench platform"""
from .dialogue import Conversation, DialogueManager
from .nlp import NaturalLanguageProcessor
from .question_answerer import QuestionAnswerer


__all__ = ['Conversation', 'DialogueManager', 'NaturalLanguageProcessor', 'QuestionAnswerer', 'EntityResolver']
