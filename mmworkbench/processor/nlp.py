# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""

from __future__ import unicode_literals
from builtins import object


'''
from mmworkbench import path, Query, ProcessedQuery

from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .ner import NamedEntityRecognizer
from .nel import NamedEntityLinker
from .parser import Parser
'''


class NaturalLanguageProcessor(object):
    """Summary

    Attributes:
        domain_classifier (DomainClassifier): Description
        domains (dict): Description
        gazetteer_factory (GazetteerFactory): Description
        preprocessor (TYPE): Description
        tokenizer (TYPE): Description
    """

    def __init__(self, app_path):
        """Initializes a natural language processor object

        Args:
            app_path (str): The path to the directory containing the app's data
        """
        self._app_path = app_path
        self.tokenizer = None
        self.preprocessor = None
        self.gazetteer_factory = None
        self.domain_classifier = None
        self.domains = {}

    def build(self):
        pass

    def load(self):
        pass

    def dump(self):
        pass

    def process(self, query):
        pass


class DomainProcessor(object):
    """Summary

    Attributes:
        gazetteer_factory (TYPE): Description
        intent_classifier (TYPE): Description
        intents (dict): Description
        preprocessor (TYPE): Description
        tokenizer (TYPE): Description
    """

    def __init__(self, app_path, tokenizer, gazetteer_factory):
        self._app_path = app_path
        self.tokenizer = None
        self.preprocessor = None
        self.gazetteer_factory = None
        self.intents = {}
        self.intent_classifier = None


class IntentProcessor(object):
    """Summary

    Attributes:
        entities (dict): Description
        linker (TYPE): Description
        preprocessor (TYPE): Description
        recognizer (TYPE): Description
        tokenizer (TYPE): Description
    """

    def __init__(self, app_path, tokenizer, gazetteer_factory):
        self._app_path = app_path
        self.tokenizer = None
        self.preprocessor = None
        self.entities = {}
        self.recognizer = None
        self.linker = None


class EntityProcessor(object):
    """Summary

    Attributes:
        preprocessor (TYPE): Description
        role_classifier (TYPE): Description
        roles (dict): Description
        tokenizer (TYPE): Description
    """

    def __init__(self, app_path, tokenizer, gazetteer_factory):
        self._app_path = app_path
        self.tokenizer = None
        self.preprocessor = None
        self.roles = {}
        self.role_classifier = None
