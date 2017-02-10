# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""

from __future__ import unicode_literals
from builtins import object


'''
from .. import path, Query, ProcessedQuery

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
        preprocessor (Preprocessor): the object responsible for processing raw text
        tokenizer (Tokenizer): the object responsible for normalizing and tokenizing processed text
    """

    def __init__(self, app_path):
        """Initializes a natural language processor object

        Args:
            app_path (str): The path to the directory containing the app's data
        """
        self._app_path = app_path
        self._tokenizer = create_tokenizer(app_path)
        self._preprocessor = create_preprocessor(app_path)
        self._resource_loader = create_resource_loader(app_path)
        self.domain_classifier = None
        self.domains = {}

    def build(self):
        pass

    def load(self):
        pass

    def dump(self):
        pass

    def process(self, query_text):
        """Summary

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        pass


class DomainProcessor(object):
    """Summary

    Attributes:
        name (str): The name of the domain
        intent_classifier (TYPE): Description
        intents (dict): Description
    """

    def __init__(self, app_path, domain, tokenizer=None, preprocessor=None, resource_loader=None):
        self._app_path = app_path
        self.name = domain
        self._tokenizer = tokenizer or create_tokenizer(app_path)
        self._preprocessor = preprocessor or create_preprocessor(app_path)
        self._resource_loader = resource_loader or create_resource_loader(app_path)
        self.intents = {}
        self.intent_classifier = None


class IntentProcessor(object):
    """Summary

    Attributes:
        domain (str): The domain this intent belongs to
        name (str): The name of this intent
        entities (dict): Description
        linker (NamedEntityLinker): Description
        recognizer (NamedEntityRecognizer): The
    """

    def __init__(self, app_path, domain, intent, tokenizer=None, preprocessor=None,
                 resource_loader=None):
        self._app_path = app_path
        self.domain = domain
        self.name = intent
        self._tokenizer = tokenizer or create_tokenizer(app_path)
        self._preprocessor = preprocessor or create_preprocessor(app_path)
        self._resource_loader = resource_loader or create_resource_loader(app_path)
        self.entities = {}
        self.recognizer = None
        self.linker = None


class EntityProcessor(object):
    """Summary

    Attributes:
        domain (str): The domain this entity belongs to
        intent (str): The intent this entity belongs to
        type (str): The type of this entity
        role_classifier (TYPE): Description
        roles (dict): Description
    """

    def __init__(self, app_path, domain, intent, entity_type, tokenizer=None, preprocessor=None,
                 resource_loader=None):
        self._app_path = app_path
        self.domain = domain
        self.intent = intent
        self.type = entity_type
        self._tokenizer = tokenizer or create_tokenizer(app_path)
        self._preprocessor = preprocessor or create_preprocessor(app_path)
        self._resource_loader = resource_loader or create_resource_loader(app_path)
        self.roles = {}
        self.role_classifier = None


def create_preprocessor(app_path):
    """Creates the preprocessor for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data

    Returns:
        Preprocessor: a preprocessor
    """
    pass


def create_tokenizer(app_path):
    """Creates the preprocessor for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data

    Returns:
        Tokenizer: a tokenizer
    """
    pass


def create_resource_loader(app_path):
    """Creates the resource loader for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data

    Returns:
        ResourceLoader: a resource loader
    """
    pass
