# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""

from __future__ import unicode_literals
from builtins import object


from .. import path, Query, ProcessedQuery

from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .nel import NamedEntityLinker
from .ner import NamedEntityRecognizer
from .parser import Parser
from .role_classifier import RoleClassifier


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
        self.domain_classifier = DomainClassifier(self._resource_loader)
        self.domains = {domain: DomainProcessor(app_path, domain, self._tokenizer,
                                                self._preprocessor, self._resource_loader)
                        for domain in path.get_domains(self.app_path)}

    def build(self):
        """Builds all models for the app."""
        # Is there a config that should be loaded here?
        self.domain_classifier.fit()
        self.domain_classifier.dump(path.get_domain_model_path(self.app_path))

        for domain_processor in self.domains.values():
            domain_processor.build()

    def process(self, query_text):
        """Summary

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = Query(query_text, self._tokenizer, self._preprocessor)
        return self.process_query(query).to_dict()

    def process_query(self, query, processed_query=None):
        processed_query = processed_query or ProcessedQuery(query)

        processed_query.domain = domain = self.domain_classifier.predict(query)

        return self.domains[domain].process_query(query, processed_query)

    def load(self):
        pass

    def dump(self):
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
        self.intent_classifier = IntentClassifier(domain, self._resource_loader)
        self.linker = NamedEntityLinker(domain, self._resource_loader)
        self.intents = {intent: IntentProcessor(app_path, domain, intent, self._tokenizer,
                                                self._preprocessor, self._resource_loader)
                        for intent in path.get_domain_intents(app_path, domain)}

    def build(self):
        # TODO: build gazetteers

        # train intent model
        self.intent_classifier.fit()

        # Something with linker?

        for intent_processor in self.intents.values():
            intent_processor.build()

    def process(self, query_text):
        """Summary

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        pass

    def process_query(self, query, processed_query=None):
        pass


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
        self.recognizer = NamedEntityRecognizer(self._resource_loader, domain, intent)
        # TODO: the parser after finishing Kwik-E-Mart demo
        self.parser = Parser(self._resource_loader, domain, intent)

        entity_types = []  # TODO: How do we get the list of entities for this intent
        self.entities = {entity_type: EntityProcessor(app_path, domain, intent, entity_type,
                                                      self._tokenizer, self._preprocessor,
                                                      self._resource_loader)
                         for entity_type in entity_types}

    def build(self):
        pass

    def process(self, query_text):
        """Summary

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        pass

    def process_query(self, query, processed_query=None):
        pass


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
        self.role_classifier = RoleClassifier(self._resource_loader, domain, intent, entity_type)


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


def create_parser(app_path):
    pass
