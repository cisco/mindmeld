# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""

from __future__ import unicode_literals
from builtins import object


from .. import path
from ..core import Query, ProcessedQuery
from ..tokenizer import Tokenizer

from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .nel import NamedEntityLinker
from .ner import NamedEntityRecognizer
from .parser import Parser
from .resource_loader import ResourceLoader
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
        self._resource_loader = create_resource_loader(app_path, self._tokenizer,
                                                       self._preprocessor)
        self.domain_classifier = DomainClassifier(self._resource_loader)
        self.domains = {domain: DomainProcessor(app_path, domain, self._tokenizer,
                                                self._preprocessor, self._resource_loader)
                        for domain in path.get_domains(self._app_path)}

    def build(self):
        """Builds all models for the app."""
        # TODO: Load configuration from some file

        if len(self.domains) > 1:
            self.domain_classifier.fit()

        for domain_processor in self.domains.values():
            domain_processor.build()

    def dump(self):
        model_path = path.get_domain_model_path(self._app_path)
        self.domain_classifier.dump(model_path)

        for domain_processor in self.domains.values():
            domain_processor.dump()

    def load(self):
        model_path = path.get_domain_model_path(self._app_path)
        self.domain_classifier.load(model_path)

        for domain_processor in self.domains.values():
            domain_processor.load()

    def process(self, query_text):
        """Processes the input text

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = Query(query_text, self._tokenizer, self._preprocessor)
        return self.process_query(query).to_dict()

    def process_query(self, query):
        """Processes the query object passed in

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: The resulting processed query
        """
        if len(self.domains) > 1:
            domain = self.domain_classifier.predict(query)
        else:
            domain = list(self.domains.keys())[0]

        processed_query = self.domains[domain].process_query(query)
        processed_query.domain = domain
        return processed_query

    def __repr__(self):
        return "<NaturalLanguageProcessor {!r}>".format(self._app_path)


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
        if resource_loader:
            self._resource_loader = resource_loader
        else:
            self._resource_loader = create_resource_loader(app_path, self._tokenizer,
                                                           self._preprocessor)
        self.intent_classifier = IntentClassifier(self._resource_loader, domain)
        self.linker = NamedEntityLinker(self._resource_loader, domain)
        self.intents = {intent: IntentProcessor(app_path, domain, intent, self.linker,
                                                self._tokenizer, self._preprocessor,
                                                self._resource_loader)
                        for intent in path.get_intents(app_path, domain)}

    def build(self):
        """Builds all models for the domain."""
        # TODO: build gazetteers

        # train intent model
        if len(self.intents) > 1:
            self.intent_classifier.fit()

        # Something with linker?

        for intent_processor in self.intents.values():
            intent_processor.build()

    def dump(self):
        # dump gazetteers?
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.dump(model_path)

        # something with linker?

        for intent_processor in self.intents.values():
            intent_processor.dump()

    def load(self):
        # load gazetteers?
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.load(model_path)

        # something with the linker?

        for intent_processor in self.intents.values():
            intent_processor.load()

    def process(self, query_text):
        """Processes the input text for this domain

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = Query(query_text, self._tokenizer, self._preprocessor)
        processed_query = self.process_query(query)
        processed_query.domain = self.name
        return processed_query.to_dict()

    def process_query(self, query):
        """Processes the query object passed in for this domain

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: The resulting processed query
        """
        if len(self.intents) > 1:
            intent = self.intent_classifier.predict(query)
        else:
            intent = list(self.intents.keys())[0]
        processed_query = self.intents[intent].process_query(query)
        processed_query.intent = intent
        return processed_query

    def __repr__(self):
        return "<DomainProcessor {!r}>".format(self.name)


class IntentProcessor(object):
    """Summary

    Attributes:
        domain (str): The domain this intent belongs to
        name (str): The name of this intent
        entities (dict): Description
        linker (NamedEntityLinker): Description
        recognizer (NamedEntityRecognizer): The
    """

    def __init__(self, app_path, domain, intent, linker, tokenizer=None, preprocessor=None,
                 resource_loader=None):
        self._app_path = app_path
        self.domain = domain
        self.name = intent
        self.linker = linker
        self._tokenizer = tokenizer or create_tokenizer(app_path)
        self._preprocessor = preprocessor or create_preprocessor(app_path)
        if resource_loader:
            self._resource_loader = resource_loader
        else:
            self._resource_loader = create_resource_loader(app_path, self._tokenizer,
                                                           self._preprocessor)
        self.recognizer = NamedEntityRecognizer(self._resource_loader, domain, intent)

        # TODO: revisit the parser after finishing Kwik-E-Mart demo
        self.parser = Parser(self._resource_loader, domain, intent)

        self.entities = {}

    def build(self):
        """Builds the models for this intent"""

        # train entity recognizer
        self.recognizer.fit()

        # TODO: something for the parser?

        entity_types = self.recognizer.entity_types
        self.entities = {}
        for entity_type in entity_types:
            processor = EntityProcessor(self._app_path, self.domain, self.intent, entity_type,
                                        self._tokenizer, self._preprocessor, self._resource_loader)
            processor.build()
            self.entities[entity_type] = processor

    def dump(self):
        model_path = path.get_entity_model_path(self._app_path, self.domain, self.name)
        self.recognizer.dump(model_path)

        # TODO: something with parser?

        for entity_processor in self.entities.values():
            entity_processor.dump()

    def load(self):
        model_path = path.get_entity_model_path(self._app_path, self.domain, self.name)
        self.recognizer.load(model_path)

        # TODO: something with the parser?

        for entity_processor in self.entities.values():
            entity_processor.load()

    def process(self, query_text):
        """Processes the input text for this intent

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = Query(query_text, self._tokenizer, self._preprocessor)
        processed_query = self.process_query(query)
        processed_query.domain = self.domain
        processed_query.intent = self.name
        return processed_query.to_dict()

    def process_query(self, query):
        """Processes the query object passed in for this intent

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: The resulting processed query
        """
        entities = self.recognizer.predict(query)

        for entity in entities:
            self.entities[entity.type].process_entity(query, entities, entity)

        # TODO: link entities
        # TODO: parse query

        return ProcessedQuery(query, entities=entities)

    def __repr__(self):
        return "<IntentProcessor {!r}>".format(self.name)


class EntityProcessor(object):
    """Summary

    Attributes:
        domain (str): The domain this entity belongs to
        intent (str): The intent this entity belongs to
        type (str): The type of this entity
        role_classifier (TYPE): Description
    """

    def __init__(self, app_path, domain, intent, entity_type, tokenizer=None, preprocessor=None,
                 resource_loader=None):
        self._app_path = app_path
        self.domain = domain
        self.intent = intent
        self.type = entity_type
        self._tokenizer = tokenizer or create_tokenizer(app_path)
        self._preprocessor = preprocessor or create_preprocessor(app_path)
        if resource_loader:
            self._resource_loader = resource_loader
        else:
            self._resource_loader = create_resource_loader(app_path, self._tokenizer,
                                                           self._preprocessor)
        self.role_classifier = RoleClassifier(self._resource_loader, domain, intent, entity_type)
        self.roles = set()

    def build(self):
        """Builds the models for this entity type"""
        self.role_classifier.fit()

    def dump(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.name)
        self.role_classifier.dump(model_path)

    def load(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.name)
        self.role_classifier.load(model_path)

    def process_entity(self, query, entities, entity):
        """Processes the given entity

        Args:
            query (Query): The query the entity originated from
            entities (list): All entities recognized in the query
            entity (Entity): The entity to process
        """
        entity.role = self.role_classifier.predict(query, entities, entity)
        return entity

    def __repr__(self):
        return "<EntityProcessor {!r}>".format(self.name)


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
    return Tokenizer()


def create_resource_loader(app_path, tokenizer, preprocessor):
    """Creates the resource loader for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data
        tokenizer (Tokenizer): The app's tokenizer
        preprocessor (Preprocessor): The app's preprocessor

    Returns:
        ResourceLoader: a resource loader
    """
    return ResourceLoader(app_path, tokenizer, preprocessor)


def create_parser(app_path):
    pass
