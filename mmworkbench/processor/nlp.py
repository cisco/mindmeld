# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""

from __future__ import unicode_literals
from builtins import object


from .. import path
from ..core import ProcessedQuery
from ..query_factory import QueryFactory
from ..tokenizer import Tokenizer

from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .linker import EntityLinker
from .recognizer import EntityRecognizer
from .parser import Parser
from .resource_loader import ResourceLoader
from .role_classifier import RoleClassifier


class NaturalLanguageProcessor(object):
    """The natural language processor is the workbench component responsible for
    all nlp.

    Attributes:
        domain_classifier (DomainClassifier): A classifier for domains
        domains (dict): Processors for each domain
        preprocessor (Preprocessor): the object responsible for processing raw text
        tokenizer (Tokenizer): the object responsible for normalizing and tokenizing processed text
    """

    def __init__(self, app_path):
        """Initializes a natural language processor object

        Args:
            app_path (str): The path to the directory containing the app's data
        """
        self._app_path = app_path
        self._query_factory = create_query_factory(app_path)
        self._resource_loader = create_resource_loader(app_path, self._query_factory)
        self.domain_classifier = DomainClassifier(self._resource_loader)
        self.domains = {domain: DomainProcessor(app_path, domain, self._query_factory,
                                                self._resource_loader)
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
        query = self._query_factory.create_query(query_text)
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
        intent_classifier (IntentClassifier): Description
        intents (dict): Description
    """

    def __init__(self, app_path, domain, query_factory=None, resource_loader=None):
        self._app_path = app_path
        self.name = domain
        self._query_factory = query_factory or create_query_factory(app_path)
        if resource_loader:
            self._resource_loader = resource_loader
        else:
            self._resource_loader = create_resource_loader(app_path, self._query_factory)
        self.intent_classifier = IntentClassifier(self._resource_loader, domain)
        self.intents = {intent: IntentProcessor(app_path, domain, intent, self._query_factory,
                                                self._resource_loader)
                        for intent in path.get_intents(app_path, domain)}

    def build(self):
        """Builds all models for the domain."""

        # train intent model
        if len(self.intents) > 1:
            self.intent_classifier.fit()

        for intent_processor in self.intents.values():
            intent_processor.build()

    def dump(self):
        # dump gazetteers?
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.dump(model_path)

        for intent_processor in self.intents.values():
            intent_processor.dump()

    def load(self):
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.load(model_path)

        for intent_processor in self.intents.values():
            intent_processor.load()

    def process(self, query_text):
        """Processes the input text for this domain

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = self.query_factory.create_query(query_text)
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
        recognizer (EntityRecognizer): The
    """

    def __init__(self, app_path, domain, intent, query_factory=None, resource_loader=None):
        self._app_path = app_path
        self.domain = domain
        self.name = intent
        self._query_factory = query_factory or create_query_factory(app_path)
        if resource_loader:
            self._resource_loader = resource_loader
        else:
            self._resource_loader = create_resource_loader(app_path, self._query_factory)
        self.recognizer = EntityRecognizer(self._resource_loader, domain, intent)

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
            processor = EntityProcessor(self._app_path, self.domain, self.name, entity_type,
                                        self._query_factory, self._resource_loader)
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

        # TODO: something with parser?

        entity_types = self.recognizer.entity_types
        self.entities = {}
        for entity_type in entity_types:
            processor = EntityProcessor(self._app_path, self.domain, self.name, entity_type,
                                        self._query_factory, self._resource_loader)
            processor.load()
            self.entities[entity_type] = processor

    def process(self, query_text):
        """Processes the input text for this intent

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = self._query_factory.create_query(query_text)
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
            self.entities[entity.entity.type].process_entity(query, entities, entity)

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

    def __init__(self, app_path, domain, intent, entity_type, query_factory=None,
                 resource_loader=None):
        self._app_path = app_path
        self.domain = domain
        self.intent = intent
        self.type = entity_type
        self._query_factory = query_factory or create_query_factory(app_path)
        if resource_loader:
            self._resource_loader = resource_loader
        else:
            self._resource_loader = create_resource_loader(app_path, self._query_factory)

        self.role_classifier = RoleClassifier(self._resource_loader, domain, intent, entity_type)
        self.linker = EntityLinker(self._resource_loader, entity_type, query_factory.normalize)

    def build(self):
        """Builds the models for this entity type"""
        # TODO: something with linker ?
        self.role_classifier.fit()

    def dump(self):
        # TODO: something with linker ?
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.dump(model_path)

    def load(self):
        # TODO: something with linker ?
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.load(model_path)

    def process(self, text):
        raise NotImplementedError('EntityProcessor objects to not support `process()`. '
                                  'Try `process_entity()`')

    def process_entity(self, query, entities, entity):
        """Processes the given entity

        Args:
            query (Query): The query the entity originated from
            entities (list): All entities recognized in the query
            entity (Entity): The entity to process
        """
        # Classify role
        entity.entity.role = self.role_classifier.predict(query, entities, entity)

        # Link entity
        entity.entity.value = self.linker.predict(entity.entity)
        return entity

    def __repr__(self):
        return "<EntityProcessor {!r}>".format(self.type)


def create_tokenizer(app_path):
    """Creates the preprocessor for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data

    Returns:
        Tokenizer: a tokenizer
    """
    return Tokenizer()


def create_preprocessor(app_path):
    """Creates the preprocessor for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data

    Returns:
        Preprocessor: a preprocessor
    """
    pass


def create_query_factory(app_path, tokenizer=None, preprocessor=None):
    tokenizer = tokenizer or create_tokenizer(app_path)
    preprocessor = preprocessor or create_preprocessor(app_path)
    return QueryFactory(tokenizer, preprocessor)


def create_resource_loader(app_path, query_factory):
    """Creates the resource loader for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data
        query_factory (QueryFactory): The app's query factory

    Returns:
        ResourceLoader: a resource loader
    """
    return ResourceLoader(app_path, query_factory)


def create_parser(app_path):
    pass
