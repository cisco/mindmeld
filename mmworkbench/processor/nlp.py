# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""

from __future__ import unicode_literals
from builtins import object, super


from .. import path
from ..core import ProcessedQuery
from ..exceptions import ProcessorError
from ..query_factory import QueryFactory
from ..tokenizer import Tokenizer

from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .entity_resolver import EntityResolver
from .entity_recognizer import EntityRecognizer
from .parser import Parser
from .resource_loader import ResourceLoader
from .role_classifier import RoleClassifier


class Processor(object):
    """A generic base class for processing queries through the workbench NLP
    components

    Attributes:
        dirty (bool): Indicates whether the processor has unsaved changes to
            its models
        ready (bool): Indicates whether the processor is ready to process
            messages
    """
    def __init__(self, app_path, resource_loader=None):
        self._app_path = app_path
        self.resource_loader = resource_loader or create_resource_loader(app_path)

        self._children = {}
        self.ready = False
        self.dirty = False

    def build(self):
        """Builds all models for this processor and its children."""
        self._build()

        for child in self._children.values():
            child.build()

        self.ready = True
        self.dirty = True

    def _build(self):
        raise NotImplementedError

    def dump(self):
        """Saves all models for this processor and its children to disk."""
        self._dump()

        for child in self._children.values():
            child.dump()

        self.dirty = True

    def _dump(self):
        raise NotImplementedError

    def load(self):
        """Loads all models for this processor and its children from disk."""
        self._load()

        for child in self._children.values():
            child.load()

        self.ready = True
        self.dirty = False

    def _load(self):
        raise NotImplementedError

    def _check_ready(self):
        if not self.ready:
            raise ProcessorError('Processor not ready, models must be built or loaded first.')

    def process(self, query_text):
        """Processes the input text

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = self.resource_loader.query_factory.create_query(query_text)
        return self.process_query(query).to_dict()

    def process_query(self, query):
        """Processes the query object passed in

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: The resulting processed query
        """
        raise NotImplementedError


class NaturalLanguageProcessor(Processor):
    """The natural language processor is the workbench component responsible for
    all nlp.

    Attributes:
        domain_classifier (DomainClassifier): A classifier for domains
        domains (dict): Processors for each domain
    """

    def __init__(self, app_path, resource_loader=None):
        """Initializes a natural language processor object

        Args:
            app_path (str): The path to the directory containing the app's data
        """
        super().__init__(app_path, resource_loader)
        self._app_path = app_path
        self.domain_classifier = DomainClassifier(self.resource_loader)

        for domain in path.get_domains(self._app_path):
            self._children[domain] = DomainProcessor(app_path, domain, self.resource_loader)

    @property
    def domains(self):
        return self._children

    def _build(self):
        if len(self.domains) > 1:
            self.domain_classifier.fit()

        for domain_processor in self.domains.values():
            domain_processor.build()

    def _dump(self):
        model_path = path.get_domain_model_path(self._app_path)
        self.domain_classifier.dump(model_path)

        for domain_processor in self.domains.values():
            domain_processor.dump()

    def _load(self):
        model_path = path.get_domain_model_path(self._app_path)
        self.domain_classifier.load(model_path)

        for domain_processor in self.domains.values():
            domain_processor.load()

    def process_query(self, query):
        """Processes the query object passed in

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: The resulting processed query
        """
        self._check_ready()
        if len(self.domains) > 1:
            domain = self.domain_classifier.predict(query)
        else:
            domain = list(self.domains.keys())[0]

        processed_query = self.domains[domain].process_query(query)
        processed_query.domain = domain
        return processed_query

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self._app_path)


class DomainProcessor(Processor):
    """Summary

    Attributes:
        name (str): The name of the domain
        intent_classifier (IntentClassifier): Description
        intents (dict): Description
    """

    @property
    def intents(self):
        return self._children

    def __init__(self, app_path, domain, resource_loader=None):
        super().__init__(app_path, resource_loader)
        self.name = domain
        self.intent_classifier = IntentClassifier(self.resource_loader, domain)
        for intent in path.get_intents(app_path, domain):
            self._children[intent] = IntentProcessor(app_path, domain, intent,
                                                     self.resource_loader)

    def _build(self):
        # train intent model
        if len(self.intents) > 1:
            self.intent_classifier.fit()

    def _dump(self):
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.dump(model_path)

    def _load(self):
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.load(model_path)

    def process(self, query_text):
        """Processes the input text for this domain

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = self.resource_loader.query_factory.create_query(query_text)
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
        self._check_ready()
        if len(self.intents) > 1:
            intent = self.intent_classifier.predict(query)
        else:
            intent = list(self.intents.keys())[0]
        processed_query = self.intents[intent].process_query(query)
        processed_query.intent = intent
        return processed_query

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)


class IntentProcessor(Processor):
    """Summary

    Attributes:
        domain (str): The domain this intent belongs to
        name (str): The name of this intent
        entities (dict): Description
        recognizer (EntityRecognizer): The
    """

    def __init__(self, app_path, domain, intent, resource_loader=None):
        super().__init__(app_path, resource_loader)
        self.domain = domain
        self.name = intent

        self.entity_recognizer = EntityRecognizer(self.resource_loader, domain, intent)
        self.parser = Parser(self.resource_loader, domain, intent)

    @property
    def entities(self):
        return self._children

    def _build(self):
        """Builds the models for this intent"""

        # train entity recognizer
        self.entity_recognizer.fit()

        # TODO: something for the parser?

        entity_types = self.entity_recognizer.entity_types
        for entity_type in entity_types:
            processor = EntityProcessor(self._app_path, self.domain, self.name, entity_type,
                                        self.resource_loader)
            self._children[entity_type] = processor

    def _dump(self):
        model_path = path.get_entity_model_path(self._app_path, self.domain, self.name)
        self.entity_recognizer.dump(model_path)

        # TODO: something with parser?

    def _load(self):
        model_path = path.get_entity_model_path(self._app_path, self.domain, self.name)
        self.entity_recognizer.load(model_path)

        # TODO: something with parser?

        entity_types = self.entity_recognizer.entity_types
        for entity_type in entity_types:
            processor = EntityProcessor(self._app_path, self.domain, self.name, entity_type,
                                        self.resource_loader)
            self._children[entity_type] = processor

    def process(self, query_text):
        """Processes the input text for this intent

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: The processed query
        """
        query = self.resource_loader.query_factory.create_query(query_text)
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
        self._check_ready()
        entities = self.entity_recognizer.predict(query)

        for entity in entities:
            self.entities[entity.entity.type].process_entity(query, entities, entity)

        # TODO: parse query

        return ProcessedQuery(query, entities=entities)

    def __repr__(self):
        return "<IntentProcessor {!r}>".format(self.name)


class EntityProcessor(Processor):
    """Summary

    Attributes:
        domain (str): The domain this entity belongs to
        intent (str): The intent this entity belongs to
        type (str): The type of this entity
        role_classifier (TYPE): Description
    """

    def __init__(self, app_path, domain, intent, entity_type, resource_loader=None):
        super().__init__(app_path, resource_loader)
        self.domain = domain
        self.intent = intent
        self.type = entity_type

        self.role_classifier = RoleClassifier(self.resource_loader, domain, intent, entity_type)
        self.entity_resolver = EntityResolver(self.resource_loader, entity_type)

    def _build(self):
        """Builds the models for this entity type"""
        self.role_classifier.fit()

    def _dump(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.dump(model_path)

    def _load(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.load(model_path)

    def process(self, text):
        raise NotImplementedError('EntityProcessor objects do not support `process()`. '
                                  'Try `process_entity()`')

    def process_entity(self, query, entities, entity):
        """Processes the given entity

        Args:
            query (Query): The query the entity originated from
            entities (list): All entities recognized in the query
            entity (Entity): The entity to process
        """
        self._check_ready()

        # Classify role
        entity.entity.role = self.role_classifier.predict(query, entities, entity)

        # Resolver entity
        entity.entity.value = self.entity_resolver.predict(entity.entity)
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


def create_resource_loader(app_path, query_factory=None):
    """Creates the resource loader for the app at app path

    Args:
        app_path (str): The path to the directory containing the app's data
        query_factory (QueryFactory): The app's query factory

    Returns:
        ResourceLoader: a resource loader
    """
    return ResourceLoader(app_path, query_factory or create_query_factory(app_path))


def create_parser(app_path):
    pass
