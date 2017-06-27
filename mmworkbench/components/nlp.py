# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""
from __future__ import absolute_import, unicode_literals
from builtins import object, super

from .. import path
from ..core import ProcessedQuery, Bunch
from ..exceptions import FileNotFoundError, ProcessorError
from ..resource_loader import ResourceLoader

from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .entity_resolver import EntityResolver
from .entity_recognizer import EntityRecognizer
from .parser import Parser
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
        """Initializes a processor

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the processor
        """
        self._app_path = app_path
        self.resource_loader = resource_loader or ResourceLoader.create_resource_loader(app_path)

        self._children = Bunch()
        self.ready = False
        self.dirty = False
        self.name = None

    def build(self):
        """Builds all the natural language processing models for this processor and its children."""
        self._build()

        for child in self._children.values():
            child.build()

        self.ready = True
        self.dirty = True

    def _build(self):
        raise NotImplementedError

    def dump(self):
        """Saves all the natural language processing models for this processor and its children to
        disk."""
        self._dump()

        for child in self._children.values():
            child.dump()

        self.dirty = False

    def _dump(self):
        raise NotImplementedError

    def load(self):
        """Loads all the natural language processing models for this processor and its children
        from disk."""
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
        """Processes the given input text using the trained natural language processing models
        for this processor and its children

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: A processed query object that contains the results from the
                application of this processor and its children to the input text
        """
        query = self.resource_loader.query_factory.create_query(query_text)
        return self.process_query(query).to_dict()

    def process_query(self, query):
        """Processes the given query using the trained natural language processing models for
        this processor and its children

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: A processed query object that contains the results from the
                application of this processor and its children to the input query
        """
        raise NotImplementedError

    def create_query(self, query_text):
        """Creates a query with the given text

        Args:
            text (str): Text to create a query object for

        Returns:
            Query: A newly constructed query
        """
        return self.resource_loader.query_factory.create_query(query_text)

    def __repr__(self):
        msg = '<{} {!r} ready: {!r}, dirty: {!r}>'
        return msg.format(self.__class__.__name__, self.name, self.ready, self.dirty)


class NaturalLanguageProcessor(Processor):
    """The natural language processor is the Workbench component responsible for understanding
    the user input using a hierarchy of natural language processing models.

    Attributes:
        domain_classifier (DomainClassifier): The domain classifier for this application
        domains (dict): The domains supported by this application
    """

    def __init__(self, app_path, resource_loader=None):
        """Initializes a natural language processor object

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the processor
        """
        super().__init__(app_path, resource_loader)
        self._app_path = app_path
        self.name = app_path
        self.domain_classifier = DomainClassifier(self.resource_loader)

        for domain in path.get_domains(self._app_path):
            self._children[domain] = DomainProcessor(app_path, domain, self.resource_loader)

    @property
    def domains(self):
        """The domains supported by this application"""
        return self._children

    def _build(self):
        if len(self.domains) > 1:
            self.domain_classifier.fit()

    def _dump(self):
        if len(self.domains) == 1:
            return
        model_path = path.get_domain_model_path(self._app_path)
        self.domain_classifier.dump(model_path)

    def _load(self):
        if len(self.domains) == 1:
            return
        model_path = path.get_domain_model_path(self._app_path)
        self.domain_classifier.load(model_path)

    def process_query(self, query):
        """Processes the given query using the full hierarchy of natural language processing models
        trained for this application

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the full hierarchy of natural language processing models to the input query
        """
        self._check_ready()
        if len(self.domains) > 1:
            domain = self.domain_classifier.predict(query)
        else:
            domain = list(self.domains.keys())[0]

        processed_query = self.domains[domain].process_query(query)
        processed_query.domain = domain
        return processed_query


class DomainProcessor(Processor):
    """The domain processor houses the hierarchy of domain-specific natural language processing
    models required for understanding the user input for a particular domain.

    Attributes:
        name (str): The name of the domain
        intent_classifier (IntentClassifier): The intent classifier for this domain
        intents (dict): The intents supported within this domain
    """

    @property
    def intents(self):
        """The intents supported within this domain"""
        return self._children

    def __init__(self, app_path, domain, resource_loader=None):
        """Initializes a domain processor object

        Args:
            app_path (str): The path to the directory containing the app's data
            domain (str): The name of the domain
            resource_loader (ResourceLoader): An object which can load resources for the processor
        """
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
        if len(self.intents) == 1:
            return
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.dump(model_path)

    def _load(self):
        if len(self.intents) == 1:
            return
        model_path = path.get_intent_model_path(self._app_path, self.name)
        self.intent_classifier.load(model_path)

    def process(self, query_text):
        """Processes the given input text using the hierarchy of natural language processing models
        trained for this domain

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input text
        """
        query = self.resource_loader.query_factory.create_query(query_text)
        processed_query = self.process_query(query)
        processed_query.domain = self.name
        return processed_query.to_dict()

    def process_query(self, query):
        """Processes the given query using the hierarchy of natural language processing models
        trained for this domain

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input query
        """
        self._check_ready()
        if len(self.intents) > 1:
            intent = self.intent_classifier.predict(query)
        else:
            intent = list(self.intents.keys())[0]
        processed_query = self.intents[intent].process_query(query)
        processed_query.intent = intent
        return processed_query


class IntentProcessor(Processor):
    """The intent processor houses the hierarchy of intent-specific natural language processing
    models required for understanding the user input for a particular intent.

    Attributes:
        domain (str): The domain this intent belongs to
        name (str): The name of this intent
        entities (dict): The entity types associated with this intent
        recognizer (EntityRecognizer): The entity recognizer for this intent
    """

    def __init__(self, app_path, domain, intent, resource_loader=None):
        """Initializes an intent processor object

        Args:
            app_path (str): The path to the directory containing the app's data
            domain (str): The domain this intent belongs to
            name (str): The name of this intent
            resource_loader (ResourceLoader): An object which can load resources for the processor
        """
        super().__init__(app_path, resource_loader)
        self.domain = domain
        self.name = intent

        self.entity_recognizer = EntityRecognizer(self.resource_loader, domain, intent)
        try:
            self.parser = Parser(self.resource_loader)
        except FileNotFoundError:
            # Unable to load parser config -> no parser
            self.parser = None

    @property
    def entities(self):
        """The entity types associated with this intent"""
        return self._children

    def _build(self):
        """Builds the models for this intent"""

        # train entity recognizer
        self.entity_recognizer.fit()

        # Create the entity processors
        entity_types = self.entity_recognizer.entity_types
        for entity_type in entity_types:
            processor = EntityProcessor(self._app_path, self.domain, self.name, entity_type,
                                        self.resource_loader)
            self._children[entity_type] = processor

    def _dump(self):
        model_path = path.get_entity_model_path(self._app_path, self.domain, self.name)
        self.entity_recognizer.dump(model_path)

    def _load(self):
        model_path = path.get_entity_model_path(self._app_path, self.domain, self.name)
        self.entity_recognizer.load(model_path)

        # Create the entity processors
        entity_types = self.entity_recognizer.entity_types
        for entity_type in entity_types:
            processor = EntityProcessor(self._app_path, self.domain, self.name, entity_type,
                                        self.resource_loader)
            self._children[entity_type] = processor

    def process(self, query_text):
        """Processes the given input text using the hierarchy of natural language processing models
        trained for this intent

        Args:
            query_text (str): The raw user text input

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input text
        """
        query = self.resource_loader.query_factory.create_query(query_text)
        processed_query = self.process_query(query)
        processed_query.domain = self.domain
        processed_query.intent = self.name
        return processed_query.to_dict()

    def process_query(self, query):
        """Processes the given query using the hierarchy of natural language processing models
        trained for this intent

        Args:
            query (Query): The query object to process

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input query
        """
        self._check_ready()
        entities = self.entity_recognizer.predict(query)

        for idx, entity in enumerate(entities):
            self.entities[entity.entity.type].process_entity(query, entities, idx)

        entities = self.parser.parse_entities(query, entities) if self.parser else entities

        return ProcessedQuery(query, entities=entities)


class EntityProcessor(Processor):
    """The entity processor houses the hierarchy of entity-specific natural language processing
    models required for analyzing a specific entity type in the user input

    Attributes:
        domain (str): The domain this entity belongs to
        intent (str): The intent this entity belongs to
        type (str): The type of this entity
        role_classifier (RoleClassifier): The role classifier for this entity type
    """

    def __init__(self, app_path, domain, intent, entity_type, resource_loader=None):
        """Initializes an entity processor object

        Args:
            app_path (str): The path to the directory containing the app's data
            domain (str): The domain this entity belongs to
            intent (str): The intent this entity belongs to
            entity_type (str): The type of this entity
            resource_loader (ResourceLoader): An object which can load resources for the processor
        """
        super().__init__(app_path, resource_loader)
        self.domain = domain
        self.intent = intent
        self.type = entity_type
        self.name = self.type

        self.role_classifier = RoleClassifier(self.resource_loader, domain, intent, entity_type)
        self.entity_resolver = EntityResolver(app_path, self.resource_loader, entity_type)

    def _build(self):
        """Builds the models for this entity type"""
        self.role_classifier.fit()
        self.entity_resolver.fit()

    def _dump(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.dump(model_path)

    def _load(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.load(model_path)
        self.entity_resolver.load()

    def process(self, text):
        raise NotImplementedError('EntityProcessor objects do not support `process()`. '
                                  'Try `process_entity()`')

    def process_entity(self, query, entities, entity_index):
        """Processes the given entity using the hierarchy of natural language processing models
        trained for this entity type

        Args:
            query (Query): The query the entity originated from
            entities (list): All entities recognized in the query
            entity (Entity): The entity to process

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input entity
        """
        self._check_ready()
        entity = entities[entity_index]

        if self.role_classifier.roles:
            # Only run role classifier if there are roles!
            entity.entity.role = self.role_classifier.predict(query, entities, entity_index)

        # Resolve entity
        entity.entity.value = self.entity_resolver.predict(entity.entity)
        return entity
