# -*- coding: utf-8 -*-
"""
This module contains the natural language processor.
"""
from __future__ import absolute_import, unicode_literals
from builtins import object, super
from concurrent.futures import ProcessPoolExecutor, as_completed

from abc import ABCMeta, abstractmethod
import logging

from future.utils import with_metaclass

from .. import path
from ..core import ProcessedQuery, Bunch
from ..exceptions import FileNotFoundError, ProcessorError
from ..resource_loader import ResourceLoader
from .._version import validate_workbench_version

from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .entity_resolver import EntityResolver
from .entity_recognizer import EntityRecognizer
from .parser import Parser
from .role_classifier import RoleClassifier
from ..exceptions import AllowedNlpClassesKeyError
from ..markup import process_markup
from ..query_factory import QueryFactory
from ._config import get_nlp_config


logger = logging.getLogger(__name__)
executor = ProcessPoolExecutor(max_workers=4)

def global_get_entities(instance_id, *args, **kwargs):
    """
    A module function used as a trampoline to call an instance function
    from within a long running child process.

    Args:
        instance_id (number): id(inst) of the IntentProcessor instance that needs called

    Returns:
        Prediction from the intent processor's  entity_recognizer
    """
    return IntentProcessor.instance_map[instance_id].entity_recognizer.predict(*args, **kwargs)


def global_create_query(instance_id, *args, **kwargs):
    """
    A module function used as a trampoline to call an instance function
    from within a long running child process.

    Args:
        instance_id (number): id(inst) of the Processor instance that needs called

    Returns:
        Query from the create_query instance method.
    """
    return Processor.instance_map[instance_id].create_query(*args, **kwargs)


class Processor(with_metaclass(ABCMeta, object)):
    """A generic base class for processing queries through the workbench NLP
    components

    Attributes:
        dirty (bool): Indicates whether the processor has unsaved changes to
            its models
        ready (bool): Indicates whether the processor is ready to process
            messages
    """

    instance_map = {}

    def __init__(self, app_path, resource_loader=None, config=None):
        """Initializes a processor

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the processor
            config (dict): A config object with processor settings (e.g. if to use n-best inference)
        """
        self._app_path = app_path
        self.resource_loader = resource_loader or ResourceLoader.create_resource_loader(app_path)

        self._children = Bunch()
        self.ready = False
        self.dirty = False
        self.name = None
        self.config = get_nlp_config(app_path, config)
        Processor.instance_map[id(self)] = self

    def build(self, incremental=False, label_set="train"):
        """Builds all the natural language processing models for this processor and its children.

        Args:
            incremental (bool, optional): When True, only build models whose training data or
                configuration has changed since the last build. Defaults to False
            label_set (string, optional): The label set from which to train all classifiers
        """
        self._build(incremental=incremental, label_set=label_set)

        for child in self._children.values():
            child.build(incremental=incremental, label_set=label_set)

        self.ready = True
        self.dirty = True

    @abstractmethod
    def _build(self, incremental=False, label_set="train"):
        raise NotImplementedError

    def dump(self):
        """Saves all the natural language processing models for this processor and its children to
        disk."""
        self._dump()

        for child in self._children.values():
            child.dump()

        self.dirty = False

    @abstractmethod
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

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    def evaluate(self, print_stats=False, label_set="test"):
        """Evaluates all the natural language processing models for this processor and its
        children.

        Args:
            print_stats (bool): If true, prints the full stats table. Otherwise prints just
                                the accuracy
            label_set (string, optional): The label set from which to evaluate
                                all classifiers

        """
        self._evaluate(print_stats, label_set=label_set)

        for child in self._children.values():
            child.evaluate(print_stats, label_set=label_set)

    @abstractmethod
    def _evaluate(self, label_set="test"):
        raise NotImplementedError

    def _check_ready(self):
        if not self.ready:
            raise ProcessorError('Processor not ready, models must be built or loaded first.')

    def _create_queries(self, query_text, language=None, time_zone=None, timestamp=None):
        """Creates the query object(s).

        Args:
            query_text (str, or tuple): The raw user text input, or a list of the n best query
                transcripts from ASR
            language (str, optional): Language as specified using a 639-2 code;
                if omitted, English is assumed.
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).

        Returns:
            Query or tuple: A newly constructed query, or tuple of queries

        """
        if isinstance(query_text, (list, tuple)):
            if not query_text:
                query_text = ['']
            query = list(query_text)
            # map the futures to the index of the query they belong to
            futures = {executor.submit(
                global_create_query, id(self), q_text, language=language, \
                time_zone=time_zone, timestamp=timestamp): idx
		       for idx, q_text in enumerate(query_text)}
            # set the completed queries into their appropriate index
            for future in as_completed(futures):
                query[futures[future]] = future.result()
            query = tuple(query)
        else:
            query = self.create_query(query_text, language=language, time_zone=time_zone,
                                      timestamp=timestamp)
        return query

    def process(self, query_text, allowed_nlp_classes=None, language=None, time_zone=None,
                timestamp=None):
        """Processes the given query using the full hierarchy of natural language processing models
        trained for this application

        Args:
            query_text (str, or tuple): The raw user text input, or a list of the n best query
                transcripts from ASR
            allowed_nlp_classes (dict, optional): A dictionary of the NLP hierarchy that is
                selected for NLP analysis. An example:

                    {
                        smart_home: {
                            close_door: {}
                        }
                    }

                where smart_home is the domain and close_door is the intent.
            language (str, optional): Language as specified using a 639-2 code;
                if omitted, English is assumed.
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the full hierarchy of natural language processing models to the input query
        """
        query = self._create_queries(query_text, language=language, time_zone=time_zone,
                                     timestamp=timestamp)
        print(query)
        tmp = self.process_query(query, allowed_nlp_classes)
        print(tmp)
        return tmp.to_dict()

    def process_query(self, query, allowed_nlp_classes=None):
        """Processes the given query using the full hierarchy of natural language processing models
        trained for this application

        Args:
            query (Query, or tuple): The user input query, or a list of the n best query objects
            allowed_nlp_classes (dict, optional): A dictionary of the NLP hierarchy that is
                selected for NLP analysis. An example:

                    {
                        smart_home: {
                            close_door: {}
                        }
                    }

                where smart_home is the domain and close_door is the intent.
            nbest_queries (list, optional): A list of Query objects, one for each of the nbest
                                            transcript from ASR.

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the full hierarchy of natural language processing models to the input query
        """
        raise NotImplementedError

    def create_query(self, query_text, language=None, time_zone=None, timestamp=None):
        """Creates a query with the given text

        Args:
            query_text (str): Text to create a query object for
            language (str, optional): Language as specified using a 639-2 code such as 'eng' or
                'spa'; if omitted, English is assumed.
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).

        Returns:
            Query: A newly constructed query
        """
        return self.resource_loader.query_factory.create_query(
            query_text, language=language, time_zone=time_zone, timestamp=timestamp
        )

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

    def __init__(self, app_path, resource_loader=None, config=None):
        """Initializes a natural language processor object

        Args:
            app_path (str): The path to the directory containing the app's data
            resource_loader (ResourceLoader): An object which can load resources for the processor
            config (dict): A config object with processor settings (e.g. if to use n-best inference)
        """
        super().__init__(app_path, resource_loader, config)
        self._app_path = app_path
        validate_workbench_version(self._app_path)
        self.name = app_path
        self.domain_classifier = DomainClassifier(self.resource_loader)

        for domain in path.get_domains(self._app_path):
            self._children[domain] = DomainProcessor(app_path, domain, self.resource_loader)

        nbest_nlp_classes = self.config.get('extract_nbest_entities', {})
        if len(nbest_nlp_classes) > 0:
            nbest_nlp_classes = self.extract_allowed_intents(nbest_nlp_classes)

            for domain in nbest_nlp_classes.keys():
                for intent in nbest_nlp_classes[domain].keys():
                    self.domains[domain].intents[intent].nbest_text_enabled = True

    @property
    def domains(self):
        """The domains supported by this application"""
        return self._children

    def _build(self, incremental=False, label_set="train"):
        if len(self.domains) == 1:
            return
        if incremental:
            model_path = path.get_domain_model_path(self._app_path)
            self.domain_classifier.fit(previous_model_path=model_path, label_set=label_set)
        else:
            self.domain_classifier.fit(label_set=label_set)

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

    def _evaluate(self, print_stats, label_set="test"):
        if len(self.domains) > 1:
            domain_eval = self.domain_classifier.evaluate(label_set=label_set)
            if domain_eval:
                print("Domain classification accuracy: '{}'".format(domain_eval.get_accuracy()))
                if print_stats:
                    domain_eval.print_stats()
            else:
                logger.info("Skipping domain classifier evaluation")

    def _process_domain(self, query, allowed_nlp_classes=None):
        if len(self.domains) > 1:
            if not allowed_nlp_classes:
                return self.domain_classifier.predict(query)
            else:
                sorted_domains = self.domain_classifier.predict_proba(query)
                for ordered_domain, _ in sorted_domains:
                    if ordered_domain in allowed_nlp_classes.keys():
                        return ordered_domain

                raise AllowedNlpClassesKeyError(
                    'Could not find user inputted domain in NLP hierarchy')
        else:
            return list(self.domains.keys())[0]

    def process_query(self, query, allowed_nlp_classes=None):
        """Processes the given query using the full hierarchy of natural language processing models
        trained for this application

        Args:
            query (Query, or tuple): The user input query, or a list of the n best query objects
            allowed_nlp_classes (dict, optional): A dictionary of the NLP hierarchy that is
            selected for NLP analysis. An example:
            {
                smart_home: {
                    close_door: {}
                }
            }
            where smart_home is the domain and close_door is the intent. If allowed_nlp_classes
            is None, we just use the normal model predict functionality.

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the full hierarchy of natural language processing models to the input query
        """
        self._check_ready()
        if isinstance(query, (list, tuple)):
            top_query = query[0]
        else:
            top_query = query
        domain = self._process_domain(top_query, allowed_nlp_classes=allowed_nlp_classes)

        allowed_intents = allowed_nlp_classes.get(domain) if allowed_nlp_classes else None

        processed_query = \
            self.domains[domain].process_query(query, allowed_intents)
        processed_query.domain = domain
        return processed_query

    def extract_allowed_intents(self, allowed_intents):
        """This function validates a user inputted list of allowed_intents against the NLP
        hierarchy and construct a hierarchy dictionary as follows: {domain: {intent: {}} if
        the validation of allowed_intents has passed.

        Args:
            allowed_intents (list): A list of allowable intents in the format "domain.intent".
            If all intents need to be included, the syntax is "domain.*".

        Returns:
            (dict): A dictionary of NLP hierarchy
        """
        nlp_components = {}

        for allowed_intent in allowed_intents:
            domain, intent = allowed_intent.split(".")

            if domain not in self.domains.keys():
                raise AllowedNlpClassesKeyError(
                    "Domain: {} is not in the NLP component hierarchy".format(domain))

            if intent != "*" and intent not in self.domains[domain].intents.keys():
                raise AllowedNlpClassesKeyError(
                    "Intent: {} is not in the NLP component hierarchy".format(intent))

            if domain not in nlp_components:
                nlp_components[domain] = {}

            if intent == "*":
                for intent in self.domains[domain].intents.keys():
                    # We initialize to an empty dictionary to extend capability for
                    # entity rules in the future
                    nlp_components[domain][intent] = {}
            else:
                nlp_components[domain][intent] = {}

        return nlp_components

    def inspect(self, markup, domain=None, intent=None):
        """ Inspect the marked up query and print the table of features and weights
        Args:
            markup (str): The marked up query string
            domain (str): The gold value for domain classification
            intent (str): The gold value for intent classification
        """
        query_factory = QueryFactory.create_query_factory()
        raw_query, query, entities = process_markup(markup, query_factory, query_options={})

        if domain:
            print('Inspecting domain classification')
            domain_inspection = self.domain_classifier.inspect(query, domain=domain)
            print(domain_inspection)
            print('')

        if intent:
            print('Inspecting intent classification')
            domain = self._process_domain(query)
            intent_inspection = self.domains[domain].inspect(query, intent=intent)
            print(intent_inspection)
            print('')


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

    def _build(self, incremental=False, label_set="train"):
        if len(self.intents) == 1:
            return
        # train intent model
        if incremental:
            model_path = path.get_intent_model_path(self._app_path, self.name)
            self.intent_classifier.fit(previous_model_path=model_path, label_set=label_set)
        else:
            self.intent_classifier.fit(label_set=label_set)

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

    def _evaluate(self, print_stats, label_set="test"):
        if len(self.intents) > 1:
            intent_eval = self.intent_classifier.evaluate(label_set=label_set)
            if intent_eval:
                print("Intent classification accuracy for the '{}' domain: {}".format(
                      self.name, intent_eval.get_accuracy()))
                if print_stats:
                    intent_eval.print_stats()
            else:
                logger.info("Skipping intent classifier evaluation for the '{}' domain".format(
                            self.name))

    def process(self, query_text, allowed_nlp_classes, time_zone=None, timestamp=None):
        """Processes the given input text using the hierarchy of natural language processing models
        trained for this domain

        Args:
            query_text (str, or list/tuple): The raw user text input, or a list of the n best query
                transcripts from ASR
            allowed_nlp_classes (dict, optional): A dictionary of the intent section of the
                NLP hierarchy that is selected for NLP analysis. An example:

                    {
                        close_door: {}
                    }

                where close_door is the intent. The intent belongs to the smart_home domain.
                If allowed_nlp_classes is None, we use the normal model predict functionality.
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input text
        """
        query = self._create_queries(query_text, time_zone=time_zone, timestamp=timestamp)
        processed_query = self.process_query(query, allowed_nlp_classes=allowed_nlp_classes)
        processed_query.domain = self.name
        return processed_query.to_dict()

    def process_query(self, query, allowed_nlp_classes=None):
        """Processes the given query using the full hierarchy of natural language processing models
        trained for this application

        Args:
            query (Query, or tuple): The user input query, or a list of the n best query objects
            allowed_nlp_classes (dict, optional): A dictionary of the intent section of the
                NLP hierarchy that is selected for NLP analysis. An example:

                    {
                        close_door: {}
                    }

                where close_door is the intent. The intent belongs to the smart_home domain.
                If allowed_nlp_classes is None, we use the normal model predict functionality.
            nbest_queries (list, optional): A list of Query objects, one for each of the nbest
                                            transcript from ASR.

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the full hierarchy of natural language processing models to the input query
        """
        self._check_ready()

        if isinstance(query, (list, tuple)):
            top_query = query[0]
        else:
            top_query = query

        if len(self.intents) > 1:
            # Check if the user has specified allowed intents
            if not allowed_nlp_classes:
                intent = self.intent_classifier.predict(top_query)
            else:
                sorted_intents = self.intent_classifier.predict_proba(top_query)
                intent = None

                for ordered_intent, _ in sorted_intents:
                    if ordered_intent in allowed_nlp_classes.keys():
                        intent = ordered_intent
                        break

                if not intent:
                    raise AllowedNlpClassesKeyError(
                        'Could not find user inputted intent in NLP hierarchy')
        else:
            intent = list(self.intents.keys())[0]
        processed_query = self.intents[intent].process_query(query)
        processed_query.intent = intent

        return processed_query

    def inspect(self, query, intent=None):
        return self.intent_classifier.inspect(query, intent=intent)


class IntentProcessor(Processor):
    """The intent processor houses the hierarchy of intent-specific natural language processing
    models required for understanding the user input for a particular intent.

    Attributes:
        domain (str): The domain this intent belongs to
        name (str): The name of this intent
        entities (dict): The entity types associated with this intent
        recognizer (EntityRecognizer): The entity recognizer for this intent
    """
    instance_map = {}

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
            self.parser = Parser(self.resource_loader, domain=domain, intent=intent)
        except FileNotFoundError:
            # Unable to load parser config -> no parser
            self.parser = None

        self._nbest_text_enabled = False
        IntentProcessor.instance_map[id(self)] = self

    @property
    def entities(self):
        """The entity types associated with this intent"""
        return self._children

    @property
    def nbest_text_enabled(self):
        """Whether or not to run nbest processing for this intent"""
        return self._nbest_text_enabled

    @nbest_text_enabled.setter
    def nbest_text_enabled(self, value):
        self._nbest_text_enabled = value

    def _build(self, incremental=False, label_set="train"):
        """Builds the models for this intent"""

        # train entity recognizer
        if incremental:
            model_path = path.get_entity_model_path(self._app_path, self.domain, self.name)
            self.entity_recognizer.fit(previous_model_path=model_path, label_set=label_set)
        else:
            self.entity_recognizer.fit(label_set=label_set)

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

    def _evaluate(self, print_stats, label_set="test"):
        if len(self.entity_recognizer.entity_types) > 1:
            entity_eval = self.entity_recognizer.evaluate(label_set=label_set)
            if entity_eval:
                print("Entity recognition accuracy for the '{}.{}' intent"
                      ": {}".format(self.domain, self.name, entity_eval.get_accuracy()))
                if print_stats:
                    entity_eval.print_stats()
            else:
                logger.info("Skipping entity recognizer evaluation for the '{}.{}' intent".format(
                            self.domain, self.name))

    def process(self, query_text, time_zone=None, timestamp=None):
        """Processes the given input text using the hierarchy of natural language processing models
        trained for this intent

        Args:
            query_text (str, or list/tuple): The raw user text input, or a list of the n best query
                transcripts from ASR
            time_zone (str, optional): The name of an IANA time zone, such as
                'America/Los_Angeles', or 'Asia/Kolkata'
                See the [tz database](https://www.iana.org/time-zones) for more information.
            timestamp (long, optional): A unix time stamp for the request (in seconds).

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input text
        """
        query = self._create_queries(query_text, time_zone=time_zone, timestamp=timestamp)
        processed_query = self.process_query(query)
        processed_query.domain = self.domain
        processed_query.intent = self.name
        return processed_query.to_dict()

    def process_query(self, query):
        """Processes the given query using the hierarchy of natural language processing models
        trained for this intent

        Args:
            query (Query, or tuple): The user input query, or a list of the n best query objects
            nbest_queries (list, optional): A list of Query objects, one for each of the nbest
                                            transcript from ASR.

        Returns:
            ProcessedQuery: A processed query object that contains the prediction results from
                applying the hierarchy of natural language processing models to the input query
        """
        self._check_ready()

        if isinstance(query, (list, tuple)):
            if self.nbest_text_enabled:
                nbest_entities = list(query)
                futures = {executor.submit(global_get_entities, id(self), tup[1]): tup for tup in enumerate(query)}
                for future in as_completed(futures):
                    entities = future.result()
                    entity_idx, n_query = futures[future]
                    for idx, entity in enumerate(entities):
                        self.entities[entity.entity.type].process_entity(n_query, entities, idx)
                    entities = self.parser.parse_entities(n_query, entities) \
                        if self.parser else entities
                    nbest_entities[entity_idx] = entities
                return ProcessedQuery(query[0], entities=nbest_entities[0], nbest_queries=query,
                                      nbest_entities=nbest_entities)
            else:
                query = query[0]

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

    def _build(self, incremental=False, label_set="train"):
        """Builds the models for this entity type"""
        if incremental:
            model_path = path.get_role_model_path(self._app_path, self.domain,
                                                  self.intent, self.type)
            self.role_classifier.fit(previous_model_path=model_path, label_set=label_set)
        else:
            self.role_classifier.fit(label_set=label_set)

        self.entity_resolver.fit()

    def _dump(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.dump(model_path)

    def _load(self):
        model_path = path.get_role_model_path(self._app_path, self.domain, self.intent, self.type)
        self.role_classifier.load(model_path)
        self.entity_resolver.load()

    def _evaluate(self, print_stats, label_set="test"):
        if len(self.role_classifier.roles) > 1:
            role_eval = self.role_classifier.evaluate(label_set=label_set)
            if role_eval:
                print("Role classification accuracy for the {}.{}.{}' entity type: {}".format(
                      self.domain, self.intent, self.type, role_eval.get_accuracy()))
                if print_stats:
                    role_eval.print_stats()
            else:
                logger.info("Skipping role classifier evaluation for the '{}.{}.{}' "
                            "entity type".format(self.domain, self.intent, self.type))

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
