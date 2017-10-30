# -*- coding: utf-8 -*-
"""
This module contains the processor resource loader.
"""
from __future__ import absolute_import, unicode_literals
from builtins import object

from copy import deepcopy
from collections import Counter

import fnmatch
import hashlib
import json
import logging
import os
import time

from . import markup, path
from .exceptions import WorkbenchError
from .gazetteer import Gazetteer
from .query_factory import QueryFactory
from .models.helpers import (GAZETTEER_RSC, QUERY_FREQ_RSC, SYS_TYPES_RSC, WORD_FREQ_RSC,
                             mask_numerics)
from .core import Entity

logger = logging.getLogger(__name__)

LABEL_SETS = {
    'train': 'train*.txt',
    'test': 'test*.txt'
}
DEFAULT_LABEL_SET = 'train'


class ResourceLoader(object):

    def __init__(self, app_path, query_factory):
        self.app_path = app_path
        self.query_factory = query_factory

        # Example Layout: {
        #   'entity_type': {
        #     'entity_data': {  # for gazetteer.txt
        #        'loaded': '',  # the time the file was last loaded
        #        'modified': '',  # the time the file was last modified
        #        'data': contents of file
        #     },
        #     'mapping': {  # for mapping.json
        #        'loaded': '',  # the time the file was last loaded
        #        'modified': '',  # the time the file was last modified
        #     },
        #     'gazetteer': {  # for pickled gazetteer
        #        'loaded': '',  # the time the file was last loaded
        #        'modified': '',  # the time the file was last modified
        #     }
        #   }
        # }
        self._entity_files = {}

        # Example layout: {
        #   'domain': {
        #     'intent': {
        #       'filename': {
        #         'modified': '',  # the time the file was last modified
        #         'loaded': '',  # the time the file was last loaded
        #         'queries': [],  # the queries loaded from the file
        #         'loaded_raw': '',  # the time the raw queries were last loaded
        #         'raw_queries': []  # the time the raw queries were last loaded
        #       }
        #     }
        #   }
        # }
        self.labeled_query_files = {}

        self._hasher = _StringHasher()

    def get_gazetteers(self, force_reload=False, **kwargs):
        """Gets all gazetteers

        Returns:
            dict: Gazetteer data keyed by entity type
        """
        # TODO: get role gazetteers
        entity_types = path.get_entity_types(self.app_path)
        return {entity_type: self.get_gazetteer(entity_type, force_reload=force_reload)
                for entity_type in entity_types}

    def get_gazetteer(self, gaz_name, force_reload=False):
        """Gets a gazetteers by name

        Args:
            gaz_name (str): The name of the entity

        Returns:
            dict: Gazetteer data
        """
        self._update_entity_file_dates(gaz_name)
        # TODO: get role gazetteers
        if self._gaz_needs_build(gaz_name) or force_reload:
            self.build_gazetteer(gaz_name, force_reload=force_reload)
        if self._entity_file_needs_load('gazetteer', gaz_name):
            self.load_gazetteer(gaz_name)

        return self._entity_files[gaz_name]['gazetteer']['data']

    def get_gazetteers_hash(self, algorithm='sha1'):
        entity_types = path.get_entity_types(self.app_path)
        gaz_hash = hashlib.new(algorithm)
        for entity_type in sorted(entity_types):
            gaz_hash.update(self.get_gazetteer_hash(entity_type,
                                                    algorithm=algorithm).encode('utf8'))

        return gaz_hash.hexdigest()

    def get_gazetteer_hash(self, gaz_name, algorithm='sha1'):
        self._update_entity_file_dates(gaz_name)
        entity_data_path = path.get_entity_gaz_path(self.app_path, gaz_name)
        entity_data_hash = self._hash_file(entity_data_path, algorithm=algorithm)

        mapping_path = path.get_entity_map_path(self.app_path, gaz_name)
        mapping_hash = self._hash_file(mapping_path, algorithm=algorithm)

        hash_obj = hashlib.new(algorithm)
        hash_obj.update(entity_data_hash.encode('utf8'))
        hash_obj.update(mapping_hash.encode('utf8'))

        return hash_obj.hexdigest()

    def build_gazetteer(self, gaz_name, exclude_ngrams=False, force_reload=False):
        """Builds the specified gazetteer using the entity data and mapping files

        Args:
            gaz_name (str): The name of the gazetteer
            exclude_ngrams (bool, optional): Whether partial matches of
                 entities should be included in the gazetteer
            force_reload (bool, optional): Whether file should be forcefully
                 reloaded from disk
        """
        popularity_cutoff = 0.0

        logger.info("Building gazetteer '{}'".format(gaz_name))

        # TODO: support role gazetteers
        gaz = Gazetteer(gaz_name, exclude_ngrams)

        entity_data_path = path.get_entity_gaz_path(self.app_path, gaz_name)
        gaz.update_with_entity_data_file(entity_data_path, popularity_cutoff,
                                         self.query_factory.normalize)
        self._entity_files[gaz_name]['entity_data']['loaded'] = time.time()

        mapping = self.get_entity_map(gaz_name, force_reload=force_reload)
        gaz.update_with_entity_map(mapping.get('entities', []), self.query_factory.normalize)

        gaz_path = path.get_gazetteer_data_path(self.app_path, gaz_name)
        gaz.dump(gaz_path)

        self._entity_files[gaz_name]['gazetteer']['data'] = gaz.to_dict()
        self._entity_files[gaz_name]['gazetteer']['loaded'] = time.time()

    def load_gazetteer(self, gaz_name):
        gaz = Gazetteer(gaz_name)
        gaz_path = path.get_gazetteer_data_path(self.app_path, gaz_name)
        gaz.load(gaz_path)
        self._entity_files[gaz_name]['gazetteer']['data'] = gaz.to_dict()
        self._entity_files[gaz_name]['gazetteer']['loaded'] = time.time()

    def get_entity_map(self, entity_type, force_reload=False):
        self._update_entity_file_dates(entity_type)
        if self._entity_file_needs_load('mapping', entity_type) or force_reload:
            # file is out of date, load it
            self.load_entity_map(entity_type)
        return deepcopy(self._entity_files[entity_type]['mapping']['data'])

    def load_entity_map(self, entity_type):
        """Loads an entity map

        Args:
            entity_type (str): The type of entity to load a mapping
        """
        file_path = path.get_entity_map_path(self.app_path, entity_type)
        logger.debug("Loading entity map from file '{}'".format(file_path))
        if not os.path.isfile(file_path):
            logger.warn("Entity map file not found at {!r}".format(file_path))
            json_data = {}
        else:
            try:
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
            except json.JSONDecodeError:
                raise WorkbenchError('Could not load entity map (Invalid JSON): {!r}'.
                                     format(file_path))

        self._entity_files[entity_type]['mapping']['data'] = json_data
        self._entity_files[entity_type]['mapping']['loaded'] = time.time()

    def _gaz_needs_build(self, gaz_name):
        try:
            build_time = self._entity_files[gaz_name]['gazetteer']['modified']
        except KeyError:
            # gazetteer hasn't been built
            return True

        data_modified = self._entity_files[gaz_name]['entity_data']['modified']
        mapping_modified = self._entity_files[gaz_name]['mapping']['modified']

        # If entity data or mapping modified after gaz build -> stale
        return build_time < data_modified or build_time < mapping_modified

    def _entity_file_needs_load(self, file_type, entity_type):
        mod_time = self._entity_files[entity_type][file_type]['modified']
        try:
            load_time = self._entity_files[entity_type][file_type]['loaded']
        except KeyError:
            # file hasn't been loaded
            return True
        return load_time < mod_time

    def _update_entity_file_dates(self, entity_type):
        # TODO: handle deleted entity
        file_table = self._entity_files.get(entity_type, {
            'entity_data': {},
            'mapping': {},
            'gazetteer': {}
        })
        self._entity_files[entity_type] = file_table

        # update entity data
        entity_data_path = path.get_entity_gaz_path(self.app_path, entity_type)
        try:
            file_table['entity_data']['modified'] = os.path.getmtime(entity_data_path)
        except (OSError, IOError):
            # required file doesnt exist -- notify and error out
            logger.warning('Entity data file not found at %r. '
                           'Proceeding with empty entity data.', entity_data_path)

            # mark the modified time as now if the entity file does not exist
            # so that the gazetteer gets rebuilt properly.
            file_table['entity_data']['modified'] = time.time()

        # update mapping
        mapping_path = path.get_entity_map_path(self.app_path, entity_type)
        try:
            file_table['mapping']['modified'] = os.path.getmtime(mapping_path)
        except (OSError, IOError):
            # required file doesnt exist
            logger.warning('Entity mapping file not found at %r. '
                           'Proceeding with empty entity data.', mapping_path)

            # mark the modified time as now if the entity mapping file does not exist
            # so that the gazetteer gets rebuilt properly.
            file_table['mapping']['modified'] = time.time()

        # update gaz data
        gazetteer_path = path.get_gazetteer_data_path(self.app_path, entity_type)
        try:
            file_table['gazetteer']['modified'] = os.path.getmtime(gazetteer_path)
        except (OSError, IOError):
            # gaz not yet built so set to a time impossibly long ago
            file_table['gazetteer']['modified'] = 0.0

    def hash_queries(self, queries, algorithm='sha1'):
        hasher = self._hasher or _StringHasher(algorithm=algorithm)
        hash_obj = hashlib.new(hasher.algorithm)
        for query in sorted(queries):
            hash_obj.update(hasher.hash(query).encode('utf8'))
        return hash_obj.hexdigest()

    def get_labeled_queries(self, domain=None, intent=None, label_set=None,
                            force_reload=False, raw=False):
        """Gets labeled queries from the cache, or loads them from disk.

        Args:
            domain (str): The domain of queries to load
            intent (str): The intent of queries to load
            force_reload (bool): Will not load queries from the cache when True
            raw (bool): Will return raw query strings instead of ProcessedQuery objects when true

        Returns:
            dict: ProcessedQuery objects (or strings) loaded from labeled query files, organized by
                domain and intent.
        """
        label_set = label_set or DEFAULT_LABEL_SET
        query_tree = {}
        loaded_key = 'loaded_raw' if raw else 'loaded'
        file_iter = self._traverse_labeled_queries_files(domain, intent, label_set)
        for a_domain, an_intent, filename in file_iter:
            file = self.labeled_query_files[a_domain][an_intent][filename]
            if force_reload or (not file[loaded_key] or file[loaded_key] < file['modified']):
                # file is out of date, load it
                self.load_query_file(a_domain, an_intent, filename, raw=raw)

            if a_domain not in query_tree:
                query_tree[a_domain] = {}

            if an_intent not in query_tree[a_domain]:
                query_tree[a_domain][an_intent] = []
            queries = query_tree[a_domain][an_intent]
            queries.extend(file['raw_queries' if raw else 'queries'])

        return query_tree

    @staticmethod
    def flatten_query_tree(query_tree):
        flattened = []
        for _, intent_queries in query_tree.items():
            for _, queries in intent_queries.items():
                flattened.extend(queries)
        return flattened

    def _traverse_labeled_queries_files(self, domain=None, intent=None, label_set='train'):
        provided_intent = intent
        try:
            file_pattern = LABEL_SETS[label_set]
        except KeyError:
            raise WorkbenchError("Unknown label set '{}'".format(label_set))
        self._update_query_file_dates(file_pattern)

        domains = [domain] if domain else self.labeled_query_files.keys()

        for a_domain in sorted(domains):
            if provided_intent:
                intents = [provided_intent]
            else:
                intents = self.labeled_query_files[a_domain].keys()
            for an_intent in sorted(intents):
                files = self.labeled_query_files[a_domain][an_intent].keys()
                # filter to files which belong to the label set
                files = fnmatch.filter(files, file_pattern)
                for filename in sorted(files):
                    yield a_domain, an_intent, filename

    def load_query_file(self, domain, intent, filename, raw=False):
        """Loads the queries from the specified file

        Args:
            domain (str): The domain of the query file
            intent (str): The intent of the query file
            filename (str): The name of the query file

        """
        logger.info("Loading %squeries from file %s/%s/%s", "raw " if raw else "", domain,
                    intent, filename)
        file_path = path.get_labeled_query_file_path(self.app_path, domain, intent, filename)
        file_data = self.labeled_query_files[domain][intent][filename]
        if raw:
            # Only load
            queries = []
            for query in markup.read_query_file(file_path):
                queries.append(query)
            file_data['raw_queries'] = queries
            file_data['loaded_raw'] = time.time()
        else:
            queries = markup.load_query_file(file_path, self.query_factory, domain, intent,
                                             is_gold=True)
            try:
                self._check_query_entities(queries)
            except WorkbenchError as exc:
                logger.warning(exc.message)
            file_data['queries'] = queries
            file_data['loaded'] = time.time()

    def _check_query_entities(self, queries):
        entity_types = path.get_entity_types(self.app_path)
        for query in queries:
            for entity in query.entities:
                if (entity.entity.type not in entity_types and
                        not entity.entity.is_system_entity):
                    msg = 'Unknown entity {!r} found in query {!r}'
                    raise WorkbenchError(msg.format(entity.entity.type, query.query.text))

    def _update_query_file_dates(self, file_pattern):
        query_tree = path.get_labeled_query_tree(self.app_path, [file_pattern])

        # Get current query table
        # We can just use this if it this is the first check
        new_app_table = {}
        for domain in query_tree.keys():
            new_app_table[domain] = {}
            for intent in query_tree[domain].keys():
                new_app_table[domain][intent] = {}

                for file, modified in query_tree[domain][intent].items():
                    new_app_table[domain][intent][file] = {
                        'modified': modified,
                        'loaded': None,
                        'loaded_raw': None
                    }

        all_domains = set(new_app_table.keys())
        all_domains.update(self.labeled_query_files.keys())
        for domain in all_domains:
            try:
                new_domain_table = new_app_table[domain]
            except KeyError:
                # an old domain that was removed
                del self.labeled_query_files[domain]
                continue

            try:
                domain_table = self.labeled_query_files[domain]
            except KeyError:
                # a new domain
                self.labeled_query_files[domain] = new_domain_table
                continue

            # domain existed before and now, resolve differences
            all_intents = set(new_domain_table.keys())
            all_intents.update(domain_table.keys())
            for intent in all_intents:
                try:
                    new_intent_table = new_domain_table[intent]
                except KeyError:
                    # an old intent that was removed
                    del domain_table[intent]
                    continue

                try:
                    intent_table = domain_table[intent]
                except KeyError:
                    # a new intent
                    domain_table[intent] = new_intent_table
                    continue

                # intent existed before and noew, resolve differences
                all_files = set(new_intent_table.keys())
                all_files.update(intent_table.keys())
                for file in all_files:
                    if file not in new_intent_table:
                        # an old file that was removed
                        del intent_table[file]
                        continue

                    if file not in intent_table:
                        intent_table[file] = new_intent_table[file]
                    else:
                        intent_table[file]['modified'] = new_intent_table[file]['modified']

    def _build_word_freq_dict(self, **kwargs):
        """Compiles unigram frequency dictionary of normalized query tokens

        Args:
            queries (list of Query): A list of all queries
        """
        # Unigram frequencies
        tokens = [mask_numerics(tok) for q in kwargs.get('queries')
                  for tok in q.normalized_tokens]
        freq_dict = Counter(tokens)

        return freq_dict

    def _build_query_freq_dict(self, **kwargs):
        """Compiles frequency dictionary of normalized query strings

        Args:
            queries (list of Query): A list of all queries
        """
        # Whole query frequencies, with singletons removed
        query_dict = Counter(['<{}>'.format(q.normalized_text) for q in kwargs.get('queries')])
        for query in query_dict:
            if query_dict[query] < 2:
                query_dict[query] = 0
        query_dict += Counter()

        return query_dict

    def _get_sys_entity_types(self, **kwargs):
        """Get all system entity types from the entity labels.

        Args:
            labels (list of QueryEntity): a list of labeled entities
        """

        # Build entity types set
        entity_types = set()
        for label in kwargs.get('labels'):
            for entity in label:
                entity_types.add(entity.entity.type)

        return set((t for t in entity_types if Entity.is_system_entity(t)))

    def load_feature_resource(self, name, **kwargs):
        """Load specified resource for feature extractor.

        Args:
            name (str): resource name
        """
        resource_loader = self.FEATURE_RSC_MAP.get(name)
        if resource_loader:
            return resource_loader(self, **kwargs)
        else:
            raise ValueError('Invalid resource name \'{}\'.'.format(name))

    def hash_feature_resource(self, name):
        return self.RSC_HASH_MAP.get(name)(self)

    @staticmethod
    def create_resource_loader(app_path, query_factory=None):
        """Creates the resource loader for the app at app path

        Args:
            app_path (str): The path to the directory containing the app's data
            query_factory (QueryFactory): The app's query factory

        Returns:
            ResourceLoader: a resource loader
        """
        query_factory = query_factory or QueryFactory.create_query_factory(app_path)
        return ResourceLoader(app_path, query_factory)

    @staticmethod
    def _hash_file(filename, algorithm='sha1'):
        """Creates a hash of the file

        Args:
            filename (str): The path of a file to hash.
            algorithm (str, optional): The hashing algorithm to use. Defaults
                to sha1. See `hashlib.algorithms_available` for a list of
                options.

        Returns:
            str: A hex digest of the files hash
        """
        hash_obj = hashlib.new(algorithm)
        with open(filename, 'rb') as file_p:
            while True:
                buf = file_p.read(4096)
                if not buf:
                    break
                buf_hash = hashlib.new(algorithm)
                buf_hash.update(buf)
                hash_obj.update(buf_hash.hexdigest().encode('utf-8'))
        return hash_obj.hexdigest()

    # resource loader map
    FEATURE_RSC_MAP = {
        GAZETTEER_RSC: get_gazetteers,
        WORD_FREQ_RSC: _build_word_freq_dict,
        QUERY_FREQ_RSC: _build_query_freq_dict,
        SYS_TYPES_RSC: _get_sys_entity_types
    }

    RSC_HASH_MAP = {
        GAZETTEER_RSC: get_gazetteers_hash,
        # the following resources only vary based on the queries used for training
        WORD_FREQ_RSC: lambda _: 'constant',
        QUERY_FREQ_RSC: lambda _: 'constant',
        SYS_TYPES_RSC: lambda _: 'constant'
    }


class _StringHasher(object):

    def __init__(self, algorithm='sha1'):
        self._algorithm = algorithm
        # TODO consider alternative data structure so cache doesnt get too big
        self._cache = {}

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        if value != self._algorithm:
            # reset cache when changing algorithm
            self._cache = {}
            self._algorithm = value

    def hash(self, string):
        if string in self._cache:
            return self._cache[string]

        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(string.encode('utf8'))
        result = hash_obj.hexdigest()
        self._cache[string] = result
        return result
