# -*- coding: utf-8 -*-
"""
This module contains the processor resource loader.
"""
from __future__ import absolute_import, unicode_literals
from builtins import object

from copy import deepcopy
import fnmatch
import json
import logging
import os
import time

from . import markup, path
from .exceptions import WorkbenchError
from .gazetteer import Gazetteer
from .query_factory import QueryFactory

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
        #        ''
        #     },
        #     'gazetteer': {  # for pickled gazetteer
        #        'loaded': '',  # the time the file was last loaded
        #        'modified': '',  # the time the file was last modified
        #        ''
        #     }
        #   }
        # }
        self._entity_files = {}

        # Example layout: {
        #   'domain': {
        #     'intent': {
        #        'filename': {
        #          'loaded': '',  # the time the file was last loaded
        #          'modified': '',  # the time the file was last modified
        #          'queries': []  # the queries loaded from the file
        #        }
        #     }
        #   }
        # }
        self.labeled_query_files = {}

    def get_gazetteers(self, force_reload=False):
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
        gaz.update_with_entity_map(mapping, self.query_factory.normalize)

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
            raise WorkbenchError('Entity map file was not found at {!r}'.format(file_path))
        try:
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
        except json.JSONDecodeError:
            raise WorkbenchError('Could not load entity map (Invalid JSON): {!r}'.format(file_path))

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
            logger.error('Entity data file not found at %r', entity_data_path)
            raise WorkbenchError('Entity data file not found at {!r}'.format(entity_data_path))

        # update mapping
        mapping_path = path.get_entity_map_path(self.app_path, entity_type)
        try:
            file_table['mapping']['modified'] = os.path.getmtime(mapping_path)
        except (OSError, IOError):
            # required file doesnt exist -- notify and error out
            logger.warning('Entity mapping file not found at %r', mapping_path)
            raise WorkbenchError('Entity mapping file not found at {!r}'.format(mapping_path))

        # update gaz data
        gazetteer_path = path.get_gazetteer_data_path(self.app_path, entity_type)
        try:
            file_table['gazetteer']['modified'] = os.path.getmtime(gazetteer_path)
        except (OSError, IOError):
            # gaz not yet built so set to a time impossibly long ago
            file_table['gazetteer']['modified'] = 0.0

    def get_labeled_queries(self, domain=None, intent=None, label_set=None, force_reload=False):
        """Gets labeled queries from the cache, or loads them from disk.

        Args:
            domain (str): The domain of queries to load
            intent (str): The intent of queries to load
            force_reload (bool): Will not load queries from the cache when True

        Returns:
            dict: ProcessedQuery objects loaded from labeled query files, organized by domain and
                intent.
        """
        label_set = label_set or DEFAULT_LABEL_SET
        query_tree = {}
        file_iter = self._traverse_labeled_queries_files(domain, intent, label_set, force_reload)
        for domain, intent, filename in file_iter:
            if domain not in query_tree:
                query_tree[domain] = {}

            if intent not in query_tree[domain]:
                query_tree[domain][intent] = []
            queries = query_tree[domain][intent]
            queries.extend(self.labeled_query_files[domain][intent][filename]['queries'])

        return query_tree

    @staticmethod
    def flatten_query_tree(query_tree):
        flattened = []
        for _, intent_queries in query_tree.items():
            for _, queries in intent_queries.items():
                for query in queries:
                    flattened.append(query)
        return flattened

    def _traverse_labeled_queries_files(self, domain=None, intent=None, label_set='train',
                                        force_reload=False):
        provided_intent = intent
        try:
            file_pattern = LABEL_SETS[label_set]
        except KeyError:
            raise WorkbenchError("Unknown label set '{}'".format(label_set))
        self._update_query_file_dates(file_pattern)

        domains = [domain] if domain else self.labeled_query_files.keys()

        for domain in domains:
            if provided_intent:
                intents = [provided_intent]
            else:
                intents = self.labeled_query_files[domain].keys()
            for intent in intents:
                files = self.labeled_query_files[domain][intent].keys()
                # filter to files which belong to the label set
                files = fnmatch.filter(files, file_pattern)
                files.sort()
                for filename in files:
                    file = self.labeled_query_files[domain][intent][filename]
                    if force_reload or (not file['loaded'] or file['loaded'] < file['modified']):
                        # file is out of date, load it
                        self.load_query_file(domain, intent, filename)
                    yield domain, intent, filename

    def load_query_file(self, domain, intent, filename):
        """Loads the queries from the specified file

        Args:
            domain (str): The domain of the query file
            intent (str): The intent of the query file
            filename (str): The name of the query file

        """
        logger.info("Loading queries from file {}/{}/{}".format(domain, intent, filename))
        file_path = path.get_labeled_query_file_path(self.app_path, domain, intent, filename)
        queries = markup.load_query_file(file_path, self.query_factory, domain, intent,
                                         is_gold=True)
        try:
            self._check_query_entities(queries)
        except WorkbenchError as exc:
            logger.warning(exc.message)
        self.labeled_query_files[domain][intent][filename]['queries'] = queries
        self.labeled_query_files[domain][intent][filename]['loaded'] = time.time()

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
                        'loaded': None
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
