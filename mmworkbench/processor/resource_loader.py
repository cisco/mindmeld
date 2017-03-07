# -*- coding: utf-8 -*-
"""
This module contains the processor resource loader.
"""

from __future__ import unicode_literals
from builtins import object

from copy import deepcopy
import fnmatch
import json
import logging
import os
import time

from .. import markup, path
from ..exceptions import FileNotFoundError

from .gazetteer import Gazetteer

logger = logging.getLogger(__name__)

QUERY_SETS = {
    'train': 'train*.txt'
}


class ResourceLoader(object):

    def __init__(self, app_path, query_factory):
        self.app_path = app_path
        self.query_factory = query_factory

        self.gazetteers = {}
        self.labeled_query_files = {}
        self.entity_maps = {}

    def get_gazetteers(self):
        """Gets all gazetteers

        Returns:
            dict: Gazetteer data keyed by entity type
        """
        # TODO: get role gazetteers
        entity_types = path.get_entity_types(self.app_path)
        return {entity_type: self.get_gazetteer(entity_type)
                for entity_type in entity_types}

    def get_gazetteer(self, gaz_name):
        """Gets a gazetteers by name

        Args:
            gaz_name (str): The name of the entity

        Returns:
            dict: Gazetteer data
        """
        # TODO: get role gazetteers
        if gaz_name not in self.gazetteers:
            gaz = Gazetteer(gaz_name)
            gaz_path = path.get_gazetteer_data_path(self.app_path, gaz_name)

            try:
                gaz.load(gaz_path)
                self.gazetteers[gaz_name] = gaz.to_dict()
            except FileNotFoundError:
                self.build_gazetteer(gaz_name)

        return self.gazetteers[gaz_name]

    def build_gazetteer(self, gaz_name, exclude_ngrams=False):
        POPULARITY_CUTOFF = 0.0

        logger.info("Building gazetteer '{}'".format(gaz_name))

        # TODO: support role gazetteers
        gaz = Gazetteer(gaz_name, exclude_ngrams)

        entity_data_path = path.get_entity_gaz_path(self.app_path, gaz_name)
        gaz.update_with_entity_data_file(entity_data_path, POPULARITY_CUTOFF,
                                         self.query_factory.normalize)

        mapping = self.get_entity_map(gaz_name)
        gaz.update_with_entity_map(mapping, self.query_factory.normalize)

        gaz_path = path.get_gazetteer_data_path(self.app_path, gaz_name)
        gaz.dump(gaz_path)

        self.gazetteers[gaz_name] = gaz.to_dict()

    def get_entity_map(self, entity_type, force_reload=False):
        file_path = path.get_entity_map_path(self.app_path, entity_type)
        last_modified = os.path.getmtime(file_path)
        never_loaded = entity_type not in self.entity_maps

        if force_reload or never_loaded or self.entity_maps[entity_type]['loaded'] < last_modified:
            # file is out of date, load it
            self.load_entity_map(entity_type)
        return deepcopy(self.entity_maps[entity_type]['mapping'])

    def load_entity_map(self, entity_type):
        """Loads an entity map

        Args:
            entity_type (str): The type of entity to load a mapping
        """
        file_path = path.get_entity_map_path(self.app_path, entity_type)
        logger.debug("Loading entity map from file '{}'".format(file_path))
        if not os.path.isfile(file_path):
            raise ValueError("Entity map file was not found at '{}'".format(file_path))
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)

        if entity_type not in self.entity_maps:
            self.entity_maps[entity_type] = {}
        self.entity_maps[entity_type]['mapping'] = json_data
        self.entity_maps[entity_type]['loaded'] = time.time()

    def get_labeled_queries(self, domain=None, intent=None, query_set='train', force_reload=False):
        """Gets labeled queries from the cache, or loads them from disk.

        Args:
            domain (str): The domain of queries to load
            intent (str): The intent of queries to load
            force_reload (bool): Will not load queries from the cache when True

        Returns:
            dict: ProcessedQuery objects loaded from labeled query files, organized by domain and
                intent.
        """
        query_tree = {}
        file_iter = self._traverse_labeled_queries_files(domain, intent, query_set, force_reload)
        for domain, intent, filename in file_iter:
            if domain not in query_tree:
                query_tree[domain] = {}

            if intent not in query_tree[domain]:
                query_tree[domain][intent] = []
            queries = query_tree[domain][intent]
            queries.extend(self.labeled_query_files[domain][intent][filename]['queries'])

        return query_tree

    def _traverse_labeled_queries_files(self, domain=None, intent=None, query_set='train',
                                        force_reload=False):
        try:
            file_pattern = QUERY_SETS[query_set]
        except KeyError:
            raise ValueError("Unknown query set '{}'".format(query_set))
        self._update_query_file_dates(file_pattern)

        domains = [domain] if domain else self.labeled_query_files.keys()

        for domain in domains:
            intents = [intent] if intent else self.labeled_query_files[domain].keys()
            for intent in intents:
                files = self.labeled_query_files[domain][intent].keys()
                # filter to files which belong to the query set
                files = fnmatch.filter(files, file_pattern)
                files.sort()
                for filename in files:
                    file = self.labeled_query_files[domain][intent][filename]
                    if force_reload or (not file['loaded'] or file['loaded'] < file['modified']):
                        # file is out of date, load it
                        self.load_query_file(domain, intent, filename)
                    yield domain, intent, filename

    @staticmethod
    def flatten_query_tree(query_tree):
        flattened = []
        for domain, intent_queries in query_tree.items():
            for intent, queries in intent_queries.items():
                for query in queries:
                    flattened.append(query)
        return flattened

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

    def load_query_file(self, domain, intent, filename):
        """Loads the queries from the specified file

        Args:
            domain (str): The domain of the query file
            intent (str): The intent of the query file
            filename (str): The name of the query file

        """
        logger.info("Loading queries from file {}/{}/{}".format(domain, intent, filename))

        file_path = path.get_labeled_query_file_path(self.app_path, domain, intent, filename)
        queries = []
        import codecs
        with codecs.open(file_path, encoding='utf-8') as queries_file:
            for line in queries_file:
                line = line.strip()
                # only create query if line is not empty string
                query_text = line.split('\t')[0].strip()
                if query_text:
                    if query_text[0] == '-':
                        continue

                    query = markup.load_query(query_text, self.query_factory,
                                              domain=domain, intent=intent, is_gold=True)
                    queries.append(query)

        self.labeled_query_files[domain][intent][filename]['queries'] = queries
        self.labeled_query_files[domain][intent][filename]['loaded'] = time.time()
