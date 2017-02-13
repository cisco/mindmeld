# -*- coding: utf-8 -*-
"""
This module contains the processor resource loader.
"""

from __future__ import unicode_literals
from builtins import object

import fnmatch
import os

from .. import markup, path

QUERY_SETS = {
    'train': '*train*.txt'
}


class ResourceLoader(object):

    def __init__(self, app_path, tokenizer, preprocessor):
        self.app_path = app_path
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

        self.gazetteers = {}
        self.labeled_query_files = {}

    def get_gazetteers(self, domain, gazeteer_names=None):
        pass

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
        try:
            file_pattern = QUERY_SETS[query_set]
        except KeyError:
            raise ValueError("Unknown query set '{}'".format(query_set))
        self._update_query_file_dates(file_pattern)

        domains = [domain] if domain else self.labeled_query_files.keys()

        query_tree = {}
        for domain in domains:
            query_tree[domain] = {}
            intents = [intent] if intent else self.labeled_query_files[domain].keys()
            for intent in intents:
                queries = []
                files = self.labeled_query_files[domain][intent].keys()
                # filter to files which belong to the query set
                files = fnmatch.filter(files, file_pattern)
                for filename in files:
                    file = self.labeled_query_files[domain][intent][filename]
                    if force_reload or (not file['loaded'] or file['loaded'] < file['modified']):
                        # file is out of date, load it
                        file['queries'] = self.load_query_file(domain, intent, filename)
                    queries.extend(file['queries'])
                query_tree[domain][intent] = queries

        return query_tree

    def _update_query_file_dates(self, file_pattern):

        new_app_table = {}
        domains = path.get_domains(self.app_path)
        for domain in domains:
            new_app_table[domain] = {}
            intents = path.get_domain_intents(self.app_path, domain)
            for intent in intents:
                new_app_table[domain][intent] = {}
                files = path.get_labeled_query_files(self.app_path, domain, intent, file_pattern)
                for file in files:
                    new_app_table[domain][intent][file] = {
                        'modified': os.path.getmtime(file),
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

                    query = markup.create_processed_query(
                        query_text, self.tokenizer, self.preprocessor, domain=domain, intent=intent,
                        is_gold=True)
                    queries.append(query)

        return queries
