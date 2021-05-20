# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the processor resource loader.
"""
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from copy import deepcopy

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from . import markup, path
from .constants import DEFAULT_TRAIN_SET_REGEX
from .core import Entity
from .exceptions import MindMeldError
from .gazetteer import Gazetteer
from .models.helpers import (CHAR_NGRAM_FREQ_RSC, ENABLE_STEMMING, GAZETTEER_RSC, QUERY_FREQ_RSC,
                             SENTIMENT_ANALYZER, SYS_TYPES_RSC, WORD_FREQ_RSC, WORD_NGRAM_FREQ_RSC,
                             mask_numerics)
from .path import MODEL_CACHE_PATH
from .query_cache import QueryCache
from .query_factory import QueryFactory

logger = logging.getLogger(__name__)


class ProcessedQueryList:
    """
    ProcessedQueryList provides a memory efficient disk backed list representation
    for a list of queries.
    """
    def __init__(self, cache=None, elements=None):
        self._cache = cache
        self.elements = elements or []

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    def append(self, query_id):
        self.elements.append(query_id)

    def extend(self, query_ids):
        self.elements.extend(query_ids)

    def __len__(self):
        return len(self.elements)

    def __bool__(self):
        return bool(len(self))

    def processed_queries(self):
        return ProcessedQueryList.Iterator(self)

    def raw_queries(self):
        return ProcessedQueryList.RawQueryIterator(self)

    def queries(self):
        return ProcessedQueryList.QueryIterator(self)

    def entities(self):
        return ProcessedQueryList.EntitiesIterator(self)

    def domains(self):
        return ProcessedQueryList.DomainIterator(self)

    def intents(self):
        return ProcessedQueryList.IntentIterator(self)

    @staticmethod
    def from_in_memory_list(queries):
        """
        Creates a ProcessedQueryList wrapper around an
        in-memory list of ProcessedQuery objects

        Args:
            queries (list(ProcessedQuery)): queries to wrap
        Returns:
            ProcessedQueryList object
        """
        return ProcessedQueryList(
            cache=ProcessedQueryList.MemoryCache(queries),
            elements=list(range(len(queries)))
        )

    class Iterator:
        def __init__(self, source, cached=False):
            self.source = source
            self.elements = source.elements
            self.result_cache = [] if not cached else [None] * len(source)
            self.iter_idx = -1

        def __iter__(self):
            self.iter_idx = -1
            return self

        def __next__(self):
            self.iter_idx += 1
            if self.iter_idx == len(self):
                raise StopIteration
            if self.result_cache and self.result_cache[self.iter_idx] is not None:
                return self.result_cache[self.iter_idx]
            result = self[self.iter_idx]
            if self.result_cache:
                self.result_cache[self.iter_idx] = result
            return result

        def __len__(self):
            return len(self.elements)

        def __getitem__(self, key):
            """
            Override this function for getting other types of cached data.
            """
            return self.source.cache.get(self.elements[key])

        def reorder(self, indices):
            self.elements = [self.elements[i] for i in indices]
            if self.result_cache:
                self.result_cache = [self.result_cache[i] for i in indices]

    class RawQueryIterator(Iterator):
        def __getitem__(self, key):
            return self.source.cache.get_raw_query(self.elements[key])

    class QueryIterator(Iterator):
        def __getitem__(self, key):
            return self.source.cache.get_query(self.elements[key])

    class EntitiesIterator(Iterator):
        def __getitem__(self, key):
            return self.source.cache.get_entities(self.elements[key])

    class DomainIterator(Iterator):
        def __init__(self, source):
            # Caches the elements in memory if they are backed by sqlite3
            cached = not isinstance(source.cache, ProcessedQueryList.MemoryCache)
            super().__init__(source, cached=cached)

        def __getitem__(self, key):
            return self.source.cache.get_domain(self.elements[key])

    class IntentIterator(Iterator):
        def __init__(self, source):
            # Caches the elements in memory if they are backed by sqlite3
            cached = not isinstance(source.cache, ProcessedQueryList.MemoryCache)
            super().__init__(source, cached=cached)

        def __getitem__(self, key):
            return self.source.cache.get_intent(self.elements[key])

    class ListIterator(Iterator):
        """
        ListIterator is a wrapper around a in memory list that supports the same
        functionality as a ProcessedQueryList.Iterator.  This allows building
        of arbitrary lists of data and presenting them as a
        ProcessedQueryList.Iterator to functions that require them.
        """
        def __init__(self, elements):
            self.source = None
            self.elements = elements
            self.result_cache = None
            self.iter_idx = -1

        def __getitem__(self, key):
            return self.elements[key]

    class MemoryCache:
        """
        A class to provide cache functionality for in-memory
        lists of ProcessedQuery objects
        """
        def __init__(self, queries):
            self.queries = queries

        def get(self, row_id):
            return self.queries[row_id]

        def get_raw_query(self, row_id):
            return self.get(row_id).query.text

        def get_query(self, row_id):
            return self.get(row_id).query

        def get_entities(self, row_id):
            return self.get(row_id).entities

        def get_domain(self, row_id):
            return self.get(row_id).domain

        def get_intent(self, row_id):
            return self.get(row_id).intent


class ResourceLoader:
    """ResourceLoader objects are responsible for loading resources necessary for nlp components
    (classifiers, entity recognizer, parsers, etc).

    Note: we need to keep resource helpers as instance methods, as ``load_feature_resource``
    assumes all helpers to be instance methods.
    """

    def __init__(self, app_path, query_factory, query_cache=None):
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
        #       }
        #     }
        #   }
        # }
        self.file_to_query_info = {}
        self._hasher = Hasher()
        self.query_cache = query_cache or QueryCache(app_path=self.app_path)
        self._hash_to_model_path = None

    @property
    def hash_to_model_path(self):
        """dict: A dictionary that maps hashes to the file path of the classifier."""
        if self._hash_to_model_path is None:
            self._load_cached_models()
        return self._hash_to_model_path

    def get_gazetteers(self, force_reload=False):
        """Gets gazetteers for all entities.

        Returns:
            dict: Gazetteer data keyed by entity type
        """
        # TODO: get role gazetteers
        entity_types = path.get_entity_types(self.app_path)
        return {
            entity_type: self.get_gazetteer(entity_type, force_reload=force_reload)
            for entity_type in entity_types
        }

    def get_gazetteer(self, gaz_name, force_reload=False):
        """Gets a gazetteers by name.

        Args:
            gaz_name (str): The name of the entity the gazetteer corresponds to

        Returns:
            dict: Gazetteer data
        """
        self._update_entity_file_dates(gaz_name)
        # TODO: get role gazetteers
        if self._gaz_needs_build(gaz_name) or force_reload:
            self.build_gazetteer(gaz_name, force_reload=force_reload)
        if self._entity_file_needs_load("gazetteer", gaz_name):
            self.load_gazetteer(gaz_name)

        return self._entity_files[gaz_name]["gazetteer"]["data"]

    def get_tokenizer(self):
        """Get the tokenizer from the query_factory attribute

        Returns:
            tokenizer (Tokenizer): The resource loaders tokenizer
        """
        return self.query_factory.tokenizer

    @staticmethod
    def get_sentiment_analyzer():
        """
        Returns a sentiment analyzer and downloads the necessary data libraries required from nltk
        """
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            logger.info("Downloading lexicon for sentiment analysis")
            nltk.download("vader_lexicon")

        return SentimentIntensityAnalyzer()

    def get_gazetteers_hash(self):
        """
        Gets a single hash of all the gazetteer ordered by alphabetical entity type.

        Returns:
            str: Hash of a list of gazetteer hashes.
        """
        entity_types = path.get_entity_types(self.app_path)
        return self._hasher.hash_list(
            (
                self.get_gazetteer_hash(entity_type)
                for entity_type in sorted(entity_types)
            )
        )

    def get_gazetteer_hash(self, gaz_name):
        """
        Gets the hash of a gazetteer by entity name.

        Args:
            gaz_name (str): The name of the entity the gazetteer corresponds to

        Returns:
            str: Hash of a gazetteer specified by name.
        """
        self._update_entity_file_dates(gaz_name)
        entity_data_path = path.get_entity_gaz_path(self.app_path, gaz_name)
        entity_data_hash = self._hasher.hash_file(entity_data_path)

        mapping_path = path.get_entity_map_path(self.app_path, gaz_name)
        mapping_hash = self._hasher.hash_file(mapping_path)

        return self._hasher.hash_list([entity_data_hash, mapping_hash])

    def build_gazetteer(self, gaz_name, exclude_ngrams=False, force_reload=False):
        """Builds the specified gazetteer using the entity data and mapping files.

        Args:
            gaz_name (str): The name of the entity the gazetteer corresponds to
            exclude_ngrams (bool, optional): Whether partial matches of
                 entities should be included in the gazetteer
            force_reload (bool, optional): Whether file should be forcefully
                 reloaded from disk
        """
        popularity_cutoff = 0.0

        logger.info("Building gazetteer '%s'", gaz_name)

        # TODO: support role gazetteers
        gaz = Gazetteer(gaz_name, self.get_tokenizer(), exclude_ngrams)

        entity_data_path = path.get_entity_gaz_path(self.app_path, gaz_name)
        gaz.update_with_entity_data_file(
            entity_data_path, popularity_cutoff, self.query_factory.normalize
        )
        self._entity_files[gaz_name]["entity_data"]["loaded"] = time.time()

        mapping = self.get_entity_map(gaz_name, force_reload=force_reload)
        gaz.update_with_entity_map(
            mapping.get("entities", []), self.query_factory.normalize
        )

        gaz_path = path.get_gazetteer_data_path(self.app_path, gaz_name)
        gaz.dump(gaz_path)

        self._entity_files[gaz_name]["gazetteer"]["data"] = gaz.to_dict()
        self._entity_files[gaz_name]["gazetteer"]["loaded"] = time.time()

    def load_gazetteer(self, gaz_name):
        """
        Loads a gazetteer specified by the entity name.

        Args:
            gaz_name (str): The name of the entity the gazetteer corresponds to
        """
        gaz = Gazetteer(gaz_name, self.get_tokenizer())
        gaz_path = path.get_gazetteer_data_path(self.app_path, gaz_name)
        gaz.load(gaz_path)
        self._entity_files[gaz_name]["gazetteer"]["data"] = gaz.to_dict()
        self._entity_files[gaz_name]["gazetteer"]["loaded"] = time.time()

    def get_entity_map(self, entity_type, force_reload=False):
        """Creates a mapping file for a given entity.

        Args:
            entity_type (str): The name of the entity
        """
        self._update_entity_file_dates(entity_type)
        if self._entity_file_needs_load("mapping", entity_type) or force_reload:
            # file is out of date, load it
            self.load_entity_map(entity_type)
        return deepcopy(self._entity_files[entity_type]["mapping"]["data"])

    def load_entity_map(self, entity_type):
        """Loads an entity mapping file.

        Args:
            entity_type (str): The name of the entity
        """
        file_path = path.get_entity_map_path(self.app_path, entity_type)
        logger.debug("Loading entity map from file '%s'", file_path)
        if not os.path.isfile(file_path):
            logger.warning("Entity map file not found at %s", file_path)
            json_data = {}
        else:
            try:
                with open(file_path, "r") as json_file:
                    json_data = json.load(json_file)
            except json.JSONDecodeError as e:
                raise MindMeldError(
                    "Could not load entity map (Invalid JSON): {!r}".format(file_path)
                ) from e

        self._entity_files[entity_type]["mapping"]["data"] = json_data
        self._entity_files[entity_type]["mapping"]["loaded"] = time.time()

    def _load_cached_models(self):
        if not self._hash_to_model_path:
            self._hash_to_model_path = {}

        cache_path = MODEL_CACHE_PATH.format(app_path=self.app_path)
        for dir_path, _, file_names in os.walk(cache_path):
            for filename in [f for f in file_names if f.endswith(".hash")]:
                file_path = os.path.join(dir_path, filename)
                hash_val = open(file_path, "r").read()
                classifier_file_path = file_path.split(".hash")[0]
                if not os.path.exists(classifier_file_path):
                    logger.warning("Could not find the serialized model")
                    continue
                self._hash_to_model_path[hash_val] = classifier_file_path

    def _gaz_needs_build(self, gaz_name):
        try:
            build_time = self._entity_files[gaz_name]["gazetteer"]["modified"]
        except KeyError:
            # gazetteer hasn't been built
            return True

        # We changed the value type for the pop_dict dict, so we check to make sure its a tuple
        pop_dict = self._entity_files[gaz_name]['gazetteer'].get('data', {}).get('pop_dict')
        if pop_dict and not all(isinstance(elem, tuple) for elem in list(pop_dict.keys())):
            return True

        data_modified = self._entity_files[gaz_name]["entity_data"]["modified"]
        mapping_modified = self._entity_files[gaz_name]["mapping"]["modified"]

        # If entity data or mapping modified after gaz build -> stale
        return build_time < data_modified or build_time < mapping_modified

    def _entity_file_needs_load(self, file_type, entity_type):
        mod_time = self._entity_files[entity_type][file_type]["modified"]
        try:
            load_time = self._entity_files[entity_type][file_type]["loaded"]
        except KeyError:
            # file hasn't been loaded
            return True
        return load_time < mod_time

    def _update_entity_file_dates(self, entity_type):
        # TODO: handle deleted entity
        file_table = self._entity_files.get(
            entity_type, {"entity_data": {}, "mapping": {}, "gazetteer": {}}
        )
        self._entity_files[entity_type] = file_table

        # update entity data
        entity_data_path = path.get_entity_gaz_path(self.app_path, entity_type)
        try:
            file_table["entity_data"]["modified"] = os.path.getmtime(entity_data_path)
        except (OSError, IOError):
            # required file doesnt exist -- notify and error out
            logger.warning(
                "Entity data file not found at %r. "
                "Proceeding with empty entity data.",
                entity_data_path,
            )

            # mark the modified time as now if the entity file does not exist
            # so that the gazetteer gets rebuilt properly.
            file_table["entity_data"]["modified"] = time.time()

        # update mapping
        mapping_path = path.get_entity_map_path(self.app_path, entity_type)
        try:
            file_table["mapping"]["modified"] = os.path.getmtime(mapping_path)
        except (OSError, IOError):
            # required file doesnt exist
            logger.warning(
                "Entity mapping file not found at %r. "
                "Proceeding with empty entity data.",
                mapping_path,
            )

            # mark the modified time as now if the entity mapping file does not exist
            # so that the gazetteer gets rebuilt properly.
            file_table["mapping"]["modified"] = time.time()

        # update gaz data
        gazetteer_path = path.get_gazetteer_data_path(self.app_path, entity_type)
        try:
            file_table["gazetteer"]["modified"] = os.path.getmtime(gazetteer_path)
        except (OSError, IOError):
            # gaz not yet built so set to a time impossibly long ago
            file_table["gazetteer"]["modified"] = 0.0

    def get_labeled_queries(
        self, domain=None, intent=None, label_set=None, force_reload=False
    ):
        """Gets labeled queries from the cache, or loads them from disk.

        Args:
            domain (str): The domain of queries to load
            intent (str): The intent of queries to load
            force_reload (bool): Will not load queries from the cache when True

        Returns:
            dict: ProcessedQuery objects (or strings) loaded from labeled query files, organized by
                domain and intent.
        """
        label_set = label_set or DEFAULT_TRAIN_SET_REGEX
        query_tree = {}
        file_iter = self._traverse_labeled_queries_files(domain, intent, label_set)
        for a_domain, an_intent, filename in file_iter:
            file_info = self.file_to_query_info[filename]
            if force_reload or (
                not file_info["loaded"]
                or file_info["loaded"] < file_info["modified"]
            ):
                # file is out of date, load it
                self.load_query_file(a_domain, an_intent, filename)

            if a_domain not in query_tree:
                query_tree[a_domain] = {}

            if an_intent not in query_tree[a_domain]:
                query_tree[a_domain][an_intent] = ProcessedQueryList(self.query_cache)
            queries = query_tree[a_domain][an_intent]
            queries.extend(file_info["query_ids"])
        return query_tree

    @staticmethod
    def flatten_query_tree(query_tree):
        """
        Takes a query tree and returns the elements in list form.

        Args:
            query_tree (dict): A nested dictionary that organizes queries by domain then intent.

        Returns:
            list: A list of Query objects.
        """
        flattened = ProcessedQueryList()
        for _, intent_queries in query_tree.items():
            for _, queries in intent_queries.items():
                if not flattened.cache:
                    flattened.cache = queries.cache
                elif flattened.cache != queries.cache:
                    logger.error("query_tree is built from incompatible query_caches")
                    raise ValueError("query_tree is built from incompatible query_caches")
                flattened.extend(queries.elements)
        return flattened

    def get_flattened_label_set(
        self, domain=None, intent=None, label_set=None, force_reload=False
    ):
        return self.flatten_query_tree(
            self.get_labeled_queries(domain, intent, label_set, force_reload)
        )

    def _traverse_labeled_queries_files(
        self, domain=None, intent=None, file_pattern=DEFAULT_TRAIN_SET_REGEX
    ):
        provided_intent = intent
        query_tree = path.get_labeled_query_tree(self.app_path)
        self._update_query_file_dates(query_tree)
        domains = [domain] if domain else query_tree.keys()

        for a_domain in sorted(domains):
            if provided_intent:
                intents = [provided_intent]
            else:
                intents = query_tree[a_domain].keys()
            for an_intent in sorted(intents):
                files = query_tree[a_domain][an_intent].keys()
                # filter to files which belong to the label set
                for file_path in sorted(files):
                    if re.match(file_pattern, os.path.basename(file_path)):
                        yield a_domain, an_intent, file_path

    def get_all_file_paths(self, file_pattern=".*.txt"):
        """ Get a list of text file paths across all intents.

        Returns:
            list: A list of all file paths.
        """
        file_iter = self._traverse_labeled_queries_files(file_pattern=file_pattern)
        return [filename for _, _, filename in file_iter]

    def filter_file_paths(self, compiled_pattern, file_paths=None):
        """ Get a list of file paths that match a specific file_pattern

        Args:
            compiled_pattern (sre.SRE_Pattern): A compiled regex pattern to filter with.
            file_paths (list): A list of file paths.

        Returns:
            list: A list of file paths.
        """
        all_file_paths = file_paths or self.get_all_file_paths()
        matched_paths = []
        for file_path in all_file_paths:
            m = compiled_pattern.match(file_path)
            if m:
                matched_paths.append(m.group())
        if len(matched_paths) == 0:
            logger.warning("No matches were found for the given compiled pattern")
        return matched_paths

    def load_query_file(self, domain, intent, file_path):
        """Loads the queries from the specified file.

        Args:
            domain (str): The domain of the query file
            intent (str): The intent of the query file
            file_path (str): The name of the query file

        """
        logger.info("Loading queries from file %s", file_path)

        file_data = self.file_to_query_info[file_path]
        query_ids = markup.cache_query_file(
            file_path,
            self.query_cache,
            self.query_factory,
            self.app_path,
            domain,
            intent,
            is_gold=True
        )
        for _id in query_ids:
            try:
                self._check_query_entities(self.query_cache.get(_id))
            except MindMeldError as exc:
                logger.warning(exc.message)
        file_data["query_ids"] = query_ids
        file_data["loaded"] = time.time()

    def _check_query_entities(self, query):
        entity_types = path.get_entity_types(self.app_path)
        for entity in query.entities:
            if (
                entity.entity.type not in entity_types
                and not entity.entity.is_system_entity
            ):
                msg = "Unknown entity {!r} found in query {!r}"
                raise MindMeldError(
                    msg.format(entity.entity.type, query.query.text)
                )

    def _update_query_file_dates(self, query_tree):
        # We can just use this if it this is the first check
        new_query_files = {}
        for domain in query_tree:
            for intent in query_tree[domain]:
                for filename in query_tree[domain][intent]:
                    # filename needs to be full path
                    new_query_files[filename] = {
                        "modified": query_tree[domain][intent][filename],
                        "loaded": None,
                    }

        all_filenames = set(new_query_files.keys())
        all_filenames.update(self.file_to_query_info.keys())
        for filename in all_filenames:
            try:
                new_file_info = new_query_files[filename]
            except KeyError:
                # an old file that was removed
                del self.file_to_query_info[filename]
                continue

            try:
                old_file_info = self.file_to_query_info[filename]
            except KeyError:
                # a new file
                self.file_to_query_info[filename] = new_file_info
                continue

            # file existed before and now -> update
            old_file_info["modified"] = new_file_info["modified"]

    class WordFreqBuilder:
        """
        Compiles unigram frequency dictionary of normalized query tokens
        """
        def __init__(self, enable_stemming=False):
            self.enable_stemming = enable_stemming
            self.tokens = []

        def add(self, query):
            for i in range(len(query.normalized_tokens)):
                tok = query.normalized_tokens[i]
                self.tokens.append(mask_numerics(tok))
                if self.enable_stemming:
                    # We only add stemmed tokens that are not the same
                    # as the original token to reduce the impact on
                    # word frequencies
                    stemmed_tok = query.stemmed_tokens[i]
                    if stemmed_tok != tok:
                        self.tokens.append(mask_numerics(stemmed_tok))

        def get_resource(self):
            return Counter(self.tokens)

    class CharNgramFreqBuilder:
        """
        Compiles n-gram character frequency dictionary of normalized query tokens
        """
        def __init__(self, lengths, thresholds):
            self.lengths = lengths
            self.thresholds = thresholds
            self.char_freq_dict = Counter()

        def add(self, query):
            for length, threshold in zip(self.lengths, self.thresholds):
                if threshold > 0:
                    character_tokens = [
                        query.normalized_text[i : i + length]
                        for i in range(len(query.normalized_text))
                        if len(query.normalized_text[i : i + length]) == length
                    ]
                    self.char_freq_dict.update(character_tokens)

        def get_resource(self):
            return self.char_freq_dict

    class WordNgramFreqBuilder:
        """
        Compiles n-gram frequency dictionary of normalized query tokens
        """
        def __init__(self, lengths, thresholds, enable_stemming=False):
            self.lengths = lengths
            self.thresholds = thresholds
            self.enable_stemming = enable_stemming
            self.word_freq_dict = Counter()

        def add(self, query):
            for length, threshold in zip(self.lengths, self.thresholds):
                if threshold > 0:
                    ngram_tokens = []
                    for i in range(len(query.normalized_tokens)):
                        ngram_query = " ".join(query.normalized_tokens[i : i + length])
                        ngram_tokens.append(ngram_query)
                        if self.enable_stemming:
                            stemmed_ngram_query = " ".join(
                                query.stemmed_tokens[i : i + length]
                            )
                            if stemmed_ngram_query != ngram_query:
                                ngram_tokens.append(stemmed_ngram_query)
                    self.word_freq_dict.update(ngram_tokens)

        def get_resource(self):
            return self.word_freq_dict

    class QueryFreqBuilder:
        """
        Compiles frequency dictionary of normalized and stemmed query strings
        """
        def __init__(self, enable_stemming=False):
            self.enable_stemming = enable_stemming
            self.query_dict = Counter()
            self.stemmed_query_dict = Counter()

        def add(self, query):
            self.query_dict.update(["<{}>".format(query.normalized_text)])

            if self.enable_stemming:
                self.stemmed_query_dict.update(["<{}>".format(query.stemmed_text)])

        def get_resource(self):
            for query in self.query_dict:
                if self.query_dict[query] < 2:
                    self.query_dict[query] = 0

            if self.enable_stemming:
                for query in self.stemmed_query_dict:
                    if self.stemmed_query_dict[query] < 2:
                        self.stemmed_query_dict[query] = 0
            self.query_dict += self.stemmed_query_dict

            return self.query_dict

    def get_sys_entity_types(self, labels):  # pylint: disable=no-self-use
        """Get all system entity types from the entity labels.

        Args:
            labels (list of QueryEntity): a list of labeled entities
        """
        # Build entity types set
        entity_types = set()
        for label in labels:
            for entity in label:
                entity_types.add(entity.entity.type)

        return set((t for t in entity_types if Entity.is_system_entity(t)))

    def hash_feature_resource(self, name):
        """Hashes the named resource.

        Args:
            name (str): The name of the resource to hash

        Returns:
            str: The hash result
        """
        hash_func = self.RSC_HASH_MAP.get(name)
        if hash_func:
            return hash_func(self)
        else:
            raise ValueError("Invalid resource name {!r}.".format(name))

    def hash_string(self, string):
        """Hashes a string.

        Args:
            string (str): The string to hash

        Returns:
            str: The hash result
        """
        return self._hasher.hash(string)

    def hash_list(self, items):
        """Hashes the list of items.

        Args:
            items (list[str]): A list of strings to hash

        Returns:
            str: The hash result
        """
        return self._hasher.hash_list(items)

    @staticmethod
    def create_resource_loader(app_path, query_factory=None, preprocessor=None):
        """Creates the resource loader for the app at app path.

        Args:
            app_path (str): The path to the directory containing the app's data
            query_factory (QueryFactory): The app's query factory
            preprocessor (Preprocessor): The app's preprocessor

        Returns:
            ResourceLoader: a resource loader
        """
        query_factory = query_factory or QueryFactory.create_query_factory(
            app_path, preprocessor=preprocessor
        )
        query_cache = QueryCache(app_path)
        return ResourceLoader(app_path, query_factory, query_cache)

    RSC_HASH_MAP = {
        GAZETTEER_RSC: get_gazetteers_hash,
        # the following resources only vary based on the queries used for training
        WORD_FREQ_RSC: lambda _: "constant",
        WORD_NGRAM_FREQ_RSC: lambda _: "constant",
        CHAR_NGRAM_FREQ_RSC: lambda _: "constant",
        QUERY_FREQ_RSC: lambda _: "constant",
        SYS_TYPES_RSC: lambda _: "constant",
        ENABLE_STEMMING: lambda _: "constant",
        SENTIMENT_ANALYZER: lambda _: "constant",
    }


class Hasher:
    """An thin wrapper around hashlib. Uses cache for commonly hashed strings.

    Attributes:
        algorithm (str): The hashing algorithm to use. Defaults to
            'sha1'. See `hashlib.algorithms_available` for a list of
            options.

    """

    def __init__(self, algorithm="sha1"):
        # TODO consider alternative data structure so cache doesnt get too big
        self._cache = {}
        # intentionally using the setter to check value
        self._algorithm = None
        self._set_algorithm(algorithm)

    def _get_algorithm(self):
        """Getter for algorithm property.

        Returns:
            str: the hashing algorithm
        """
        return self._algorithm

    def _set_algorithm(self, value):
        """Setter for algorithm property.

        Args:
            value (str): The hashing algorithm to use. Defaults
                to sha1. See `hashlib.algorithms_available` for a list of
                options.
            value (str): The hashing algorithm to use.
        """
        if value not in hashlib.algorithms_available:
            raise ValueError("Invalid hashing algorithm: {!r}".format(value))

        if value != self._algorithm:
            # reset cache when changing algorithm
            self._cache = {}
            self._algorithm = value

    algorithm = property(_get_algorithm, _set_algorithm)

    def hash(self, string):
        """Hashes a string.

        Args:
            string (str): The string to hash

        Returns:
            str: The hash result
        """
        if string in self._cache:
            return self._cache[string]

        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(string.encode("utf8"))
        result = hash_obj.hexdigest()
        self._cache[string] = result
        return result

    def hash_list(self, strings):
        """Hashes a list of strings.

        Args:
            strings (list[str]): The strings to hash

        Returns:
            str: The hash result
        """
        hash_obj = hashlib.new(self.algorithm)
        for string in strings:
            hash_obj.update(self.hash(string).encode("utf8"))
        return hash_obj.hexdigest()

    def hash_file(self, filename):
        """Creates a hash of the file. If the file does not exist, use the empty string instead
        and return the resulting hash digest.

        Args:
            filename (str): The path of a file to hash.

        Returns:
            str: A hex digest of the file hash
        """
        hash_obj = hashlib.new(self.algorithm)
        try:
            with open(filename, "rb") as file_p:
                while True:
                    buf = file_p.read(4096)
                    if not buf:
                        break
                    hash_obj.update(buf)
        except IOError:
            hash_obj.update("".encode("utf-8"))
        return hash_obj.hexdigest()
