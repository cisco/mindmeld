# -*- coding: utf-8 -*-
"""
This module contains the query cache implementation
"""
import os
import shutil
import logging
from sklearn.externals import joblib

from .path import QUERY_CACHE_PATH, QUERY_CACHE_TMP_PATH, GEN_FOLDER

logger = logging.getLogger(__name__)


class QueryCache:
    def __init__(self, app_path):
        self.app_path = app_path
        self.is_dirty = False
        # We initialize cached_queries to None instead of {} since
        # we want to lazy load it from disk only when necessary ie during
        # set, get and dump ops. This allows us to run the application
        # faster.
        self._cached_queries = None

    @property
    def cached_queries(self):
        if self._cached_queries is None:
            self.load()

        return self._cached_queries

    def set_value(self, domain, intent, query_text, processed_query):
        """
        Set value for the corresponding argument parameters
        Args:
            domain (str): The domain
            intent (str): The intent
            query_text (str): The query text
            processed_query (ProcessedQuery): The ProcessedQuery
                object corresponding to the domain, intent and query_text
        """
        if (domain, intent, query_text) in self.cached_queries:
            return

        self.cached_queries[(domain, intent, query_text)] = processed_query
        self.is_dirty = True

    def get_value(self, domain, intent, query_text):
        """
        Gets the value associated with the triple key
        """
        try:
            return self.cached_queries[(domain, intent, query_text)]
        except KeyError:
            return

    def dump(self):
        """
        This function dumps the query cache mapping to disk. THIS OPERATION IS EXPENSIVE,
        SO USE IT SPARINGLY!
        """
        if not self.is_dirty:
            return

        # make generated directory if necessary
        folder = GEN_FOLDER.format(app_path=self.app_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        main_cache_location = QUERY_CACHE_PATH.format(app_path=self.app_path)
        tmp_cache_location = QUERY_CACHE_TMP_PATH.format(app_path=self.app_path)

        try:
            if os.path.isfile(main_cache_location):
                # We write to a new cache temp file and then rename it to prevent file corruption
                # due to the user cancelling the training operation midway during the
                # file write.
                joblib.dump(self.cached_queries, tmp_cache_location)
                os.remove(main_cache_location)
                shutil.move(tmp_cache_location, main_cache_location)
            else:
                joblib.dump(self.cached_queries, main_cache_location)

            self.is_dirty = False
        except (OSError, IOError, KeyboardInterrupt):
            if os.path.exists(main_cache_location):
                os.remove(main_cache_location)

            if os.path.exists(tmp_cache_location):
                os.remove(tmp_cache_location)

            logger.error("Couldn't dump query cache to disk properly, "
                         "so deleting cache due to possible corruption.")

    def load(self):
        """
        Loads the generated query cache into memory
        """
        file_location = QUERY_CACHE_PATH.format(app_path=self.app_path)
        try:
            self._cached_queries = joblib.load(file_location)
        except (OSError, IOError, KeyboardInterrupt):
            self._cached_queries = {}
        self.is_dirty = False
