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
        self.query_cache_dict = {}
        self.load()

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
        if (domain, intent, query_text) in self.query_cache_dict:
            return

        self.query_cache_dict[(domain, intent, query_text)] = processed_query
        self.is_dirty = True

    def get_value(self, domain, intent, query_text):
        """
        Gets the value associated with the triple key
        """
        try:
            return self.query_cache_dict[(domain, intent, query_text)]
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

        try:
            file_location = QUERY_CACHE_PATH.format(app_path=self.app_path)
            if os.path.isfile(file_location):
                # We write to a new cache temp file and then rename it to prevent file corruption
                # due to the user cancelling the training operation midway during the
                # file write.
                file_location_tmp = QUERY_CACHE_TMP_PATH.format(app_path=self.app_path)
                joblib.dump(self.query_cache_dict, file_location_tmp)
                os.remove(file_location)
                shutil.move(file_location_tmp, file_location)
            else:
                joblib.dump(self.query_cache_dict, file_location)
            self.is_dirty = False
        except (OSError, IOError):
            logger.error("Couldn't dump query cache to disk.")

    def load(self):
        """
        Loads the generated query cache into memory
        """
        file_location = QUERY_CACHE_PATH.format(app_path=self.app_path)
        try:
            self.query_cache_dict = joblib.load(file_location)
        except (OSError, IOError):
            pass
        self.is_dirty = False
