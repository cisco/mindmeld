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
This module contains the query cache implementation.
"""
import logging
import os
import shutil

from sklearn.externals import joblib

from ._version import get_mm_version
from .path import GEN_FOLDER, QUERY_CACHE_PATH, QUERY_CACHE_TMP_PATH

logger = logging.getLogger(__name__)


class QueryCache:
    """
    An object that stores ProcessedQuery objects in memory to save time on reloading.
    ProcessedQuery objects consist of the query itself, the domain/intent classifications,
    recognized entities in the query, and more.
    """

    def __init__(self, app_path):
        self.app_path = app_path
        self.is_dirty = False
        # We initialize cached_queries to None instead of {} since
        # we want to lazy load it from disk only when necessary ie during
        # set, get and dump ops. This allows us to run the application
        # faster.
        self._cached_queries = None
        self.gen_folder = GEN_FOLDER.format(app_path=self.app_path)
        self.main_cache_location = QUERY_CACHE_PATH.format(app_path=self.app_path)
        self.tmp_cache_location = QUERY_CACHE_TMP_PATH.format(app_path=self.app_path)

    @property
    def cached_queries(self):
        """A dictionary containing all the cached queries"""
        if self._cached_queries is None:
            self.load()

        return self._cached_queries

    @property
    def versioned_data(self):
        """A dictionary containing the MindMeld version in addition to any cached queries."""
        return {"mm_version": get_mm_version(), "cached_queries": self.cached_queries}

    def set_value(self, domain, intent, query_text, processed_query):
        """
        Set value for the corresponding argument parameters.

        Args:
            domain (str): The domain
            intent (str): The intent
            query_text (str): The query text
            processed_query (ProcessedQuery): The ProcessedQuery \
                object corresponding to the domain, intent and query_text
        """
        if (domain, intent, query_text) in self.cached_queries:
            return

        self.cached_queries[(domain, intent, query_text)] = processed_query
        self.is_dirty = True

    def get_value(self, domain, intent, query_text):
        """
        Gets the value associated with the triplet key (domain, intent, query_text).

        Args:
            domain (str): The domain
            intent (str): The intent
            query_text (str): The query text
        """
        try:
            return self.cached_queries[(domain, intent, query_text)]
        except KeyError:
            return

    def dump(self):
        """
        This function dumps the query cache mapping to disk. This operation is expensive,
        so use it sparingly!
        """
        if not self.is_dirty:
            return

        # make generated directory if necessary
        if not os.path.isdir(self.gen_folder):
            os.makedirs(self.gen_folder)

        try:
            # We write to a new cache temp file and then rename it to prevent file corruption
            # due to the user cancelling the training operation midway during the
            # file write.
            joblib.dump(self.versioned_data, self.tmp_cache_location)
            if os.path.isfile(self.main_cache_location):
                os.remove(self.main_cache_location)
            shutil.move(self.tmp_cache_location, self.main_cache_location)
            self.is_dirty = False
        except (OSError, IOError, KeyboardInterrupt):
            if os.path.exists(self.main_cache_location):
                os.remove(self.main_cache_location)

            if os.path.exists(self.tmp_cache_location):
                os.remove(self.tmp_cache_location)

            logger.error(
                "Couldn't dump query cache to disk properly, "
                "so deleting query cache due to possible corruption."
            )

    def load(self):
        """
        Loads a generated query cache into memory.
        """
        file_location = QUERY_CACHE_PATH.format(app_path=self.app_path)
        try:
            versioned_data = joblib.load(file_location)
            if "cached_queries" not in versioned_data:
                # The old version of caching did not have versions
                logger.warning(
                    "The cache contains deprecated versions of queries. Please "
                    "run this command to clear the query cache: "
                    '"python -m <app_name> clean -q"'
                )
                self._cached_queries = versioned_data
            else:
                self._cached_queries = versioned_data["cached_queries"]
        except (OSError, IOError, KeyboardInterrupt):
            self._cached_queries = {}
        self.is_dirty = False
