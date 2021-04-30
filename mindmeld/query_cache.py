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
from hashlib import md5
import json
import logging
import os
import shutil
import sqlite3

from sklearn.externals import joblib

from ._version import get_mm_version
from .path import GEN_FOLDER, QUERY_CACHE_DB_PATH
from .core import ProcessedQuery

logger = logging.getLogger(__name__)

class QueryCache:
    """
    An object that stores ProcessedQuery objects in memory to save time on reloading.
    ProcessedQuery objects consist of the query itself, the domain/intent classifications,
    recognized entities in the query, and more.
    """

    def __init__(self, app_path):
        # make generated directory if necessary
        gen_folder = GEN_FOLDER.format(app_path=app_path)
        if not os.path.isdir(gen_folder):
            os.makedirs(gen_folder)

        db_file_location = QUERY_CACHE_DB_PATH.format(app_path=app_path)
        self.connection = sqlite3.connect(db_file_location)
        cursor = self.connection.cursor()

        if not self.compatible_version():
            cursor.execute('''
            DROP TABLE IF EXISTS queries, version;
            ''')
        # Create table to store queries
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries
        (hash_id TEXT PRIMARY KEY, query TEXT)
        WITHOUT ROWID;
        ''')
        # Create table to store the data version
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS version
        (version_string TEXT PRIMARY KEY)
        WITHOUT ROWID;
        ''')
        cursor.execute('''
        INSERT OR IGNORE INTO version values (?);
        ''', (get_mm_version(),))
        self.connection.commit()


    def compatible_version(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute('''
            SELECT version_string FROM version WHERE version_string=(?)
            ''', (get_mm_version(),))
            if len(cursor.fetchall()) == 0:
                # version does not match
                return False
        except Exception:
            pass
        return True


    @staticmethod
    def get_key(domain, intent, query_text):
        h = md5(domain.encode())
        h.update(intent.encode())
        h.update(query_text.encode())
        return h.hexdigest()


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
        key = self.get_key(domain, intent, query_text)
        cursor = self.connection.cursor()
        cursor.execute('''
        INSERT OR IGNORE into queries values (?, ?)
        ''', (key, json.dumps(processed_query.to_cache())))
        self.connection.commit()


    def get_value(self, domain, intent, query_text):
        """
        Gets the value associated with the triplet key (domain, intent, query_text).

        Args:
            domain (str): The domain
            intent (str): The intent
            query_text (str): The query text
        """
        key = self.get_key(domain, intent, query_text)
        cursor = self.connection.cursor()
        cursor.execute('''
        SELECT query FROM queries WHERE hash_id=(?);
        ''', (key,))
        row = cursor.fetchone()
        if row:
            return ProcessedQuery.from_cache(json.loads(row[0]))
        return None

    def dump(self):
        """
        This function dumps the query cache mapping to disk. This operation is expensive,
        so use it sparingly!
        """
        pass

    def load(self):
        pass
