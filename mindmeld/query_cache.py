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
from distutils.util import strtobool
from functools import lru_cache
from hashlib import sha256
import json
import logging
import os
import sqlite3

from .path import GEN_FOLDER, QUERY_CACHE_DB_PATH
from .core import ProcessedQuery

logger = logging.getLogger(__name__)


class QueryCache:
    """
    QueryCache stores ProcessedQuerys and associated metadata in a sqlite3 backed
    cache to save processing time on reloading the examples later.
    """

    @staticmethod
    def copy_db(source, dest):
        cursor = dest.cursor()
        for statement in source.iterdump():
            cursor.execute(statement)
        dest.commit()

    def __init__(self, app_path):
        # make generated directory if necessary
        gen_folder = GEN_FOLDER.format(app_path=app_path)
        if not os.path.isdir(gen_folder):
            os.makedirs(gen_folder)

        db_file_location = QUERY_CACHE_DB_PATH.format(app_path=app_path)
        self.disk_connection = sqlite3.connect(db_file_location)
        self.batch_write_size = int(os.environ.get("MM_QUERY_CACHE_WRITE_SIZE", "1000"))

        cursor = self.disk_connection.cursor()

        if not self.compatible_version():
            cursor.execute("""
            DROP TABLE IF EXISTS queries;
            """)
            cursor.execute("""
            DROP TABLE IF EXISTS version;
            """)
        # Create table to store queries
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries
        (hash_id TEXT PRIMARY KEY, query TEXT, raw_query TEXT, domain TEXT, intent TEXT);
        """)
        # Create table to store the data version
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS version
        (version_number INTEGER PRIMARY KEY);
        """)
        cursor.execute("""
        INSERT OR IGNORE INTO version values (?);
        """, (ProcessedQuery.version,))
        self.disk_connection.commit()

        in_memory = bool(strtobool(os.environ.get("MM_QUERY_CACHE_IN_MEMORY", "1").lower()))

        if in_memory:
            logger.info("Loading query cache into memory")
            self.memory_connection = sqlite3.connect(":memory:")
            self.copy_db(self.disk_connection, self.memory_connection)
            self.batch_writes = []
        else:
            self.memory_connection = None
            self.batch_writes = None

    def flush_to_disk(self):
        """
        Flushes data from the in-memory cache into the disk-backed cache
        """
        logger.info("Flushing %s queries from in-memory cache to disk", len(self.batch_writes))
        rows = self.memory_connection.execute(f"""
        SELECT hash_id, query, raw_query, domain, intent FROM queries
        WHERE rowid IN ({",".join(self.batch_writes)});
        """)
        self.disk_connection.executemany("""
        INSERT OR IGNORE into queries values (?, ?, ?, ?, ?);
        """, rows)
        self.disk_connection.commit()
        self.batch_writes = []

    def __del__(self):
        if self.memory_connection and self.batch_writes:
            self.flush_to_disk()
            self.memory_connection.close()

        self.disk_connection.close()

    def compatible_version(self):
        """
        Checks to see if the cache db file exists and that the data version
        matches the current data version.
        """

        cursor = self.disk_connection.cursor()
        try:
            cursor.execute("""
            SELECT version_number FROM version WHERE version_number=(?);
            """, (ProcessedQuery.version,))
            if len(cursor.fetchall()) == 0:
                # version does not match
                return False
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    @staticmethod
    def get_key(domain, intent, query_text):
        """
        Calculates a hash key for the domain, intent and text of an example.
        This key is required for further interactions with the query cache.

        Args:
            domain(str): The domain of the example
            intent(str): The intent of the example
            query_text(str): The raw text of the example
        Returns:
            str: Hash id representing this query
        """

        h = sha256(domain.encode())
        h.update(b"###")
        h.update(intent.encode())
        h.update(b"###")
        h.update(query_text.encode())
        return h.hexdigest()

    @property
    def connection(self):
        return self.memory_connection or self.disk_connection

    def key_to_row_id(self, key):
        """
        Args:
            key(str): A key generated by the QueryCache.get_key() function
        Returns:
            Optional(Integer): Unique id of the query in the cache if it exists
        """

        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT rowid FROM queries where hash_id=(?);
        """, (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def put(self, key, processed_query):
        """
        Adds a ProcessedQuery to the cache

        Args:
            key(str): A key generated by QueryCache.get_key() for this example
            processed_query(ProcessedQuery): The ProcessedQuery generated for this example
        Returns:
            integer: The unique id of the query in the cache.
        """
        data = json.dumps(processed_query.to_cache())

        def commit_to_db(connection):
            cursor = connection.cursor()
            cursor.execute("""
            INSERT OR IGNORE into queries values (?, ?, ?, ?, ?);
            """, (key,
                  data,
                  processed_query.query.text,
                  processed_query.domain,
                  processed_query.intent,
                  ))
            connection.commit()

        if self.memory_connection:
            commit_to_db(self.memory_connection)
            rowid = self.key_to_row_id(key)
            self.batch_writes.append(str(rowid))
            if len(self.batch_writes) == self.batch_write_size:
                self.flush_to_disk()
        else:
            commit_to_db(self.disk_connection)

        return self.key_to_row_id(key)

    @lru_cache(maxsize=1)
    def get(self, row_id):
        """
        Get a cached ProcessedQuery by id.  Note: this call should never fail as
        it is required that the row_id exist in the database before this method
        is called. Exceptions may be thrown in the case of database corruption.

        The method caches the previously retrieved element because it is common
        for a set of iterators (examples and lables) to retrieve the same row_id
        in sequence.  The cache prevents extra db lookups in this case.

        Args:
            row_id(integer): The unique id returned by QueryCache.key_to_row_id() or
                             QueryCache.put().
        Returns:
            ProcessedQuery: The ProcessedQuery associated with the identifier.
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT query FROM queries WHERE rowid=(?);
        """, (row_id,))
        row = cursor.fetchone()
        return ProcessedQuery.from_cache(json.loads(row[0]))

    def get_value(self, domain, intent, query_text):
        row_id = self.key_to_row_id(self.get_key(domain, intent, query_text))
        return None if row_id is None else self.get(row_id)

    def get_raw_query(self, row_id):
        """
        Get the raw text only from a cached example.  See notes on get().
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT raw_query FROM queries WHERE rowid=(?);
        """, (row_id,))
        return cursor.fetchone()[0]

    def get_query(self, row_id):
        """
        Get the Query from a cached example. See notes on get().
        """
        return self.get(row_id).query

    def get_entities(self, row_id):
        """
        Get entities from a cached example. See notes on get().
        """
        return self.get(row_id).entities

    def get_domain(self, row_id):
        """
        Get the domain only from a cached example.  See notes on get().
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT domain FROM queries WHERE rowid=(?);
        """, (row_id,))
        return cursor.fetchone()[0]

    def get_intent(self, row_id):
        """
        Get the intent only from a cached example.  See notes on get().
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT intent FROM queries WHERE rowid=(?);
        """, (row_id,))
        return cursor.fetchone()[0]
