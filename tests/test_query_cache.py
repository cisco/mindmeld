#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_query_cache
----------------------------------

Tests the generated query cache has to correct key value types.

"""
# pylint: disable=locally-disabled,redefined-outer-name
import os
import sqlite3
from tempfile import TemporaryDirectory
from unittest.mock import patch
from hashlib import sha256

from mindmeld.core import ProcessedQuery
from mindmeld.query_cache import QueryCache
from mindmeld.text_preparation.text_preparation_pipeline import TextPreparationPipelineFactory


def test_query_cache_has_the_correct_format(kwik_e_mart_app_path):
    text_prep_pipeline = TextPreparationPipelineFactory.create_from_app_path(kwik_e_mart_app_path)
    cache = QueryCache(kwik_e_mart_app_path, text_prep_pipeline.get_hashid())
    key = QueryCache.get_key("store_info", "help", "User manual")
    row_id = cache.key_to_row_id(key)
    assert row_id is not None
    pq = cache.get(row_id)
    assert type(pq) == ProcessedQuery
    assert pq.domain == "store_info"
    assert pq.intent == "help"


def compare_dbs(db1, db2):
    pass


def get_query_from_disk(tmpdir, key):
    conn = sqlite3.connect(f"{tmpdir}/.generated/query_cache.db")
    res = conn.execute("""
    SELECT domain, intent, raw_query FROM queries WHERE hash_id=(?);
    """, (key,)).fetchone()
    conn.close()
    return tuple(res) if res else None


def test_disk_query_cache(processed_queries):
    environ = {"MM_QUERY_CACHE_IN_MEMORY": "0"}

    with TemporaryDirectory() as tmpdir, patch.dict(os.environ, environ):
        text_prep_pipeline = TextPreparationPipelineFactory.create_from_app_path(tmpdir)
        cache = QueryCache(tmpdir, text_prep_pipeline.get_hashid())

        # Verify that there is no in-memory caching
        assert cache.memory_connection is None
        for q in processed_queries:
            key = QueryCache.get_key(q.domain, q.intent, q.query.text)
            cache.put(key, q)
            # Verify that queries are written to disk immediately
            assert get_query_from_disk(tmpdir, key) == (q.domain, q.intent, q.query.text)


def test_memory_query_cache(processed_queries):
    environ = {
        "MM_QUERY_CACHE_IN_MEMORY": "1",
        "MM_QUERY_CACHE_WRITE_SIZE": "10"
    }

    with TemporaryDirectory() as tmpdir, patch.dict(os.environ, environ):
        text_prep_pipeline = TextPreparationPipelineFactory.create_from_app_path(tmpdir)
        cache = QueryCache(tmpdir, text_prep_pipeline.get_hashid())

        # Verify that there is in-memory caching
        assert cache.memory_connection is not None
        for q in processed_queries[:9]:
            key = QueryCache.get_key(q.domain, q.intent, q.query.text)
            cache.put(key, q)
            # Verify that the first 9 queries are not written to disk
            assert not get_query_from_disk(tmpdir, key)

        # Verify that the 10th query triggers a flush to disk
        q = processed_queries[9]
        key = QueryCache.get_key(q.domain, q.intent, q.query.text)
        cache.put(key, q)
        for q in processed_queries[:10]:
            key = QueryCache.get_key(q.domain, q.intent, q.query.text)
            assert get_query_from_disk(tmpdir, key) == (q.domain, q.intent, q.query.text)

        # Verify that a GC triggers a flush to disk
        for q in processed_queries[10:15]:
            key = QueryCache.get_key(q.domain, q.intent, q.query.text)
            cache.put(key, q)
            assert not get_query_from_disk(tmpdir, key)
        cache = None
        for q in processed_queries[10:15]:
            key = QueryCache.get_key(q.domain, q.intent, q.query.text)
            assert get_query_from_disk(tmpdir, key)
