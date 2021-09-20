#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_resource_loader
----------------------------------

Tests functionality of the resource_loader module

"""
import pytest

from mindmeld.core import ProcessedQuery, Query, NestedEntity
from mindmeld.query_cache import QueryCache
from mindmeld.resource_loader import ProcessedQueryList


def test_get_labeled_queries(resource_loader):
    qmap = resource_loader.get_labeled_queries()
    # verify the domains are correct
    assert set(qmap.keys()) == {"banking", "store_info"}

    # verify the intents are correct
    assert set(qmap["banking"].keys()) == {"transfer_money"}
    assert set(qmap["store_info"].keys()) == {"greet", "get_store_number",
                                              "find_nearest_store", "exit",
                                              "get_store_hours", "help"}


def test_flatten_query_tree(resource_loader):
    qmap = resource_loader.get_labeled_queries()
    count = 0
    for intent_dict in qmap.values():
        for query_ids in intent_dict.values():
            count += len(query_ids)
    flattened = resource_loader.flatten_query_tree(qmap)
    # verify that all queries in tree are flattened
    assert count == len(flattened)

    # verify that query trees built from different query_caches will raise an error
    hash_id = resource_loader.query_factory.text_preparation_pipeline.get_hashid()
    qmap["banking"]["transfer_money"].cache = QueryCache(app_path=resource_loader.app_path,
                                                         schema_version_hash=hash_id)
    with pytest.raises(ValueError):
        flattened = resource_loader.flatten_query_tree(qmap)


def _run_tests_on_pql(queries):
    # verify the correct types are returned by the iterators
    assert isinstance(next(queries.processed_queries()), ProcessedQuery)
    assert isinstance(next(queries.raw_queries()), str)
    assert isinstance(next(queries.queries()), Query)
    assert isinstance(next(queries.entities())[0], NestedEntity)
    assert isinstance(next(queries.domains()), str)
    assert isinstance(next(queries.intents()), str)
    int_iterator = ProcessedQueryList.ListIterator(list(range(len(queries))))
    assert isinstance(next(int_iterator), int)

    # verify the domains are correct
    assert set(queries.domains()) == {"banking", "store_info"}

    # verify the intents are correct
    assert set(queries.intents()) == {"transfer_money", "greet", "get_store_number",
                                      "find_nearest_store", "exit", "get_store_hours", "help"}

    def test_reordering(iterator):
        # verify that reordering works properly
        indices = list(range(len(iterator)))
        indices.reverse()
        _reversed = list(iterator)
        _reversed.reverse()
        iterator.reorder(indices)
        for q1, q2 in zip(_reversed, iterator):
            assert q1 == q2

    # uncached iterator
    test_reordering(queries.raw_queries())
    # cached iterator
    test_reordering(queries.intents())
    # custom list iterator
    test_reordering(int_iterator)


def test_processed_query_list(resource_loader):
    queries = resource_loader.get_flattened_label_set()
    # verify that the query cache is correct
    assert queries.cache == resource_loader.query_cache

    # test sqlite backed list
    _run_tests_on_pql(queries)

    # test in-memory list
    mem_pql = ProcessedQueryList.from_in_memory_list(
        list(queries.processed_queries())
    )
    assert isinstance(mem_pql.cache, ProcessedQueryList.MemoryCache)
    _run_tests_on_pql(mem_pql)
