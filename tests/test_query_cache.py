#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_query_cache
----------------------------------

Tests the generated query cache has to correct key value types.

"""
# pylint: disable=locally-disabled,redefined-outer-name
import os

from sklearn.externals import joblib

from mindmeld.core import ProcessedQuery
from mindmeld.query_cache import QueryCache


def test_query_cache_has_the_correct_format(kwik_e_mart_app_path):
    cache = QueryCache(kwik_e_mart_app_path)
    key = cache.get_key('store_info', 'help', 'User manual')
    row_id = cache.key_to_row_id(key)
    assert row_id is not None
    pq = cache.get(row_id)
    assert type(pq) == ProcessedQuery
    assert pq.domain == 'store_info'
    assert pq.intent == 'help'
