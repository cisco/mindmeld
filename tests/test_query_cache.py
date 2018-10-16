#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dialogue
----------------------------------

Tests for app_manager module.

These tests apply only when async/await are supported.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import os
from sklearn.externals import joblib

QUERY_CACHE_RELATIVE_PATH = '.generated/query_cache.pkl'


def test_query_cache_has_the_correct_format(kwik_e_mart_app_path):
    query_cache_location = os.path.join(kwik_e_mart_app_path, QUERY_CACHE_RELATIVE_PATH)
    query_cache = joblib.load(query_cache_location)
    assert ('store_info', 'help', 'User manual') in query_cache
    assert query_cache[('store_info', 'help', 'User manual')].domain == 'store_info'
    assert query_cache[('store_info', 'help', 'User manual')].intent == 'help'
