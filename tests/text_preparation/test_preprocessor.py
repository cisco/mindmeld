#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Preprocessor
----------------------------------

Tests for Preprocessors in the `text_preparation.preprocessors` module.
"""

import pytest


@pytest.mark.parametrize(
    "query, expected",
    [
        ("a ghost girl", "a  girl"),
        ("a girl", "a girl"),
        ("a ghosghostt girl", "a  girl"),
    ],
)
def test_preprocessor(query_factory, query, expected):
    processed_query = query_factory.create_query(query)
    assert expected == processed_query.processed_text
