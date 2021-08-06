#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Stemmers
----------------------------------

Tests for Stemmers in the `text_preparation.stemmers` module.
"""
import pytest
from mindmeld.text_preparation.stemmers import (
    StemmerFactory,
    EnglishNLTKStemmer,
    NoOpStemmer,
    SnowballNLTKStemmer,
)
from mindmeld.query_factory import QueryFactory


@pytest.mark.parametrize(
    "language_code,stemmer_class",
    [
        ("en", EnglishNLTKStemmer),
        ("EN", EnglishNLTKStemmer),
        ("es", SnowballNLTKStemmer),
        ("AFG", NoOpStemmer),
        ("INVALID_CODE", NoOpStemmer),
    ],
)
def test_get_stemmer_by_language(language_code, stemmer_class):
    stemmer = StemmerFactory.get_stemmer_by_language(language_code)
    assert isinstance(stemmer, stemmer_class)


test_data_not_stemmed = [
    "airliner",
    "gyroscopic",
    "adjustable",
    "defensible",
    "irritant",
    "replacement",
    "adjustment",
    "dependent",
    "adoption",
    "communism",
    "activate",
    "effective",
    "bowdlerize",
    "manager",
    "proceed",
    "exceed",
    "succeed",
    "outing",
    "inning",
    "news",
    "sky",
]


@pytest.mark.parametrize("query", test_data_not_stemmed)
def test_nlp_for_non_stemmed_queries(kwik_e_mart_app_path, query):
    """Tests queries that are NOT in the training data but have their stemmed
    versions in the training data"""
    query_factory = QueryFactory.create_query_factory(kwik_e_mart_app_path)
    stemmed_tokens = query_factory.create_query(text=query).stemmed_tokens
    assert query == stemmed_tokens[0]


test_data_need_stemming = [
    ("cancelled", "cancel"),
    ("aborted", "abort"),
    ("backwards", "backward"),
    ("exitted", "exit"),
    ("finished", "finish"),
]


@pytest.mark.parametrize("query,stemmed_query", test_data_need_stemming)
def test_nlp_for_stemmed_queries(kwik_e_mart_app_path, query, stemmed_query):
    """Tests queries that are NOT in the training data but have their stemmed
    versions in the training data"""
    query_factory = QueryFactory.create_query_factory(kwik_e_mart_app_path)
    stemmed_tokens = query_factory.create_query(text=query).stemmed_tokens
    assert stemmed_query == stemmed_tokens[0]


test_data_stemmed = ["cancelled", "exited", "aborted"]


@pytest.mark.parametrize("query", test_data_stemmed)
def test_nlp_hierarchy_for_stemmed_queries(kwik_e_mart_nlp, query):
    """Tests queries that are NOT in the training data but have their stemmed
    versions in the training data"""
    response = kwik_e_mart_nlp.process(query)
    assert response["text"] == query
    assert response["domain"] == "store_info"
    assert response["intent"] == "exit"


def test_no_op_stemmer():
    assert NoOpStemmer().stem_word("Running") == "Running"


@pytest.mark.parametrize(
    "language_code, language",
    [("en", "English"), ("far", "Fataleka"), ("fr", "French")],
)
def test_get_language_from_language_code(language_code, language):
    assert (
        language == StemmerFactory.get_language_from_language_code(language_code).name
    )
