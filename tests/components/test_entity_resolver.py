#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_entity_resolver
----------------------------------

Tests for `entity_resolver` module.
"""
# import mock

# pylint: disable=locally-disabled,redefined-outer-name
import pytest
# from mock import PropertyMock

from mindmeld.components._elasticsearch_helpers import create_es_client
from mindmeld.components.entity_resolver import EntityResolver
from mindmeld.core import Entity

ENTITY_TYPE = "store_name"
APP_PATH = "../kwik_e_mart"


@pytest.fixture
def es_client():
    """An Elasticsearch client"""
    return create_es_client()


@pytest.fixture
def resolver_exact_match(resource_loader, es_client):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    er_config = {
        'model_type': 'exact_match'
    }
    resolver = EntityResolver(
        APP_PATH, resource_loader, ENTITY_TYPE,
        es_client=es_client, er_config=er_config
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_elastic_search(resource_loader, es_client):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    er_config = {
        'model_type': 'text_relevance',
        'phonetic_match_types': [
            # "double_metaphone"
        ],
    }
    resolver = EntityResolver(
        APP_PATH, resource_loader, ENTITY_TYPE,
        es_client=es_client, er_config=er_config
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_sbert(resource_loader):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    er_config = {
        'model_type': 'sbert_cosine_similarity',
        'model_settings': {
            "batch_size": 16,
        }
    }
    resolver = EntityResolver(
        APP_PATH, resource_loader, ENTITY_TYPE, er_config=er_config
    )
    resolver.fit()
    return resolver


def test_canonical_exact_match(resolver_exact_match):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_exact_match.predict(Entity("Pine and Market", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_canonical_elastic_search(resolver_elastic_search):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_elastic_search.predict(Entity("Pine and Market", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


@pytest.mark.extras
@pytest.mark.bert
def test_canonical_sbert(resolver_sbert):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_sbert.predict(Entity("Pine and Market", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_synonym_elastic_search(resolver_elastic_search):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_elastic_search.predict(Entity("Pine St", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


@pytest.mark.extras
@pytest.mark.bert
def test_synonym_sbert(resolver_sbert):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_sbert.predict(Entity("Pine St", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]
