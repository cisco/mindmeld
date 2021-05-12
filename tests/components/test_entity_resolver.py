#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_entity_resolver
----------------------------------

Tests for `entity_resolver` module.
"""

# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mindmeld.components._elasticsearch_helpers import create_es_client
from mindmeld.components.entity_resolver import EntityResolverFactory, EntityResolver
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
        'model_type': 'resolver',
        "model_settings": {
            "resolver_type": "exact_match",
        }
    }
    resolver = EntityResolverFactory.create_resolver(
        APP_PATH, ENTITY_TYPE, resource_loader=resource_loader,
        es_client=es_client, er_config=er_config
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_elastic_search(resource_loader, es_client):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    er_config = {
        'model_type': 'resolver',
        "model_settings": {
            "resolver_type": "text_relevance",
        }
    }
    resolver = EntityResolverFactory.create_resolver(
        APP_PATH, ENTITY_TYPE, resource_loader=resource_loader,
        es_client=es_client, er_config=er_config
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_sbert(resource_loader):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    er_config = {
        'model_type': 'resolver',
        "model_settings": {
            "resolver_type": "sbert_cosine_similarity",
            "pretrained_name_or_abspath": "distilbert-base-nli-stsb-mean-tokens",
            "batch_size": 16,
            "concat_last_n_layers": 4,
            "normalize_token_embs": True,
            "bert_output_type": "mean",
            "augment_lower_case": False,
            "quantize_model": True,
            "augment_average_synonyms_embeddings": True
        }
    }
    resolver = EntityResolverFactory.create_resolver(
        APP_PATH, ENTITY_TYPE, resource_loader=resource_loader, er_config=er_config
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_tfidf(resource_loader, es_client):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    er_config = {
        'model_type': 'resolver',
        "model_settings": {
            "resolver_type": "tfidf_cosine_similarity",
        }
    }
    resolver = EntityResolverFactory.create_resolver(
        APP_PATH, ENTITY_TYPE, resource_loader=resource_loader,
        es_client=es_client, er_config=er_config
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_default(resource_loader):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    resolver = EntityResolverFactory.create_resolver(
        APP_PATH, ENTITY_TYPE, resource_loader=resource_loader
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_deprecated_configs(resource_loader, es_client):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    er_config = {
        'model_type': 'text_relevance',
    }
    resolver = EntityResolverFactory.create_resolver(
        APP_PATH, ENTITY_TYPE, resource_loader=resource_loader,
        es_client=es_client, er_config=er_config
    )
    resolver.fit()
    return resolver


@pytest.fixture
def resolver_deprecated_class(resource_loader, es_client):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    resolver = EntityResolver(
        APP_PATH, resource_loader, ENTITY_TYPE, es_client=es_client
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


def test_canonical_tfidf(resolver_tfidf):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_tfidf.predict(Entity("Pine and Market", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_canonical_default(resolver_default):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_default.predict(Entity("Pine and Market", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_canonical_deprecated_configs(resolver_deprecated_configs):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_deprecated_configs.predict(Entity("Pine and Market", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_canonical_deprecated_class(resolver_deprecated_class):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_deprecated_class.predict(Entity("Pine and Market", ENTITY_TYPE))[0]
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


def test_synonym_tfidf(resolver_tfidf):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_tfidf.predict(Entity("Pine St", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_synonym_default(resolver_default):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_default.predict(Entity("Pine St", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_synonym_deprecated_configs(resolver_deprecated_configs):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_deprecated_configs.predict(Entity("Pine St", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]


def test_synonym_deprecated_class(resolver_deprecated_class):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = resolver_deprecated_class.predict(Entity("Pine St", ENTITY_TYPE))[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]
