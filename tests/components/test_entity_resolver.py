#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_entity_resolver
----------------------------------

Tests for `entity_resolver` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mmworkbench.core import Entity

from mmworkbench.components.entity_resolver import EntityResolver
from mmworkbench.components._elasticsearch_helpers import create_es_client

ENTITY_TYPE = 'store_name'
APP_PATH = '../kwik_e_mart'


@pytest.fixture
def es_client():
    """An Elasticsearch client"""
    return create_es_client()


@pytest.fixture
def resolver(resource_loader, es_client):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    resolver = EntityResolver(APP_PATH, resource_loader, ENTITY_TYPE, es_client=es_client)
    resolver.fit()
    es_client.indices.flush(index='_all')
    return resolver


def test_canonical(resolver):
    """Tests that entity resolution works for a canonical entity in the map"""
    expected = {'id': '2', 'cname': 'Pine and Market'}
    predicted = resolver.predict(Entity('Pine and Market', ENTITY_TYPE))[0]
    print(predicted)
    assert predicted['id'] == expected['id']
    assert predicted['cname'] == expected['cname']


def test_synonym(resolver):
    """Tests that entity resolution works for an entity synonym in the map"""
    expected = {'id': '2', 'cname': 'Pine and Market'}
    predicted = resolver.predict(Entity('Pine St', ENTITY_TYPE))[0]
    assert predicted['id'] == expected['id']
    assert predicted['cname'] == expected['cname']
