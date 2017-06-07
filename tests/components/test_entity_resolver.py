#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_entity_resolver
----------------------------------

Tests for `entity_resolver` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import pytest

from mmworkbench.core import Entity

from mmworkbench.components.entity_resolver import EntityResolver

ENTITY_TYPE = 'location'
APP_PATH = '../kwik_e_mart'


@pytest.fixture
def resolver(resource_loader):
    """An entity resolver for 'location' on the Kwik-E-Mart app"""
    resolver = EntityResolver(APP_PATH, resource_loader, ENTITY_TYPE)
    resolver.fit()
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
